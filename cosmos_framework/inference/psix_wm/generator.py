"""Image-WM generation from a checkpoint: a condition frame and a caption in, five frames out.

Used by offline evaluation, goal-video generation and the ``/wm`` server, through the stock
Cosmos inference path. The condition frame is fitted to the canvas as in training (aspect kept,
reflection-padded) and passed as a lossless PNG; the outputs are cropped back to that content.
Two speed-ups of ``generate_samples_from_batch`` are on: ``batched_cfg`` (both CFG branches in one
forward) and ``und_kv_cache`` (the text tower computed once per request).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from cosmos_framework.inference.psix_wm.sampler import NEGATIVE_PROMPT, Sampler
from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import _SYSTEM_PROMPT_VIDEO
from cosmos_framework.psix_wm.caption import NUM_FRAMES
from cosmos_framework.psix_wm.checkpoint import latest_checkpoint
from cosmos_framework.utils import log


class Generator:
    """One loaded checkpoint: ``checkpoint_path`` or, for a run's folder, its latest one, with the
    ``config.yaml`` of its folder unless ``config_file`` names another.

    ``quantize="fp8"`` runs the language model's Linears in fp8, weights and matmuls (about half the
    memory, and faster where batch-1 inference is compute-bound); ``compile`` compiles the language
    model and captures CUDA graphs (a slow first request, then much faster ones).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_file: str | Path | None,
        output_dir: Path,
        *,
        quantize: str = "none",
        compile: bool = False,
        max_batch: int = 1,
    ) -> None:
        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.common.init import init_output_dir

        if quantize not in ("none", "fp8"):
            raise ValueError(f"quantize must be 'none' or 'fp8', got {quantize!r}")
        if compile and "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            raise ValueError(
                "compile captures CUDA graphs, which fail under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            )
        checkpoint = latest_checkpoint(checkpoint_path)
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        batching = {"max_num_seqs": max_batch} if max_batch > 1 else {}
        setup = OmniSetupOverrides(
            checkpoint_path=str(checkpoint),
            config_file=str(config_file or checkpoint.parents[1] / "config.yaml"),
            output_dir=output_dir,
            guardrails=False,
            parallelism_preset="latency",
            sampler="unipc",
            warmup=0,
            use_ema_weights=False,
            use_torch_compile=compile,
            use_cuda_graphs=compile,
            compiled_region="language",
            **batching,
        ).build_setup()
        init_output_dir(output_dir, resume=True, job_name="psix_image_world_model")
        self.pipe = setup.get_inference_cls().create(setup)
        self._output_dir = output_dir
        model = self.pipe.model
        model.eval()
        self.resolution = str(model.config.resolution)
        _release_text_head(model)
        if quantize == "fp8":
            _quantize_fp8(model)
        else:  # fp8 already frees more than the vocabulary table would
            _embedding_to_host(model)

    def canvas(self, image_shape: tuple[int, int]) -> tuple[int, int]:
        """(W, H) the model generates at for frames of ``image_shape`` (H, W), as in training."""

        from cosmos_framework.psix_wm.queries import training_canvas

        return training_canvas(image_shape, self.resolution)

    def generate(
        self, items: list[tuple[Image.Image, str, float, int]], *, canvas: tuple[int, int], sampler: Sampler
    ) -> list[list[Image.Image]]:
        """The five frames for each ``(condition, caption, mrope_fps, seed)``, generated at ``canvas``."""

        import torch

        from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
        from cosmos_framework.inference.inference import batch_sample_data

        model = self.pipe.model
        aspect = next(
            (key for key, size in VIDEO_RES_SIZE_INFO[self.resolution].items() if tuple(size) == tuple(canvas)), None
        )
        if aspect is None:
            raise ValueError(f"canvas {canvas} is not a size of resolution {self.resolution}")

        # Per item: (the condition on the canvas, the (W, H) of its content).
        fitted = [fit_to_canvas(condition, canvas) for condition, *_ in items]

        # Condition PNGs go to local memory when there is some: one file per sample.
        with (
            torch.no_grad(),
            tempfile.TemporaryDirectory(dir="/dev/shm" if os.path.isdir("/dev/shm") else None) as frames_dir,
        ):
            samples = [  # (sample args, sample data) per item
                self._sample(
                    Path(frames_dir) / f"condition_{index}.png",
                    fitted[index][0],
                    caption,
                    mrope_fps,
                    seed,
                    aspect=aspect,
                    sampler=sampler,
                )
                for index, (_, caption, mrope_fps, seed) in enumerate(items)
            ]
            batch = batch_sample_data([data for _, data in samples], model)
            outputs = model.generate_samples_from_batch(
                batch,
                sampler=None,
                guidance=sampler.guidance,
                seed=[seed for *_, seed in items],
                num_steps=sampler.num_steps,
                shift=sampler.shift,
                sigma_max=samples[0][0].sigma_max,
                has_negative_prompt=True,
                n_sample=len(items),
                normalize_cfg=False,
                batched_cfg=True,
                und_kv_cache=True,
                upsample_task=None,
            )
            video = model.decode(torch.cat(outputs["vision"], dim=0))  # [B, C, T, H, W] in [-1, 1]
            pixels = ((1.0 + video) / 2).clamp(0, 1).permute(0, 2, 3, 4, 1).float().mul(255).round().to(torch.uint8)
        return [
            [Image.fromarray(frame[:height, :width], mode="RGB") for frame in clip]
            for (_, (width, height)), clip in zip(fitted, pixels.cpu().numpy())
        ]

    def _sample(
        self,
        condition_path: Path,
        condition: Image.Image,
        caption: str,
        mrope_fps: float,
        seed: int,
        *,
        aspect: str,
        sampler: Sampler,
    ) -> tuple[Any, dict]:
        """The stock sample args and sample data of one item, its condition saved to ``condition_path``."""

        import torch

        from cosmos_framework.inference.args import ModelMode, NegativeMetadataMode, OmniSampleOverrides
        from cosmos_framework.inference.inference import get_sample_data

        model = self.pipe.model
        condition.save(condition_path)
        args = OmniSampleOverrides(
            name="wm",
            output_dir=self._output_dir,
            model_mode=ModelMode.IMAGE2VIDEO,
            prompt=caption,
            prompt_upsampling=False,
            negative_prompt=NEGATIVE_PROMPT,
            negative_metadata_mode=NegativeMetadataMode.NONE,
            negative_prompt_keep_metadata=False,
            vision_path=str(condition_path),
            condition_frame_indexes_vision=[0],
            resolution=self.resolution,
            aspect_ratio=aspect,
            fps=24,  # the output video's frame rate; the model is timed by the mRoPE rate set below
            num_frames=NUM_FRAMES,
            num_steps=sampler.num_steps,
            guidance=sampler.guidance,
            shift=sampler.shift,
            normalize_cfg=False,
            seed=seed,
            enable_sound=False,
        ).build_sample(model_config=self.pipe.model_config)
        data = get_sample_data(sample_args=args, model=model)
        # As in training: the caption's exact bytes and the clip's mRoPE rate.
        data[model.input_caption_key] = [caption]
        if model.config.vlm_config.use_system_prompt:  # checkpoints trained with Qwen's video system prompt
            data["system_prompt"] = _SYSTEM_PROMPT_VIDEO
        data["conditioning_fps"] = torch.full_like(data["conditioning_fps"], mrope_fps)
        return args, data


def fit_to_canvas(image: Image.Image, canvas: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    """``image`` as training fits a frame to the (W, H) canvas: resized with its aspect kept (never up),
    then reflection-padded on the bottom and right. Returns the canvas image and the (W, H) of its content."""

    import numpy as np
    import torch

    from cosmos_framework.data.vfm.action.transforms import reflection_pad_to_target

    pixels = torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(1)  # [3, 1, H, W] uint8
    data = reflection_pad_to_target({"video": pixels}, ["video"], True, target_w=canvas[0], target_h=canvas[1])
    _, _, content_h, content_w = (int(v) for v in data["image_size"])
    return Image.fromarray(data["video"][:, 0].permute(1, 2, 0).numpy(), mode="RGB"), (content_w, content_h)


def _release_text_head(model: Any) -> None:
    """The LM head never runs here, so its memory is freed."""

    for name, _ in list(model.named_modules()):
        parent, _, attribute = name.rpartition(".")
        if attribute == "lm_head":
            setattr(model.get_submodule(parent) if parent else model, attribute, None)


def _quantize_fp8(model: Any) -> None:
    """fp8 weights and fp8 matmuls for the language model's Linears; the VAE stays in bf16."""

    import torch
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

    class EmptySafeLinear(torch.nn.Module):
        """A quantized Linear; a zero-row input (the cached text lane) has no maximum to scale by."""

        def __init__(self, linear: torch.nn.Module) -> None:
            super().__init__()
            self.linear = linear

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.numel() == 0:
                return x.new_empty(*x.shape[:-1], self.linear.out_features)
            return self.linear(x)

    for name, child in model.named_children():
        if (
            "tokenizer" not in name.lower()
            and "vae" not in name.lower()
            and any(isinstance(m, torch.nn.Linear) for m in child.modules())
        ):
            quantize_(child, Float8DynamicActivationFloat8WeightConfig())
    quantized = [  # a quantized weight is a torchao tensor, no longer a plain Parameter
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and type(module.weight) is not torch.nn.Parameter
    ]
    if not quantized:
        raise RuntimeError("fp8: torchao quantized no Linear")
    for name in quantized:
        parent, _, attribute = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        setattr(owner, attribute, EmptySafeLinear(getattr(owner, attribute)))
    torch.cuda.empty_cache()
    log.info(f"fp8 weights and matmuls; resident {torch.cuda.memory_allocated() / 2**30:.2f} GiB")


def _embedding_to_host(model: Any) -> None:
    """Keep the vocabulary table in host memory (a lookup, not a matmul), returning device tensors."""

    import torch

    class HostEmbedding(torch.nn.Module):
        def __init__(self, embedding: torch.nn.Embedding, device: torch.device) -> None:
            super().__init__()
            self.embedding, self.target_device = embedding, device

        @property
        def weight(self) -> torch.Tensor:
            return self.embedding.weight

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embedding(input_ids.to("cpu")).to(self.target_device)

    for name, module in list(model.named_modules()):
        parent, _, attribute = name.rpartition(".")
        if attribute == "embed_tokens" and isinstance(module, torch.nn.Embedding) and module.weight.is_cuda:
            device = module.weight.device
            setattr(
                model.get_submodule(parent) if parent else model, attribute, HostEmbedding(module.to("cpu"), device)
            )
    torch.cuda.empty_cache()
