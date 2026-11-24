"""Cosmos3-Nano Image-WM SFT, one experiment per run TOML in ``$PSIX_WM_RUN_TOMLS``
(default ``examples/toml/sft_config/psix_image_wm/``).

Each run TOML names its experiment (``[job].experiment``), holds its data under ``[custom.data]``
(``cosmos_framework/psix_wm/data_config.py``) and its training settings under ``[custom.train]``
(``TrainSettings``). A sample is five frames, the current observation and four futures. The 4x causal
VAE turns them into two latents and only latent 1 carries flow-matching loss.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import tomllib
from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.psix_wm_val_metrics import PsixWmValMetricsCallback
from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.data.vfm.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.inference.psix_wm.sampler import PRESETS
from cosmos_framework.psix_wm.data_config import Table, load
from cosmos_framework.psix_wm.mixture import build_mixture
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.lazy_config import LazyCall as L

# Every run TOML in this directory registers an experiment; another repository can point it at its own.
RUN_TOMLS = Path(
    os.environ.get("PSIX_WM_RUN_TOMLS", Path(__file__).resolve().parents[5] / "examples/toml/sft_config/psix_image_wm")
)

# Only the generation tower trains: the parameters whose names contain one of these.
GEN_TOWER_KEYS = ["moe_gen", "time_embedder", "vae2llm", "llm2vae"]

# The job's output directory (JobConfig.path_local), where a resumed run finds its latest checkpoint.
_RUN_DIR = "${oc.env:IMAGINAIRE_OUTPUT_ROOT,/tmp/imaginaire4-output}/${job.project}/${job.group}/${job.name}"

# Per training resolution, the resolution keys whose VAE chunks are 4 frames and whose training shift is 1:
# its own, and the two that its canvases are looked up under (448x256 as "256", 384x288 as "480").
_RESOLUTION_KEYS = {"448": ("256", "480", "448")}

# Diagnostics that cost thousands of W&B keys per run and are not read.
_SILENCED_CALLBACKS = (
    "norm_monitor",
    "training_stats",
    "device_monitor",
    "sigma_loss_analysis",
    "sequence_packing_padding",
    "mfu",
    "ofu",
    "expert_heatmap",
    "moe_specialization",
    "moe_stability",
)


class TrainSettings(Table):
    """``[custom.train]`` of a run TOML: training speed-ups."""

    detach_frozen_und: bool = True  # no backward through the frozen text lane (trainable gradients unchanged)
    vae_encode_batch: int = 32  # samples of one shape VAE-encoded per call
    fsdp_replicate_frozen: bool = True  # frozen weights replicated on every rank instead of sharded (no all-gather)
    fsdp_reshard_after_forward: bool = False  # False: block weights stay gathered from forward to backward


class NoOpCallback(Callback):
    """A callback that does nothing and accepts any arguments, to switch a configured one off."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        super().__init__()


def build_experiment(run_toml: Path, raw: dict) -> dict:
    data = load(run_toml)
    train = TrainSettings.read(run_toml, "train")
    if raw.get("optimizer", {}).get("keys_to_select", GEN_TOWER_KEYS) != GEN_TOWER_KEYS and (
        train.detach_frozen_und or train.fsdp_replicate_frozen
    ):
        raise ValueError(
            f"{run_toml}: detach_frozen_und and fsdp_replicate_frozen need only the generation tower to train"
        )
    if data.validation.sampler not in PRESETS:
        raise ValueError(f"{run_toml}: validation.sampler must be one of {sorted(PRESETS)}")
    if data.resolution not in _RESOLUTION_KEYS:
        raise ValueError(f"{run_toml}: resolution {data.resolution!r} is not one of {sorted(_RESOLUTION_KEYS)}")
    sample_kwargs = dict(tokenizer_config="${model.config.vlm_config.tokenizer}", video_backend="pyav")
    config = copy.deepcopy(vision_sft_nano)
    config["job"]["group"] = "psix_image_wm"
    config["job"]["name"] = data.name

    # Only the generation tower trains. FusedAdam keeps FP32 master weights for the BF16 net.
    optimizer = config["optimizer"]
    optimizer["optimizer_type"] = "FusedAdam"
    optimizer["eps"] = 1.0e-8
    optimizer["weight_decay"] = 1.0e-8
    optimizer["keys_to_select"] = list(GEN_TOWER_KEYS)

    trainer = config["trainer"]
    trainer["grad_accum_iter"] = 1
    callbacks = trainer["callbacks"]
    callbacks["grad_clip"]["clip_norm"] = 1.0
    callbacks["every_n_sample_reg"] = dict(save_s3=False)
    callbacks["every_n_sample_ema"] = dict(save_s3=False)
    for name in _SILENCED_CALLBACKS:
        callbacks[name] = L(NoOpCallback)()
    validation = data.validation
    if data.held_out:
        callbacks["psix_wm_val_metrics"] = L(PsixWmValMetricsCallback)(
            run_toml=str(run_toml),
            dataset_kwargs=dict(resolution=data.resolution, **sample_kwargs),
            every_n=validation.every,
            generation_every_n=validation.generate_every,
            num_steps=PRESETS[validation.sampler].num_steps,
            guidance=PRESETS[validation.sampler].guidance,
            shift=PRESETS[validation.sampler].shift,
        )

    model = config["model"]["config"]
    model["resolution"] = data.resolution
    model["ema"]["enabled"] = False
    # freeze_und only informs FLOP accounting; detach_frozen_und is what skips the text lane's backward.
    model["vlm_config"]["model_instance"]["config"]["freeze_und"] = True
    model["vlm_config"]["model_instance"]["config"]["text_config_overrides"] = {
        "detach_frozen_und": train.detach_frozen_und
    }
    model["vae_encode_batch"] = train.vae_encode_batch
    model["parallelism"]["fsdp_reshard_after_forward"] = train.fsdp_reshard_after_forward
    model["parallelism"]["fsdp_trainable_keys"] = "${optimizer.keys_to_select}" if train.fsdp_replicate_frozen else []
    resolution_keys = _RESOLUTION_KEYS[data.resolution]
    tokenizer = model["tokenizer"]
    tokenizer["encode_chunk_frames"] = {**tokenizer["encode_chunk_frames"], **dict.fromkeys(resolution_keys, 4)}
    tokenizer["encode_exact_durations"] = [5]
    flow = model["rectified_flow_training_config"]
    flow["shift"] = {**flow["shift"], **dict.fromkeys(resolution_keys, 1)}
    flow["train_time_video_distribution"] = "uniform"
    # Latent 0 is the clean condition; average the loss over latent 1 only.
    flow["normalize_loss_by_active"] = True

    config["dataloader_train"] = L(PackingDataLoader)(
        audio_sample_rate=48000,
        dataset_name="psix_image_wm_lerobot",
        max_samples_per_batch=1,
        max_sequence_length=None,
        patch_spatial=2,
        sound_latent_fps=0,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        dataloader=L(RankPartitionedDataLoader)(
            batch_size=4,
            in_order=True,
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=16,
            sampler=None,
            datasets=dict(
                psix_wm_mix=dict(
                    ratio=1,
                    dataset=L(build_mixture)(run_toml=str(run_toml), **sample_kwargs, run_dir=_RUN_DIR),
                ),
            ),
        ),
    )
    return config


def register_run_tomls(run_tomls: Path) -> dict[str, Path]:
    """Register the experiment of every run TOML in ``run_tomls``; a TOML that fails is logged and skipped,
    so it does not break importing the other experiments."""

    registered: dict[str, Path] = {}
    for run_toml in sorted(run_tomls.glob("*.toml")):
        try:
            raw = tomllib.loads(run_toml.read_text())
            experiment = raw["job"]["experiment"]
            if experiment in registered:
                raise ValueError(f"{registered[experiment]} declares experiment {experiment!r} too")
            node = build_experiment(run_toml, raw)
        except Exception as error:
            log.error(f"Psi0.1 image WM: run TOML {run_toml} is not registered: {type(error).__name__}: {error}")
            continue
        registered[experiment] = run_toml
        ConfigStore.instance().store(group="experiment", package="_global_", name=experiment, node=node)
    return registered


register_run_tomls(RUN_TOMLS)
