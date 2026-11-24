"""In-loop validation of the Image-WM on the run's fixed validation samples (``psix_wm.validation``).

Every ``every_n`` steps it computes the training loss (``training_step`` under ``no_grad``, with a
fixed noise seed) on every sample, and every ``generation_every_n`` steps it generates the first
samples and scores them (``psix_wm.metrics``). Logged to W&B under ``inloop_val/``:

    loss, loss/<set>, loss/<bucket>
    gen/<metric>, gen/<set>/<metric>, gen/<bucket>/<metric>

A failure on any rank disables the callback on every rank; training continues.
"""

from __future__ import annotations

import traceback
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.data.vfm.dataflow.collators import VFMListCollator
from cosmos_framework.inference.psix_wm.sampler import NEGATIVE_PROMPT
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.psix_wm.data_config import load
from cosmos_framework.psix_wm.metrics import LAST_FRAME_METRICS, frame_metrics, score_array
from cosmos_framework.psix_wm.validation import bucket_of, load_plan, plan_table, prepare_plan
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import distributed, log, misc


class PsixWmValMetricsCallback(EveryN):
    def __init__(
        self,
        run_toml: str,
        dataset_kwargs: dict,
        every_n: int,
        generation_every_n: int,
        num_steps: int,
        guidance: float,
        shift: float,
    ) -> None:
        super().__init__(every_n=every_n, barrier_after_run=True)
        self.run_toml = run_toml
        self.dataset_kwargs = dataset_kwargs
        self.generation_every_n = generation_every_n
        self.num_steps, self.guidance, self.shift = num_steps, guidance, shift
        self._disabled = False
        self._plan: dict | None = None
        self._local: list[dict] = []
        self._items: list[dict] = []
        self._n_generation_local = 0

    # ------------------------------------------------------------------ preparation
    def _any_rank_failed(self, failed: bool) -> bool:
        """Whether any rank failed; every rank calls it, so all of them take the same branch after it."""

        if not (dist.is_available() and dist.is_initialized()):
            return failed
        flag = torch.tensor([int(failed)], device="cuda", dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _disable(self, reason: str) -> None:
        log.critical(f"{type(self).__name__}: disabled ({reason}); training continues", rank0_only=False)
        self._disabled, self._plan, self._local, self._items = True, None, [], []

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        failed = False
        try:
            self._prepare()
        except Exception:
            log.critical(f"{type(self).__name__}: preparation failed:\n{traceback.format_exc()}", rank0_only=False)
            failed = True
        if self._any_rank_failed(failed):
            self._disable("preparation failed")

    def _prepare(self) -> None:
        from cosmos_framework.psix_wm.dataset import ImageWMDataset

        config = load(self.run_toml)
        failed = False
        if distributed.get_rank() == 0:
            try:
                prepare_plan(config)
            except Exception:
                log.critical(f"{type(self).__name__}: the validation plan failed:\n{traceback.format_exc()}")
                failed = True
        if self._any_rank_failed(failed):  # also the barrier: the other ranks wait for rank 0's plan
            raise RuntimeError("rank 0 could not prepare the validation plan")
        plan = load_plan(config)
        samples, n_generation = plan["samples"], plan["n_generation"]
        rank, world = distributed.get_rank(), max(1, distributed.get_world_size())

        # Every rank runs the same number of forwards: a short rank repeats a sample, marked as padding.
        def own(positions: list[int], per_rank: int, fallback: int) -> list[dict]:
            owned = [dict(samples[p], _pad=False) for p in positions]
            while len(owned) < per_rank:
                owned.append(dict(owned[-1] if owned else samples[fallback], _pad=True))
            return owned

        per_rank_generation = -(-n_generation // world)
        per_rank_rest = -(-(len(samples) - n_generation) // world)
        self._local = own(list(range(rank, n_generation, world)), per_rank_generation, 0) + own(
            list(range(n_generation + rank, len(samples), world)), per_rank_rest, min(n_generation, len(samples) - 1)
        )
        self._n_generation_local = per_rank_generation

        by_set: dict[str, dict[str, dict]] = {}
        for entry in self._local:
            by_set.setdefault(entry["set"], {})[entry["query_id"]] = {k: v for k, v in entry.items() if k != "_pad"}
        datasets = {}
        for name, rows in by_set.items():
            dataset = ImageWMDataset(plan_table(config, name, list(rows.values())), **self.dataset_kwargs)
            datasets[name] = (dataset, {query_id: i for i, query_id in enumerate(dataset.query_ids())})
        # Decoded once; every trigger after that is GPU work only.
        self._items = [datasets[e["set"]][0][datasets[e["set"]][1][e["query_id"]]] for e in self._local]
        self._plan = plan
        log.info(
            f"{type(self).__name__}: {len(samples)} validation samples ({n_generation} generated), "
            f"{len(self._local)} on each rank"
        )

    # ------------------------------------------------------------------ firing
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict,
        output_batch: dict,
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if self._disabled:
            return
        failed, measured = False, ([], [])
        try:
            measured = self._measure(model, iteration)
        except Exception:
            log.critical(f"{type(self).__name__}: step {iteration} failed:\n{traceback.format_exc()}", rank0_only=False)
            failed = True
        if self._any_rank_failed(failed):
            self._disable(f"failure at step {iteration}")
            return
        metrics = self._reduce(*measured)
        if distributed.get_rank() == 0:
            import wandb

            if wandb.run is not None:
                wandb.log({f"inloop_val/{key}": value for key, value in metrics.items()}, step=iteration)
            headline = ("loss", *(f"gen/{metric}" for metric in LAST_FRAME_METRICS))
            log.info(
                f"{type(self).__name__} @ {iteration}: "
                + "  ".join(f"{key}={metrics[key]:.6g}" for key in headline if key in metrics)
            )

    def _measure(self, model: ImaginaireModel, iteration: int) -> tuple[list, list]:
        """This rank's losses and generation records."""

        was_training = model.training
        try:
            model.eval()
            losses = self._losses(model)
            generated = self._generate(model) if iteration % self.generation_every_n == 0 else []
        finally:
            if was_training:
                model.train()
        return losses, generated

    def _batch(self, items: Sequence[dict]) -> dict:
        """The packed batch the training loader hands the model."""

        batch = VFMListCollator().collate(list(items))
        batch["dataset_name"] = [self.config.dataloader_train.dataset_name] * len(items)
        return misc.to(batch, device="cuda")

    @torch.no_grad()
    def _losses(self, model: ImaginaireModel) -> list[tuple[dict, float]]:
        """Per-sample flow-matching loss of the run's own ``training_step``, the noise drawn from a fixed seed."""

        micro = int(self.config.dataloader_train.max_samples_per_batch)
        records = []
        cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        try:
            for start in range(0, len(self._items), micro):
                # One seed per rank and micro-batch: the same noise at every trigger of a run.
                torch.manual_seed(1234 + 1_000_003 * distributed.get_rank() + start)
                output, _ = model.training_step(self._batch(self._items[start : start + micro]), 0)
                values = output["flow_matching_loss_vision_per_instance"].float().reshape(-1).cpu().tolist()
                records += [
                    (entry, value)
                    for entry, value in zip(self._local[start : start + micro], values)
                    if not entry["_pad"]
                ]
        finally:
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_state)
        return records

    @torch.no_grad()
    def _generate(self, model: ImaginaireModel) -> list[dict]:
        """One generation call per rank (the sampler issues its own collectives)."""

        count = self._n_generation_local
        if count <= 0:
            return []
        items = self._items[:count]
        batch = self._batch(items)
        batch["neg_ai_caption"] = [NEGATIVE_PROMPT] * count
        samples = model.generate_samples_from_batch(
            batch,
            guidance=self.guidance,
            shift=self.shift,
            num_steps=self.num_steps,
            seed=[0] * count,
            n_sample=count,
            has_negative_prompt=True,
            normalize_cfg=False,
        )
        records = []
        for entry, item, latent in zip(self._local[:count], items, samples["vision"]):
            if entry["_pad"]:
                continue
            predicted = _uint8_frames(model.decode(latent))
            truth = _uint8_frames(item["video"])
            span = min(predicted.shape[1], truth.shape[1])
            records.append(
                dict(
                    entry,
                    **frame_metrics(
                        [score_array(_image(predicted[:, step])) for step in range(1, span)],
                        [score_array(_image(truth[:, step])) for step in range(1, span)],
                    ),
                )
            )
        return records

    def _reduce(self, losses: list[tuple[dict, float]], generated: list[dict]) -> dict:
        """Means over all ranks of every series; the keys come from the plan, so every rank's vector lines up."""

        def suffixes(entry: dict) -> list[str]:
            return ["", f"/{entry['set']}", f"/{bucket_of(entry)}"]

        all_suffixes = sorted({suffix for sample in self._plan["samples"] for suffix in suffixes(sample)})
        keys = [f"loss{suffix}" for suffix in all_suffixes]
        keys += [f"gen{suffix}/{metric}" for metric in LAST_FRAME_METRICS for suffix in all_suffixes]
        position = {key: i for i, key in enumerate(keys)}
        sums = torch.zeros(len(keys), dtype=torch.float64, device="cuda")
        counts = torch.zeros_like(sums)
        for entry, value in losses:
            for suffix in suffixes(entry):
                sums[position["loss" + suffix]] += value
                counts[position["loss" + suffix]] += 1
        for record in generated:
            for metric in LAST_FRAME_METRICS:
                for suffix in suffixes(record):
                    sums[position[f"gen{suffix}/{metric}"]] += record[metric]
                    counts[position[f"gen{suffix}/{metric}"]] += 1
        if dist.is_available() and dist.is_initialized():
            for tensor in (sums, counts):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        means, totals = (sums / counts.clamp(min=1)).tolist(), counts.tolist()
        return {key: means[i] for i, key in enumerate(keys) if totals[i] > 0}


def _uint8_frames(video: torch.Tensor) -> np.ndarray:
    """A [C, T, H, W] uint8 array from a uint8 video or one in [-1, 1]."""

    while video.ndim > 4:
        video = video[0]
    if video.dtype != torch.uint8:
        video = ((video.float() + 1.0) / 2.0).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return video.detach().cpu().numpy()


def _image(chw: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(chw.transpose(1, 2, 0)), mode="RGB")
