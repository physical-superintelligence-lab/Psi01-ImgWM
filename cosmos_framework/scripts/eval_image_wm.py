"""Offline evaluation of an Image-WM checkpoint on its run's validation sets, or on given datasets.

Up to ``--per-bucket`` rows per (set, bucket) are drawn as in-loop validation draws them, generated and
scored against the real frames. Every sample is saved as PNGs and ``meta.json``; ``summary.json`` holds
the means overall, per set and per (set, bucket).

    python -m cosmos_framework.scripts.eval_image_wm --run-toml examples/toml/sft_config/psix_image_wm/base.toml \\
        --checkpoint-path <run> --output-dir <out>

``<run>`` is a run's folder (its latest checkpoint) or one ``checkpoints/iter_*``. ``--datasets path ...``
evaluates those datasets instead of the run's validation sets.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from cosmos_framework.inference.psix_wm.generator import Generator, fit_to_canvas
from cosmos_framework.inference.psix_wm.sampler import PRESETS
from cosmos_framework.psix_wm import lerobot
from cosmos_framework.psix_wm.caption import SCHEMA, Shot
from cosmos_framework.psix_wm.checkpoint import latest_checkpoint
from cosmos_framework.psix_wm.data_config import load
from cosmos_framework.psix_wm.dataset import ImageWMDataset
from cosmos_framework.psix_wm.metrics import METRICS, frame_metrics, score_array
from cosmos_framework.psix_wm.validation import check_samples, draw_plan, set_table


def plan_samples(args, config) -> tuple[list[dict], dict[str, tuple[ImageWMDataset, Shot]]]:
    """Every sample to generate, and the frame decoder and shot of each validation set."""

    per_bucket = {"loss_per_bucket": args.per_bucket, "generate_per_bucket": args.per_bucket}
    changes = {"validation": config.validation.model_copy(update=per_bucket)}
    if args.datasets:
        changes["val_dataset_paths"] = tuple(args.datasets)
    config = config.model_copy(update=changes)
    for root in dict.fromkeys(dataset.path for dataset in config.held_out):
        print(f"index {lerobot.build_index(root, config.index_cache)}", flush=True)
    # A directory's plan is drawn once; a rerun checks it against the tables and continues it.
    plan_file = args.output_dir / "plan.json"
    if plan_file.is_file():
        samples = json.loads(plan_file.read_text())
        check_samples(config, samples, plan_file)
    else:
        samples = [dict(row, sample_id=row["query_id"]) for row in draw_plan(config)["samples"]]
        plan_file.write_text(json.dumps(samples, indent=1))
    sets = {}
    for name in config.validation_sets():
        table = set_table(config, name)
        sets[name] = ImageWMDataset(table, resolution=config.resolution, video_backend="pyav"), table.shot
    return samples, sets


def _images(video) -> list[Image.Image]:
    frames = (video * 255.0).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(frame, mode="RGB") for frame in frames]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-toml", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", help="default: config.yaml of the checkpoint's folder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", help="datasets to evaluate instead of the run's validation sets")
    parser.add_argument("--per-bucket", type=int, default=8)
    parser.add_argument("--preset", default="quality", choices=sorted(PRESETS))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    config = load(args.run_toml)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The plan and finished samples are reused on a rerun, so a directory belongs to one checkpoint and settings.
    checkpoint = latest_checkpoint(args.checkpoint_path)
    run = {
        "checkpoint": str(checkpoint.resolve()),
        "preset": args.preset,
        "caption_schema": SCHEMA,
        "per_bucket": args.per_bucket,
        "datasets": args.datasets,
    }
    run_file = args.output_dir / "run.json"
    if run_file.is_file() and json.loads(run_file.read_text()) != run:
        raise SystemExit(f"{args.output_dir} holds samples of another run ({run_file}); use a new --output-dir")
    run_file.write_text(json.dumps(run, indent=1))
    samples, sets = plan_samples(args, config)
    print(f"{len(samples)} samples")
    if args.plan_only:
        return

    generator = Generator(checkpoint, args.config_file, args.output_dir / "_pipe", max_batch=args.batch)
    if generator.resolution != config.resolution:
        raise SystemExit(
            f"the checkpoint generates at {generator.resolution}, the run's captions use {config.resolution}"
        )
    pending = [s for s in samples if not (args.output_dir / s["sample_id"] / "meta.json").is_file()]
    for name in dict.fromkeys(s["set"] for s in pending):
        group, (decoder, shot) = [s for s in pending if s["set"] == name], sets[name]
        canvas = shot.canvas
        for start in range(0, len(group), args.batch):
            batch = group[start : start + args.batch]
            truths = [_images(decoder.decode(s["episode_index"], s["frame_indexes"])) for s in batch]
            outputs = generator.generate(
                [(truth[0], s["caption_json"], s["mrope_fps"], 0) for truth, s in zip(truths, batch)],
                canvas=canvas,
                sampler=PRESETS[args.preset],
            )
            for sample, frames, truth in zip(batch, outputs, truths):
                _save(args.output_dir / sample["sample_id"], sample, frames, truth, canvas)
            print(f"{name}: {start + len(batch)}/{len(group)}", flush=True)

    metas = [json.loads((args.output_dir / s["sample_id"] / "meta.json").read_text()) for s in samples]
    cells = defaultdict(list)
    for meta in metas:
        for key in ("all", meta["set"], f"{meta['set']}/{meta['bucket']}"):
            cells[key].append(meta)
    means = {
        key: {
            "n": len(rows),
            **{m: round(float(np.mean([r[m] for r in rows])), 5) for m in METRICS},
        }
        for key, rows in sorted(cells.items())
    }
    (args.output_dir / "summary.json").write_text(json.dumps({"means": means, "samples": metas}, indent=1))
    for key, row in means.items():
        print(f"{key:60s} n={row['n']:4d}  mse_last={row['mse_last']:.5f}  psnr_last={row['psnr_last']:.2f}")


def _save(
    directory: Path, sample: dict, frames: list[Image.Image], truth: list[Image.Image], canvas: tuple[int, int]
) -> None:
    """PNGs of the condition, the predicted frames and the real ones; ``meta.json`` (written last) with the scores.

    The real frames are scored as the model sees frames: fitted to its canvas and cropped to their content.
    """

    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(score_array(truth[0])).save(directory / "condition.png")
    predicted = [score_array(frame) for frame in frames[1:]]
    for step, frame in enumerate(predicted, start=1):
        Image.fromarray(frame).save(directory / f"pred_{step}.png")
    real = [score_array(_on_canvas(frame, canvas)) for frame in truth[1:]]
    for step, frame in enumerate(real, start=1):
        Image.fromarray(frame).save(directory / f"gt_{step}.png")
    meta = {key: value for key, value in sample.items() if key != "caption_json"} | {"caption": sample["caption_json"]}
    meta |= frame_metrics(predicted, real)
    partial = directory / "meta.json.partial"  # meta.json marks the sample finished, so it appears whole
    partial.write_text(json.dumps(meta, indent=1))
    partial.replace(directory / "meta.json")


def _on_canvas(frame: Image.Image, canvas: tuple[int, int]) -> Image.Image:
    fitted, (width, height) = fit_to_canvas(frame, canvas)
    return fitted.crop((0, 0, width, height))


if __name__ == "__main__":
    main()
