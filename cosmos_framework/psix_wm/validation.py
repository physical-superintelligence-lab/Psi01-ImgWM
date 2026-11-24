"""The fixed validation samples of an Image-WM run: its plan, drawn once and kept.

For every held-out dataset (``val_dataset_paths``) and bucket (goal, or future at one horizon), up
to ``loss_per_bucket`` rows are drawn, distinct episodes first; the first ``generate_per_bucket`` of
them are also generated. A plan is never drawn again: every use checks that its samples are still the
rows the tables give, and stops when one is not.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import socket
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cosmos_framework.psix_wm.caption import FUTURE_HORIZONS
from cosmos_framework.psix_wm.data_config import DataConfig
from cosmos_framework.psix_wm.queries import QueryTable, load_query_table


def plan_path(config: DataConfig) -> Path:
    """The plan's file, named by the run's settings that say what to draw: the held-out datasets, how many
    samples per bucket, and how rows are built. Nothing else names it, so a plan outlives code and data edits."""

    key = json.dumps(
        {
            "held_out": [dataclasses.asdict(dataset) for dataset in config.held_out],
            "loss_per_bucket": config.validation.loss_per_bucket,
            "generate_per_bucket": config.validation.generate_per_bucket,
            "stored_goal_every_frames": config.sampling.stored_goal_every_frames,
            "min_goal_seconds": config.sampling.min_goal_seconds,
            "resolution": config.resolution,
        },
        sort_keys=True,
    )
    return Path(config.validation_dir) / f"plan_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def prepare_plan(config: DataConfig) -> Path:
    """The run's plan file: drawn when there is none, checked against the tables either way."""

    path = plan_path(config)
    if not path.is_file():
        plan = draw_plan(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp-{socket.gethostname()}-{os.getpid()}")  # every node may draw it at once
        tmp.write_text(json.dumps(plan))
        tmp.replace(path)
    check_samples(config, json.loads(path.read_text())["samples"], path)
    return path


def check_samples(config: DataConfig, samples: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Stop unless every sample of the plan at ``path`` is still the row its table gives: the same frames
    and caption. A plan is not drawn again by itself; delete the file to draw a new one."""

    tables: dict[str, QueryTable] = {}
    for sample in samples:
        name, row = sample["set"], sample["row"]
        if name not in tables:
            tables[name] = set_table(config, name)
        try:
            now = tables[name].describe(row, sample["horizon_seconds"])
        except (IndexError, ValueError) as error:
            now = {"row": str(error)}
        if changed := sorted(key for key, value in now.items() if sample.get(key) != value):
            raise ValueError(
                f"{path}: sample {sample['query_id']} of {name} is not row {row} of its table any more "
                f"({', '.join(changed)} differ). The held-out data or the rows built from it changed; "
                "delete the file to draw a new plan."
            )


def load_plan(config: DataConfig) -> dict[str, Any]:
    return json.loads(prepare_plan(config).read_text())


def set_table(config: DataConfig, name: str) -> QueryTable:
    spec = config.validation_sets()[name]
    return load_query_table(spec["root"], config.query_cache, config.index_cache, **spec["queries"])


def plan_table(config: DataConfig, name: str, samples: Sequence[Mapping[str, Any]]) -> QueryTable:
    """The rows of ``samples``, all of set ``name``, each future at its planned horizon."""

    return set_table(config, name).take([s["row"] for s in samples], [s["horizon_seconds"] for s in samples])


def draw_plan(config: DataConfig) -> dict[str, Any]:
    validation = config.validation
    if validation.generate_per_bucket > validation.loss_per_bucket:
        raise ValueError("generate_per_bucket must not exceed loss_per_bucket")
    sets = config.validation_sets()
    rng = np.random.default_rng(0)
    loss_cells: dict[tuple[str, str], list[dict]] = {}
    generate_cells: dict[tuple[str, str], list[dict]] = {}
    for name in sets:
        table = set_table(config, name)
        for bucket, (rows, horizon) in sorted(_buckets(table).items()):
            rows = _distinct_episodes_first(rows, np.asarray(table.episode)[rows], validation.loss_per_bucket, rng)
            picked = [dict(table.describe(int(row), horizon), set=name, bucket=bucket) for row in rows]
            generate_cells[(name, bucket)] = picked[: validation.generate_per_bucket]
            loss_cells[(name, bucket)] = picked[validation.generate_per_bucket :]
    generated = [row for cell in sorted(generate_cells) for row in generate_cells[cell]]
    return {
        "sets": {name: spec["root"] for name, spec in sets.items()},
        "n_generation": len(generated),
        "samples": generated + [row for cell in sorted(loss_cells) for row in loss_cells[cell]],
    }


def bucket_of(row: Mapping[str, Any]) -> str:
    return f"future_{float(row['horizon_seconds']):.1f}" if row["query_type"] == "future" else "subgoal"


def _buckets(table: QueryTable) -> dict[str, tuple[np.ndarray, float | None]]:
    """Per bucket, the rows it may draw and the horizon they are drawn at: goal rows, and the future rows
    that fit each horizon."""

    buckets = {"subgoal": (np.flatnonzero(table.is_goal), None)}
    bins = np.asarray(table.bins)
    for bit, horizon in enumerate(FUTURE_HORIZONS):
        buckets[f"future_{horizon:.1f}"] = (np.flatnonzero(bins >> bit & 1), horizon)
    return {bucket: entry for bucket, entry in buckets.items() if len(entry[0])}


def _distinct_episodes_first(
    rows: np.ndarray, episodes: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """``count`` of ``rows`` in a random order: one per episode first, then the rest."""

    if count <= 0 or not len(rows):
        return rows[:0]
    order = rng.permutation(len(rows))
    _, first = np.unique(episodes[order], return_index=True)
    distinct = np.zeros(len(order), dtype=bool)
    distinct[first] = True
    ranked = np.r_[np.flatnonzero(distinct), np.flatnonzero(~distinct)]
    return rows[order[ranked[:count]]]
