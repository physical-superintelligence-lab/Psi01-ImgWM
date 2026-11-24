"""The ``[custom.data]`` table of a run TOML: the LeRobot datasets a run mixes, how much, and how
captions are drawn (see ``docs/psix_image_world_model.md``)."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tomllib
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from typing_extensions import Self

from cosmos_framework.psix_wm import lerobot
from cosmos_framework.psix_wm.caption import FUTURE_HORIZONS, TEXT_FORMS


class Table(BaseModel):
    """A ``[custom.*]`` table of a run TOML: an unknown key or a value of the wrong type is an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def read(cls, run_toml: str | Path, table: str, **given: Any) -> Self:
        """``[custom.<table>]`` of ``run_toml``; ``given`` are the fields the file does not hold."""

        raw = tomllib.loads(Path(run_toml).read_text()).get("custom", {}).get(table, {})
        if taken := sorted(set(raw) & set(given)):
            raise ValueError(f"{run_toml}: [custom.{table}] cannot set {taken}")
        try:
            return cls.model_validate({**raw, **given})
        except ValidationError as error:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or table}: {e['msg']}" for e in error.errors())
            raise ValueError(f"{run_toml}: [custom.{table}] {problems}") from None


# ------------------------------------------------------------------ a dataset of a run
@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    weight: float = 1.0
    tasks: tuple[str, ...] = ()  # globs over tasks.jsonl task names: only their rows are used; empty: all
    categories: tuple[str, ...] = ()  # only the tasks whose tasks.jsonl category is one of these

    def task_names(self) -> tuple[str, ...]:
        """The globs that select its tasks: ``tasks``, or with ``categories`` the names of the tasks they
        leave (read from ``meta/tasks.jsonl``). Empty: every task."""

        if not self.categories:
            return self.tasks
        names = tuple(
            row["task"]
            for row in lerobot.read_jsonl(Path(self.path) / "meta" / "tasks.jsonl")
            if row.get("category", "") in self.categories
            and (not self.tasks or any(fnmatch.fnmatchcase(row["task"], glob) for glob in self.tasks))
        )
        if not names:
            raise ValueError(f"{self.name}: no tasks match its category/task filters")
        return names


_NAME_LEAVES = ("g1", "h1", "val", "test")  # kept in a dataset's name
_SKIPPED_LEAVES = ("train", "human")


def parse_dataset(entry: str | dict[str, Any], weight: float = 1.0, data_root: str = "") -> Dataset:
    """A dataset at a path (under ``data_root`` when relative), named after the path's last meaningful
    components (``cups_g1``). ``entry`` is the path, or a table with ``path`` and optionally ``tasks``
    (use only the rows of the tasks whose name matches one of these globs), ``categories`` (only the tasks
    whose ``tasks.jsonl`` category is one of these) and ``name`` (instead of the one from the path)."""

    entry = {"path": entry} if isinstance(entry, str) else dict(entry)
    path, tasks = entry.pop("path"), tuple(entry.pop("tasks", ()))
    name = entry.pop("name", None)
    categories = tuple(entry.pop("categories", ()))
    if entry:
        raise ValueError(f"dataset entry {path}: unknown keys {sorted(entry)}")
    path = path if path.startswith("/") else os.path.join(data_root, path)
    parts = list(Path(path).parts)
    leaves = []
    while parts[-1] in _NAME_LEAVES + _SKIPPED_LEAVES:
        leaf = parts.pop()
        if leaf in _NAME_LEAVES:
            leaves.insert(0, leaf)
    return Dataset(name or "_".join([parts[-1], *leaves]), path, float(weight), tasks, categories)


# ------------------------------------------------------------------ the tables
class Prompt(Table):
    """``[custom.data.prompt]``: how often each text form is drawn (``caption.narrow``)."""

    goal_forms: dict[str, float] = {"both": 0.25, "subtask": 0.75}
    future_forms: dict[str, float] = {"both": 0.35, "subtask": 0.45, "task": 0.20}

    @model_validator(mode="after")
    def _check(self) -> Self:
        for kind, forms in (("goal", self.goal_forms), ("future", self.future_forms)):
            if set(forms) - set(TEXT_FORMS[kind]) or abs(sum(forms.values()) - 1.0) > 1e-6:
                raise ValueError(f"{kind}_forms must give {TEXT_FORMS[kind]} probabilities summing to 1, got {forms}")
        return self


class Sampling(Table):
    """``[custom.data.sampling]``: the query rows built once per dataset, and how samples are drawn from them."""

    # Building rows. A goal row is stored per this many frames of a subtask; a draw starts at any of them.
    stored_goal_every_frames: int = 9
    min_goal_seconds: float = 0.8  # a goal clip starts at least this long before its subtask ends
    # Drawing samples
    goal_sample_share: float = 0.36  # goals' share of a dataset's samples (when it has any)
    # How often each nominal future horizon (seconds) is drawn, among those that fit in the subtask.
    horizon_weights: dict[str, float] = {"0.8": 0.125, "1.6": 0.375, "3.2": 0.5}

    @model_validator(mode="after")
    def _check(self) -> Self:
        if sorted(float(h) for h in self.horizon_weights) != sorted(FUTURE_HORIZONS):
            raise ValueError(f"horizon_weights must weigh each of {FUTURE_HORIZONS}, got {self.horizon_weights}")
        if (
            min(self.horizon_weights.values()) <= 0
            or self.stored_goal_every_frames <= 0
            or not 0 < self.goal_sample_share < 1
        ):
            raise ValueError(
                "horizon_weights and stored_goal_every_frames must be positive, goal_sample_share in (0, 1)"
            )
        return self


class Validation(Table):
    """``[custom.data.validation]``: how often the held-out datasets are scored, and on how many samples."""

    every: int = 5000
    generate_every: int = 5000
    loss_per_bucket: int = 16
    generate_per_bucket: int = 4
    sampler: str = "fast"  # the inference.psix_wm.sampler preset that generates

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.every <= 0 or self.generate_every <= 0 or self.generate_every % self.every:
            raise ValueError("generate_every must be a positive multiple of every")
        return self


class DataConfig(Table):
    """``[custom.data]``, under the names the file uses."""

    name: str  # the run's name: the file's stem
    cache: str  # index caches, query tables and validation plans
    dataset_paths: tuple[str | dict[str, Any], ...]  # each a path or a table (``parse_dataset``)
    dataset_weights: tuple[float, ...] | None = None  # one per dataset; 1.0 each when left out
    weighting: Literal["rows", "share"]
    val_dataset_paths: tuple[str | dict[str, Any], ...] = ()  # held out, only to monitor training
    prompt: Prompt = Prompt()
    sampling: Sampling = Sampling()
    validation: Validation = Validation()
    seed: int = 42
    data_root: str = ""  # relative dataset paths are under it
    resolution: str = "448"  # the model's resolution tier: each dataset's canvas is its closest size

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.dataset_weights is not None and len(self.dataset_weights) != len(self.dataset_paths):
            raise ValueError(f"{len(self.dataset_paths)} dataset_paths but {len(self.dataset_weights)} dataset_weights")
        for key, group in (("dataset_paths", self.datasets), ("val_dataset_paths", self.held_out)):
            names = [dataset.name for dataset in group]
            if repeated := sorted({name for name in names if names.count(name) > 1}):
                raise ValueError(f"two {key} have one name: {repeated}")
        trained = {os.path.realpath(dataset.path) for dataset in self.datasets}
        if leaked := [d.path for d in self.held_out if os.path.realpath(d.path) in trained]:
            raise ValueError(f"held-out datasets are also trained on: {leaked}")
        return self

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        weights = self.dataset_weights or (1.0,) * len(self.dataset_paths)
        return tuple(parse_dataset(entry, weight, self.data_root) for entry, weight in zip(self.dataset_paths, weights))

    @property
    def held_out(self) -> tuple[Dataset, ...]:
        return tuple(parse_dataset(entry, data_root=self.data_root) for entry in self.val_dataset_paths)

    @property
    def index_cache(self) -> str:
        return f"{self.cache}/psix_wm_index"

    @property
    def query_cache(self) -> str:
        return f"{self.cache}/psix_wm_queries"

    @property
    def validation_dir(self) -> str:
        return f"{self.cache}/psix_wm_validation/{self.name}"

    def build_args(self, dataset: Dataset) -> dict[str, Any]:
        """The ``queries.build_query_table`` arguments of ``dataset``: its ``meta/psix.json``, its task
        filters and this run's settings."""

        metadata = lerobot.load_dataset_metadata(dataset.path)
        return {
            "dataset_id": dataset.name,
            "robot_type": metadata.agent,
            "camera": metadata.camera,
            "task_cameras": metadata.task_cameras,
            "control": metadata.control,
            "anchor_every": metadata.anchor_every,
            "tasks": list(dataset.task_names()),
            "skipped_episodes": list(metadata.skipped_episodes),
            "ends_done": metadata.ends_done,
            "max_subtask_seconds": metadata.max_subtask_seconds,
            "resolution": self.resolution,
            "stored_goal_every_frames": self.sampling.stored_goal_every_frames,
            "min_goal_seconds": self.sampling.min_goal_seconds,
        }

    def validation_sets(self) -> dict[str, dict[str, Any]]:
        """Every held-out dataset: its root and table build arguments."""

        return {dataset.name: {"root": dataset.path, "queries": self.build_args(dataset)} for dataset in self.held_out}


def load(run_toml: str | Path) -> DataConfig:
    """The ``[custom.data]`` table of a run TOML; the run's name is the file's stem."""

    return DataConfig.read(run_toml, "data", name=Path(run_toml).stem)
