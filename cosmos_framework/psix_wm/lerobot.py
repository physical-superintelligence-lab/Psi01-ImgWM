"""Read a LeRobot v2.1 dataset: one ``data/chunk-*/episode_*.parquet`` and one mp4 per episode,
``meta/episodes.jsonl`` and ``meta/tasks.jsonl``, and its ``meta/psix.json``.

Psi0.1 exports keep a short slug per task in ``tasks.jsonl`` and the instruction itself in every
frame's ``task_description``; the instruction is the task text.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import socket
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_framework.psix_wm.caption import CAMERAS, CONTROLS, ROBOT_TYPES

EGO_VIDEO = "observation.images.egocentric"  # the camera every sample is decoded from

T = TypeVar("T")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


@functools.cache  # read once per process: every decoded sample looks up its video path in it
def info(root: str | Path) -> dict[str, Any]:
    return json.loads((Path(root) / "meta" / "info.json").read_text())


@dataclass(frozen=True)
class DatasetMetadata:
    """Capture metadata and episode/subtask filtering rules read from ``meta/psix.json``."""

    agent: str  # caption.ROBOT_TYPES
    camera: str  # caption.CAMERAS, of every task not in task_cameras
    control: Literal["teleop", "natural"]  # "natural": a person doing the task
    anchor_every: int = 1  # an anchor every this many frames
    task_cameras: dict[int, str] = field(default_factory=dict)  # task_index: its camera, where not camera
    skipped_episodes: tuple[int, ...] = ()  # episodes no model uses
    ends_done: bool = True  # an episode's last subtask is annotated as finished, not cut off by the recording
    max_subtask_seconds: float | None = None  # a longer subtask is not used as one

    def __post_init__(self) -> None:
        if self.agent not in ROBOT_TYPES:
            raise ValueError(f"agent {self.agent!r} not in {ROBOT_TYPES}")
        if bad := sorted({self.camera, *self.task_cameras.values()} - set(CAMERAS)):
            raise ValueError(f"cameras {bad} not in {CAMERAS}")
        if self.control not in CONTROLS or self.anchor_every <= 0:
            raise ValueError(f"control {self.control!r} must be one of {CONTROLS}, anchor_every positive")


def load_dataset_metadata(root: str | Path) -> DatasetMetadata:
    path = Path(root) / "meta/psix.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; every dataset needs one (docs/psix_image_world_model.md)")
    meta = json.loads(path.read_text())
    if not meta.get("capabilities", {}).get("imgwm_future"):
        raise ValueError(f"{path}: the dataset lacks the columns image-WM training reads")
    return DatasetMetadata(
        agent=meta["agent"],
        camera=meta["camera"],
        control=meta["control"],
        anchor_every=int(meta.get("anchor_every", 1)),
        task_cameras={int(task): camera for task, camera in meta.get("task_cameras", {}).items()},
        skipped_episodes=tuple(sorted(int(episode) for episode in meta["skipped_episodes"])),
        ends_done=bool(meta["terminal_is_done"]),
        max_subtask_seconds=meta["max_subtask_seconds"],
    )


def episode_file(root: str | Path, key: str, episode_index: int) -> Path:
    """The data parquet (``key="data"``) or a video's mp4 (``key`` = its feature) of one episode."""

    meta = info(root)
    chunk = episode_index // int(meta["chunks_size"])
    if key == "data":
        return Path(root) / meta["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return Path(root) / meta["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=episode_index)


def data_files(root: str | Path) -> list[Path]:
    """The per-episode data parquet files, in episode order."""

    files = sorted((Path(root) / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet files under {Path(root) / 'data'}")
    return files


def frame_file_groups(
    root: str | Path, episode_files_per_group: int, index_cache: str | Path | None
) -> list[list[Path]]:
    """The root's frames as groups of whole-episode parquet files: each part of its index under
    ``index_cache`` alone when one is built, else ``episode_files_per_group`` episode files a group."""

    index = index_dir(root, index_cache) if index_cache else None
    if index is not None and (index / "manifest.json").is_file():
        return [[index / part["file"]] for part in json.loads((index / "manifest.json").read_text())["parts"]]
    files = data_files(root)
    return [files[start : start + episode_files_per_group] for start in range(0, len(files), episode_files_per_group)]


def _read_each(read: Callable[[Path], T], files: Sequence[Path]) -> Iterator[T]:
    """``read(file)`` per file, in order, on a few threads."""

    window = 256  # bounds the results waiting for the consumer
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(files), window):
            yield from pool.map(read, files[start : start + window])


def episode_rows(root: str | Path, index_cache: str | Path | None) -> list[dict[str, Any]]:
    """Every episode's ``episodes.jsonl`` row, its ``tasks`` labelled with task text."""

    root = Path(root)
    rows = read_jsonl(root / "meta" / "episodes.jsonl")
    label = dict(_task_texts(str(root.resolve()), str(index_cache) if index_cache else None))
    for row in rows:
        if "tasks" in row:
            row["tasks"] = [label.get(task, task) if isinstance(task, int) else task for task in row["tasks"]]
    return rows


def pick_task_texts(tasks: Mapping[int, str], described: Iterable[tuple[int, str]]) -> dict[int, str]:
    """Per task its text: the first description its frames carry (``described``: task_index and
    description, in frame order), else its ``tasks.jsonl`` text."""

    first: dict[int, str] = {}
    for index, text in described:
        first.setdefault(index, text)
    return {index: first.get(index, task) for index, task in tasks.items()}


@functools.cache  # one pass over the data per process: several datasets can share a root
def _task_texts(root: str, index_cache: str | None) -> tuple[tuple[int, str], ...]:
    """``(task_index, task text)`` per task of ``tasks.jsonl``."""

    def read(file: Path):
        handle = pq.ParquetFile(file)
        if "task_description" not in handle.schema_arrow.names:
            return None
        return handle.read(columns=["task_index", "task_description"])

    files = [path for group in frame_file_groups(root, 1, index_cache) for path in group]
    described = (
        (int(index), str(text))
        for table in _read_each(read, files)
        if table is not None
        for index, text in zip(table["task_index"].to_pylist(), table["task_description"].to_pylist())
        if text is not None
    )
    tasks = {row["task_index"]: row["task"] for row in read_jsonl(Path(root) / "meta" / "tasks.jsonl")}
    return tuple(pick_task_texts(tasks, described).items())


# ------------------------------------------------------------------ index cache
# Opening every per-episode file on every rank is slow for large datasets, so the columns the Image-WM
# reads are consolidated into a few parquet parts per dataset, built before training (``build_index``, the
# one place that looks at every data file). Without an index cache directory, or with no index built for a
# dataset, reads use the per-episode files.

# The frame columns the Image-WM reads, and the ones a dataset may lack.
REQUIRED_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "sub_task_index",
    "subtask_prompt",
    "sub_goal_frame_index",
)
OPTIONAL_COLUMNS = ("task_description",)
INDEX_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def index_dir(root: str | Path, index_cache: str | Path) -> Path:
    """Where ``root``'s index lives under ``index_cache``, keyed by its real path, its metadata stamps and
    the columns it holds (so a new index column rebuilds it)."""

    meta = Path(os.path.realpath(Path(root) / "meta" / "episodes.jsonl")).parent
    stamps = [str(meta.parent), ",".join(INDEX_COLUMNS)]
    for name in ("episodes.jsonl", "tasks.jsonl", "info.json"):
        stat = (meta / name).stat()
        stamps.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    key = hashlib.sha256("\n".join(stamps).encode()).hexdigest()[:16]
    return Path(index_cache) / f"{meta.parent.parent.name}-{meta.parent.name}-{key}"


def data_stamp(root: str | Path) -> str:
    """A digest of every data file's name, size and modification time: it changes when one is rewritten."""

    digest = hashlib.sha256()
    for file in data_files(root):
        stat = file.stat()
        digest.update(f"{file.parent.name}/{file.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def indexed_data(root: str | Path, index_cache: str | Path | None) -> str | None:
    """The ``data_stamp`` of the files ``root``'s index was built from; None without an index."""

    manifest = index_dir(root, index_cache) / "manifest.json" if index_cache else None
    return json.loads(manifest.read_text())["data"] if manifest and manifest.is_file() else None


def build_index(root: str | Path, index_cache: str | Path) -> Path:
    """``root``'s index under ``index_cache``, built unless one of the data files as they are now exists."""

    root = Path(root)
    out = index_dir(root, index_cache)
    stamp = data_stamp(root)
    if (out / "manifest.json").is_file():
        if json.loads((out / "manifest.json").read_text()).get("data") == stamp:
            return out
        shutil.rmtree(out)
    files = data_files(root)
    rows = read_jsonl(root / "meta" / "episodes.jsonl")
    missing = [
        row["episode_index"] for row in rows if not episode_file(root, EGO_VIDEO, int(row["episode_index"])).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{root}: {len(missing)} episodes have no {EGO_VIDEO} video, e.g. {missing[:5]}")

    def read(path: Path) -> pa.Table:
        handle = pq.ParquetFile(path)
        return handle.read(columns=[column for column in INDEX_COLUMNS if column in handle.schema_arrow.names])

    tmp = out.with_name(f"{out.name}.tmp-{socket.gethostname()}-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    parts: list[dict[str, Any]] = []
    pending: list[pa.Table] = []

    def flush() -> None:
        table = pa.concat_tables(pending, promote_options="permissive")  # files may differ in optional columns
        name = f"part-{len(parts):05d}.parquet"
        pq.write_table(table, tmp / name)
        episodes = table["episode_index"]
        parts.append(
            {
                "file": name,
                "rows": table.num_rows,
                "first_episode": int(pc.min(episodes).as_py()),
                "last_episode": int(pc.max(episodes).as_py()),
            }
        )
        pending.clear()

    total = 0
    columns: set[str] = set()
    for table in _read_each(read, files):
        pending.append(table)
        total += table.num_rows
        columns.update(table.column_names)
        if sum(t.num_rows for t in pending) >= 2_000_000:  # rows of one index part
            flush()
    if pending:
        flush()
    expected = sum(int(row["length"]) for row in rows)
    if total != expected:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"{root}: index holds {total} rows, episodes.jsonl declares {expected}")
    (tmp / "manifest.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "data": stamp,
                "columns": sorted(columns),
                "rows": total,
                "episodes": len(files),
                "parts": parts,
            },
            indent=1,
        )
    )
    if (out / "manifest.json").is_file():  # another job built it meanwhile: keep theirs
        shutil.rmtree(tmp, ignore_errors=True)
        return out
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out
