"""The Image-WM query rows of one LeRobot v2.1 dataset.

Every anchor frame yields a FUTURE row, valid at the nominal horizons (``caption.FUTURE_HORIZONS``)
whose clip ends within the frame's subtask, and every ``stored_goal_every_frames`` frames of a
subtask yield a GOAL row that ends on the subtask's last frame. An episode without subtask
annotation yields future rows of its task only.

A dataset's rows are the columns of a ``QueryTable``. Building one scans the frames with numpy, one
job per index part (or per few episode files), on as many processes as it is given. A built table is
cached on disk and opened memory-mapped, so the processes of a node share one copy.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import functools
import hashlib
import json
import math
import os
import re
import shutil
import socket
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cosmos_framework.psix_wm import lerobot
from cosmos_framework.psix_wm.caption import (
    CAMERAS,
    FUTURE_HORIZONS,
    GOAL_MROPE_FPS,
    NUM_FRAMES,
    Shot,
    canonical,
    frame_indexes,
    future_caption,
    future_mrope_fps,
    goal_caption,
)

# Part of the cache key: a table is rebuilt by itself when the rules it is built by change.
_RULES = hashlib.sha256(
    b"".join(Path(__file__).with_name(name).read_bytes() for name in ("queries.py", "caption.py", "lerobot.py"))
).hexdigest()
_ROW_COLUMNS = ("episode", "current", "is_goal", "end", "latest_start", "bins", "task", "subtask", "camera")
_TIMELINE_COLUMNS = ("episodes", "offsets", "timestamps")
_TEXT_COLUMNS = ("subtask_prompt", "task_description")


@dataclasses.dataclass(frozen=True)
class Query:
    """One sample of a row: its frames, and its caption in the widest text form its data allows."""

    query_id: str
    episode_index: int
    query_type: str  # "subgoal" or "future"
    frame_indexes: tuple[int, ...]
    caption: str
    mrope_fps: float
    horizon: float | None = None  # a future's nominal horizon
    subtask_end: int | None = None  # the last frame of the subtask a future caption names
    latest_start: int | None = None  # a goal clip may start anywhere from its first frame to this one


# ------------------------------------------------------------------ the table
@dataclasses.dataclass(frozen=True)
class Timelines:
    """The timestamps of every episode's frames, as one array in episode order."""

    episodes: np.ndarray  # sorted episode indexes
    offsets: np.ndarray  # episode k's frames are timestamps[offsets[k] : offsets[k + 1]]
    timestamps: np.ndarray

    def __getitem__(self, episode: int) -> np.ndarray:
        position = int(np.searchsorted(self.episodes, episode))
        if position == len(self.episodes) or self.episodes[position] != episode:
            raise KeyError(f"no episode {episode}")
        return self.timestamps[self.offsets[position] : self.offsets[position + 1]]


@dataclasses.dataclass(frozen=True)
class QueryTable:
    """One dataset's query rows in episode and frame order, a goal row before the future row of its frame."""

    root: str
    dataset_id: str
    fps: float
    shot: Shot  # its camera is the dataset's; a row's own is in camera
    texts: tuple[str, ...]
    episode: np.ndarray
    current: np.ndarray  # the anchor frame
    is_goal: np.ndarray
    end: np.ndarray  # the anchor's subtask's last frame (its episode's, without annotation): no clip runs past it
    latest_start: np.ndarray  # a goal clip starts at most here
    bins: np.ndarray  # bit k set: FUTURE_HORIZONS[k] fits in the subtask (future rows)
    task: np.ndarray  # an index into texts
    subtask: np.ndarray  # an index into texts; -1 without annotation
    camera: np.ndarray  # an index into caption.CAMERAS
    timelines: Timelines

    def __len__(self) -> int:
        return len(self.episode)

    def caption_shot_for_row(self, row: int) -> Shot:
        """Caption capture settings for this row, including its task-specific camera override."""

        camera = CAMERAS[int(self.camera[row])]
        return self.shot if camera == self.shot.camera else dataclasses.replace(self.shot, camera=camera)

    def horizons(self, row: int) -> tuple[float, ...]:
        bits = int(self.bins[row])
        return tuple(horizon for bit, horizon in enumerate(FUTURE_HORIZONS) if bits >> bit & 1)

    def take(self, rows: Sequence[int], horizons: Sequence[float | None] | None = None) -> QueryTable:
        """The table of ``rows``; a future row with an entry in ``horizons`` keeps that horizon only."""

        rows = np.asarray(rows, dtype=np.int64)
        columns = {name: np.asarray(getattr(self, name)[rows]) for name in _ROW_COLUMNS}
        for position, horizon in enumerate(horizons or ()):
            if horizon is None:
                continue
            if horizon not in self.horizons(int(rows[position])):
                raise ValueError(f"row {rows[position]} does not fit horizon {horizon}")
            columns["bins"][position] = 1 << FUTURE_HORIZONS.index(horizon)
        return dataclasses.replace(self, **columns)

    def query_id(self, row: int, horizon: float | None = None) -> str:
        """A future's id names its horizon: ``horizon``, else the row's only one."""

        kind = "subgoal" if self.is_goal[row] else "future"
        query_id = f"{self.dataset_id}__ep{int(self.episode[row]):06d}__{kind}__c{int(self.current[row]):06d}"
        horizon = self._horizon(row, horizon)
        return query_id if horizon is None else f"{query_id}__h{horizon:.1f}"

    def _horizon(self, row: int, horizon: float | None) -> float | None:
        """``horizon``, else the row's only one (None for a goal row or a row valid at several)."""

        horizons = self.horizons(row)
        return horizons[0] if horizon is None and len(horizons) == 1 else horizon

    def query(self, row: int, horizon: float | None = None) -> Query:
        """The row as a sample; a future at ``horizon``, which a row valid at one horizon only may omit."""

        episode, current, end = int(self.episode[row]), int(self.current[row]), int(self.end[row])
        task = self.texts[self.task[row]]
        subtask = self.texts[self.subtask[row]] if self.subtask[row] >= 0 else None
        if self.is_goal[row]:
            return Query(
                query_id=self.query_id(row),
                episode_index=episode,
                query_type="subgoal",
                frame_indexes=frame_indexes(current, end),
                caption=canonical(goal_caption(self.caption_shot_for_row(row), task, subtask)),
                mrope_fps=GOAL_MROPE_FPS,
                latest_start=int(self.latest_start[row]),
            )
        horizons, horizon = self.horizons(row), self._horizon(row, horizon)
        if horizon not in horizons:
            raise ValueError(f"row {row} fits horizons {horizons}, not {horizon}")
        return Query(
            query_id=self.query_id(row, horizon),
            episode_index=episode,
            query_type="future",
            frame_indexes=tuple(current + step for step in _horizon_steps(self.fps)[horizon]),
            caption=self.future_caption_at(row, horizon, horizon),
            mrope_fps=future_mrope_fps(horizon),
            horizon=horizon,
            subtask_end=end if self._future_subtask(row, horizon) else None,
        )

    def future_caption_at(self, row: int, horizon: float, seconds: float) -> str:
        """The caption of future row ``row`` at its nominal ``horizon``, stating ``seconds`` (in that horizon's bin)."""

        subtask = self._future_subtask(row, horizon)
        return canonical(
            future_caption(self.caption_shot_for_row(row), seconds, self.texts[self.task[row]], subtask=subtask)
        )

    def _future_subtask(self, row: int, horizon: float) -> str | None:
        """The subtask a future at nominal ``horizon`` names: the row's, unless it ends within the horizon."""

        if self.subtask[row] < 0 or (int(self.end[row]) - int(self.current[row])) / self.fps < horizon:
            return None
        return self.texts[self.subtask[row]]

    def describe(self, row: int, horizon: float | None = None) -> dict[str, Any]:
        """The row as a plain record, for validation plans and evaluation."""

        query = self.query(row, horizon)
        task = self.texts[self.task[row]]
        return {
            "row": int(row),
            "query_id": query.query_id,
            "episode_index": query.episode_index,
            "query_type": query.query_type,
            "current_frame_index": int(self.current[row]),
            "frame_indexes": list(query.frame_indexes),
            "horizon_seconds": query.horizon,
            "mrope_fps": query.mrope_fps,
            "caption_json": query.caption,
            "task_text": task,
            "subgoal_text": self.texts[self.subtask[row]] if self.subtask[row] >= 0 else task,
        }


# ------------------------------------------------------------------ building
def build_query_table(
    root: str | Path,
    *,
    dataset_id: str,
    robot_type: str,
    camera: str,
    control: str,
    resolution: str,
    stored_goal_every_frames: int,
    min_goal_seconds: float,
    anchor_every: int = 1,
    task_cameras: Mapping[int, str] | None = None,
    tasks: Sequence[str] = (),
    skipped_episodes: Sequence[int] = (),
    ends_done: bool = True,
    max_subtask_seconds: float | None = None,
    index_cache: str | Path | None = None,
    pool: ProcessPoolExecutor | None = None,
) -> QueryTable:
    """``root``'s rows, its frames read from its index under ``index_cache`` when one is built and scanned
    on ``pool`` (in this process without one). ``camera`` films every task but those in ``task_cameras``
    (task_index: camera); only the tasks whose name matches a ``tasks`` glob give rows (every task when
    empty), and ``skipped_episodes`` give none. Without ``ends_done`` an episode's last subtask ends where
    the recording stops, so it gives no goal rows. A subtask longer than ``max_subtask_seconds`` is not
    used as one: its frames count as unannotated."""

    build = _Build.of(
        root,
        dataset_id=dataset_id,
        shot=dataset_shot(root, robot_type, camera, control, resolution),
        task_cameras=task_cameras or {},
        tasks=tasks,
        skipped_episodes=skipped_episodes,
        ends_done=ends_done,
        max_subtask_seconds=max_subtask_seconds,
        anchor_every=anchor_every,
        stored_goal_every_frames=stored_goal_every_frames,
        min_goal_seconds=min_goal_seconds,
    )
    jobs = lerobot.frame_file_groups(root, 256, index_cache)  # episode files per job
    scan = pool.map if pool else map
    return _merge(build, list(scan(_scan, [build] * len(jobs), jobs)))


def load_query_table(
    root: str | Path, cache_dir: str | Path, index_cache: str | Path | None, **build_args: Any
) -> QueryTable:
    """``root``'s cached table, built in this process and cached first when it is missing."""

    path = cache_path(root, cache_dir, index_cache, build_args)
    if not (path / "meta.json").is_file():
        _save(build_query_table(root, **build_args, index_cache=index_cache), path)
    return _open(path)


def prepare_query_tables(
    datasets: Mapping[str, Mapping[str, Any]], cache_dir: str | Path, index_cache: str | Path | None, workers: int
) -> list[Path]:
    """The cached tables of ``datasets`` (root: build arguments), those missing built on ``workers`` processes."""

    paths = {root: cache_path(root, cache_dir, index_cache, args) for root, args in datasets.items()}
    missing = [root for root, path in paths.items() if not (path / "meta.json").is_file()]
    if missing:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            for root in missing:
                _save(build_query_table(root, **datasets[root], index_cache=index_cache, pool=pool), paths[root])
    return list(paths.values())


def dataset_shot(root: str | Path, robot_type: str, camera: str, control: str, resolution: str) -> Shot:
    """The caption's shot for a dataset: its robot, its camera, and its training canvas (``meta/info.json``)."""

    return Shot(robot_type, camera, control, training_canvas(_load_info(Path(root))[1], resolution))


def training_canvas(image_shape: tuple[int, int], resolution: str) -> tuple[int, int]:
    """(W, H) the training transform hands the model: the aspect kept, snapped to the tier's closest size."""

    from cosmos_framework.data.vfm.action.transforms import find_closest_target_size

    height, width = image_shape
    return find_closest_target_size(height, width, resolution)


def _load_info(root: Path) -> tuple[float, tuple[int, int]]:
    """fps and the (H, W) of the egocentric camera."""

    info = lerobot.info(root)
    fps = float(info.get("fps", 0.0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{root}: meta/info.json has invalid fps={fps!r}")
    feature = info.get("features", {}).get(lerobot.EGO_VIDEO)
    if feature is None:
        raise ValueError(f"{root}: meta/info.json has no {lerobot.EGO_VIDEO} feature")
    return fps, (int(feature["shape"][0]), int(feature["shape"][1]))


@functools.cache
def _horizon_steps(fps: float) -> dict[float, tuple[int, ...]]:
    """Per nominal horizon, the five frames' offsets from the anchor: the frames nearest the five target
    times. A horizon whose frames would miss a target by more than half a frame is left out."""

    steps = {}
    for horizon in FUTURE_HORIZONS:
        offsets = tuple(int(round(step * horizon * fps / (NUM_FRAMES - 1))) for step in range(NUM_FRAMES))
        errors = [abs(offset / fps - step / future_mrope_fps(horizon)) for step, offset in enumerate(offsets)]
        if all(a < b for a, b in zip(offsets, offsets[1:])) and max(errors) <= 0.5 / fps + 1e-6:
            steps[horizon] = offsets
    return steps


# ------------------------------------------------------------------ _Build.of, then _scan per job, then _merge
@dataclasses.dataclass(frozen=True)
class _Part:
    """The rows and frames of one scan job, its texts as codes into its own word lists."""

    rows: dict[str, np.ndarray]  # "task" holds a description code (-1: none), "subtask" a subtask code
    task_index: np.ndarray  # per row
    subtask_words: list[str]
    description_words: list[str]
    first_description: dict[int, str]  # task_index: the first description its frames carry here
    episodes: np.ndarray
    lengths: np.ndarray
    timestamps: np.ndarray


@dataclasses.dataclass(frozen=True)
class _Build:
    """A dataset's metadata and build arguments, as every scan job and the merge read them."""

    root: str
    dataset_id: str
    shot: Shot
    fps: float
    episodes: np.ndarray  # sorted episode indexes
    lengths: np.ndarray
    task_text: dict[int, str]  # task_index: tasks.jsonl text
    task_cameras: dict[int, int]  # task_index: an index into CAMERAS, where not the shot's camera
    kept_tasks: np.ndarray  # task indexes that give rows
    skipped_episodes: np.ndarray  # episode indexes that give none
    ends_done: bool  # an episode's last subtask is finished where it ends
    max_subtask_frames: float  # a longer subtask is not used as one (inf: no limit)
    anchor_every: int
    stored_goal_every_frames: int
    min_goal_frames: float

    @classmethod
    def of(
        cls,
        root: str | Path,
        *,
        dataset_id: str,
        shot: Shot,
        task_cameras: Mapping[int, str],
        tasks: Sequence[str],
        skipped_episodes: Sequence[int],
        ends_done: bool,
        max_subtask_seconds: float | None,
        anchor_every: int,
        stored_goal_every_frames: int,
        min_goal_seconds: float,
    ) -> _Build:
        """``build_query_table``'s arguments under their own names, checked against the dataset's metadata."""

        if anchor_every <= 0 or stored_goal_every_frames <= 0:
            raise ValueError("anchor_every and stored_goal_every_frames must be positive")
        root = Path(root).expanduser()
        episodes: dict[int, int] = {}
        for row in lerobot.read_jsonl(root / "meta" / "episodes.jsonl"):
            index, length = _integer(row.get("episode_index")), _integer(row.get("length"))
            if length <= 0 or index in episodes:
                raise ValueError(f"{root}: episode {index} is repeated or has length {length}")
            episodes[index] = length
        task_text: dict[int, str] = {}
        for row in lerobot.read_jsonl(root / "meta" / "tasks.jsonl"):
            index, task = _integer(row.get("task_index")), row.get("task")
            if not isinstance(task, str) or not task.strip() or index in task_text:
                raise ValueError(f"{root}: task_index={index} is repeated or has an empty task")
            task_text[index] = task.strip()
        if not task_text:
            raise ValueError(f"{root}: empty task table")
        task_cameras = {int(task): camera for task, camera in task_cameras.items()}
        if unknown := sorted(set(task_cameras) - set(task_text)):
            raise ValueError(f"{root}: task_cameras names unknown task indexes {unknown}")
        kept_tasks = sorted(
            index
            for index, task in task_text.items()
            if not tasks or any(fnmatch.fnmatchcase(task, glob) for glob in tasks)
        )
        if not kept_tasks:
            raise ValueError(f"{root}: no task matches {list(tasks)}")
        fps = _load_info(root)[0]
        if min_goal_seconds * fps < NUM_FRAMES - 1:
            raise ValueError(f"{root}: min_goal_seconds={min_goal_seconds} is under {NUM_FRAMES - 1} frames")
        return cls(
            root=str(root),
            dataset_id=_safe_id(dataset_id),
            shot=shot,
            fps=fps,
            episodes=np.asarray(sorted(episodes), dtype=np.int64),
            lengths=np.asarray([episodes[index] for index in sorted(episodes)], dtype=np.int64),
            task_text=task_text,
            task_cameras={task: CAMERAS.index(camera) for task, camera in task_cameras.items()},
            kept_tasks=np.asarray(kept_tasks, dtype=np.int64),
            skipped_episodes=np.asarray(sorted(skipped_episodes), dtype=np.int64),
            ends_done=ends_done,
            max_subtask_frames=math.inf if max_subtask_seconds is None else max_subtask_seconds * fps,
            anchor_every=anchor_every,
            stored_goal_every_frames=stored_goal_every_frames,
            min_goal_frames=min_goal_seconds * fps,
        )


def _scan(build: _Build, files: list[Path]) -> _Part:
    """The rows of ``files``, which hold whole episodes, after checking every rule the rows rely on."""

    # ---- 1. Frames: every array below holds one value per frame, in (episode, frame) order.
    table = pa.concat_tables([_read(path) for path in files], promote_options="permissive").unify_dictionaries()
    episode = _integers(table, "episode_index")
    position = np.minimum(np.searchsorted(build.episodes, episode), len(build.episodes) - 1)
    if unknown := np.flatnonzero(build.episodes[position] != episode).tolist():
        raise ValueError(f"{build.root}: data refers to unknown episode_index={episode[unknown[0]]}")
    episodes = np.unique(episode)
    lengths = build.lengths[np.searchsorted(build.episodes, episodes)]
    table = _clamp_final_subgoal(table, dict(zip(episodes.tolist(), lengths.tolist())))

    frame = _integers(table, "frame_index")
    order = np.lexsort((frame, episode))
    episode, frame = episode[order], frame[order]
    timestamp = table["timestamp"].to_numpy().astype(np.float64)[order]
    task_index = _integers(table, "task_index")[order]
    sub_task_index = _integers(table, "sub_task_index")[order]
    subgoal = _integers(table, "sub_goal_frame_index", nullable=True)[order]
    subtask, subtask_words = _text_codes(table["subtask_prompt"])
    subtask = subtask[order]
    description, description_words = np.full(len(order), -1), []
    if "task_description" in table.column_names:
        description, description_words = _text_codes(table["task_description"])
        description = description[order]

    # ---- 2. Checks: each ``require`` names the first frame breaking its rule.
    def require(ok: np.ndarray, problem: Callable[[int], str]) -> None:
        if not ok.all():
            row = int(np.flatnonzero(~ok)[0])
            raise ValueError(f"{build.root}: episode {episode[row]} frame {frame[row]}: {problem(row)}")

    _, first, counts = np.unique(episode, return_index=True, return_counts=True)
    length, data_rows = np.repeat(lengths, counts), np.repeat(counts, counts)
    require(data_rows == length, lambda row: f"{data_rows[row]} data rows, episodes.jsonl says {length[row]}")
    require(
        frame == np.arange(len(frame)) - np.repeat(first, counts), lambda row: "frame indexes are not contiguous from 0"
    )
    expected = np.repeat(timestamp[first], counts) + frame / build.fps
    require(
        np.isfinite(timestamp) & (np.abs(timestamp - expected) <= max(1e-6, 0.01 / build.fps)),
        lambda row: f"timestamp={timestamp[row]!r} does not follow fps={build.fps}",
    )
    require(np.isin(task_index, list(build.task_text)), lambda row: f"unknown task_index={task_index[row]}")

    annotated = np.repeat(np.logical_or.reduceat(subtask >= 0, first), counts)
    require(~annotated | (subtask >= 0), lambda row: "empty subtask_prompt in an annotated episode")
    # A frame's end: its subtask's completion frame, or without annotation its episode's last frame.
    end = np.where(annotated, subgoal, length - 1)
    require((frame <= end) & (end < length), lambda row: f"invalid subgoal frame {end[row]}")

    # ---- 3. Segments: a segment is a subtask of an episode; its end and text are fixed, and it ends on its
    # end. A subtask longer than max_subtask_frames counts as unannotated.
    key = episode * (int(sub_task_index.max() - sub_task_index.min()) + 1) + sub_task_index - sub_task_index.min()
    _, segment_first, segment = np.unique(key, return_index=True, return_inverse=True)
    segment_last = len(key) - 1 - np.unique(key[::-1], return_index=True)[1]
    text = np.where(annotated, subtask, description)
    require(
        (end == end[segment_first][segment]) & (text == text[segment_first][segment]),
        lambda row: f"subtask {sub_task_index[row]} changes within its segment",
    )
    require(end == frame[segment_last][segment], lambda row: f"subtask {sub_task_index[row]} does not end on its goal")
    long_step = annotated & ((frame[segment_last] - frame[segment_first] + 1)[segment] > build.max_subtask_frames)
    annotated, end = annotated & ~long_step, np.where(long_step, length - 1, end)

    # ---- 4. Rows: an anchor is a frame of a kept task and episode, every anchor_every frames of its segment.
    offset = frame - frame[segment_first][segment]
    anchor = (offset % build.anchor_every == 0) & np.isin(task_index, build.kept_tasks)
    anchor &= ~np.isin(episode, build.skipped_episodes)
    # A future row: an anchor where at least one horizon's last frame is still within its end (bit k: horizon k).
    bins = np.zeros(len(frame), dtype=np.uint8)
    for bit, horizon in enumerate(FUTURE_HORIZONS):
        if steps := _horizon_steps(build.fps).get(horizon):
            bins |= (frame + steps[-1] <= end).astype(np.uint8) << bit
    # A goal row: an annotated anchor every stored_goal_every_frames frames of its segment, at least
    # min_goal_frames before the end. Its clip starts anywhere up to latest_start (before the next goal row)
    # and ends on the end.
    emits_goal = anchor & annotated & (offset % build.stored_goal_every_frames == 0)
    emits_goal &= end - frame >= build.min_goal_frames
    if not build.ends_done:  # the last subtask's end is only where the recording stopped
        emits_goal &= end != length - 1
    latest_start = np.minimum(frame + build.stored_goal_every_frames - 1, end - math.ceil(build.min_goal_frames))

    # ---- 5. The part: the rows' columns, a goal row before the future row of its frame.
    at = np.r_[np.flatnonzero(emits_goal), np.flatnonzero(anchor & (bins > 0))]
    is_goal = np.arange(len(at)) < emits_goal.sum()
    by_frame = np.lexsort((~is_goal, at))
    at, is_goal = at[by_frame], is_goal[by_frame]
    camera = np.full(len(at), CAMERAS.index(build.shot.camera), dtype=np.uint8)
    for task, code in build.task_cameras.items():
        camera[task_index[at] == task] = code
    described = description >= 0
    described_tasks, first_described = np.unique(task_index[described], return_index=True)
    return _Part(
        rows={
            "episode": episode[at].astype(np.int32),
            "current": frame[at].astype(np.int32),
            "is_goal": is_goal,
            "end": end[at].astype(np.int32),
            "latest_start": latest_start[at].astype(np.int32),
            "bins": np.where(is_goal, 0, bins[at]).astype(np.uint8),
            "task": description[at].astype(np.int32),
            "subtask": np.where(annotated[at], subtask[at], -1).astype(np.int32),
            "camera": camera,
        },
        task_index=task_index[at],
        subtask_words=subtask_words,
        description_words=description_words,
        first_description={
            int(task): description_words[code]
            for task, code in zip(described_tasks, description[described][first_described])
        },
        episodes=episodes,
        lengths=counts,
        timestamps=timestamp,
    )


def _merge(build: _Build, parts: list[_Part]) -> QueryTable:
    """One table of all parts: their words in one text table, and every row's task resolved (a frame
    without a description takes its task's text, ``lerobot.pick_task_texts``)."""

    texts: dict[str, int] = {}

    def text_ids(words: list[str]) -> np.ndarray:
        """Per code, its index into ``texts``; code -1 maps to -1."""

        return np.asarray([texts.setdefault(word, len(texts)) for word in words] + [-1], dtype=np.int32)

    subtask_ids = [text_ids(part.subtask_words) for part in parts]
    description_ids = [text_ids(part.description_words) for part in parts]
    described = ((task, text) for part in parts for task, text in part.first_description.items())
    task_text = lerobot.pick_task_texts(build.task_text, described)
    task_indexes = np.asarray(sorted(task_text), dtype=np.int64)
    task_ids = np.asarray([texts.setdefault(task_text[index], len(texts)) for index in task_indexes.tolist()], np.int32)

    columns: dict[str, list[np.ndarray]] = {name: [] for name in _ROW_COLUMNS}
    for part, subtask, description in zip(parts, subtask_ids, description_ids):
        rows = dict(part.rows)
        own, of_task = description[rows["task"]], task_ids[np.searchsorted(task_indexes, part.task_index)]
        rows["task"] = np.where(own >= 0, own, of_task).astype(np.int32)
        rows["subtask"] = subtask[rows["subtask"]]
        for name in _ROW_COLUMNS:
            columns[name].append(rows[name])

    episodes = np.concatenate([part.episodes for part in parts])
    if np.any(np.diff(episodes) <= 0):
        raise ValueError(f"{build.root}: data files are not in episode order or repeat an episode")
    if missing := sorted(set(build.episodes.tolist()) - set(episodes.tolist())):
        raise ValueError(f"{build.root}: episodes {missing[:5]} have no data rows")
    lengths = np.concatenate([part.lengths for part in parts])
    return QueryTable(
        root=build.root,
        dataset_id=build.dataset_id,
        fps=build.fps,
        shot=build.shot,
        texts=tuple(texts),
        **{name: np.concatenate(values) for name, values in columns.items()},
        timelines=Timelines(
            episodes=episodes,
            offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64),
            timestamps=np.concatenate([part.timestamps for part in parts]),
        ),
    )


def _clamp_final_subgoal(table: pa.Table, lengths: Mapping[int, int]) -> pa.Table:
    """Read a final segment's ``sub_goal_frame_index`` past the episode as its last frame.

    Some Psi0.1 exports aim the last segment one or two frames past the episode; that
    segment ends with the episode, so its completion frame is the last one. A non-final segment
    aimed past the episode is left to fail validation.
    """

    if table.num_rows == 0:
        return table
    episode = table["episode_index"].to_numpy()
    frame = table["frame_index"].to_numpy()
    segment = table["sub_task_index"].to_numpy()
    unique, inverse = np.unique(episode, return_inverse=True)
    length = np.asarray([lengths.get(int(index), -1) for index in unique], dtype=np.int64)[inverse]
    # each episode's final segment is the one holding its highest frame
    order = np.lexsort((frame, inverse))
    last_rows = order[np.r_[np.flatnonzero(np.diff(inverse[order])), len(order) - 1]]
    final_segment = np.empty(len(unique), dtype=segment.dtype)
    final_segment[inverse[last_rows]] = segment[last_rows]
    goal_column = table["sub_goal_frame_index"]
    goal = pc.fill_null(goal_column, -1).to_numpy()  # null in an episode without annotation
    hit = (length > 0) & (goal >= length) & (segment == final_segment[inverse])
    if not hit.any():
        return table
    fixed = np.where(hit, length - 1, goal)
    return table.set_column(
        table.schema.get_field_index("sub_goal_frame_index"),
        "sub_goal_frame_index",
        pa.array(fixed, type=goal_column.type, mask=pc.is_null(goal_column).to_numpy(zero_copy_only=False)),
    )


def _read(path: Path) -> pa.Table:
    names = pq.read_schema(path).names
    if missing := [column for column in lerobot.REQUIRED_COLUMNS if column not in names]:
        raise KeyError(f"{path} is missing columns {missing}")
    columns = [*lerobot.REQUIRED_COLUMNS, *(column for column in lerobot.OPTIONAL_COLUMNS if column in names)]
    texts = [column for column in _TEXT_COLUMNS if column in columns]  # read as dictionaries: few distinct values
    return pq.read_table(path, columns=columns, read_dictionary=texts, use_threads=False)


def _integers(table: pa.Table, name: str, *, nullable: bool = False) -> np.ndarray:
    """An integer column; a null reads -1 when ``nullable``."""

    column = table[name]
    if not pa.types.is_integer(column.type):
        raise ValueError(f"{name} must be integers, got {column.type}")
    if column.null_count and not nullable:
        raise ValueError(f"{name} has {column.null_count} null values")
    return pc.fill_null(column, -1).to_numpy().astype(np.int64)


def _text_codes(column: pa.ChunkedArray) -> tuple[np.ndarray, list[str]]:
    """Each value's index into the column's distinct stripped texts (-1: null or blank), and those texts."""

    if pa.types.is_null(column.type):
        return np.full(len(column), -1, dtype=np.int64), []
    array = column.combine_chunks()
    if not pa.types.is_dictionary(array.type):
        array = array.dictionary_encode()
    codes = pc.fill_null(array.indices, -1).to_numpy().astype(np.int64)
    texts: dict[str, int] = {}
    words = [str(word).strip() for word in array.dictionary.to_pylist()]
    remap = [texts.setdefault(word, len(texts)) if word else -1 for word in words]
    return np.asarray(remap + [-1], dtype=np.int64)[codes], list(texts)


# ------------------------------------------------------------------ cache
def cache_path(
    root: str | Path, cache_dir: str | Path, index_cache: str | Path | None, build_args: Mapping[str, Any]
) -> Path:
    """Where ``root``'s table is cached, keyed by its real path, its metadata stamps, the data files its index
    was built from and the build arguments."""

    meta = Path(os.path.realpath(Path(root) / "meta" / "episodes.jsonl")).parent
    stamps = []
    for name in ("episodes.jsonl", "tasks.jsonl", "info.json"):
        stat = (meta / name).stat()
        stamps.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    data = lerobot.indexed_data(root, index_cache)
    key = json.dumps([_RULES, str(meta.parent), stamps, data, dict(build_args)], sort_keys=True)
    return Path(cache_dir) / f"{_safe_id(build_args['dataset_id'])}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def _save(table: QueryTable, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{socket.gethostname()}-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in _ROW_COLUMNS:
        np.save(tmp / f"{name}.npy", getattr(table, name))
    for name in _TIMELINE_COLUMNS:
        np.save(tmp / f"timeline_{name}.npy", getattr(table.timelines, name))
    meta = {
        "root": table.root,
        "dataset_id": table.dataset_id,
        "fps": table.fps,
        "shot": dataclasses.asdict(table.shot),
        "texts": table.texts,
    }
    (tmp / "meta.json").write_text(json.dumps(meta))
    if (path / "meta.json").is_file():  # another process cached it meanwhile: keep theirs
        shutil.rmtree(tmp, ignore_errors=True)
        return
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def _open(path: Path) -> QueryTable:
    meta = json.loads((path / "meta.json").read_text())
    return QueryTable(
        root=meta["root"],
        dataset_id=meta["dataset_id"],
        fps=meta["fps"],
        shot=Shot(**meta["shot"] | {"canvas": tuple(meta["shot"]["canvas"])}),
        texts=tuple(meta["texts"]),
        **{name: np.load(path / f"{name}.npy", mmap_mode="r") for name in _ROW_COLUMNS},
        timelines=Timelines(
            **{name: np.load(path / f"timeline_{name}.npy", mmap_mode="r") for name in _TIMELINE_COLUMNS}
        ),
    )


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected an integer, got {value!r}")
    return value


def _safe_id(dataset_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_id.strip()).strip("._")
    if not safe:
        raise ValueError("dataset_id must contain a letter or a number")
    return safe
