"""Five-frame Image-WM samples from one dataset's query table, and their tokenized form.

A sample is a query row (``cosmos_framework.psix_wm.queries``): its caption and the five source
frames decoded from the egocentric video. In training each draw also picks a future's horizon (a bin
its row fits, then a horizon inside it), a goal's first frame, and one text form of the caption.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.psix_wm import lerobot
from cosmos_framework.psix_wm.caption import (
    HORIZON_BINS,
    HORIZON_STEP,
    NUM_FRAMES,
    TEXT_FORMS,
    canonical,
    frame_indexes,
    narrow,
)
from cosmos_framework.psix_wm.queries import Query, QueryTable


class ImageWMDataset(Dataset):
    """The rows of one query table with their frames, through the model's text and image transform.

    ``dataset[row]`` is the row as built: a future row must fit one horizon only. ``draw(row, occurrence)``
    picks a future's nominal horizon among those its row fits (by ``horizon_weights``) and a horizon inside
    its bin, a goal's start up to the next goal row's, and one text form (``text_forms``: the probability
    of each form of ``caption.TEXT_FORMS``), each from its own generator (``_rng``).
    """

    def __init__(
        self,
        table: QueryTable,
        *,
        resolution: str,
        tokenizer_config: dict | None = None,
        video_backend: str | None = None,
        text_forms: Mapping[str, Mapping[str, float]] | None = None,
        horizon_weights: Mapping[str, float] | None = None,
        draw_seed: int | None = None,
    ) -> None:
        super().__init__()
        if not len(table):
            raise ValueError(f"{table.root}: no query rows")
        self.table = table
        self._root = Path(table.root)
        self._resolution = resolution
        # Upstream's transform, here for the resize and pad, the caption's tokens (``tokenizer_config`` is the
        # text tokenizer) and the sequence plan; the samples' mode "image2video" carries no action.
        self._transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=0.0,
            append_viewpoint_info=False,
            append_duration_fps_timestamps=False,
            append_resolution_info=False,
            append_idle_frames=False,
        )
        self._video_backend = video_backend
        self._draw_seed = draw_seed
        self._text_forms = text_forms and {
            kind: {form: float(text_forms[kind].get(form, 0.0)) for form in forms} for kind, forms in TEXT_FORMS.items()
        }
        self._horizon_weights = {float(horizon): float(weight) for horizon, weight in (horizon_weights or {}).items()}
        self._fallback_episodes: set[int] = set()  # warned about once each
        episodes = np.asarray(table.episode)
        starts = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
        self._blocks = (starts, np.diff(np.r_[starts, len(episodes)]))

    def _timestamps(self, episode: int, frame_indexes: Sequence[int]) -> np.ndarray:
        """The timestamps of an episode's frames, which the video is decoded at."""

        timestamps = self.table.timelines[episode]
        requested = np.asarray(frame_indexes, dtype=np.int64)
        if requested.min() < 0 or requested.max() >= len(timestamps):
            raise ValueError(f"{self._root}: episode {episode} has no frames {requested.tolist()}")
        return np.asarray(timestamps[requested], dtype=np.float64)

    def _rng(self, stream: str, idx: int, occurrence: tuple[int, ...]) -> np.random.Generator:
        """The generator of one draw, seeded by (seed, row, occurrence, stream): a row gives a new sample at
        each occurrence, the same in every run and on any number of workers, and its streams are independent."""

        salt = int.from_bytes(stream.encode(), "big")
        return np.random.default_rng((self._draw_seed, idx, *occurrence, salt))

    # ------------------------------------------------------------------ draw-time rewrites
    def _draw_horizon(self, idx: int, occurrence: tuple[int, ...]) -> float:
        """A nominal horizon among those the row fits, by ``horizon_weights`` (even when unset)."""

        horizons = self.table.horizons(idx)
        if len(horizons) == 1:
            return horizons[0]
        weights = np.asarray([self._horizon_weights.get(horizon, 1.0) for horizon in horizons])
        return horizons[int(self._rng("bin", idx, occurrence).choice(len(horizons), p=weights / weights.sum()))]

    def _resample_horizon(self, query: Query, idx: int, occurrence: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """A horizon drawn inside the row's bin, and the frames nearest it; the bin stays the mRoPE rate."""

        low, high = HORIZON_BINS[query.horizon]
        margin = self._seconds_left(query.episode_index, query.frame_indexes[0])
        tick_low = math.ceil(low / HORIZON_STEP - 1e-9)
        tick_high = math.floor(min(high, margin) / HORIZON_STEP + 1e-9)
        if tick_high < tick_low:
            return query.caption, query.frame_indexes
        sampled = float(self._rng("horizon", idx, occurrence).uniform(low, min(high, margin)))
        horizon = round(min(max(int(round(sampled / HORIZON_STEP)), tick_low), tick_high) * HORIZON_STEP, 6)
        if math.isclose(horizon, query.horizon, abs_tol=1e-9):
            return query.caption, query.frame_indexes
        try:
            new_frames = self._future_frames(query.episode_index, query.frame_indexes[0], horizon)
        except ValueError:
            if query.episode_index not in self._fallback_episodes:
                self._fallback_episodes.add(query.episode_index)
                logging.warning("episode %s keeps its nominal horizon frames (uneven timeline)", query.episode_index)
            return query.caption, query.frame_indexes
        return self.table.future_caption_at(idx, query.horizon, horizon), new_frames

    def _resample_goal_start(self, query: Query, idx: int, occurrence: tuple[int, ...]) -> tuple[int, ...]:
        """The goal clip from a start drawn between the row's anchor and the next goal row's."""

        start = int(self._rng("goal", idx, occurrence).integers(query.frame_indexes[0], query.latest_start + 1))
        return frame_indexes(start, query.frame_indexes[-1])

    def _narrow_text(self, caption: str, idx: int, occurrence: tuple[int, ...], past_subtask: bool) -> str:
        """One text form for this draw; the row holds the widest form its data allows."""

        decoded = json.loads(caption)
        kind = decoded["prediction"]["type"]
        if kind == "future" and "context" not in decoded:
            return caption
        if past_subtask:
            return canonical(narrow(decoded, "task"))
        if not self._text_forms:
            return caption
        forms = self._text_forms["goal" if kind == "subgoal" else "future"]
        names = list(forms)
        rng = self._rng("text", idx, occurrence)
        form = names[int(rng.choice(len(names), p=[forms[name] for name in names]))]
        return caption if form == "both" else canonical(narrow(decoded, form))

    def _future_frames(self, episode: int, start_frame: int, seconds: float) -> tuple[int, ...]:
        """The five frames spanning ``seconds`` from ``start_frame`` at even steps, nearest the target times."""

        timestamps = np.asarray(self.table.timelines[episode], dtype=np.float64)
        rate = (NUM_FRAMES - 1) / seconds
        # Off-grid horizons land up to half a frame from a source frame; 0.75 frame admits those.
        return _frames_at_rate(
            np.arange(len(timestamps)), timestamps, start_frame, rate, tolerance=0.75 / self.table.fps
        )

    def _seconds_left(self, episode: int, frame: int) -> float:
        """From ``frame`` to the episode's last frame."""

        return float(self.table.timelines[episode][-1]) - float(self._timestamps(episode, [frame])[0])

    def decode(self, episode: int, frame_indexes: Sequence[int]) -> torch.Tensor:
        """Frames of an episode, [T, C, H, W] in [0, 1]."""

        return decode_video_frames(
            lerobot.episode_file(self._root, lerobot.EGO_VIDEO, episode),
            [float(t) for t in self._timestamps(episode, frame_indexes)],
            2e-3,  # how far a decoded frame's timestamp may be from the requested one, in seconds
            backend=self._video_backend,
        )

    # ------------------------------------------------------------------ dataset protocol
    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._sample(int(idx), None)

    def draw(self, idx: int, occurrence: tuple[int, ...]) -> dict[str, Any]:
        """The row with the draws of one occurrence of it in the training stream (``EpisodeShuffle``)."""

        return self._sample(idx, occurrence)

    def _sample(self, idx: int, occurrence: tuple[int, ...] | None) -> dict[str, Any]:
        """The row as built, with an ``occurrence`` its horizon or goal start and its text form drawn; then
        the frames decoded and the sample through the transform."""

        drawn = occurrence is not None
        horizon = self._draw_horizon(idx, occurrence) if drawn and not self.table.is_goal[idx] else None
        query = self.table.query(idx, horizon)
        caption, frame_indexes = query.caption, query.frame_indexes
        if drawn and query.query_type == "future":
            caption, frame_indexes = self._resample_horizon(query, idx, occurrence)
        if drawn and query.query_type == "subgoal":
            frame_indexes = self._resample_goal_start(query, idx, occurrence)
        if drawn:
            # A horizon resampled past the subtask's end shows the next one, so the caption names only the task.
            past_subtask = query.subtask_end is not None and frame_indexes[-1] > query.subtask_end
            caption = self._narrow_text(caption, idx, occurrence, past_subtask)
        video = self.decode(query.episode_index, frame_indexes)
        sample = {
            "__key__": query.query_id,
            "ai_caption": caption,
            "video": (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3),
            # mRoPE keeps the bin's rate; the caption seconds and pixels carry a resampled horizon.
            "conditioning_fps": torch.tensor(query.mrope_fps, dtype=torch.float32),
            "mode": "image2video",
            "episode_index": torch.tensor(query.episode_index, dtype=torch.long),
            "source_frame_indexes": torch.tensor(frame_indexes, dtype=torch.long),
        }
        return self._transform(sample, self._resolution)

    def query_ids(self) -> list[str]:
        """Every row's id as built, in row order: validation finds its planned rows by them."""

        return [self.table.query_id(row) for row in range(len(self.table))]

    def episode_blocks(self) -> tuple[np.ndarray, np.ndarray]:
        """The start and length of each run of one episode's rows: ``EpisodeShuffle`` deals whole episodes."""

        return self._blocks

    def get_goal_mask(self) -> np.ndarray:
        """Per row, whether it is a goal row: ``EpisodeShuffle`` repeats those."""

        return np.asarray(self.table.is_goal)


def _frames_at_rate(
    frames: np.ndarray, timestamps: np.ndarray, start_frame: int, rate: float, *, tolerance: float
) -> tuple[int, ...]:
    """Five strictly increasing frames from ``start_frame`` nearest ``start + k / rate`` (least total error)."""

    start = int(np.searchsorted(frames, start_frame))
    targets = float(timestamps[start]) + np.arange(NUM_FRAMES, dtype=np.float64) / rate
    states: dict[int, tuple[float, tuple[int, ...]]] = {start: (0.0, (start,))}
    for target in targets[1:]:
        first = int(np.searchsorted(timestamps, target - tolerance, side="left"))
        last = int(np.searchsorted(timestamps, target + tolerance, side="right"))
        previous = sorted(states.items())
        cursor, best, next_states = 0, None, {}
        for candidate in range(first, last):
            while cursor < len(previous) and previous[cursor][0] < candidate:
                if best is None or previous[cursor][1] < best:
                    best = previous[cursor][1]
                cursor += 1
            if best is not None:
                next_states[candidate] = (
                    best[0] + abs(float(timestamps[candidate]) - float(target)),
                    best[1] + (candidate,),
                )
        if not next_states:
            raise ValueError("no strictly increasing frame assignment near the target timestamps")
        states = next_states
    return tuple(int(frames[position]) for position in min(states.values())[1])
