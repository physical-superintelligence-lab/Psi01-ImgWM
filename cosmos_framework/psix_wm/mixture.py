"""The training stream of an Image-WM run: one episode-shuffled stream per dataset, mixed per sample."""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from cosmos_framework.psix_wm.data_config import load
from cosmos_framework.psix_wm.dataset import ImageWMDataset
from cosmos_framework.psix_wm.queries import load_query_table
from cosmos_framework.utils import log


def _assign_blocks_by_size(lengths: np.ndarray, total_shards: int, seed: int) -> list[np.ndarray]:
    """Every episode block to one shard, largest first to the shard with the fewest queries.

    A dataset with fewer episodes than shards gives every shard one episode, dealt
    round-robin largest first, so some episodes are owned by more than one shard.
    """

    lengths = np.asarray(lengths, dtype=np.int64)
    if len(lengths) < total_shards:
        largest_first = np.argsort(-lengths, kind="stable")
        return [largest_first[[shard % len(lengths)]] for shard in range(total_shards)]
    if np.any(lengths <= 0):
        raise ValueError("every episode block needs at least one query")
    order = np.random.default_rng(seed).permutation(len(lengths))  # ties broken at random
    order = order[np.argsort(-lengths[order], kind="stable")]
    shards = [(0, 0, shard) for shard in range(total_shards)]  # (queries, blocks, shard), a heap
    owned: list[list[int]] = [[] for _ in range(total_shards)]
    for block in order.tolist():
        queries, blocks, shard = heapq.heappop(shards)
        owned[shard].append(block)
        heapq.heappush(shards, (queries + int(lengths[block]), blocks + 1, shard))
    return [np.asarray(blocks, dtype=np.int64) for blocks in owned]


class EpisodeShuffle(IterableDataset):
    """An endless stream over one dataset's queries.

    Each rank owns fixed, whole episodes (``shard_rank`` / ``shard_world_size``, set by the dataloader);
    its workers walk disjoint slices of one epoch order. An epoch walks the owned queries in a fresh
    random order: every future row once and every goal row ``goal_repeat`` times on average. A sample's
    draws follow from its epoch and its place in that order, so they do not depend on the workers.
    """

    def __init__(self, dataset: ImageWMDataset, seed: int = 42, goal_repeat: float = 1.0) -> None:
        super().__init__()
        if goal_repeat <= 0:
            raise ValueError(f"goal_repeat must be positive, got {goal_repeat}")
        self._dataset = dataset
        self._seed = int(seed)
        self._goal_repeat = float(goal_repeat)
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:  # informational only; iteration is infinite
        return len(self._dataset)

    def __iter__(self):
        starts, lengths = (np.asarray(column, dtype=np.int64) for column in self._dataset.episode_blocks())
        goal_mask = np.asarray(self._dataset.get_goal_mask(), dtype=bool)
        worker_info = get_worker_info()
        stride = worker_info.num_workers if worker_info is not None else 1
        phase = worker_info.id if worker_info is not None else 0
        shard, total_shards = int(self.shard_rank), max(1, int(self.shard_world_size))
        owned = _assign_blocks_by_size(lengths, total_shards, self._seed)[shard]
        owned_lengths = lengths[owned]
        block = np.repeat(np.arange(len(owned)), owned_lengths)  # each owned row's position in ``owned``
        first = np.cumsum(owned_lengths) - owned_lengths
        rows = starts[owned][block] + np.arange(len(block)) - first[block]
        goals = goal_mask[rows]
        epoch = 0
        while True:
            rng = np.random.default_rng((self._seed, epoch, shard))
            copies = np.ones(len(rows), dtype=np.int64)
            copies[goals] = int(self._goal_repeat) + (rng.random(int(goals.sum())) < self._goal_repeat % 1)
            walked = rng.permutation(np.repeat(rows, copies))
            # Fewer rows than workers: every worker walks the whole epoch rather than none at all, as its own copy.
            whole = len(walked) < stride
            for position in range(len(walked)) if whole else range(phase, len(walked), stride):
                yield self._dataset.draw(int(walked[position]), (epoch, position, phase if whole else 0))
            epoch += 1


class WeightedMixture(IterableDataset):
    """Every sample drawn from one stream, with probability proportional to its weight."""

    def __init__(
        self,
        members: dict[str, EpisodeShuffle],
        weights: dict[str, float],
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("WeightedMixture needs at least one member")
        if set(members) != set(weights):
            raise ValueError(f"weights keys {sorted(weights)} must match member keys {sorted(members)}")
        total = float(sum(weights.values()))
        if not total > 0 or any(weight < 0 for weight in weights.values()):
            raise ValueError(f"weights must be non-negative with a positive sum, got {weights}")
        self._names = sorted(members)
        self._members = [members[name] for name in self._names]
        self._weights = torch.tensor([weights[name] / total for name in self._names], dtype=torch.float64)
        self._seed = int(seed)
        self.shard_world_size = 1
        self.shard_rank = 0

    def __len__(self) -> int:  # informational only; iteration is infinite
        return sum(len(member) for member in self._members)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        for member in self._members:
            member.shard_world_size = self.shard_world_size
            member.shard_rank = self.shard_rank
        iterators = [iter(member) for member in self._members]
        generator = torch.Generator().manual_seed(
            self._seed + 1_000_003 * (int(self.shard_rank) * num_workers + worker_id)
        )
        while True:
            member_index = int(torch.multinomial(self._weights, 1, generator=generator).item())
            yield next(iterators[member_index])


def build_mixture(
    *, run_toml: str, tokenizer_config: dict | None, video_backend: str | None = None, run_dir: str | None = None
) -> WeightedMixture:
    """The training stream of a run TOML's ``[custom.data]``: one episode-shuffled stream per dataset.

    With ``weighting="share"`` a dataset draws ``weight / sum(weights)`` of the samples; with ``"rows"`` it
    weighs ``weight x`` the draws of one pass over its rows. Goal rows are repeated (or thinned) per pass
    to make ``sampling.goal_sample_share`` of a dataset's draws.
    """

    config = load(run_toml)
    # A resumed run draws a fresh stream instead of replaying the one it started with.
    seed = config.seed + _resumed_iteration(run_dir)
    text_forms = {"goal": dict(config.prompt.goal_forms), "future": dict(config.prompt.future_forms)}
    streams: dict[str, EpisodeShuffle] = {}
    weights: dict[str, float] = {}
    for dataset in config.datasets:
        name = dataset.name
        table = load_query_table(dataset.path, config.query_cache, config.index_cache, **config.build_args(dataset))
        rows = ImageWMDataset(
            table,
            resolution=config.resolution,
            tokenizer_config=tokenizer_config,
            video_backend=video_backend,
            text_forms=text_forms,
            horizon_weights=config.sampling.horizon_weights,
            draw_seed=seed,
        )
        goals = int(np.count_nonzero(table.is_goal))
        futures = len(table) - goals
        share = config.sampling.goal_sample_share
        goal_repeat = share / (1 - share) * futures / goals if goals and futures else 1.0
        units = futures + goal_repeat * goals  # the draws of one pass over the dataset
        weights[name] = dataset.weight * (units if config.weighting == "rows" else 1.0)
        streams[name] = EpisodeShuffle(rows, seed=seed, goal_repeat=goal_repeat)
        log.info(f"dataset {name}: {len(table):,} rows ({goals:,} goals, each drawn {goal_repeat:.2f}x per pass)")
    total = sum(weights.values())
    log.info("dataset shares: " + ", ".join(f"{name} {weight / total:.2%}" for name, weight in weights.items()))
    return WeightedMixture(streams, weights, seed=seed)


def _resumed_iteration(run_dir: str | None) -> int:
    """The iteration the run writing to ``run_dir`` resumes from: its latest checkpoint, else 0."""

    if not run_dir:
        return 0
    latest = Path(run_dir) / "checkpoints" / "latest_checkpoint.txt"
    return int(latest.read_text().strip().removeprefix("iter_")) if latest.is_file() else 0
