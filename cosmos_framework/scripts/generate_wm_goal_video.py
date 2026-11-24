"""World-model goal tracks for LeRobot v2.1 datasets: one mp4 per episode, aligned with its source video.

    wm[t] = source[t]                        for t <  seconds x fps
    wm[t] = WM(source[t - seconds x fps])    for t >= seconds x fps

Frame ``t`` is the same instant in both tracks, so the goal ``seconds`` ahead of step ``c`` is
``wm[c + seconds x fps]``: what the ``/wm`` server returns for observation ``c``, captioned the same
way (the future caption of the episode's task). A dataset's track is written to
``<output-dir>/<dataset>/videos/chunk-XXX/observation.images.egocentric_imgwm/episode_XXXXXX.mp4``;
finished episodes (here or installed) are skipped, so a rerun resumes. ``install_wm_goal_tracks`` then adds
them to their datasets.

    python -m cosmos_framework.scripts.generate_wm_goal_video --checkpoint-path <run> --output-dir <out> \\
        --datasets /data/cups/g1 [--shard 0 --num-shards 8]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import socket
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image

from cosmos_framework.inference.psix_wm.generator import Generator
from cosmos_framework.inference.psix_wm.sampler import PRESETS, Sampler
from cosmos_framework.psix_wm import lerobot
from cosmos_framework.psix_wm.caption import DEPLOYMENT_SECONDS, FUTURE_HORIZONS, SCHEMA, Shot, future_prompt
from cosmos_framework.psix_wm.checkpoint import latest_checkpoint
from cosmos_framework.psix_wm.data_config import Dataset, parse_dataset
from cosmos_framework.psix_wm.queries import dataset_shot

WM_KEY = "observation.images.egocentric_imgwm"


@dataclass(frozen=True)
class Episode:
    dataset: Dataset
    index: int
    length: int
    task: str
    camera: str


def plan_episodes(datasets: list[Dataset], shard: int, num_shards: int, index_cache: Path) -> list[Episode]:
    """This shard's episodes; shards are balanced by frame count (longest first, to the lightest shard)."""

    episodes = []
    for dataset in datasets:
        metadata = lerobot.load_dataset_metadata(dataset.path)
        skipped = set(metadata.skipped_episodes)
        patterns = dataset.task_names()
        tasks = lerobot.read_jsonl(Path(dataset.path) / "meta/tasks.jsonl")
        selected_tasks = {
            row["task_index"]
            for row in tasks
            if not patterns or any(fnmatch.fnmatchcase(row["task"], pattern) for pattern in patterns)
        }
        index_of_task = {row["task"]: row["task_index"] for row in tasks}
        task_index = {
            row["episode_index"]: _episode_task_index(dataset, row, index_of_task)
            for row in lerobot.read_jsonl(Path(dataset.path) / "meta" / "episodes.jsonl")
        }
        for row in lerobot.episode_rows(dataset.path, index_cache):
            if row["episode_index"] in skipped or task_index[row["episode_index"]] not in selected_tasks:
                continue
            # The episode's own instruction, as in training; its task's text when it has none.
            task = str(row.get("task_description") or "").strip() or str((row.get("tasks") or [""])[0]).strip()
            if not task:
                raise ValueError(f"{dataset.path}: episode {row['episode_index']} has no task text")
            camera = metadata.task_cameras.get(task_index[row["episode_index"]], metadata.camera)
            episodes.append(Episode(dataset, int(row["episode_index"]), int(row["length"]), task, camera))
    shards: list[list[Episode]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for episode in sorted(episodes, key=lambda e: (-e.length, e.dataset.name, e.index)):
        lightest = loads.index(min(loads))
        shards[lightest].append(episode)
        loads[lightest] += episode.length
    return shards[shard]


def _episode_task_index(dataset: Dataset, row: dict, index_of_task: dict[str, int]) -> int:
    """The task_index of an ``episodes.jsonl`` row's first task: an index (Psi0.1 exports) or a ``tasks.jsonl``
    task text (stock LeRobot v2.1)."""

    task = (row.get("tasks") or [None])[0]
    if isinstance(task, str):
        task = index_of_task.get(task)
    if not isinstance(task, int):
        raise ValueError(f"{dataset.path}: episode {row['episode_index']} names no task of meta/tasks.jsonl")
    return task


def predict_track(
    generator: Generator,
    episode: Episode,
    frames: list[Image.Image],
    *,
    shot: Shot,
    fps: float,
    seconds: float,
    sampler: Sampler,
    batch: int,
    pad: bool,
) -> list[Image.Image]:
    """The episode's goal track: its first ``seconds`` copied from the source, then frame ``t`` predicted
    from source frame ``t - seconds x fps``. ``pad`` fills a short last batch to ``batch``."""

    offset = round(seconds * fps)
    caption, mrope_fps = future_prompt(shot, episode.task, seconds)
    items = [
        (frames[t - offset], caption, mrope_fps, (episode.index * 1_000_003 + t) % (2**31 - 1))
        for t in range(offset, len(frames))
    ]
    predicted = []
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        padded = chunk + [chunk[-1]] * (batch - len(chunk)) if pad else chunk
        clips = generator.generate(padded, canvas=shot.canvas, sampler=sampler)
        predicted += [clip[-1].resize(frames[0].size, Image.BICUBIC) for clip in clips[: len(chunk)]]
    return frames[:offset] + predicted


def track_path(output_dir: Path, episode: Episode) -> Path:
    return (
        output_dir
        / episode.dataset.name
        / "videos"
        / f"chunk-{episode.index // 1000:03d}"
        / WM_KEY
        / f"episode_{episode.index:06d}.mp4"
    )


def read_frames(path: Path) -> list[Image.Image]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return [frame.to_image() for frame in container.decode(stream)]


def write_video(path: Path, frames: list[Image.Image], fps: float, crf: int) -> None:
    """Encode to a temporary file, then rename: the final path only ever appears complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.mp4.partial")
    try:
        with av.open(str(partial), mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
            stream.width, stream.height = frames[0].size
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf)}
            for image in frames:
                container.mux(stream.encode(av.VideoFrame.from_image(image)))
            container.mux(stream.encode())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", help="default: config.yaml of the checkpoint's folder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True, help="LeRobot v2.1 datasets, each with meta/psix.json")
    parser.add_argument(
        "--index-cache",
        type=Path,
        help="reuse the training psix_wm_index directory; default: <output-dir>/_cache/psix_wm_index",
    )
    parser.add_argument("--seconds", type=float, default=DEPLOYMENT_SECONDS, choices=FUTURE_HORIZONS)
    parser.add_argument("--preset", default="fast", choices=sorted(PRESETS))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--compile", action="store_true", help="compile the model and capture CUDA graphs (pays off on long runs)"
    )
    args = parser.parse_args()

    datasets = [parse_dataset(entry) for entry in args.datasets]
    index_cache = args.index_cache or args.output_dir / "_cache" / "psix_wm_index"
    for root in dict.fromkeys(dataset.path for dataset in datasets):
        print(f"index {lerobot.build_index(root, index_cache)}", flush=True)
    episodes = plan_episodes(datasets, args.shard, args.num_shards, index_cache)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    generator = Generator(
        args.checkpoint_path,
        args.config_file,
        args.output_dir / "_work" / f"shard{args.shard:03d}",
        compile=args.compile,
        max_batch=args.batch,
    )
    sampler = PRESETS[args.preset]
    shots, fps = {}, {}
    for dataset in datasets:
        metadata = lerobot.load_dataset_metadata(dataset.path)
        shots[dataset.name] = dataset_shot(
            dataset.path, metadata.agent, metadata.camera, metadata.control, generator.resolution
        )
        fps[dataset.name] = float(lerobot.info(dataset.path)["fps"])
        meta = {
            "video_key": WM_KEY,
            "source_video_key": lerobot.EGO_VIDEO,
            "source_root": str(Path(dataset.path).resolve()),
            "camera": metadata.camera,
            "seconds": args.seconds,
            "head_frames": round(args.seconds * fps[dataset.name]),
            "fps": fps[dataset.name],
            "canvas": list(shots[dataset.name].canvas),
            "sampler": vars(sampler),
            "caption_schema": SCHEMA,
            "checkpoint": str(latest_checkpoint(args.checkpoint_path)),
            "crf": args.crf,
        }
        info_file = args.output_dir / dataset.name / "meta" / "wm_info.json"
        # Finished tracks are kept on a rerun, so one dataset's tracks must come from one model and setting.
        if info_file.is_file() and json.loads(info_file.read_text()) != json.loads(json.dumps(meta)):
            raise SystemExit(f"{info_file} describes tracks made with other settings; use a new --output-dir")
        info_file.parent.mkdir(parents=True, exist_ok=True)
        # Shards start together, on several nodes: never half a file, never another shard's partial.
        partial = info_file.with_suffix(f".{socket.gethostname()}-{os.getpid()}.json.partial")
        partial.write_text(json.dumps(meta, indent=2))
        os.replace(partial, info_file)
    # Finished: written here, or already installed into the dataset by install_wm_goal_tracks.
    todo = [
        episode
        for episode in episodes
        if not track_path(args.output_dir, episode).is_file()
        and not lerobot.episode_file(episode.dataset.path, WM_KEY, episode.index).is_file()
    ]
    print(f"shard {args.shard}/{args.num_shards}: {len(episodes)} episodes, {len(todo)} to generate", flush=True)
    # This run's failures only: an episode that failed before is in ``todo`` again.
    failed_list = args.output_dir / f"failed_shard{args.shard:03d}.txt"
    failed_list.unlink(missing_ok=True)

    def source_frames(episode: Episode) -> list[Image.Image]:
        return read_frames(lerobot.episode_file(episode.dataset.path, lerobot.EGO_VIDEO, episode.index))

    # Reading the next episode and encoding the last one run beside the GPU.
    reader, writer = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    upcoming = reader.submit(source_frames, todo[0]) if todo else None
    writing: Future | None = None
    start, generated = time.perf_counter(), 0
    for number, episode in enumerate(todo):
        reading = upcoming
        if number + 1 < len(todo):
            upcoming = reader.submit(source_frames, todo[number + 1])
        name = episode.dataset.name
        try:
            frames = reading.result()
            # The track mirrors the source video frame for frame, so a timestamp reads the same instant from both;
            # a video may hold a few frames more than the episode has rows (rows trimmed from its start).
            if len(frames) < episode.length:
                raise ValueError(f"{len(frames)} frames, metadata says {episode.length}")
        except (ValueError, OSError, av.error.FFmpegError) as error:  # an unreadable episode is listed and skipped
            with failed_list.open("a") as failed:
                failed.write(f"{name} {episode.index}: {error}\n")
            continue
        # Compiled graphs are captured per batch size, so a short last batch is padded.
        track = predict_track(
            generator,
            episode,
            frames,
            shot=replace(shots[name], camera=episode.camera),
            fps=fps[name],
            seconds=args.seconds,
            sampler=sampler,
            batch=args.batch,
            pad=args.compile,
        )
        if writing is not None:
            writing.result()
        writing = writer.submit(write_video, track_path(args.output_dir, episode), track, fps[name], args.crf)
        generated += len(frames)
        rate = (time.perf_counter() - start) / generated
        print(
            f"{name} episode {episode.index}: {len(frames)} frames, {rate:.3f} s/frame, "
            f"{number + 1}/{len(todo)} episodes",
            flush=True,
        )
    if writing is not None:
        writing.result()
    reader.shutdown()
    writer.shutdown()
    print(f"shard {args.shard} done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
