"""Adds the goal tracks of ``generate_wm_goal_video`` to their source LeRobot v2.1 datasets, ready to load.

For each dataset under ``--tracks`` (its ``meta/wm_info.json`` names the source), nothing changes unless every
episode the dataset uses has a track (here or already installed), any existing registration matches, and every
new track decodes, has its source video's frame count and repeats the source's first ``head_frames``. Then the
tracks move into the source's ``videos/`` (no existing video is replaced) and the source registers them:

    meta/wm_info.json   how the tracks were made
    meta/info.json      the track's feature (the source camera's, under the track key); total_videos
    meta/modality.json  video.<--modality-key> -> the track key, the name a training recipe asks for

The first install copies ``info.json`` and ``modality.json`` to ``<tracks>/<dataset>/meta_backup/``. A rerun
finishes an interrupted one. Run one install per dataset at a time.

    python -m cosmos_framework.scripts.install_wm_goal_tracks --tracks <generate output-dir> \\
        [--modality-key egocentric_wm] [--workers 8]
"""

from __future__ import annotations

import argparse
import errno
import filecmp
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np

from cosmos_framework.psix_wm import lerobot


def read_head(path: Path, head: int) -> tuple[np.ndarray, int]:
    """The first ``head`` frames of a video and its frame count, decoding without keeping the rest."""

    frames, count = [], 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if count < head:
                frames.append(frame.to_ndarray(format="rgb24"))
            count += 1
    return np.stack(frames).astype(np.float32), count


def check_track(track: Path, source: Path, head: int) -> str | None:
    """Why ``track`` cannot be installed, or None."""

    try:
        (wm, wm_count), (src, src_count) = read_head(track, head), read_head(source, head)
    except (OSError, ValueError, av.error.FFmpegError) as error:
        return f"{track}: {error}"
    if wm_count != src_count:
        return f"{track}: {wm_count} frames, source {src_count}"
    psnr = 10 * np.log10(255.0**2 / max(float(np.mean((wm - src) ** 2)), 1e-10))
    # 35 dB: the copied head after one more h264 pass; a different video scores far below.
    return f"{track}: head PSNR {psnr:.1f} dB against the source" if psnr < 35.0 else None


def move(track: Path, target: Path) -> None:
    """``track`` to ``target``: renamed on one filesystem, else copied beside it first so it appears whole."""

    target.parent.mkdir(exist_ok=True)
    try:
        os.rename(track, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        partial = target.with_name(target.name + ".partial")
        shutil.copyfile(track, partial)
        os.replace(partial, target)
        track.unlink()


def write_json(path: Path, value: dict) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=4))  # as LeRobot writes them
    os.replace(partial, path)


def install(output: Path, modality_key: str, workers: int) -> None:
    wm_info = json.loads((output / "meta/wm_info.json").read_text())
    root, key, source_key = Path(wm_info["source_root"]), wm_info["video_key"], wm_info["source_video_key"]
    meta = root / "meta"
    info, modality = json.loads((meta / "info.json").read_text()), json.loads((meta / "modality.json").read_text())
    feature, entry = info["features"][source_key], {"original_key": key}
    if (meta / "wm_info.json").exists() and json.loads((meta / "wm_info.json").read_text()) != wm_info:
        raise SystemExit(f"{meta / 'wm_info.json'} describes tracks made with other settings")
    for registered, expected, where in (
        (info["features"].get(key), feature, f"info.json features.{key}"),
        (modality["video"].get(modality_key), entry, f"modality.json video.{modality_key}"),
    ):
        if registered not in (None, expected):
            raise SystemExit(f"{meta}: {where} is {registered}, expected {expected}")

    tracks = {int(t.stem.removeprefix("episode_")): t for t in sorted(output.glob(f"videos/*/{key}/*.mp4"))}
    skipped = set(lerobot.load_dataset_metadata(root).skipped_episodes)
    episodes = [row["episode_index"] for row in lerobot.read_jsonl(Path(root) / "meta" / "episodes.jsonl")]
    used = [episode for episode in episodes if episode not in skipped]
    if missing := [e for e in used if e not in tracks and not lerobot.episode_file(root, key, e).exists()]:
        raise SystemExit(f"{root}: {len(missing)} episodes have no track yet, e.g. {missing[:10]}")
    new, done = {}, []  # done: moved by an interrupted install, its copy still here
    for episode, track in tracks.items():
        target = lerobot.episode_file(root, key, episode)
        if not target.exists():
            new[episode] = track
        elif filecmp.cmp(track, target, shallow=False):
            done.append(track)
        else:
            raise SystemExit(f"{target} exists and differs from {track}")
    with ProcessPoolExecutor(workers) as pool:
        sources = [lerobot.episode_file(root, source_key, episode) for episode in new]
        checks = pool.map(check_track, new.values(), sources, [wm_info["head_frames"]] * len(new), chunksize=4)
        if bad := [error for error in checks if error]:
            raise SystemExit(f"{root}: {len(bad)} tracks fail the check:\n" + "\n".join(bad))

    backup = output / "meta_backup"
    if not backup.exists():
        partial = output / "meta_backup.partial"  # renamed when complete
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir()
        for name in ("info.json", "modality.json"):
            shutil.copy2(meta / name, partial / name)
        partial.rename(backup)
    for episode, track in new.items():
        move(track, lerobot.episode_file(root, key, episode))
    for track in done:
        track.unlink()
    info["features"][key] = feature
    info["total_videos"] = info["total_episodes"] * sum(f["dtype"] == "video" for f in info["features"].values())
    modality["video"][modality_key] = entry
    write_json(meta / "info.json", info)
    write_json(meta / "modality.json", modality)
    if not (meta / "wm_info.json").exists():
        write_json(meta / "wm_info.json", wm_info)
    print(f"{root}: {len(new)} tracks installed, every episode it uses now has one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", type=Path, required=True, help="generate_wm_goal_video's --output-dir")
    parser.add_argument("--modality-key", default="egocentric_wm", help="the video name a training recipe asks for")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for wm_info in sorted(args.tracks.glob("*/meta/wm_info.json")):
        install(wm_info.parent.parent, args.modality_key, args.workers)


if __name__ == "__main__":
    main()
