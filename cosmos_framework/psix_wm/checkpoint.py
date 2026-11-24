"""A run's folder, or a released checkpoint's: ``config.yaml`` and ``checkpoints/iter_*``.

python -m cosmos_framework.psix_wm.checkpoint <folder>    # prints its latest checkpoint
"""

from __future__ import annotations

import sys
from pathlib import Path


def latest_checkpoint(path: str | Path) -> Path:
    """``path`` when it is a checkpoint itself; for a folder with ``checkpoints/``, the one its
    ``latest_checkpoint.txt`` names, else its highest ``iter_*``."""

    folder = Path(path) / "checkpoints"
    if not folder.is_dir():
        return Path(path)
    named = folder / "latest_checkpoint.txt"
    found = [folder / named.read_text().strip()] if named.is_file() else sorted(folder.glob("iter_*"))
    if not found:
        raise FileNotFoundError(f"{folder} holds no checkpoint")
    return found[-1]


if __name__ == "__main__":
    print(latest_checkpoint(sys.argv[1]))
