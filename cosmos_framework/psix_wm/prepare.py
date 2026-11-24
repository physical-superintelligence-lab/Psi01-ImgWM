"""Build a run's caches before it starts: the index and the query table of every dataset, and the fixed
validation plan. What exists is left alone::

    python -m cosmos_framework.psix_wm.prepare <run.toml>
"""

from __future__ import annotations

import os
import sys

from cosmos_framework.psix_wm import lerobot, queries, validation
from cosmos_framework.psix_wm.data_config import DataConfig, load


def prepare(config: DataConfig, workers: int) -> None:
    datasets = (*config.datasets, *config.held_out)
    for root in dict.fromkeys(dataset.path for dataset in datasets):
        print(f"index {lerobot.build_index(root, config.index_cache)}", flush=True)
    for dataset in datasets:
        tables = queries.prepare_query_tables(
            {dataset.path: config.build_args(dataset)}, config.query_cache, config.index_cache, workers
        )
        for path in tables:
            print(f"queries {path}", flush=True)
    if config.held_out:
        print(f"validation plan {validation.prepare_plan(config)}", flush=True)


if __name__ == "__main__":
    prepare(load(sys.argv[1]), workers=len(os.sched_getaffinity(0)))
