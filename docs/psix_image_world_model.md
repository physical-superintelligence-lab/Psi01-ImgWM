# Ψ₀.₁ Image World Model: reference

From one egocentric frame and an instruction, the model generates the next four frames of a five-frame clip:
a **future** (the scene `h` seconds ahead, 0.8 to 4.0 s) or a **goal** (the current subtask complete).

| Path under `cosmos_framework/` | Role |
| --- | --- |
| [`psix_wm/`](../cosmos_framework/psix_wm) | Captions, LeRobot reading, query tables, the dataset and mixture, `[custom.data]`, the prepare step, validation |
| `configs/base/experiment/sft/psix_image_wm.py` | One experiment per run TOML; `[custom.train]` |
| `callbacks/psix_wm_val_metrics.py` | In-loop validation |
| `inference/psix_wm/` | A loaded checkpoint, sampler presets, the `POST /wm` server |
| `scripts/eval_image_wm.py`, `serve_image_wm.py` | Offline evaluation; serving |
| `scripts/generate_wm_goal_video.py`, `install_wm_goal_tracks.py` | Goal tracks |

Shared Cosmos code changed in a few places: the `448` resolution tier, the sampling speed-ups `batched_cfg` and
`und_kv_cache`, and the training speed-ups of `[custom.train]`.

## Data

In the Ψ₀.₁ repository, `python scripts/data/prepare_psix_data.py` prepares the data root; its
[`DATA_PREPARATION.md`](https://github.com/physical-superintelligence-lab/Psi0/blob/main/scripts/data/DATA_PREPARATION.md) has the layout and the format for your own data.

A dataset is a LeRobot v2.1 directory with a `meta/psix.json`, of which the model reads
([`lerobot.DatasetMetadata`](../cosmos_framework/psix_wm/lerobot.py)):

```json
{"agent": "g1", "camera": "zed_mini_neck", "control": "teleop", "anchor_every": 1, "task_cameras": {"21": "zed_mini"},
 "skipped_episodes": {}, "terminal_is_done": true, "max_subtask_seconds": null, "capabilities": {"imgwm_future": true}}
```

[`queries.py`](../cosmos_framework/psix_wm/queries.py) builds each dataset's rows once: a future row at every
`anchor_every`-th frame and goal rows inside each subtask. An episode without subtask annotation gives only
futures of its task; one under `skipped_episodes` gives nothing. To use a dataset you cannot write into, point
the run at a folder whose `data/`, `videos/` and `meta/*` link to it and which adds `meta/psix.json` (goal
tracks are written into the dataset's `videos/`, so they need one you can write).

## A run

A run is one TOML in [`examples/toml/sft_config/psix_image_wm/`](../examples/toml/sft_config/psix_image_wm)
(`PSIX_WM_RUN_TOMLS` names another folder): the schedule, as in any SFT recipe, and the data:

```toml
[job]
experiment = "psix_image_wm_my_run"

[custom.data]
resolution = "448"                    # the model's resolution tier
data_root = "/data/lerobot"           # relative dataset paths are under it
cache = ".cache/psix_wm"              # index caches, query tables and validation plans
# a {path, tasks} entry uses only its tasks matching a glob
dataset_paths = ["egodex", { path = "finetune/cups/g1", tasks = ["cup_*"] }]
dataset_weights = [1.0, 2.0]          # optional, 1.0 each
weighting = "rows"                    # or "share"
val_dataset_paths = ["finetune/cups_val/g1"]   # held-out datasets, to monitor training

[custom.data.sampling]
stored_goal_every_frames = 9          # building rows: store one goal row per 9 frames of a subtask
min_goal_seconds = 0.8                # a goal clip starts at least this long before its subtask ends
goal_sample_share = 0.36              # drawing samples: goals' share of a dataset's samples
horizon_weights = { "0.8" = 0.125, "1.6" = 0.375, "3.2" = 0.5 }

[custom.data.prompt]                  # how often each text form is drawn
goal_forms = { both = 0.25, subtask = 0.75 }
future_forms = { both = 0.35, subtask = 0.45, task = 0.20 }

[custom.data.validation]
every = 1000                          # validation loss every N steps
generate_every = 1000                 # generation metrics every N steps
loss_per_bucket = 32
generate_per_bucket = 8
```

With `weighting = "rows"` a dataset is drawn in proportion to `weight x` its rows; with `"share"` the weights
are shares of all draws. A dataset is named after its path (`finetune/cups/g1` is `cups_g1`); `name`, `tasks`
and `categories` tell subsets of one path apart.

Two runs ship. [`base.toml`](../examples/toml/sft_config/psix_image_wm/base.toml) is the base recipe: 80k steps
from Cosmos3-Nano, LR 7e-6, cosine to 10%.
[`robot.toml`](../examples/toml/sft_config/psix_image_wm/robot.toml) adapts the released checkpoint to your
robot.

### Training

```shell
bash examples/launch_sft_psix_image_wm.sh examples/toml/sft_config/psix_image_wm/base.toml [hydra.key=value ...]
```

The Cosmos3-Nano checkpoint and the VAE are read from `examples/checkpoints/` (`BASE_CHECKPOINT_PATH`,
`WAN_VAE_PATH`), and outputs go to `outputs/train` (`OUTPUT_ROOT`). On several nodes run it on each with
`NNODES`, `NODE_RANK` and `MASTER_ADDR` set. Before training, every node builds what is missing under the run's
`cache`: the dataset indexes, the query tables and the validation plan.

Only the generation tower trains. `[custom.train]` sets four speed-ups:

```toml
[custom.train]
detach_frozen_und          = true     # no backward through the frozen text lane (trainable gradients unchanged)
vae_encode_batch           = 32       # samples of one shape VAE-encoded per call
fsdp_replicate_frozen      = true     # frozen weights replicated on every rank instead of sharded (no all-gather)
fsdp_reshard_after_forward = false    # block weights stay gathered from forward to backward
```

### Fine-tuning

The command is in the [README](../README.md#fine-tuning). `BASE_CHECKPOINT_PATH`, like every checkpoint argument
here, takes a run's or the released checkpoint's folder (its latest checkpoint) or one `checkpoints/iter_*`.
[`robot.toml`](../examples/toml/sft_config/psix_image_wm/robot.toml) is a starting point, not a benchmarked
recipe:

- your data make 20% of the draws; the rest replays `g1_sonic`, `h1_sonic` and `g1_sonic_neck`;
- 2k steps at global batch 512, LR 5e-6, a 200-step warmup; aim for one to three passes over your rows, and
  set both `trainer.max_iter` and `scheduler.cycle_lengths` when you change the steps;
- it validates and saves every 500 steps: keep the checkpoint with the best scores on your held-out episodes.

### In-loop validation

From each `val_dataset_paths` dataset and bucket (goal, or future at one horizon), `loss_per_bucket` rows are
drawn once and the first `generate_per_bucket` are also generated. W&B gets `inloop_val/loss` and the last
frame's `inloop_val/gen/mse_last` and `inloop_val/gen/psnr_last`, and the same per set and per bucket
(`inloop_val/gen/my_robot_val/psnr_last`; a set is named after its path, `my_robot/val` is `my_robot_val`).

## Offline evaluation

```shell
python -m cosmos_framework.scripts.eval_image_wm \
    --run-toml examples/toml/sft_config/psix_image_wm/base.toml \
    --checkpoint-path <run> --output-dir <out> [--per-bucket 8] [--datasets path ...]
```

It generates the run's validation samples with the `quality` sampler and scores them against the real frames:
PNGs and a `meta.json` per sample, the means in `summary.json`. A rerun skips finished samples; `--datasets`
evaluates other datasets. The model's configuration is the `config.yaml` of the checkpoint's folder
(`--config-file` names another).

## Serving

```shell
python -m cosmos_framework.scripts.serve_image_wm --checkpoint-path <run> --host 0.0.0.0 --quantize fp8 --compile
```

The defaults are the deployment's: port 8016, the G1's neck camera (`--camera`), the `fast` sampler
(`--preset`) and 1.6 s for a request that names no horizon (`--seconds`). `POST /wm` takes

```json
{"transport": "jpeg", "ego_jpeg": "<base64 JPEG>", "task": "...", "subtask": "...",
 "mode": "future", "seconds": 1.6, "seed": 0,
 "robot_episode_session_id": "...", "req_id": 1, "prompt_gen": 0}
```

and returns the predicted frame as `subgoal_jpeg`. `mode` is `future` (`seconds` ahead, 0.8–4.0) or `subgoal`
(`subtask` complete); a future uses only `task`, and `subtask` is read only for a subgoal. A repeated `req_id`
within a session replays its answer (the last four are kept); the same `req_id` with other inputs is refused
(409 `conflict`), and one request runs at a time (429 otherwise).
`GET /health`, `/ready` and `/state`; `POST /reset`.

## Goal tracks

The Ψ₀.₁ VLA takes part of its training goals from this model. A goal track is one video per episode, frame for
frame beside the camera video:

    wm[t] = source[t]                          for t <  seconds x fps
    wm[t] = WM(source[t - seconds x fps])      for t >= seconds x fps

```shell
bash examples/generate_wm_goal_tracks.sh <run> <output dir> <dataset> [<dataset> ...]
```

Each dataset needs `meta/psix.json` and `meta/modality.json`. It generates the tracks at 1.6 s with the `fast`
sampler, one process per GPU, and a rerun continues. It then
checks each track against its camera video, moves it into the dataset under
`videos/chunk-*/observation.images.egocentric_imgwm/` and registers it as `video.egocentric_wm`, the name the
Ψ₀.₁ training scripts read; `meta/info.json` and `meta/modality.json` are first copied to
`<output dir>/<dataset>/meta_backup/`. With `NNODES` above 1, run it on every node and then
`python -m cosmos_framework.scripts.install_wm_goal_tracks --tracks <output dir>` once.
