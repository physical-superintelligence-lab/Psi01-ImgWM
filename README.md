# Ψ₀.₁ Image World Model

A Cosmos3-Nano fine-tune. From one egocentric frame and the task, it generates the frame a few seconds ahead
(**future**, 0.8 to 4.0 s) or, given the current step, the frame when that step is complete (**goal**). In
[Ψ₀.₁](https://github.com/physical-superintelligence-lab/Psi0/blob/main/README_PSI01.md) this frame is the
VLA's goal image; the robot asks for the future at 1.6 s.

This repository is a fork of NVIDIA's [Cosmos-Framework](https://github.com/NVIDIA/cosmos-framework); the
model's code is in [`cosmos_framework/psix_wm/`](./cosmos_framework/psix_wm) and the details are in the
[reference](./docs/psix_image_world_model.md).

## Setup

```shell
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev tree wget
uv sync --frozen --all-extras --group=cu130-train        # or --group=cu128-train
source .venv/bin/activate && export LD_LIBRARY_PATH=
bash examples/download_psix_image_wm.sh
```

The last line downloads the released checkpoint and the Wan2.2 VAE into `examples/checkpoints/`.

## Serving

```shell
python -m cosmos_framework.scripts.serve_image_wm --host 0.0.0.0 --quantize fp8 --compile \
    --checkpoint-path examples/checkpoints/imgwm/psix_image_wm_nano
```

It listens on port 8016 and expects the G1's neck camera; for a G1 without the neck add `--camera realsense`.
`POST /wm` takes a JPEG frame and the task (a future uses only the task; the step is read for a goal) and
returns the predicted frame ([request and reply](./docs/psix_image_world_model.md#serving)). On an RTX 5090 a
reply takes about 0.3 s; the server needs about 31 GB of GPU memory while loading and 19 GB after.

## Goal tracks

The Ψ₀.₁ VLA trains on goal images from this model. Generate them once for whole datasets:

```shell
bash examples/generate_wm_goal_tracks.sh examples/checkpoints/imgwm/psix_image_wm_nano \
    <output dir> <dataset> [<dataset> ...]
```

Each dataset needs its `meta/psix.json` (for the released humanoid packs, `prepare_psix_data.py --only robot`
in Ψ₀ writes it) and its `meta/modality.json`. This adds `video.egocentric_wm` to every dataset, one process per
GPU; a rerun continues
([details](./docs/psix_image_world_model.md#goal-tracks)).

## Fine-tuning

The released checkpoint is a general base. For a new robot or scene, fine-tune it on your data. In
[Ψ₀](https://github.com/physical-superintelligence-lab/Psi0), prepare the humanoid packs it replays into this
repository's data root:

```shell
python scripts/data/prepare_psix_data.py --only robot --root <this repository>/.data/psix_lerobot
```

Put your own data at `.data/my_robot/train` and `.data/my_robot/val`, in the format of
[Your own data](https://github.com/physical-superintelligence-lab/Psi0/blob/main/scripts/data/DATA_PREPARATION.md#your-own-data).
Then:

```shell
BASE_CHECKPOINT_PATH=examples/checkpoints/imgwm/psix_image_wm_nano \
    bash examples/launch_sft_psix_image_wm.sh examples/toml/sft_config/psix_image_wm/robot.toml
```

[`robot.toml`](./examples/toml/sft_config/psix_image_wm/robot.toml) runs 2k steps on 8 GPUs at global batch
512, with your data at 20% of the draws, and validates and saves every 500 steps. Keep the checkpoint that scores
best on `my_robot/val` (`my_robot_val` in W&B). This is a starting point, not a benchmarked recipe. Runs are
written to `outputs/train` and logged to W&B: set `WANDB_API_KEY`, or add `job.wandb_mode=disabled` to the command.

Your data's share is the first entry of `dataset_weights` in `robot.toml`: percent of the draws, in the order of
`dataset_paths` (the other three replay `g1_sonic`, `h1_sonic` and `g1_sonic_neck`). With several datasets of
your own, list them all and split that share among them.

> 🖥️ Change the GPUs with `NPROC_PER_NODE=4 bash examples/launch_sft_psix_image_wm.sh ...`; the global batch
> is 64 per GPU.

## Evaluation

```shell
python -m cosmos_framework.scripts.eval_image_wm --run-toml examples/toml/sft_config/psix_image_wm/robot.toml \
    --checkpoint-path <run> --output-dir <out>
```

`<run>` is a run's folder (its latest checkpoint) or one of its `checkpoints/iter_*`. It generates the run's
held-out samples and scores them against the real frames; `summary.json` holds the means.

## Training the base

The base trains from Cosmos3-Nano on all the data: run `prepare_psix_data.py` in Ψ₀ with the same `--root` and
without `--only` (the human video needs the setup in its
[guide](https://github.com/physical-superintelligence-lab/Psi0/blob/main/scripts/data/DATA_PREPARATION.md)).
Then convert Cosmos3-Nano ([details](./docs/training.md#step-2--prepare-checkpoint)) and train:

```shell
python -m cosmos_framework.scripts.convert_model_to_dcp -o examples/checkpoints/Cosmos3-Nano \
    --checkpoint-path Cosmos3-Nano
bash examples/launch_sft_psix_image_wm.sh examples/toml/sft_config/psix_image_wm/base.toml
```

[`base.toml`](./examples/toml/sft_config/psix_image_wm/base.toml) runs 80k steps on 8 GPUs at global batch 512.
For several nodes, see the [reference](./docs/psix_image_world_model.md#training).

## License

Cosmos-Framework is released by NVIDIA under the [OpenMDW License](./LICENSE); see also
[ATTRIBUTIONS.md](./ATTRIBUTIONS.md) and [NOTICE](./NOTICE).
