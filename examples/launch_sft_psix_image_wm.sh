#!/usr/bin/env bash
# Image-WM SFT from a run TOML in $PSIX_WM_RUN_TOMLS: its schedule, and its data under [custom.data]
# (docs/psix_image_world_model.md). Drives cosmos_framework.scripts.train.
#
# Optional env vars:
#   PSIX_WM_RUN_TOMLS     the run-TOML directory; default: examples/toml/sft_config/psix_image_wm
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano; a run's or a released checkpoint's folder
#                         stands for its latest checkpoint
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   OUTPUT_ROOT           default: outputs/train
#   NNODES, NODE_RANK, MASTER_ADDR  to train on several nodes; run this on each of them
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_psix_image_wm.sh examples/toml/sft_config/psix_image_wm/base.toml [hydra.key=value ...]

TOML_FILE="$(realpath "${1:?usage: launch_sft_psix_image_wm.sh <run toml> [hydra overrides]}")"
TAIL_OVERRIDES=("${@:2}")
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# Every TOML in this directory is registered as an experiment, so a run must live there.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PSIX_WM_RUN_TOMLS="$(cd "${PSIX_WM_RUN_TOMLS:-$REPO/examples/toml/sft_config/psix_image_wm}" && pwd -P)" || exit 1
export PSIX_WM_RUN_TOMLS
[[ "$(cd "$(dirname "$TOML_FILE")" && pwd -P)" == "$PSIX_WM_RUN_TOMLS" ]] ||
    { echo "ERROR: run TOMLs live in $PSIX_WM_RUN_TOMLS/" >&2; exit 1; }

# Build the run's missing dataset indexes and validation plan; a no-op once they exist.
(cd "$REPO" && PYTHONPATH=. python -m cosmos_framework.psix_wm.prepare "$TOML_FILE") || exit 1
BASE_CHECKPOINT_PATH="$(cd "$REPO" && PYTHONPATH=. python -m cosmos_framework.psix_wm.checkpoint "$BASE_CHECKPOINT_PATH")" || exit 1

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
