#!/usr/bin/env bash
# Goal tracks for LeRobot datasets (cosmos_framework/scripts/generate_wm_goal_video.py): one generate_wm_goal_video
# shard per GPU of this node, shards NODE_RANK x GPUS .. of NNODES x GPUS (every node the same GPU count).
# <checkpoint> is a run's or a released checkpoint's folder (its latest checkpoint) or one checkpoints/iter_N; its
# config.yaml is the folder's. On one node the tracks are then added to their datasets
# (cosmos_framework/scripts/install_wm_goal_tracks.py). On several, run it on every node; when all have exited 0
# (a failed episode is listed in <output dir>/failed_shard*.txt), run once:
#   python -m cosmos_framework.scripts.install_wm_goal_tracks --tracks <output dir>
#
# Optional env vars:
#   NNODES, NODE_RANK  default 1, 0
#   GPUS               default: the GPUs in CUDA_VISIBLE_DEVICES, else all
#   WAN_VAE_PATH       default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#
# Usage (inside the training container, from the repo root; a rerun resumes):
#   bash examples/generate_wm_goal_tracks.sh <checkpoint> <output dir> <dataset> [<dataset> ...]
set -euo pipefail
ckpt=${1:?usage: generate_wm_goal_tracks.sh <checkpoint> <output dir> <dataset> ...} out=${2:?output dir}
shift 2
(($#)) || { echo "ERROR: no datasets" >&2; exit 2; }
for dataset in "$@"; do  # the tracks are registered in it once they are generated
    [[ -f $dataset/meta/modality.json ]] || { echo "ERROR: $dataset/meta/modality.json is missing" >&2; exit 2; }
done
nnodes=${NNODES:-1} node=${NODE_RANK:-0}
IFS=, read -ra devices <<< "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
gpus=${GPUS:-${#devices[@]}}
: "${WAN_VAE_PATH:=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
export WAN_VAE_PATH TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=8
unset TRITON_CACHE_DIR
index=$out/_cache/psix_wm_index
mkdir -p "$out/_logs"

# Index the datasets once here, not in every shard (a no-op once built).
PYTHONPATH=. python -c "
import sys
from cosmos_framework.psix_wm import lerobot
for root in sys.argv[2:]:
    lerobot.build_index(root, sys.argv[1])
" "$index" "$@"

run_shard() {  # one GPU with its own compile caches: a compile cache must have one writer
    local slot=$1 shard=$((node * gpus + $1)) cache=$out/_work/cache/node${node}_gpu$1
    shift
    mkdir -p "$cache"/{tmp,triton,inductor,nv,xdg}
    # exec: the shard's pid is python's, so the trap below stops the python itself.
    CUDA_VISIBLE_DEVICES=${devices[$slot]} TMPDIR=$cache/tmp TRITON_HOME=$cache/triton \
        TORCHINDUCTOR_CACHE_DIR=$cache/inductor CUDA_CACHE_PATH=$cache/nv XDG_CACHE_HOME=$cache/xdg PYTHONPATH=. \
        exec python -m cosmos_framework.scripts.generate_wm_goal_video --checkpoint-path "$ckpt" \
        --output-dir "$out" --index-cache "$index" \
        --compile --shard "$shard" --num-shards $((nnodes * gpus)) --datasets "$@" \
        < /dev/null > "$out/_logs/shard$(printf %03d "$shard").log" 2>&1
}

pids=()
trap 'kill "${pids[@]}" 2> /dev/null' INT TERM
for ((slot = 0; slot < gpus; slot++)); do
    run_shard "$slot" "$@" &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
for list in "$out"/failed_shard*.txt; do
    [[ -s $list ]] && { echo "episodes failed, see $list" >&2; failed=1; }
done
((failed)) && exit 1
((nnodes > 1)) && { echo "tracks written; when every node is done, run install_wm_goal_tracks once"; exit 0; }
PYTHONPATH=. python -m cosmos_framework.scripts.install_wm_goal_tracks --tracks "$out"
