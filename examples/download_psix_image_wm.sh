#!/usr/bin/env bash
# The released Psi0.1 image world model and the Wan2.2 VAE, into examples/checkpoints, with the checkpoint's
# config.yaml pointed at that VAE. A rerun downloads only what is missing.
#
# Usage (from anywhere, in the environment of the repository):
#   bash examples/download_psix_image_wm.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
hf download USC-PSI-Lab/psix-model --include 'imgwm/psix_image_wm_nano/*' --local-dir examples/checkpoints
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir examples/checkpoints/wan22_vae
sed -i "s#^\( *vae_path:\).*#\1 $PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth#" \
    examples/checkpoints/imgwm/psix_image_wm_nano/config.yaml
