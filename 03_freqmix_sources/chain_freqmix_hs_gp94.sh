#!/usr/bin/env bash
# Run a fresh E1 preflight, then start the frozen E400 cell only if it passes.
# Usage: chain_freqmix_hs_gp94.sh GPU_ID SEED ARM CAMPAIGN_ROOT
set -euo pipefail

GPU_ID=${1:?GPU_ID}
SEED=${2:?SEED}
ARM=${3:?ARM}
ROOT=${4:?CAMPAIGN_ROOT}
LAUNCHER=/home/zyk/projects/ydf/freqmix_hs_gp94_20260819/run_freqmix_hs_gp94.sh

"$LAUNCHER" "$GPU_ID" "$SEED" 1 "$ARM" "$ROOT/e1_${ARM}_s${SEED}"
"$LAUNCHER" "$GPU_ID" "$SEED" 400 "$ARM" "$ROOT/e400_${ARM}_s${SEED}"

