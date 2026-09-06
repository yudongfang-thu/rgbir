#!/usr/bin/env bash
set -euo pipefail
dataset="$1"
stage="$2"
gpu="$3"
root=/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_transfer_diagnosis_20260906
repo=/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction
py="$repo/environments/sn6-int8-kd/bin/python"
export MPLCONFIGDIR="$root/matplotlib_cache"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
extra=()
if [[ "$stage" == canary ]]; then extra=(--n 2 --canary); else extra=(--n 200); fi
cd "$root"
"$py" "$repo/tools/project_resource_guard.py" run \
  --job-id "rgbir_probe_${dataset}_${stage}_s42" --kind feature --candidate-gpu "$gpu" \
  --expected-vram-mib 6000 --expected-rss-mib 8192 --free-safety-mib 2048 \
  -- "$py" -u "$root/probe_rgbir.py" --config "$root/probe_config.json" \
  --dataset "$dataset" --out "$root/${dataset}_${stage}" "${extra[@]}" \
  > "$root/${dataset}_${stage}.log" 2>&1
