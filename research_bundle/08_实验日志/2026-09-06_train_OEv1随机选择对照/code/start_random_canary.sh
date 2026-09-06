#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
GPU=${1:?physical GPU required}
SEED=${2:?student seed required}
ATTEMPT=${3:-attempt1}
[[ "$GPU" =~ ^[0-7]$ ]] || exit 2
[[ "$SEED" == 0 || "$SEED" == 42 || "$SEED" == 123 ]] || exit 2
[[ "$ATTEMPT" =~ ^attempt[0-9]+$ ]] || exit 2
REPO=/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction
ROOT=/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_random_20260906
RUNS=/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_random_20260906
PY=$REPO/environments/sn6-int8-kd/bin/python
SOURCE=$ROOT/release_v1
OUT=$RUNS/canary_paired_random_s${SEED}_${ATTEMPT}
LOG=$ROOT/canary_paired_random_s${SEED}_${ATTEMPT}.log
[[ ! -e "$OUT" && ! -e "$LOG" ]] || exit 2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$PY" "$REPO/tools/project_resource_guard.py" run \
  --job-id "rgbir_oev1_random_canary_s${SEED}_${ATTEMPT}" --kind train \
  --candidate-gpu "$GPU" --gpu-count 1 --cuda-processes-per-gpu 1 \
  --expected-vram-mib 10000 --expected-rss-mib 49152 --free-safety-mib 2048 \
  --non-formal-train -- "$PY" "$SOURCE/train_object_evidence.py" \
  --config "$SOURCE/config_drone.yaml" --output "$OUT" \
  --arm paired_random --seed "$SEED" --max-steps 24 > "$LOG" 2>&1
