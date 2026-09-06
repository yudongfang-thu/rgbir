#!/usr/bin/env bash
set -eu
STAGE=$1
SEED=$2
GPU=$3
ROOT=/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_expand_20260906
RUNS=/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906
SOURCE=/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906/release_v2
PY=/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python
exec "$PY" -u "$ROOT/release_v1/expansion_worker.py" --stage "$STAGE" --seed "$SEED" --gpu "$GPU" --root "$ROOT" --run-root "$RUNS" --source "$SOURCE" --attempt attempt1 > "$ROOT/${STAGE}_s${SEED}_attempt1.log" 2>&1
