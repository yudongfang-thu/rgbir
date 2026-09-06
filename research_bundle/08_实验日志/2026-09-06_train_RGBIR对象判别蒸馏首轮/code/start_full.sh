#!/usr/bin/env bash
set -eu
ROOT=/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906
RUNS=/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906
PY=/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python
exec "$PY" -u "$ROOT/release_v2/campaign_queue.py" --stage full --gpu 4 --vram-mib 10000 --rss-mib 49152 --root "$ROOT" --run-root "$RUNS" --attempt attempt1 > "$ROOT/full_attempt1.log" 2>&1
