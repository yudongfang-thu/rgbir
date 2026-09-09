#!/bin/bash
ROOT=/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction
OUT=/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/availability94_20260909_attempt1
"$ROOT/environments/sn6-int8-kd/bin/python" -B -u "$ROOT/tools/project_resource_guard.py" run --job-id availability94_20260909 --kind other --expected-vram-mib 1024 --expected-rss-mib 2048 --free-safety-mib 2048 -- "$ROOT/environments/sn6-int8-kd/bin/python" -B -u "$OUT/cuda_health94.py" "$OUT/cuda_receipt.json" > "$OUT/cuda_health.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/cuda_health.exit"
exit "$rc"
