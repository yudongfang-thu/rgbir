#!/bin/bash
ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
OUT=$ROOT/artifacts/original_repro_continue_20260909_attempt1
export RGBIR90_PROJECT_ROOT=$ROOT
"$ROOT/environments/cft90/bin/python" -B -u "$ROOT/tools/project_resource_guard.py" --lease-file "$ROOT/runs/.project_resource_leases.json" run --job-id cft_author_dp3_canary_20260909 --kind train --gpu-count 3 --expected-vram-mib 17152 --expected-rss-mib 65536 --free-safety-mib 2048 -- "$ROOT/environments/cft90/bin/python" -B -u "$OUT/train_canary_dp3_diagnostic.py" --gpu-count 3 --memory-fraction .68 --successful-steps 2 --max-batches 16 --weights "$ROOT/external_reproductions/cft/author_initialization/yolov5l.pt" --output "$OUT/cft_dp3_canary_attempt1" > "$OUT/cft_dp3_canary.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/cft_dp3_canary.exit"
exit "$rc"
