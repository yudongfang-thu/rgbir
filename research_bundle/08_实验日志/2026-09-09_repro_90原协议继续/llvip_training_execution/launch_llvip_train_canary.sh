#!/bin/bash
ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
OUT=$ROOT/artifacts/original_repro_continue_20260909_attempt1
export RGBIR90_PROJECT_ROOT=$ROOT
export YOLOV5_CONFIG_DIR=$ROOT/cache/llvip_yolov5
export TMPDIR=$ROOT/cache/tmp
"$ROOT/environments/cft90/bin/python" -B -u "$ROOT/tools/project_resource_guard.py" --lease-file "$ROOT/runs/.project_resource_leases.json" run --job-id llvip_author_train_canary_20260909 --kind train --expected-vram-mib 17152 --expected-rss-mib 65536 --free-safety-mib 2048 -- "$ROOT/environments/cft90/bin/python" -B -u "$OUT/train_author_canary.py" canary --weights "$ROOT/external_reproductions/cft/author_initialization/yolov5l.pt" --reference-weights "$ROOT/external_reproductions/llvip_author_baseline/extracted_attempt1/yolov5_trained_model/yolov5_visible.pt" --modality visible --successful-updates 24 --max-batches 256 --memory-fraction .68 --output "$OUT/llvip_train_canary_attempt1" > "$OUT/llvip_train_canary.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/llvip_train_canary.exit"
exit "$rc"
