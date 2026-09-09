#!/bin/bash
ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
OUT=$ROOT/artifacts/original_repro_continue_20260909_attempt1
"$ROOT/environments/cft90/bin/python" -B -u "$OUT/run_llvip_author_evaluation_queue.py" > "$OUT/llvip_eval_queue.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/llvip_eval_queue.exit"
exit "$rc"
