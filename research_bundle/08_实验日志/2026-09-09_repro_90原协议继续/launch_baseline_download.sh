#!/bin/bash
ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
OUT=$ROOT/artifacts/original_repro_continue_20260909_attempt1
export TMPDIR=$ROOT/cache/tmp
"$ROOT/environments/cft90/bin/python" -B -u "$OUT/fetch_llvip_author_models.py" > "$OUT/llvip_author_download.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/llvip_author_download.exit"
exit "$rc"
