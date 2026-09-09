#!/bin/bash
ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
OUT=$ROOT/artifacts/original_repro_continue_20260909_attempt1
git clone --depth 1 https://github.com/bupt-ai-cz/LLVIP.git "$ROOT/external_reproductions/llvip_author_baseline/author_source" > "$OUT/llvip_source_clone.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/llvip_source_clone.exit"
if [ "$rc" -eq 0 ]; then
    git -C "$ROOT/external_reproductions/llvip_author_baseline/author_source" rev-parse HEAD > "$OUT/llvip_source_revision.txt"
fi
exit "$rc"
