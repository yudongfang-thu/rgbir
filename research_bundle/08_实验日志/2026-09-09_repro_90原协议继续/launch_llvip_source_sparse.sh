#!/bin/bash
ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90
OUT=$ROOT/artifacts/original_repro_continue_20260909_attempt1
DST=$ROOT/external_reproductions/llvip_author_baseline/author_code_attempt2
{
    git clone --depth 1 --filter=blob:none --no-checkout https://github.com/bupt-ai-cz/LLVIP.git "$DST" &&
    git -C "$DST" sparse-checkout set --no-cone '/yolov5/*.py' '/yolov5/models/' '/yolov5/utils/' '/yolov5/data/*.yaml' '/yolov5/requirements.txt' '/yolov5/LICENSE' '/yolov5/README*' '/README*' '/LICENSE*' &&
    git -C "$DST" checkout &&
    git -C "$DST" rev-parse HEAD > "$OUT/llvip_source_sparse_revision.txt"
} > "$OUT/llvip_source_sparse.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/llvip_source_sparse.exit"
exit "$rc"
