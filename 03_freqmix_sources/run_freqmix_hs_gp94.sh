#!/usr/bin/env bash
# M-FM-HS-v1 single-cell launcher on gp94.
# Usage: run_freqmix_hs_gp94.sh GPU_ID SEED EPOCHS ARM OUTPUT_ROOT
set -euo pipefail

GPU_ID=${1:?GPU_ID}
SEED=${2:?SEED}
EPOCHS=${3:?EPOCHS}
ARM=${4:?ARM}
OUTPUT=${5:?OUTPUT_ROOT}

BASE=/home/zyk/projects/ydf
CAMPAIGN=$BASE/freqmix_hs_gp94_20260819
CODE=$CAMPAIGN/code_run
RUNTIME=/home/zyk/data/projects/ydf/projects/LADD_public/comparison/runtime/current_hbb
SHARED=/home/zyk/data/projects/ydf/projects/LADD_public/shared
DATA=$BASE/datasets/OGSOD-1.0
GRAY_DATA=$BASE/datasets/OGSOD-1.0-gray-gp94
GRAY_YAML=$BASE/ogsod_gray_highdim_gp94_20260818/ogsod_gray_gp94.yaml
PY=/home/zyk/data/projects/ydf/conda/envs/ladd/bin/python

case "$SEED" in
  42|123) ;;
  *) echo "seed must be 42 or 123" >&2; exit 2 ;;
esac
case "$ARM" in
  H_S)
    REFERENCE_ROOT=$DATA/rgb
    REFERENCE_YAML=$CODE/configs/datasets/ogsod_94_rgb.yaml
    ;;
  h_s_freqmix_gray_shuffled|h_s_freqmix_sar_shuffled|h_s_freqmix_gray_sign_randomized)
    REFERENCE_ROOT=$GRAY_DATA/rgb
    REFERENCE_YAML=$GRAY_YAML
    ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

RGB_TEACHER=$BASE/tri_rank_94/anchors/ogsod_lr_diag_90_rgb_yolo11n_rgb_s${SEED}_e400_lr0.005_lrf0.01_cos0_best.pt
SAR_TEACHER=$BASE/tri_rank_94/anchors/pemt_ogsod_90_native_only_s${SEED}_e400_best.pt

mkdir -p "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"

"$PY" "$CODE/tools/run_pemt_v5_direct.py" \
  --sar-root "$DATA/sar" \
  --rgb-root "$REFERENCE_ROOT" \
  --output-root "$OUTPUT" \
  --runtime-root "$RUNTIME" \
  --canonical-yolo-root "$SHARED/yolo" \
  --canonical-shared-root "$SHARED" \
  --canonical-ladd-shared-root "$SHARED" \
  --sar-data "$CODE/configs/datasets/ogsod_94_sar.yaml" \
  --rgb-data "$REFERENCE_YAML" \
  --model "$BASE/tri_rank_94/anchors/yolo11n.pt" \
  --rgb-teacher "$RGB_TEACHER" \
  --sar-teacher "$SAR_TEACHER" \
  --arm "$ARM" \
  --seed "$SEED" \
  --device "$GPU_ID" \
  --epochs "$EPOCHS" \
  --imgsz 256 \
  --lr0 0.005

PYTHONPATH="$RUNTIME/src:$SHARED/yolo:$SHARED" "$PY" -c '
import csv
import json
import math
import sys
from pathlib import Path

from ultralytics import YOLO

output = Path(sys.argv[1])
sar_data = sys.argv[2]
device = int(sys.argv[3])
status = json.loads((output / "run_status.json").read_text())
if status.get("status") != "COMPLETED":
    raise RuntimeError(f"training did not complete: {status}")
with (output / "results.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
losses = {
    key: float(value)
    for key, value in rows[-1].items()
    if "loss" in key and value not in (None, "")
}
if not losses or any(not math.isfinite(value) for value in losses.values()):
    raise RuntimeError("endpoint losses are missing or non-finite")
model = YOLO(str(output / "best.pt"))
metrics = model.val(
    data=sar_data,
    imgsz=256,
    batch=64,
    workers=8,
    device=device,
    project=str(output),
    name="sar_only_val",
    exist_ok=False,
    plots=False,
    verbose=False,
)
values = {str(key): float(value) for key, value in metrics.results_dict.items()}
if not values or any(not math.isfinite(value) for value in values.values()):
    raise RuntimeError("clean-SAR validation returned non-finite metrics")
evidence = {
    "arm": status["arm"],
    "seed": status["seed"],
    "epochs": len(rows),
    "finite_endpoint_losses": losses,
    "sar_only_metrics": values,
}
(output / "preflight_evidence.json").write_text(
    json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
' "$OUTPUT" "$CODE/configs/datasets/ogsod_94_sar.yaml" "$GPU_ID"
