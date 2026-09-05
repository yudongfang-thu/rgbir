#!/usr/bin/env bash

set -uo pipefail

if [[ "$#" -lt 9 || "$#" -gt 10 ]]; then
  echo "usage: $0 REPO PYTHON CONFIG TEACHER OUTPUT DATASET SEED GPU_A GPU_B [profiled-second]" >&2
  exit 64
fi

repo=$1
python_bin=$2
config=$3
teacher=$4
output=$5
dataset=$6
seed=$7
gpu_a=$8
gpu_b=$9
profile_mode=${10:-}
job_base="cmdistill-adapted-v2-${dataset}-seed${seed}"

expected_train_vram_mib=10000
second_train_args=()
if [[ "$profile_mode" == "profiled-second" ]]; then
  expected_train_vram_mib=7000
  second_train_args=(--profiled-second-train)
fi

run_with_retry() {
  local label=$1
  shift
  while true; do
    echo "[$(date -Is)] ${label}: requesting resource lease"
    "$@"
    local rc=$?
    if [[ "$rc" -eq 0 ]]; then
      echo "[$(date -Is)] ${label}: completed"
      return 0
    fi
    if [[ "$rc" -ne 2 ]]; then
      echo "[$(date -Is)] ${label}: failed with exit code ${rc}" >&2
      return "$rc"
    fi
    echo "[$(date -Is)] ${label}: resources busy; retrying in 60 seconds"
    sleep 60
  done
}

mkdir -p "$output"

if [[ ! -f "$output/completion_receipt.json" || ! -f "$output/weights/last.pt" ]]; then
  run_with_retry train \
    "$python_bin" "$repo/tools/project_resource_guard.py" run \
      --job-id "${job_base}-train" \
      --kind train \
      --candidate-gpu "$gpu_a" \
      --candidate-gpu "$gpu_b" \
      --gpu-count 1 \
      --expected-vram-mib "$expected_train_vram_mib" \
      --expected-rss-mib 32768 \
      --cuda-processes-per-gpu 1 \
      --formal-train \
      "${second_train_args[@]}" \
      --free-safety-mib 2048 \
      -- \
      "$python_bin" "$repo/tools/train_rgbt_cmdistill.py" \
        --config "$config" \
        --arm cmdistill_corrected \
        --teacher-weights "$teacher" \
        --output "$output" \
        --device 0 \
        --seed "$seed" \
        --batch 32 \
        --workers 8 || exit $?
fi

metric_record="$output/metrics_record.json"
if [[ ! -f "$metric_record" ]]; then
  run_with_retry eval \
    "$python_bin" "$repo/tools/project_resource_guard.py" run \
      --job-id "${job_base}-eval" \
      --kind eval \
      --candidate-gpu "$gpu_a" \
      --candidate-gpu "$gpu_b" \
      --gpu-count 1 \
      --expected-vram-mib 6000 \
      --expected-rss-mib 16384 \
      --cuda-processes-per-gpu 1 \
      --non-formal-train \
      --free-safety-mib 2048 \
      -- \
      "$python_bin" "$repo/tools/eval_rgbt_detector.py" \
        --config "$config" \
        --checkpoint "$output/weights/last.pt" \
        --arm cmdistill_corrected \
        --seed "$seed" \
        --output "$metric_record" \
        --imgsz 640 \
        --batch 32 \
        --workers 8 \
        --device 0 || exit $?
fi

echo "[$(date -Is)] ${job_base}: train and dev evaluation complete"
