#!/usr/bin/env bash
set -u
S=/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction
R=/mnt/dataset/yudongfang/projects/RGBT_campaign
P=$S/environments/sn6-int8-kd/bin/python
CFG=$S/configs/research/rgbt_cmdistill_protocol_drone.yaml
for s in 42 0 123; do
  D=$R/runs/cgkd_w1/native_rgb_s${s}_e200
  if [ ! -f "$D/metrics_record.json" ]; then
    CUDA_VISIBLE_DEVICES=4 $P $S/tools/project_resource_guard.py run \
      --job-id cgkd-w1-native-s${s}-eval --kind eval --candidate-gpu 4 --gpu-count 1 \
      --expected-vram-mib 6000 --expected-rss-mib 16384 --cuda-processes-per-gpu 1 \
      --non-formal-train --free-safety-mib 2048 -- \
      $P $S/tools/eval_rgbt_detector.py --config $CFG --checkpoint $D/weights/last.pt \
        --arm native_weight0 --seed $s --output $D/metrics_record.json \
        --imgsz 640 --batch 32 --workers 8 --device 0
    echo "[eval] s$s exit=$?"
  fi
done
echo "NATIVE_EVALS_DONE"
