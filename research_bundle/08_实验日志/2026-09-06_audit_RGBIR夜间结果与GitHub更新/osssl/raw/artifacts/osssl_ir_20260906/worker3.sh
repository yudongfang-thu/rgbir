#!/usr/bin/env bash
set -u
S=/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction
R=/mnt/dataset/yudongfang/projects/RGBT_campaign
P=$S/environments/sn6-int8-kd/bin/python
O=$R/artifacts/osssl_ir_20260906
GPU=$1; JOBS=$2
while IFS= read -r line; do
  [ -z "$line" ] && continue
  arm=${line%,*}; seed=${line#*,}
  out=$R/runs/osssl_ir_20260906/${arm}_rgb_s${seed}_e200
  [ -f "$out/completion_receipt.json" ] && continue
  echo "[w$GPU] start $arm s$seed $(date -Is)"
  rc=2
  while [ "$rc" -eq 2 ]; do
    CUDA_VISIBLE_DEVICES=$GPU $P $S/tools/project_resource_guard.py run \
      --job-id osssl-ir-$arm-s$seed --kind train --candidate-gpu $GPU --gpu-count 1 \
      --expected-vram-mib 10000 --expected-rss-mib 32768 --cuda-processes-per-gpu 1 \
      --formal-train --free-safety-mib 2048 -- \
      $P $R/artifacts/cgkd_20260905/train_native_rgbt.py --config $O/protocol_clean_$arm.yaml \
      --output $out --device 0 --seed $seed >> $O/ft_${arm}_s${seed}.log 2>&1
    rc=$?
    if [ "$rc" -eq 2 ]; then echo "[w$GPU] busy $arm s$seed, retry in 60s $(date -Is)"; sleep 60; fi
  done
  echo "[w$GPU] done $arm s$seed exit=$rc $(date -Is)"
done < "$JOBS"
echo "WORKER3_GPU${GPU}_DONE"
