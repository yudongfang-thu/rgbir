#!/usr/bin/env bash
set -u
S=/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction
R=/mnt/dataset/yudongfang/projects/RGBT_campaign
P=$S/environments/sn6-int8-kd/bin/python
O=$R/artifacts/osssl_ir_20260906
W=$S/artifacts/int8_cross_modal_stage2_v1/weights/yolo11n.pt
export PYTHONPATH=$S
# 1. patch final.pt markers
$P - << "PY"
import torch
for arm in ("paired", "sar_only", "shuffled"):
    p = f"/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/osssl_ir_20260906/ssl/{arm}/final.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    if d.get("checkpoint_kind") != "final":
        d["checkpoint_kind"] = "final"
        d["checkpoint_format"] = "yolo-osssl-ssl-v1-legacy"
        torch.save(d, p)
        print("patched", arm)
PY
# 2. inject x3
for arm in paired sar_only shuffled; do
  CUDA_VISIBLE_DEVICES= $P $O/inject_patched.py --weights $W --nc 5 \
    --template-state $O/template_nc5.pt --ssl-final-checkpoint $O/ssl/$arm/final.pt \
    --output $O/injected_$arm.pt 2>&1 | tail -1
done
ls -la $O/injected_*.pt
# 3. per-arm yamls (model field = injected template)
for arm in paired sar_only shuffled; do
  sed "s|^model: .*|model: $O/injected_$arm.pt|" $S/configs/research/rgbt_cmdistill_protocol_drone.yaml > $O/protocol_$arm.yaml
done
grep -H "^model:" $O/protocol_*.yaml
echo "PIPELINE_W2_PREP_DONE"
