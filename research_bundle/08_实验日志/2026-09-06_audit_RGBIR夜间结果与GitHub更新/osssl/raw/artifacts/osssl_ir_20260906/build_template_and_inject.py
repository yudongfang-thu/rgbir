import json, sys
from pathlib import Path
import torch
S = Path("/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction")
sys.path.insert(0, str(S))
from yolo_osssl.yolo import build_yolo11n_template, load_template_state
R = Path("/mnt/dataset/yudongfang/projects/RGBT_campaign")
O = R / "artifacts/osssl_ir_20260906"
W = S / "artifacts/int8_cross_modal_stage2_v1/weights/yolo11n.pt"
template, info1, info2 = build_yolo11n_template(str(W), nc=5)
tpath = O / "template_nc5.pt"
torch.save({"template_state": template.state_dict(), "meta": {"info1": str(info1), "info2": str(info2)}}, tpath)
check = template
load_template_state(tpath, check)
print("template saved:", tpath)
for arm in ("paired", "sar_only", "shuffled"):
    out = O / f"injected_{arm}.pt"
    print("[inject]", arm, flush=True)
