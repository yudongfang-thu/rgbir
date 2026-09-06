# -*- coding: utf-8 -*-
"""P3 probe: day/night bucketed evaluation on DroneVehicle val.

Splits the 1,469 val pairs by RGB-image mean luminance (median threshold),
builds hardlinked bucket subsets (rgb side + infrared side), then evaluates
three YOLO11 models with ultralytics val:
  T        = infrared teacher  (evaluated on IR bucket images)
  S_native = native RGB y11    (evaluated on RGB bucket images)
  S_dist   = adapted_v2 student(RGB, distilled; evaluated on RGB bucket images)
Buckets: day / night / full. Metrics: mAP50, mAP75, mAP50-95 per bucket.
Read-only wrt original data (hardlinks only). Outputs summary.json + luminance csv.
"""
import csv
import json
import os
import shutil

import numpy as np
from PIL import Image

R = "/mnt/dataset/yudongfang/projects/RGBT_campaign"
BASE = f"{R}/data/processed/dronevehicle/yolo/hbb_v1"
OUT_DIR = f"{R}/artifacts/p3_daynight_20260905"
IMG_DIR_NAMES = ("images", "labels")
NAMES = {0: "car", 1: "freight car", 2: "truck", 3: "bus", 4: "van"}


def find_split_dir(root, kind, split):
    """kind in {images, labels}; supports images/val or val/images layouts."""
    for a, b in ((kind, split), (split, kind)):
        p = os.path.join(root, a, b)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"{kind}/{split} under {root}")


def hardlink_tree(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        if not os.path.exists(dst):
            os.link(src, dst)
    return dst_dir


def make_yaml(path, root, names):
    with open(path, "w") as f:
        f.write(f"path: {root}\ntrain: images/val\nval: images/val\nnames: {names}\n")


def eval_model(ckpt, yaml_path, device=0):
    from ultralytics import YOLO
    m = YOLO(ckpt)
    res = m.val(data=yaml_path, imgsz=640, device=device, plots=False,
                verbose=False, workers=4)
    return {"mAP50": round(res.box.map50 * 100, 3),
            "mAP75": round(res.box.map75 * 100, 3),
            "mAP50_95": round(res.box.map * 100, 3)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rgb_img = find_split_dir(f"{BASE}/rgb", "images", "val")
    ir_img = find_split_dir(f"{BASE}/infrared", "images", "val")
    rgb_lbl = find_split_dir(f"{BASE}/rgb", "labels", "val")
    ir_lbl = find_split_dir(f"{BASE}/infrared", "labels", "val")

    files = sorted(f for f in os.listdir(rgb_img) if f.lower().endswith((".jpg", ".png")))
    print(f"val pairs: {len(files)}", flush=True)

    # luminance and median split
    lums = {}
    for f in files:
        lums[f] = float(np.asarray(Image.open(os.path.join(rgb_img, f)).convert("L")).mean())
    vals = sorted(lums.values())
    thr = vals[len(vals) // 2]
    day = [f for f in files if lums[f] >= thr]
    night = [f for f in files if lums[f] < thr]
    print(f"threshold(median luminance)={thr:.2f}  day={len(day)} night={len(night)}", flush=True)

    with open(os.path.join(OUT_DIR, "per_image_luminance.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "luminance", "bucket"])
        for f in files:
            w.writerow([f, round(lums[f], 3), "day" if lums[f] >= thr else "night"])

    # build bucket trees (hardlinks)
    buckets = {}
    for bname, fl in (("day", day), ("night", night), ("full", files)):
        root = os.path.join(OUT_DIR, bname)
        rgb_root = os.path.join(root, "rgb")
        ir_root = os.path.join(root, "infrared")
        for f in fl:
            src_i = os.path.join(rgb_img, f)
            dst_i = os.path.join(rgb_root, "images", "val", f)
            os.makedirs(os.path.dirname(dst_i), exist_ok=True)
            if not os.path.exists(dst_i):
                os.link(src_i, dst_i)
            src_l = os.path.join(rgb_lbl, os.path.splitext(f)[0] + ".txt")
            dst_l = os.path.join(rgb_root, "labels", "val", os.path.splitext(f)[0] + ".txt")
            os.makedirs(os.path.dirname(dst_l), exist_ok=True)
            if os.path.exists(src_l) and not os.path.exists(dst_l):
                os.link(src_l, dst_l)
            src_i2 = os.path.join(ir_img, f)
            dst_i2 = os.path.join(ir_root, "images", "val", f)
            os.makedirs(os.path.dirname(dst_i2), exist_ok=True)
            if not os.path.exists(dst_i2):
                os.link(src_i2, dst_i2)
            src_l2 = os.path.join(ir_lbl, os.path.splitext(f)[0] + ".txt")
            dst_l2 = os.path.join(ir_root, "labels", "val", os.path.splitext(f)[0] + ".txt")
            os.makedirs(os.path.dirname(dst_l2), exist_ok=True)
            if os.path.exists(src_l2) and not os.path.exists(dst_l2):
                os.link(src_l2, dst_l2)
        make_yaml(os.path.join(rgb_root, "bucket.yaml"), rgb_root, NAMES)
        make_yaml(os.path.join(ir_root, "bucket.yaml"), ir_root, NAMES)
        buckets[bname] = {"rgb_yaml": os.path.join(rgb_root, "bucket.yaml"),
                          "ir_yaml": os.path.join(ir_root, "bucket.yaml")}

    summary = {"threshold_median_luminance": round(thr, 3),
               "n_day": len(day), "n_night": len(night), "n_full": len(files),
               "models": {}}
    MODELS = {
        "T_ir": (f"{R}/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/best.pt", "ir"),
        "S_native_rgb": (f"{R}/runs/rgbt_cmdistill_paper_reconstructed_v2/native_rgb_s42_b64_e200/weights/best.pt", "rgb"),
        "S_dist_rgb": (f"{R}/runs/rgbt_cmdistill_adapted_v2/dronevehicle_seed42_b32_e200/weights/best.pt", "rgb"),
    }
    for mname, (ck, side) in MODELS.items():
        summary["models"][mname] = {}
        for bname in ("day", "night", "full"):
            key = f"{bname}_{side}"
            print(f"[val] {mname} on {key} ...", flush=True)
            summary["models"][mname][bname] = eval_model(ck, buckets[bname][f"{side}_yaml"])
            print(json.dumps(summary["models"][mname][bname]), flush=True)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
