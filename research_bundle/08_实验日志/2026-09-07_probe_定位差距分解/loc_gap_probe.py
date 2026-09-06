# -*- coding: utf-8 -*-
"""Localization gap decomposition probe (read-only, no training).

For every val image: run IR teacher + RGB student, match predictions to GT
(IoU>=0.5 greedy, same logic family as oev1), then bucket each GT object:

  A  student-detected, student IoU in [0.5, 0.75), teacher IoU >= 0.75
     -> recoverable localization quality (oev2-loc target market)
  B  student-detected, both IoU >= 0.75                    -> already good
  C  student-detected, teacher IoU < 0.75                   -> teacher no better
  D  student missed (IoU < 0.5), teacher IoU >= 0.5         -> no-observation
  E  both missed                                            -> hard for both

Also record: teacher-vs-student box IoU for A-class (direct distillation
feasibility), per-class counts, and day/night bucket via RGB luminance median
(frozen threshold 78.283 from p3 probe).

Outputs: summary.json + per_image.csv under OUT_DIR.
"""
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

R = "/mnt/dataset/yudongfang/projects/RGBT_campaign"
IR_VAL = R + "/data/processed/dronevehicle/yolo/hbb_v1/infrared/images/val"
RGB_VAL = R + "/data/processed/dronevehicle/yolo/hbb_v1/rgb/images/val"
RGB_LBL = R + "/data/processed/dronevehicle/yolo/hbb_v1/rgb/labels/val"
TEACHER = R + "/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt"
STUDENT = R + "/runs/cgkd_w1/native_rgb_s42_e200/weights/last.pt"
OUT_DIR = R + "/artifacts/loc_gap_probe_20260907"
IMGSZ = 640
CONF = 0.10          # prediction threshold (low enough to see weak candidates)
IOU_DET = 0.50       # "detected" threshold
IOU_GOOD = 0.75      # "good localization" threshold
LUM_THRESHOLD = 78.283  # frozen from p3_daynight probe
NAMES = {0: "car", 1: "freight car", 2: "truck", 3: "bus", 4: "van"}


def load_gt(txt_path, img_w, img_h):
    """YOLO txt -> array of (cls, x1, y1, x2, y2) pixel xyxy."""
    out = []
    if not os.path.isfile(txt_path):
        return np.zeros((0, 5))
    for line in open(txt_path):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        x1, y1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
        x2, y2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
        out.append([c, x1, y1, x2, y2])
    return np.array(out).reshape(-1, 5)


def iou_matrix(a, b):
    """(N,4) x (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pred_boxes, gt_boxes):
    """Greedy IoU>=0.5 matching, class-aware. Returns per-GT (matched_pred_idx, iou)."""
    n = len(gt_boxes)
    res = [(-1, 0.0)] * n
    if len(pred_boxes) == 0 or n == 0:
        return res
    ious = iou_matrix(gt_boxes[:, 1:5], pred_boxes[:, :4])
    cls_ok = gt_boxes[:, 0][:, None] == pred_boxes[:, 4][None, :]
    ious = ious * cls_ok
    order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    used_p, used_g = set(), set()
    for gi, pi in order:
        if gi in used_g or pi in used_p or ious[gi, pi] < IOU_DET:
            continue
        used_g.add(int(gi)); used_p.add(int(pi))
        res[int(gi)] = (int(pi), float(ious[gi, pi]))
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda:0"
    teacher = YOLO(TEACHER)
    student = YOLO(STUDENT)

    files = sorted(f for f in os.listdir(RGB_VAL) if f.endswith(".jpg"))
    print(f"val images: {len(files)}", flush=True)

    buckets = {"day": {k: 0 for k in "ABCDE"}, "night": {k: 0 for k in "ABCDE"}}
    cls_A = {}
    ts_iou_A = []      # teacher-student box IoU on A-class objects
    s_iou_A, t_iou_A = [], []

    with open(OUT_DIR + "/per_image.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "bucket", "n_gt", "A", "B", "C", "D", "E"])
        for k, fname in enumerate(files):
            rgb_path = os.path.join(RGB_VAL, fname)
            ir_path = os.path.join(IR_VAL, fname)
            lbl_path = os.path.join(RGB_LBL, os.path.splitext(fname)[0] + ".txt")
            im = Image.open(rgb_path)
            gt = load_gt(lbl_path, *im.size)
            lum = float(np.asarray(im.convert("L")).mean())
            bk = "day" if lum >= LUM_THRESHOLD else "night"

            tp = teacher.predict(ir_path, imgsz=IMGSZ, conf=CONF, verbose=False, device=device)[0]
            sp = student.predict(rgb_path, imgsz=IMGSZ, conf=CONF, verbose=False, device=device)[0]
            tboxes = np.column_stack([tp.boxes.xyxy.cpu().numpy(),
                                      tp.boxes.cls.cpu().numpy()]) if len(tp.boxes) else np.zeros((0, 5))
            sboxes = np.column_stack([sp.boxes.xyxy.cpu().numpy(),
                                      sp.boxes.cls.cpu().numpy()]) if len(sp.boxes) else np.zeros((0, 5))
            # rescale pred to original image coords (predict already returns orig coords)
            tmatch = match(tboxes, gt)
            smatch = match(sboxes, gt)

            cnt = {k2: 0 for k2 in "ABCDE"}
            for gi in range(len(gt)):
                si, s_iou = smatch[gi]
                ti, t_iou = tmatch[gi]
                c = int(gt[gi, 0])
                if si >= 0:
                    if s_iou >= IOU_GOOD:
                        cnt["B"] += 1
                    elif t_iou >= IOU_GOOD:
                        cnt["A"] += 1
                        cls_A[c] = cls_A.get(c, 0) + 1
                        s_iou_A.append(s_iou); t_iou_A.append(t_iou)
                        if ti >= 0:
                            ts_iou_A.append(float(iou_matrix(
                                sboxes[si:si+1, :4], tboxes[ti:ti+1, :4])[0, 0]))
                    else:
                        cnt["C"] += 1
                else:
                    if t_iou >= IOU_DET:
                        cnt["D"] += 1
                    else:
                        cnt["E"] += 1
            for k2 in "ABCDE":
                buckets[bk][k2] += cnt[k2]
            w.writerow([fname, bk, len(gt)] + [cnt[k2] for k2 in "ABCDE"])
            if (k + 1) % 200 == 0:
                print(f"{k+1}/{len(files)}", flush=True)

    def pct(d):
        tot = sum(d.values())
        return {k2: (d[k2], round(100 * d[k2] / max(tot, 1), 1)) for k2 in "ABCDE"}, tot

    summary = {"thresholds": {"conf": CONF, "iou_det": IOU_DET, "iou_good": IOU_GOOD},
               "luminance_threshold": LUM_THRESHOLD,
               "teacher": TEACHER, "student": STUDENT, "n_images": len(files)}
    for bk in ("day", "night",):
        p, tot = pct(buckets[bk])
        summary[bk] = {"total_gt": tot, "buckets": p}
    summary["A_class_detail"] = {
        "per_class": {NAMES.get(c, c): n for c, n in sorted(cls_A.items())},
        "student_iou_mean": round(float(np.mean(s_iou_A)), 3) if s_iou_A else None,
        "teacher_iou_mean": round(float(np.mean(t_iou_A)), 3) if t_iou_A else None,
        "teacher_student_box_iou_mean": round(float(np.mean(ts_iou_A)), 3) if ts_iou_A else None,
        "n_A_total": len(s_iou_A),
    }
    with open(OUT_DIR + "/summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    print("LOC_GAP_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
