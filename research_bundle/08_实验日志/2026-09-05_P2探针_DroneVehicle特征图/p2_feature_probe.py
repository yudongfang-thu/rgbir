# -*- coding: utf-8 -*-
"""P2 feature-map probe v2 (DroneVehicle YOLO11 trio).

Question: did cross-modal distillation move the RGB student's features toward the
IR teacher, and where (which level, which frequency band, which input condition)?

Models (all YOLO11, ultralytics 8.4.115):
  T        = infrared_seed42_native_b32a2   (IR-input teacher, p3_causal formal_native)
  S_native = native_rgb_s42_b64_e200        (RGB student, NO distillation)
  S_dist   = adapted_v2 dronevehicle_seed42 (RGB student, distilled)

Conditions on DroneVehicle val (paired RGB/IR by filename):
  paired      T(IR_i)   vs S(RGB_i)
  shuffled    T(IR_i)   vs S(RGB_perm(i))
  same_input  T(RGB_i)  vs S(RGB_i)   (modality gap removed from the input side)

Per level P3/P4/P5: linear CKA, saliency-map Pearson corr (gray_sar-probe compat),
FFT band divergence ratio D_high/D_low (rc=0.5, FreqKD eq.8 style).
Read-only inference. Outputs JSON + per-image CSV.
"""
import csv
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

R = "/mnt/dataset/yudongfang/projects/RGBT_campaign"
OUT_DIR = os.path.join(R, "artifacts", "p2_feature_probe_20260905")
N_MAX = 300
IMGSZ = 640
PERM_SEED = 20260905
RC = 0.5
EPS = 1e-6
LEVELS = ("P3", "P4", "P5")

IR_VAL = f"{R}/data/processed/dronevehicle/yolo/hbb_v1/infrared/images/val"
RGB_VAL = f"{R}/data/processed/dronevehicle/yolo/hbb_v1/rgb/images/val"
MODELS = {
    "T": f"{R}/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/best.pt",
    "S_native": f"{R}/runs/rgbt_cmdistill_paper_reconstructed_v2/native_rgb_s42_b64_e200/weights/best.pt",
    "S_dist": f"{R}/runs/rgbt_cmdistill_adapted_v2/dronevehicle_seed42_b32_e200/weights/best.pt",
}
STUDENTS = ("S_native", "S_dist")
CONDITIONS = ("paired", "shuffled", "same_input")


def load_model(ckpt, device):
    from ultralytics import YOLO
    m = YOLO(ckpt)
    net = m.model.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    captured = {}

    def pre_hook(module, args):
        feats = args[0]
        if isinstance(feats, (list, tuple)) and len(feats) >= 3:
            captured["feats"] = [f.detach().float() for f in feats[-3:]]

    net.model[-1].register_forward_pre_hook(pre_hook)
    return net, captured


def preprocess(path, device):
    im = Image.open(path).convert("RGB").resize((IMGSZ, IMGSZ), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0
    return x.unsqueeze(0).to(device)


@torch.no_grad()
def forward_feats(net, captured, x):
    net(x)
    return captured["feats"]


def linear_cka(x, y):
    num = torch.norm(x.T @ y, p="fro") ** 2
    den = torch.norm(x.T @ x, p="fro") * torch.norm(y.T @ y, p="fro") + EPS
    return (num / den).item()


def saliency_corr(a, b):
    s1 = a.norm(dim=0).flatten()
    s2 = b.norm(dim=0).flatten()
    s1 = s1 - s1.mean()
    s2 = s2 - s2.mean()
    return ((s1 @ s2) / (s1.norm() * s2.norm() + EPS)).item()


def match_size(f, ref):
    if f.shape[1:] == ref.shape[1:]:
        return f
    return F.interpolate(f.unsqueeze(0), size=ref.shape[1:], mode="bilinear",
                         align_corners=False)[0]


def band_divergence(ft, fs):
    def spec(f):
        f = f - f.mean(dim=(2, 3), keepdim=True)
        f = f / (f.norm(dim=(2, 3), keepdim=True) + EPS)
        return torch.fft.fftshift(torch.fft.fft2(f, dim=(2, 3)), dim=(2, 3))

    ft_s, fs_s = spec(ft), spec(fs)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, ft.shape[2], device=ft.device),
        torch.linspace(-1, 1, ft.shape[3], device=ft.device), indexing="ij")
    radius = torch.sqrt(xx ** 2 + yy ** 2)
    out = {}
    for name, msk in (("low", (radius <= RC).float()), ("high", (radius > RC).float())):
        out[name] = ((ft_s - fs_s).abs() ** 2 * msk).mean().item()
    return out


def mean_sd(a):
    a = np.asarray(a, dtype=np.float64)
    return round(float(a.mean()), 5), round(float(a.std()), 5), int(len(a))


def main():
    device = "cuda:0"
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(PERM_SEED)
    nets, caps = {}, {}
    for k, ck in MODELS.items():
        nets[k], caps[k] = load_model(ck, device)

    ir_files = sorted(os.listdir(IR_VAL))
    rgb_set = set(os.listdir(RGB_VAL))
    common = [f for f in ir_files if f in rgb_set]
    if len(common) > N_MAX:
        common = common[:: int(np.ceil(len(common) / N_MAX))]
    n = len(common)
    perm = list(range(n))[::-1]
    print(f"[dronevehicle] paired val images: {n}", flush=True)

    agg = {s: {c: {lv: {"cka": [], "corr": []} for lv in LEVELS} for c in CONDITIONS}
           for s in STUDENTS}
    fft_ratio = {s: {lv: [] for lv in LEVELS} for s in STUDENTS}

    with open(os.path.join(OUT_DIR, "per_image.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "image", "level", "student"] +
            [f"cka_{c}" for c in CONDITIONS] + [f"corr_{c}" for c in CONDITIONS] +
            ["fft_hi_lo"])
        writer.writeheader()

        for k, fname in enumerate(common):
            x_ir = preprocess(os.path.join(IR_VAL, fname), device)
            x_rgb = preprocess(os.path.join(RGB_VAL, fname), device)
            x_rgb_perm = preprocess(os.path.join(RGB_VAL, common[perm[k]]), device)

            f_t = forward_feats(nets["T"], caps["T"], x_ir)        # paired & shuffled teacher side
            f_t_rgb = forward_feats(nets["T"], caps["T"], x_rgb)   # same-input teacher side
            f_s = {}
            for s in STUDENTS:
                f_s[s] = forward_feats(nets[s], caps[s], x_rgb)
            f_s_perm = {s: forward_feats(nets[s], caps[s], x_rgb_perm) for s in STUDENTS}

            for lv_i, lv in enumerate(LEVELS):
                ft = f_t[lv_i][0]
                ft_rgb = f_t_rgb[lv_i][0]
                for s in STUDENTS:
                    fs = match_size(f_s[s][lv_i][0], ft)
                    fs_perm = match_size(f_s_perm[s][lv_i][0], ft)
                    fs_si = match_size(f_s[s][lv_i][0], ft_rgb)
                    conds = {
                        "paired": (ft, fs),
                        "shuffled": (ft, fs_perm),
                        "same_input": (ft_rgb, fs_si),
                    }
                    row = {"image": fname, "level": lv, "student": s}
                    for c, (a, b) in conds.items():
                        cka = linear_cka(a.flatten(1).T.float(), b.flatten(1).T.float())
                        corr = saliency_corr(a, b)
                        agg[s][c][lv]["cka"].append(cka)
                        agg[s][c][lv]["corr"].append(corr)
                        row[f"cka_{c}"] = round(cka, 5)
                        row[f"corr_{c}"] = round(corr, 5)
                    bd = band_divergence(ft.unsqueeze(0), fs.unsqueeze(0))
                    ratio = bd["high"] / (bd["low"] + EPS)
                    fft_ratio[s][lv].append(ratio)
                    row["fft_hi_lo"] = round(ratio, 4)
                    writer.writerow(row)

            if (k + 1) % 50 == 0:
                print(f"[dronevehicle] {k + 1}/{n}", flush=True)

    summary = {"dataset": "dronevehicle", "n_images": n, "imgsz": IMGSZ,
               "teacher": MODELS["T"], "levels": {}}
    for lv in LEVELS:
        summary["levels"][lv] = {}
        for s in STUDENTS:
            entry = {}
            for c in CONDITIONS:
                m, sd, n_ = mean_sd(agg[s][c][lv]["cka"])
                m2, sd2, _ = mean_sd(agg[s][c][lv]["corr"])
                entry[c] = {"cka_mean": m, "cka_sd": sd, "corr_mean": m2, "corr_sd": sd2, "n": n_}
            m, sd, _ = mean_sd(fft_ratio[s][lv])
            entry["fft_high_over_low"] = {"mean": m, "sd": sd}
            summary["levels"][lv][s] = entry
    # distilled-minus-native contrast on the paired condition
    for lv in LEVELS:
        pn = summary["levels"][lv]["S_native"]["paired"]["cka_mean"]
        pd_ = summary["levels"][lv]["S_dist"]["paired"]["cka_mean"]
        summary["levels"][lv]["distill_pull_paired"] = round(pd_ - pn, 5)
        sn = summary["levels"][lv]["S_native"]["same_input"]["cka_mean"]
        sdg = summary["levels"][lv]["S_dist"]["same_input"]["cka_mean"]
        summary["levels"][lv]["distill_pull_same_input"] = round(sdg - sn, 5)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1), flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
