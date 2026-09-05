#!/usr/bin/env python3
"""CGA-KD trainer (pre-registered 2026-09-05): Condition-aware Geometry-anchored KD.

Arms (frozen in 07_研究分析/方法预注册_CGA-KD_20260905.md):
  geom      geometry-anchored KD: IoU + class terms only (no feature mimicry)
  invariant learned invariant/specific decomposition g/h on teacher P4/P5 feats
            (+ pairing-adversarial verification via GRL; student aligns to g only)
  gated     corrected terms with per-sample ISP-proxy gate
            lambda_i = sigmoid(w0 + w1*(1 - luminance_i)) scaling teacher feats
Reuses the frozen cmdistill protocol yaml / dataset / guard / receipt machinery.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

REPO_ROOT = Path("/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo_osssl.rgbt_cmdistill_kd import CMDistillError, cmdistill_terms  # noqa: E402
from yolo_osssl.rgbt_hnewa_pairing import RGBTSharedGeometryDataset  # noqa: E402
from tools.project_resource_guard import require_bound_lease_from_environment  # noqa: E402
from tools.write_jstars_run_receipt import (  # noqa: E402
    METHOD_IDENTITIES,
    emit_bound_run_receipt,
)

ARMS = ("geom", "invariant", "gated")
LEVEL_ALIGN = (1, 2)  # P4, P5 only (P3 architecture-saturated, see P2 probe)
EPS = 1e-6


class CMDistillTrainerError(RuntimeError):
    pass


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lamb * grad, None


def grad_reverse(x, lamb=1.0):
    return _GradReverse.apply(x, lamb)


def _mean_stats(rows):
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {k: sum(float(r.get(k, 0.0)) for r in rows) / len(rows) for k in keys}


def _preds_mapping(prediction):
    if isinstance(prediction, (tuple, list)):
        mappings = [i for i in prediction if isinstance(i, dict)]
        if len(mappings) != 1:
            raise CMDistillTrainerError("prediction tuple must contain exactly one mapping")
        prediction = mappings[0]
    if not isinstance(prediction, dict) or not {"boxes", "scores", "feats"} <= set(prediction):
        raise CMDistillTrainerError("prediction mapping lacks boxes/scores/feats")
    return prediction


class InvariantHeads(nn.Module):
    """g/h projection heads + pairing discriminator over GAP embeddings."""

    def __init__(self, level_channels, student_channels):
        super().__init__()
        mid = 128
        self.g = nn.ModuleDict({str(l): nn.Sequential(nn.Conv2d(c, mid, 1), nn.GELU(), nn.Conv2d(mid, 64, 1))
                                for l, c in level_channels.items()})
        self.h = nn.ModuleDict({str(l): nn.Sequential(nn.Conv2d(c, mid, 1), nn.GELU(), nn.Conv2d(mid, 64, 1))
                                for l, c in level_channels.items()})
        self.g_proj = nn.ModuleDict({str(l): nn.Linear(64, student_channels[str(l)])
                                     for l in level_channels})
        self.disc = nn.Sequential(nn.Linear(64 + student_channels, 64), nn.GELU(), nn.Linear(64, 1))

    def embed(self, lvl, feats):
        g = self.g[str(lvl)](feats).mean(dim=(2, 3))
        h = self.h[str(lvl)](feats).mean(dim=(2, 3))
        return g, h


class CGACriterion:
    def __init__(self, native, teacher, *, strides, arm, host_model):
        self.native = native
        self.teacher = teacher
        self.strides = tuple(float(v) for v in strides)
        self.arm = arm
        self.stats_rows = []
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.aux = None
        self.host_model = host_model

    def _ensure_heads(self, teacher_feats, student_feats):
        if self.arm != "invariant" or self.aux is not None:
            return
        level_channels = {str(l): teacher_feats[l].shape[1] for l in LEVEL_ALIGN}
        student_channels = {str(l): student_feats[l].shape[1] for l in LEVEL_ALIGN}
        self.aux = InvariantHeads(level_channels, student_channels).to(teacher_feats[0].device)
        self.host_model.add_module("cga_aux", self.aux)

    @staticmethod
    def _gate(batch):
        img = batch["img"].float()
        lum = img.mean(dim=(1, 2, 3)) / img.max().clamp(min=EPS) if img.max() > 0 else img.mean(dim=(1, 2, 3))
        darkness = 1.0 - lum / (lum.max() + EPS)
        return darkness  # (B,) in [0,1]-ish

    def __call__(self, prediction, batch):
        native_total, native_items = self.native(prediction, batch)
        if "strong_img" not in batch:
            return native_total, native_items
        student = _preds_mapping(prediction)
        self.teacher.eval()
        with torch.no_grad():
            teacher = _preds_mapping(self.teacher(batch["strong_img"]))

        if self.arm == "geom":
            try:
                terms = cmdistill_terms(student, teacher, self.strides, variant="corrected")
            except CMDistillError as e:
                raise CMDistillTrainerError(str(e)) from e
            kd = terms.iou + terms.cls
            values = {"kd_iou": terms.iou.detach(), "kd_cls": terms.cls.detach(),
                      "kd_fg_ratio": terms.fg_ratio}

        elif self.arm == "invariant":
            self._ensure_heads(teacher["feats"], student["feats"])
            with torch.no_grad():
                teacher_shuf = _preds_mapping(self.teacher(torch.roll(batch["strong_img"], 1, dims=0)))
            bsz = int(batch["img"].shape[0])
            student_emb = student["feats"][LEVEL_ALIGN[0]].mean(dim=(2, 3))
            g_p, h_p = self.aux.embed(str(LEVEL_ALIGN[0]), teacher["feats"][LEVEL_ALIGN[0]])
            g_s, h_s = self.aux.embed(str(LEVEL_ALIGN[0]), teacher_shuf["feats"][LEVEL_ALIGN[0]])
            # pairing discriminator: 1 = paired-with-student, 0 = shuffled
            d_pair = self.aux.disc(torch.cat([grad_reverse(h_p), student_emb], dim=1))
            d_shuf = self.aux.disc(torch.cat([grad_reverse(h_s), student_emb], dim=1))
            l_disc = (F.binary_cross_entropy_with_logits(d_pair, torch.ones_like(d_pair))
                      + F.binary_cross_entropy_with_logits(d_shuf, torch.zeros_like(d_shuf))) * 0.5
            # g must be indistinguishable (GRL pushes g away from pairing info)
            dg_pair = self.aux.disc(torch.cat([grad_reverse(g_p), student_emb], dim=1))
            dg_shuf = self.aux.disc(torch.cat([grad_reverse(g_s), student_emb], dim=1))
            l_grl = (F.binary_cross_entropy_with_logits(dg_pair, torch.zeros_like(dg_pair))
                     + F.binary_cross_entropy_with_logits(dg_shuf, torch.ones_like(dg_shuf))) * 0.5
            l_orth = (F.cosine_similarity(g_p, h_p, dim=1) ** 2).mean()
            # student aligns to g only (scale-free cosine), P4/P5
            l_align = 0.0
            for li in LEVEL_ALIGN:
                sf = student["feats"][li].mean(dim=(2, 3))
                gf = self.aux.g_proj(self.aux.g[str(li)](teacher["feats"][li]).mean(dim=(2, 3)))
                sf = sf / (sf.norm(dim=1, keepdim=True) + EPS)
                gf = gf / (gf.norm(dim=1, keepdim=True) + EPS)
                l_align = l_align + (1 - (sf * gf).sum(dim=1)).mean()
            l_align = l_align / len(LEVEL_ALIGN)
            kd = l_align + 0.1 * l_orth + 0.1 * l_grl + 0.1 * l_disc
            values = {"cga_align": l_align.detach(), "cga_orth": l_orth.detach(),
                      "cga_grl": l_grl.detach(), "cga_disc": l_disc.detach()}

        elif self.arm == "gated":
            darkness = self._gate(batch)
            gate = torch.sigmoid(-2.0 + 4.0 * darkness)  # (B,) high at night
            gated = dict(teacher)
            gated["feats"] = tuple(f * gate.view(-1, 1, 1, 1) for f in teacher["feats"])
            try:
                terms = cmdistill_terms(student, gated, self.strides, variant="corrected")
            except CMDistillError as e:
                raise CMDistillTrainerError(str(e)) from e
            kd = terms.pccfd + terms.slrd + terms.iou + terms.cls
            values = {"kd_pccfd": terms.pccfd.detach(), "kd_slrd": terms.slrd.detach(),
                      "kd_iou": terms.iou.detach(), "kd_cls": terms.cls.detach(),
                      "gate_mean": float(gate.mean())}
        else:
            raise CMDistillTrainerError(f"unknown arm {self.arm}")

        items = dict(native_items) if isinstance(native_items, dict) else {}
        items.update({k: v for k, v in values.items()})
        self.stats_rows.append({k: float(v) for k, v in values.items()})
        bsz = int(student["boxes"].shape[0])
        return native_total + float(bsz) * kd, items


def load_teacher(teacher_weights):
    payload = torch.load(str(teacher_weights), map_location="cpu", weights_only=False)
    model = payload.get("model") or payload.get("ema") if isinstance(payload, dict) else payload
    if not isinstance(model, torch.nn.Module):
        raise CMDistillTrainerError(f"teacher checkpoint has no model object: {teacher_weights}")
    return model.float()


def build_trainer(*, arm, overrides, paired_mapping, teacher_ckpt, max_steps, imgsz):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class CGAKDTrainer(DetectionTrainer):
        def __init__(self, *a, **k):
            self.cga_started = time.time()
            self.cga_steps = 0
            super().__init__(*a, **k)

        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode=mode, batch=batch)
            if mode == "train":
                return RGBTSharedGeometryDataset(ds, paired_mapping)
            return ds

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            if "strong_img" in batch:
                batch["strong_img"] = batch["strong_img"].to(self.device, non_blocking=True).float() / 255
            return batch

        def set_model_attributes(self):
            super().set_model_attributes()
            teacher = load_teacher(teacher_ckpt).to(self.device)
            self.model.criterion = CGACriterion(
                self.model.init_criterion(), teacher,
                strides=tuple(float(v) for v in self.model.stride), arm=arm,
                host_model=self.model)

        def validate(self):
            return {}, 0.0

        def final_eval(self):
            return {}, 0.0

        def optimizer_step(self):
            super().optimizer_step()
            self.cga_steps += 1
            if max_steps is not None and self.cga_steps >= max_steps:
                self.stop = True

    return CGAKDTrainer(overrides=overrides)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--teacher-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--skip-lease", action="store_true",
                        help="canary mode: skip resource-guard lease check")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    method_identity = str(config.get("method_identity", "PROTOCOL-ADAPTED"))
    if method_identity not in METHOD_IDENTITIES:
        raise CMDistillTrainerError(f"unsupported method identity: {method_identity}")
    paths = config["paths"]
    mapping_file = Path(paths["paired_train_mapping"])
    if not mapping_file.is_file():
        raise CMDistillTrainerError(f"paired mapping missing: {mapping_file}")
    if not args.teacher_weights.is_file():
        raise CMDistillTrainerError(f"teacher missing: {args.teacher_weights}")
    if args.validate_only:
        print(json.dumps({"status": "validated", "arm": args.arm}))
        return 0
    if not args.skip_lease:
        require_bound_lease_from_environment()
    paired_mapping = {str(k): str(v) for k, v in json.loads(mapping_file.read_text()).items()}

    aug = {k: float(v) for k, v in config["augmentation"].items()}
    overrides = {
        "model": str(Path(config["model"])),
        "data": str(Path(paths["student_data_yaml"])),
        "imgsz": int(config["imgsz"]), "epochs": int(config["epochs"]),
        "batch": int(args.batch), "workers": int(args.workers),
        "optimizer": str(config["optimizer"]), "lr0": float(config["lr0"]),
        "lrf": float(config["lrf"]), "momentum": float(config["momentum"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_epochs": float(config["warmup_epochs"]),
        "warmup_momentum": float(config["warmup_momentum"]),
        "warmup_bias_lr": float(config["warmup_bias_lr"]),
        "cos_lr": bool(config["cos_lr"]), "close_mosaic": int(config["close_mosaic"]),
        "nbs": int(config["nbs"]), "patience": int(config["patience"]),
        "amp": bool(config["amp"]), "deterministic": bool(config["deterministic"]),
        "seed": int(args.seed), "device": args.device,
        "val": False, "save": True, "save_period": -1, "pretrained": True,
        "plots": False, "project": str(args.output.parent), "name": args.output.name,
        "exist_ok": True, **aug,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(arm=args.arm, overrides=overrides, paired_mapping=paired_mapping,
                            teacher_ckpt=args.teacher_weights, max_steps=args.max_steps,
                            imgsz=int(config["imgsz"]))
    started = time.time()
    trainer.train()
    criterion = getattr(trainer.model, "criterion", None)
    stats_rows = list(getattr(criterion, "stats_rows", []))
    receipt = {
        "schema": "cga-kd-receipt-v1", "status": "completed",
        "method_identity": "CGA-KD-PRE-REG-20260905",
        "arm": args.arm,
        "losses": {"geom": ["iou", "cls"], "invariant": ["align_g", "orth", "grl", "disc"],
                   "gated": ["PCCFD", "SLRD", "IBCLD", "gate"]}[args.arm],
        "dataset": config["dataset"], "seed": int(args.seed),
        "epochs_configured": int(config["epochs"]),
        "optimizer_steps": int(getattr(trainer, "cga_steps", 0)),
        "kd_batches_logged": len(stats_rows),
        "kd_stats_mean": _mean_stats(stats_rows),
        "kd_stats_last": stats_rows[-1] if stats_rows else {},
        "teacher_weights": str(args.teacher_weights),
        "official_test_accessed": False,
        "walltime_seconds": time.time() - started,
    }
    (args.output / "completion_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not args.skip_lease:
        import ultralytics
        emit_bound_run_receipt(
        run_dir=args.output.resolve() / "run_evidence",
        method_identity=method_identity, dataset=str(config["dataset"]),
        data_role="development_train", seed=args.seed, run_kind="train",
        trainers=[Path(__file__)],
        losses=[Path(__file__)],
        configs=[args.config],
        split_rosters=[mapping_file],
        metric_files=[args.output / "completion_receipt.json"],
        environment={"torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__)},
        inputs={"arm": args.arm,
                "initial_weights": str(Path(config["model"]).resolve()),
                "teacher_weights": str(args.teacher_weights.resolve())},
    )
    print(json.dumps({"status": "completed", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
