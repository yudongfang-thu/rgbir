#!/usr/bin/env python3
"""CCLKD-adapted trainer: KD + CCL terms on native YOLO11 detection loss.

v1 scope (per ALIGN_AND_IMPROVE_PLAN_20260902): LLD (Eq 8) + CCL (Eq 16-18)
computed on per-anchor teacher/student class scores and DFL-projected boxes;
FLD/RLD require per-anchor feature plumbing and stay disabled until the
OGSOD-native sanity run closes. Teacher is a frozen single-modality detector
on the privileged modality image (paper: easy-modality teacher).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo_osssl.rgbt_cclkd_kd import (  # type: ignore[name-error]
    CCLKDLoss,
    ccl_class_weights,
    ccl_infonce,
    ccl_literal,
    cop_partition,
    lkd_logit,
    masked_box_distributions,
    patm_temperature,
)
from yolo_osssl.rgbt_hnewa_pairing import RGBTSharedGeometryDataset  # type: ignore[name-error]
from tools.project_resource_guard import require_bound_lease_from_environment
from tools.write_jstars_run_receipt import emit_bound_run_receipt, implementation_files

ARMS = ("cclkd_literal", "cclkd_infonce")
DFL_BINS = 16


class CCLKDTrainerError(RuntimeError):
    pass


def dfl_project(box_raw: torch.Tensor) -> torch.Tensor:
    """(B, 4*reg_max, N) raw DFL logits -> (B, N, 4) decoded offsets."""

    b, _, n = box_raw.shape
    dist = box_raw.view(b, 4, DFL_BINS, n).softmax(2)
    proj = dist.new_tensor(list(range(DFL_BINS))).view(1, 1, DFL_BINS, 1)
    return (dist * proj).sum(2).permute(0, 2, 1)  # (B, N, 4)


def _preds_mapping(prediction) -> dict:
    """Resolve the Ultralytics 8.4 Detect output to its raw boxes/scores/feats mapping.

    Training batches hand the criterion ``{"boxes", "scores", "feats"}``; an
    eval-mode teacher returns ``(y_decoded, preds_mapping)``.
    """

    if isinstance(prediction, (tuple, list)):
        mappings = [item for item in prediction if isinstance(item, dict)]
        if len(mappings) != 1:
            raise CCLKDTrainerError("prediction tuple must contain exactly one mapping")
        prediction = mappings[0]
    if not isinstance(prediction, dict) or not {"boxes", "scores", "feats"} <= set(prediction):
        raise CCLKDTrainerError("prediction mapping lacks boxes/scores/feats: %s" % type(prediction))
    return prediction


class CCLKDCriterion:
    """Wraps the native YOLO loss with per-image CCLKD LLD+CCL terms.

    Teacher and student anchors share one grid, so both sides enter the KD
    terms in the same declared representation: ``sigmoid(dfl_project(raw
    DFL logits))`` for boxes and ``sigmoid(raw logits)`` for class scores.
    The student side keeps its gradient; only the teacher branch is detached.
    KD terms are batch means scaled by ``B`` to match the native ``loss * B``.
    """

    def __init__(self, native, teacher: nn.Module, *, arm: str, alpha: float = 5.0) -> None:
        self.native = native
        self.teacher = teacher
        self.arm = arm
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.stats_rows: list[dict[str, float]] = []

    def __call__(self, prediction, batch):
        native_total, native_items = self.native(prediction, batch)
        if "strong_img" not in batch:
            return native_total, native_items
        s_preds = _preds_mapping(prediction)
        reg_max4 = 4 * DFL_BINS
        student_box = torch.sigmoid(dfl_project(s_preds["boxes"][:, :reg_max4, :]))
        self.teacher.eval()
        with torch.no_grad():
            t_preds = _preds_mapping(self.teacher(batch["strong_img"]))
            teacher_box = torch.sigmoid(dfl_project(t_preds["boxes"].detach()[:, :reg_max4, :]))
            teacher_cls = torch.sigmoid(t_preds["scores"].detach()).transpose(1, 2)  # (B, N, nc)
        kd_total = native_total.new_zeros(())
        ccl_total = native_total.new_zeros(())
        bsz = int(s_preds["boxes"].shape[0])
        for i in range(bsz):
            lld, ccl, _, _ = self._per_image(teacher_cls[i], teacher_box[i], student_box[i], bsz)
            kd_total = kd_total + lld
            ccl_total = ccl_total + ccl
        detached = dict(native_items) if isinstance(native_items, dict) else {}
        detached["kd_lld"] = kd_total.detach()
        detached["kd_ccl"] = ccl_total.detach()
        self.stats_rows.append(
            {"kd_lld": float(detached["kd_lld"]), "kd_ccl": float(detached["kd_ccl"])}
        )
        return native_total + float(bsz) * (kd_total + ccl_total), detached

    def _per_image(self, t_cls, t_box, s_box, bsz):
        parts = cop_partition(t_cls)
        temperature = patm_temperature(t_cls, parts.mask, self.alpha)
        lld = lkd_logit(t_cls, t_box, s_box, parts.mask, temperature, bsz)
        b_pos_t, b_pos_s, b_neg_t, b_neg_s = masked_box_distributions(t_box, s_box, parts.mask, temperature)
        if self.arm == "cclkd_literal":
            ccl = ccl_literal(b_pos_t, b_pos_s, b_neg_t, b_neg_s, parts.mask, bsz)
        else:
            ccl = ccl_infonce(b_pos_t, b_pos_s, b_neg_t, b_neg_s, parts.mask, bsz)
        return lld, ccl, temperature, parts.mask


def build_cclkd_trainer(*, arm, overrides, paired_mapping, teacher_ckpt, base_weights, max_steps):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class CCLKDTrainer(DetectionTrainer):
        def __init__(self, *a, **k):
            self.cclkd_started = time.time()
            self.cclkd_steps = 0
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
            teacher = YOLO(str(teacher_ckpt), task="detect").model.to(self.device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            self.model.criterion = CCLKDCriterion(self.model.init_criterion(), teacher, arm=arm)

        def optimizer_step(self):
            super().optimizer_step()
            self.cclkd_steps += 1
            if max_steps is not None and self.cclkd_steps >= max_steps:
                self.stop = True

    return CCLKDTrainer(overrides=overrides)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--teacher-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    mapping_file = Path(paths["paired_train_mapping"])
    if not mapping_file.is_file():
        raise CCLKDTrainerError(f"paired mapping missing: {mapping_file}")
    if not args.teacher_weights.is_file():
        raise CCLKDTrainerError(f"teacher checkpoint missing: {args.teacher_weights}")
    if args.validate_only:
        print(json.dumps({"status": "validated", "arm": args.arm}))
        return 0
    require_bound_lease_from_environment()
    paired_mapping = {str(k): str(v) for k, v in json.loads(mapping_file.read_text()).items()}

    aug = {k: float(v) for k, v in config["augmentation"].items()}
    overrides = {
        "model": str(Path(config["model"])),
        "data": str(Path(paths["student_data_yaml"])),
        "imgsz": int(config["imgsz"]),
        "epochs": int(config["epochs"]),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "optimizer": str(config["optimizer"]),
        "lr0": float(config["lr0"]), "lrf": float(config["lrf"]),
        "momentum": float(config["momentum"]), "weight_decay": float(config["weight_decay"]),
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
    trainer = build_cclkd_trainer(
        arm=args.arm, overrides=overrides, paired_mapping=paired_mapping,
        teacher_ckpt=args.teacher_weights, base_weights=Path(config["model"]),
        max_steps=args.max_steps,
    )
    started = time.time()
    trainer.train()
    receipt = {
        "schema": "rgbt-cclkd-adapted-receipt-v1", "status": "completed",
        "method_track": "CCLKD-adapted (GIS 2026); not an exact reproduction",
        "arm": args.arm, "kd_terms_v1": ["LLD", "CCL"],
        "dataset": config["dataset"], "seed": int(args.seed),
        "epochs_configured": int(config["epochs"]),
        "optimizer_steps": int(getattr(trainer, "cclkd_steps", 0)),
        "teacher_weights": str(args.teacher_weights),
        "lambda_kd": 1.0, "lambda_cc": 1.0,
        "kd_dose": "batch_mean_x_batch_size",
        "official_test_accessed": False,
        "walltime_seconds": time.time() - started,
    }
    rows = getattr(getattr(trainer.model, "criterion", None), "stats_rows", [])
    if rows:
        receipt["kd_last_steps"] = rows[-5:]
    (args.output / "completion_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    import ultralytics

    criterion = getattr(trainer.model, "criterion", None)
    emit_bound_run_receipt(
        run_dir=args.output.resolve() / "run_evidence",
        method_identity="ADAPTED-PARTIAL",
        dataset=str(config["dataset"]),
        data_role="development_train",
        seed=args.seed,
        run_kind="train",
        trainers=[Path(__file__)],
        losses=implementation_files(criterion, getattr(criterion, "native", None), CCLKDLoss),
        configs=[args.config],
        split_rosters=[mapping_file],
        metric_files=[args.output / "completion_receipt.json"],
        environment={"torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__)},
        inputs={
            "arm": args.arm,
            "initial_weights": str(Path(config["model"]).resolve()),
            "teacher_weights": str(args.teacher_weights.resolve()),
        },
    )
    print(json.dumps({"status": "completed", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
