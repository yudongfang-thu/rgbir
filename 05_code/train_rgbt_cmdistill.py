#!/usr/bin/env python3
"""CMDistill-adapted trainer: native YOLO11 loss + PCCFD/SLRD/IBCLD.

Modes follow the ALIGN_AND_IMPROVE plan: IR-teacher (privileged) to weak
student on the frozen paired dataset config. The teacher is a frozen
single-modality detector; inference keeps the plain student graph.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo_osssl.rgbt_cmdistill_kd import CMDistillError, cmdistill_terms  # noqa: E402
from yolo_osssl.rgbt_hnewa_pairing import RGBTSharedGeometryDataset  # noqa: E402
from tools.project_resource_guard import require_bound_lease_from_environment  # noqa: E402
from tools.write_jstars_run_receipt import (  # noqa: E402
    METHOD_IDENTITIES,
    emit_bound_run_receipt,
    implementation_files,
)

ARMS = ("cmdistill_literal", "cmdistill_corrected")


class CMDistillTrainerError(RuntimeError):
    pass


def _mean_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}


def _preds_mapping(prediction) -> dict:
    if isinstance(prediction, (tuple, list)):
        mappings = [item for item in prediction if isinstance(item, dict)]
        if len(mappings) != 1:
            raise CMDistillTrainerError("prediction tuple must contain exactly one mapping")
        prediction = mappings[0]
    if not isinstance(prediction, dict) or not {"boxes", "scores", "feats"} <= set(prediction):
        raise CMDistillTrainerError("prediction mapping lacks boxes/scores/feats")
    return prediction


class CMDistillCriterion:
    """Native YOLO loss plus the declared CMDistill-adapted terms."""

    def __init__(
        self,
        native,
        teacher: torch.nn.Module,
        *,
        strides: tuple[float, ...],
        variant: str = "corrected",
    ) -> None:
        self.native = native
        self.teacher = teacher
        self.strides = tuple(float(value) for value in strides)
        self.variant = variant
        self.stats_rows: list[dict[str, float]] = []
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    def __call__(self, prediction, batch):
        native_total, native_items = self.native(prediction, batch)
        if "strong_img" not in batch:
            return native_total, native_items
        student = _preds_mapping(prediction)
        self.teacher.eval()
        with torch.no_grad():
            teacher = _preds_mapping(self.teacher(batch["strong_img"]))
        try:
            terms = cmdistill_terms(
                student,
                teacher,
                self.strides,
                variant=self.variant,
            )
        except CMDistillError as error:
            raise CMDistillTrainerError(str(error)) from error
        batch_size = int(student["boxes"].shape[0])
        kd_total = terms.pccfd + terms.slrd + terms.iou + terms.cls
        items = dict(native_items) if isinstance(native_items, dict) else {}
        values = {
            "kd_pccfd": terms.pccfd.detach(),
            "kd_slrd": terms.slrd.detach(),
            "kd_iou": terms.iou.detach(),
            "kd_cls": terms.cls.detach(),
            "kd_fg_ratio": terms.fg_ratio,
        }
        items.update(values)
        self.stats_rows.append(
            {
                key: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                for key, value in values.items()
            }
        )
        return native_total + float(batch_size) * kd_total, items


def load_teacher(teacher_weights: Path) -> torch.nn.Module:
    payload = torch.load(str(teacher_weights), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        model = payload.get("model") or payload.get("ema")
    else:
        model = payload
    if not isinstance(model, torch.nn.Module):
        raise CMDistillTrainerError(f"teacher checkpoint has no model object: {teacher_weights}")
    return model.float()


def build_cmdistill_trainer(*, arm, overrides, paired_mapping, teacher_ckpt, max_steps):
    from ultralytics.models.yolo.detect import DetectionTrainer

    variant = "literal" if arm == "cmdistill_literal" else "corrected"

    class CMDistillTrainer(DetectionTrainer):
        def __init__(self, *a, **k):
            self.cmd_started = time.time()
            self.cmd_steps = 0
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
            self.model.criterion = CMDistillCriterion(
                self.model.init_criterion(),
                teacher,
                strides=tuple(float(value) for value in self.model.stride),
                variant=variant,
            )

        def validate(self):
            # Weak student eval is owned by tools/eval_rgbt_detector.py on last.pt.
            return {}, 0.0

        def final_eval(self):
            return {}, 0.0

        def optimizer_step(self):
            super().optimizer_step()
            self.cmd_steps += 1
            if max_steps is not None and self.cmd_steps >= max_steps:
                self.stop = True

    return CMDistillTrainer(overrides=overrides)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--teacher-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int)
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
    trainer = build_cmdistill_trainer(
        arm=args.arm, overrides=overrides, paired_mapping=paired_mapping,
        teacher_ckpt=args.teacher_weights, max_steps=args.max_steps,
    )
    started = time.time()
    trainer.train()
    criterion = getattr(trainer.model, "criterion", None)
    stats_rows = list(getattr(criterion, "stats_rows", []))
    receipt = {
        "schema": "rgbt-cmdistill-adapted-receipt-v1", "status": "completed",
        "method_track": f"CMDistill {method_identity} (JSTARS 2025)",
        "method_identity": method_identity,
        "arm": args.arm, "losses": ["PCCFD", "SLRD", "IBCLD"],
        "lambda1": 1.0, "lambda2": 1.0, "lambda3": 1.0,
        "dataset": config["dataset"], "seed": int(args.seed),
        "epochs_configured": int(config["epochs"]),
        "optimizer_steps": int(getattr(trainer, "cmd_steps", 0)),
        "kd_batches_logged": len(stats_rows),
        "kd_stats_mean": _mean_stats(stats_rows),
        "kd_stats_last": stats_rows[-1] if stats_rows else {},
        "teacher_weights": str(args.teacher_weights),
        "official_test_accessed": False,
        "walltime_seconds": time.time() - started,
    }
    (args.output / "completion_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    import ultralytics

    emit_bound_run_receipt(
        run_dir=args.output.resolve() / "run_evidence",
        method_identity=method_identity,
        dataset=str(config["dataset"]),
        data_role="development_train",
        seed=args.seed,
        run_kind="train",
        trainers=[Path(__file__)],
        losses=implementation_files(criterion, getattr(criterion, "native", None), cmdistill_terms),
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
