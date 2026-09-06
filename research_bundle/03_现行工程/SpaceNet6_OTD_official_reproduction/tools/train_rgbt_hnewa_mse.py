#!/usr/bin/env python3
"""Frozen Hnewa-CMKD-MSE-inspired YOLO11n trainer for the RGB-T stage-1 track.

Two modes follow ``refine-logs/rgbt_literature_v1/EXPERIMENT_PLAN.md`` §5:

- ``fusion-teacher`` (T0): train one six-channel weak+strong fusion teacher
  on the paired train split with shared geometry, E200, ``last.pt`` endpoint.
- ``student`` (H1/H2/H3): train the ordinary weak-modality student with
  ``L_YOLO + B * 0.5 * mean(MSE(P3), MSE(P4), MSE(P5))``.  ``h1_paired`` and
  ``h2_shuffled`` read the frozen fusion teacher on ``cat(weak, strong)`` or
  ``cat(weak, shuffled_strong)``; ``h3_same_modal`` reads a frozen 3-channel
  weak-native seed42 teacher on the weak image only.

Inference remains the plain weak-modality student graph; validation is
disabled here and always performed separately by ``tools/eval_rgbt_detector.py``
on ``last.pt``.
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

from yolo_osssl.rgbt_hnewa_kd import (
    HNEWA_FEATURE_MSE_ALPHA,
    HNEWA_YOLO11_TAP_LAYERS,
    HnewaFeatureDistiller,
    build_six_channel_yolo11_fusion_teacher,
    combine_hnewa_total_loss,
    freeze_hnewa_teacher,
    hnewa_feature_mse,
    six_channel_fusion,
    yolo11_prediction_features,
)
from tools.project_resource_guard import require_bound_lease_from_environment
from tools.write_jstars_run_receipt import emit_bound_run_receipt, implementation_files
from yolo_osssl.rgbt_hnewa_pairing import RGBTSharedGeometryDataset

MODES = ("fusion-teacher", "student")
STUDENT_ARMS = ("h1_paired", "h2_shuffled", "h3_same_modal")
PAIRED_STUDENT_ARMS = frozenset({"h1_paired", "h2_shuffled"})
SHUFFLED_STUDENT_ARMS = frozenset({"h2_shuffled"})


class HnewaTrainerError(RuntimeError):
    """Raised when the frozen Hnewa track contract is violated."""


def load_mapping(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise HnewaTrainerError(f"mapping must be a non-empty JSON object: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def load_fusion_teacher(teacher_weights: Path, base_weights: Path) -> nn.Module:
    """Load the 6ch teacher from a T0 ``last.pt`` Ultralytics checkpoint.

    The pickled model object is used directly: rebuilding from ``yolo11n.pt``
    would restore the COCO 80-class head while the trained teacher carries the
    dataset's ``nc``.
    """

    payload = torch.load(str(teacher_weights), map_location="cpu", weights_only=False)
    model = payload.get("model") if isinstance(payload, dict) else payload
    if not isinstance(model, nn.Module):
        raise HnewaTrainerError(f"fusion teacher checkpoint has no model object: {teacher_weights}")
    return model.float()


class HnewaMSELoss:
    """Ultralytics criterion façade adding the frozen feature-MSE term."""

    def __init__(
        self,
        native: Any,
        student: nn.Module,
        *,
        arm: str,
        fusion_teacher: nn.Module | None,
        same_modal_teacher: nn.Module | None,
    ) -> None:
        self.native = native
        self.arm = arm
        self.student = student
        self.fusion_teacher = fusion_teacher
        self.same_modal_teacher = same_modal_teacher
        self.stats_rows: list[dict[str, float]] = []
        if arm in PAIRED_STUDENT_ARMS:
            if fusion_teacher is None:
                raise HnewaTrainerError(f"{arm} requires a fusion teacher")
            self.distiller = HnewaFeatureDistiller(student, fusion_teacher)
        elif arm == "h3_same_modal":
            if same_modal_teacher is None:
                raise HnewaTrainerError("h3_same_modal requires a same-modal teacher")
            freeze_hnewa_teacher(same_modal_teacher)
        else:
            raise HnewaTrainerError(f"unknown student arm {arm}")

    def __call__(self, prediction: Any, batch: dict[str, torch.Tensor]):
        # Validator batches carry the 3ch val view only; KD applies to training
        # batches that the paired loader equipped with the strong half.
        strong_key = "shuffled_strong_img" if self.arm in SHUFFLED_STUDENT_ARMS else "strong_img"
        if self.arm in PAIRED_STUDENT_ARMS and strong_key not in batch:
            return self.native(prediction, batch)
        native_total, native_items = self.native(prediction, batch)
        batch_size = int(batch["img"].shape[0])
        if self.arm in PAIRED_STUDENT_ARMS:
            strong_key = "shuffled_strong_img" if self.arm in SHUFFLED_STUDENT_ARMS else "strong_img"
            if strong_key not in batch:
                raise HnewaTrainerError(f"{self.arm} batch is missing {strong_key}")
            fusion_images = six_channel_fusion(batch["img"], batch[strong_key])
            result = self.distiller.loss(prediction, fusion_images)
        else:
            student_features = yolo11_prediction_features(self.student, prediction)
            self.same_modal_teacher.eval()
            with torch.no_grad():
                teacher_prediction = self.same_modal_teacher(batch["img"])
            teacher_features = tuple(
                feature.detach()
                for feature in yolo11_prediction_features(self.same_modal_teacher, teacher_prediction)
            )
            result = hnewa_feature_mse(student_features, teacher_features)
        total = combine_hnewa_total_loss(native_total, result, batch_size=batch_size)
        self.stats_rows.append(
            {
                "kd_feature_mse_weighted": float(result.weighted.detach()),
                "kd_feature_mse_p3": float(result.per_level[0].detach()),
                "kd_feature_mse_p4": float(result.per_level[1].detach()),
                "kd_feature_mse_p5": float(result.per_level[2].detach()),
                "batch_size": float(batch_size),
            }
        )
        detached = dict(native_items)
        detached["kd_feature_mse"] = result.weighted.detach()
        return total, detached


def build_hnewa_trainer(
    *,
    mode: str,
    arm: str,
    overrides: dict[str, Any],
    strong_by_weak: dict[str, str] | None,
    shuffled_strong_by_weak: dict[str, str] | None,
    teacher_init: str | None,
    teacher_weights: Path | None,
    same_modal_weights: Path | None,
    base_weights: Path,
    max_steps: int | None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    needs_pairs = mode == "fusion-teacher" or arm in PAIRED_STUDENT_ARMS

    class HnewaTrainer(DetectionTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.hnewa_started_at = time.time()
            self.hnewa_optimizer_steps = 0
            super().__init__(*args, **kwargs)

        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            dataset = super().build_dataset(img_path, mode=mode, batch=batch)
            if mode != "train" or not needs_pairs:
                return dataset
            donors = shuffled_strong_by_weak if arm in SHUFFLED_STUDENT_ARMS else None
            return RGBTSharedGeometryDataset(dataset, strong_by_weak or {}, donors)

        def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
            batch = super().preprocess_batch(batch)
            for key in ("strong_img", "shuffled_strong_img"):
                if key in batch:
                    batch[key] = batch[key].to(self.device, non_blocking=True).float() / 255
            if mode == "fusion-teacher":
                if "strong_img" not in batch:
                    raise HnewaTrainerError("fusion-teacher batch is missing strong_img")
                batch["img"] = six_channel_fusion(batch["img"], batch["strong_img"])
            return batch

        def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True):
            base = super().get_model(cfg, weights, verbose)
            if mode != "fusion-teacher":
                return base
            fusion = build_six_channel_yolo11_fusion_teacher(base)
            for attribute in ("args", "names", "pt_path", "save_dir"):
                if hasattr(base, attribute):
                    setattr(fusion, attribute, getattr(base, attribute))
            return fusion

        def set_model_attributes(self) -> None:
            super().set_model_attributes()
            if mode != "student":
                return
            fusion = None
            same_modal = None
            if arm in PAIRED_STUDENT_ARMS:
                if teacher_init == "random":
                    from ultralytics import YOLO

                    fusion = build_six_channel_yolo11_fusion_teacher(
                        YOLO(str(base_weights), task="detect").model
                    ).to(self.device)
                else:
                    assert teacher_weights is not None
                    fusion = load_fusion_teacher(teacher_weights, base_weights).to(self.device)
                freeze_hnewa_teacher(fusion)
            else:
                from ultralytics import YOLO

                assert same_modal_weights is not None
                same_modal = YOLO(str(same_modal_weights), task="detect").model.to(self.device)
            self.model.criterion = HnewaMSELoss(
                self.model.init_criterion(),
                self.model,
                arm=arm,
                fusion_teacher=fusion,
                same_modal_teacher=same_modal,
            )

        def validate(self):
            # The 6ch fusion teacher cannot consume the 3ch val view; T0/H-arm
            # evaluation is owned by tools/eval_rgbt_detector.py on last.pt.
            if mode == "fusion-teacher":
                return {}, 0.0
            return super().validate()

        def final_eval(self):
            # The end-of-training validator warmup feeds a 3ch [1,3,640,640]
            # tensor, which the 6ch teacher cannot consume.
            if mode == "fusion-teacher":
                return {}, 0.0
            return super().final_eval()

        def optimizer_step(self) -> None:
            super().optimizer_step()
            self.hnewa_optimizer_steps += 1
            if max_steps is not None and self.hnewa_optimizer_steps >= max_steps:
                self.stop = True

    return HnewaTrainer(overrides=overrides)


def _overrides(config: dict[str, Any], args: argparse.Namespace, output: Path, base_weights: Path) -> dict[str, Any]:
    augmentation = config["augmentation"]
    return {
        "model": str(base_weights),
        "data": str(Path(config["paths"]["student_data_yaml"])),
        "imgsz": int(config["imgsz"]),
        "epochs": int(config["epochs"]),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "optimizer": str(config["optimizer"]),
        "lr0": float(config["lr0"]),
        "lrf": float(config["lrf"]),
        "momentum": float(config["momentum"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_epochs": float(config["warmup_epochs"]),
        "warmup_momentum": float(config["warmup_momentum"]),
        "warmup_bias_lr": float(config["warmup_bias_lr"]),
        "cos_lr": bool(config["cos_lr"]),
        "close_mosaic": int(config["close_mosaic"]),
        "nbs": int(config["nbs"]),
        "patience": int(config["patience"]),
        "amp": bool(config["amp"]),
        "deterministic": bool(config["deterministic"]),
        "seed": int(args.seed),
        "device": args.device,
        "val": False,
        "save": True,
        "save_period": -1,
        "pretrained": True,
        "plots": False,
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
        **{key: float(value) for key, value in augmentation.items()},
    }


def validate_inputs(config_path: Path, *, mode: str, arm: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise HnewaTrainerError("config must be a YAML mapping")
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise HnewaTrainerError("config is missing the paths block")
    required = ("student_data_yaml", "paired_train_mapping")
    if mode == "student" and arm in SHUFFLED_STUDENT_ARMS:
        required = required + ("shuffled_train_mapping",)
    for key in required:
        if not Path(paths[key]).is_file():
            raise HnewaTrainerError(f"paths.{key} does not exist: {paths[key]}")
    if mode == "student" and arm in PAIRED_STUDENT_ARMS:
        strong_yaml = Path(paths["privileged_data_yaml"])
        if not strong_yaml.is_file():
            raise HnewaTrainerError(f"paths.privileged_data_yaml does not exist: {strong_yaml}")
    import ultralytics

    if str(ultralytics.__version__) != str(config["ultralytics_version"]):
        raise HnewaTrainerError(
            f"ultralytics version drifted: {ultralytics.__version__} != {config['ultralytics_version']}"
        )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--arm", choices=STUDENT_ARMS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-weights", type=Path, help="T0 fusion teacher last.pt, or 'random' for engineering mini-E2E")
    parser.add_argument("--same-modal-teacher", type=Path, help="frozen weak-native seed42 last.pt for h3_same_modal")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "student" and args.arm is None:
        raise HnewaTrainerError("--arm is required in student mode")
    if args.mode == "fusion-teacher" and args.arm is not None:
        raise HnewaTrainerError("--arm is only valid in student mode")
    config = validate_inputs(args.config, mode=args.mode, arm=args.arm or "")
    if args.validate_only:
        print(json.dumps({"status": "validated", "mode": args.mode, "arm": args.arm}))
        return 0
    require_bound_lease_from_environment()

    paths = config["paths"]
    strong_by_weak = None
    shuffled_strong_by_weak = None
    needs_pairs = args.mode == "fusion-teacher" or (args.arm or "") in PAIRED_STUDENT_ARMS
    if needs_pairs:
        strong_by_weak = load_mapping(Path(paths["paired_train_mapping"]))
        if (args.arm or "") in SHUFFLED_STUDENT_ARMS:
            shuffled_strong_by_weak = load_mapping(Path(paths["shuffled_train_mapping"]))

    base_weights = Path(config["model"])
    teacher_init = None
    teacher_weights = None
    same_modal_weights = None
    if args.mode == "student":
        if args.arm in PAIRED_STUDENT_ARMS:
            if args.teacher_weights is None:
                raise HnewaTrainerError(f"{args.arm} requires --teacher-weights (T0 last.pt or 'random')")
            if str(args.teacher_weights) == "random":
                teacher_init = "random"
            else:
                teacher_init = "checkpoint"
                teacher_weights = Path(args.teacher_weights)
                if not teacher_weights.is_file():
                    raise HnewaTrainerError(f"teacher checkpoint does not exist: {teacher_weights}")
        elif args.arm == "h3_same_modal":
            if args.same_modal_teacher is None or not Path(args.same_modal_teacher).is_file():
                raise HnewaTrainerError("h3_same_modal requires --same-modal-teacher")
            same_modal_weights = Path(args.same_modal_teacher)

    args.output.mkdir(parents=True, exist_ok=True)
    overrides = _overrides(config, args, args.output, base_weights)
    trainer = build_hnewa_trainer(
        mode=args.mode,
        arm=args.arm or "",
        overrides=overrides,
        strong_by_weak=strong_by_weak,
        shuffled_strong_by_weak=shuffled_strong_by_weak,
        teacher_init=teacher_init,
        teacher_weights=teacher_weights,
        same_modal_weights=same_modal_weights,
        base_weights=base_weights,
        max_steps=args.max_steps,
    )
    started = time.time()
    trainer.train()
    walltime = time.time() - started

    criterion = getattr(trainer.model, "criterion", None)
    rows = getattr(criterion, "stats_rows", []) if args.mode == "student" else []
    mean_rows = {
        key: float(sum(row.get(key, 0.0) for row in rows) / len(rows)) for key in sorted({k for r in rows for k in r})
    } if rows else {}
    import ultralytics

    receipt = {
        "schema": "rgbt-hnewa-mse-completion-receipt-v1",
        "status": "completed",
        "mode": args.mode,
        "arm": args.arm,
        "method_track": "Hnewa-CMKD-MSE-inspired-YOLO11n",
        "method_source": "inspired port; not an exact reproduction",
        "dataset": config["dataset"],
        "student_modality": config["student_modality"],
        "privileged_modality": config["privileged_modality"],
        "evaluation_role": config["evaluation_role"],
        "seed": int(args.seed),
        "epochs_configured": int(config["epochs"]),
        "optimizer_steps": int(getattr(trainer, "hnewa_optimizer_steps", 0)),
        "max_steps": args.max_steps,
        "taps": list(HNEWA_YOLO11_TAP_LAYERS),
        "alpha": HNEWA_FEATURE_MSE_ALPHA,
        "batch": int(args.batch),
        "nbs": int(config["nbs"]),
        "teacher_weights": str(args.teacher_weights) if args.teacher_weights else None,
        "same_modal_teacher": str(args.same_modal_teacher) if args.same_modal_teacher else None,
        "official_test_accessed": False,
        "torch_version": str(torch.__version__),
        "ultralytics_version": str(ultralytics.__version__),
        "walltime_seconds": walltime,
        **mean_rows,
    }
    (args.output / "completion_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        (args.output / "method_stats.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    criterion = getattr(trainer.model, "criterion", None)
    split_rosters = [Path(paths["paired_train_mapping"])]
    if (args.arm or "") in SHUFFLED_STUDENT_ARMS:
        split_rosters.append(Path(paths["shuffled_train_mapping"]))
    emit_bound_run_receipt(
        run_dir=args.output.resolve() / "run_evidence",
        method_identity="HNEWA-INSPIRED",
        dataset=str(config["dataset"]),
        data_role="development_train",
        seed=args.seed,
        run_kind="train",
        trainers=[Path(__file__)],
        losses=implementation_files(criterion, getattr(criterion, "native", None)),
        configs=[args.config],
        split_rosters=split_rosters,
        metric_files=[args.output / "completion_receipt.json"] + ([args.output / "method_stats.json"] if rows else []),
        environment={"torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__)},
        inputs={
            "mode": args.mode,
            "arm": args.arm,
            "initial_weights": str(base_weights.resolve()),
            "teacher_weights": str(teacher_weights.resolve()) if teacher_weights else teacher_init,
            "same_modal_teacher": str(same_modal_weights.resolve()) if same_modal_weights else None,
        },
    )
    print(json.dumps({"status": "completed", "output": str(args.output), "walltime_seconds": walltime}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
