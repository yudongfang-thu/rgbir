#!/usr/bin/env python3
"""Train the frozen RGBT-P3-CAUSAL-v1 detector arms from ordinary YOLO11n weights."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo_osssl.fp_selective_distillation import FPSelectiveLoss
from yolo_osssl.paired_detection import PairedDetectionDataset
from tools.project_resource_guard import require_bound_lease_from_environment
from tools.write_jstars_run_receipt import emit_bound_run_receipt, implementation_files


ARMS = ("native", "p3", "p3_same_modal", "p3_shuffled", "p3_random_dose")
CROSS_MODAL_ARMS = frozenset({"p3", "p3_shuffled", "p3_random_dose"})
SHUFFLED_ARMS = frozenset({"p3_shuffled"})
SAME_MODAL_ARMS = frozenset({"p3_same_modal"})
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


@dataclass(frozen=True)
class ExperimentInputs:
    config: dict[str, Any]
    config_path: Path
    weights: Path
    student_data: Path
    privileged_data: Path
    paired_mapping: dict[str, str]
    shuffled_mapping: dict[str, str]


def _require_frozen_batch(expected: int, actual: int, configured: int) -> None:
    if actual != expected or configured != expected:
        raise RuntimeError(
            f"Ultralytics changed frozen batch {expected} to batch_size={actual}, args.batch={configured}"
        )


def _validate_run_args(*, batch: int, max_steps: int | None) -> None:
    if batch != 32:
        raise RuntimeError("RGBT-P3-CAUSAL-v1 freezes physical batch 32 with nbs=64")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max-steps must be positive")


def _load_mapping(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in payload.items()):
        raise RuntimeError(f"mapping must be a JSON string-to-string object: {path}")
    return {str(Path(key).resolve()): str(Path(value).resolve()) for key, value in payload.items()}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML must be a mapping: {path}")
    return payload


def _data_nc(data_path: Path) -> int:
    payload = _load_yaml_mapping(data_path)
    if "test" in payload:
        raise RuntimeError(f"RGB-T production data YAML must omit test: {data_path}")
    names = payload.get("names")
    if isinstance(names, dict):
        inferred = len(names)
    elif isinstance(names, list):
        inferred = len(names)
    else:
        raise RuntimeError(f"data YAML names must define class count: {data_path}")
    declared = payload.get("nc")
    if declared is not None and int(declared) != inferred:
        raise RuntimeError(f"data YAML nc/names disagree: {data_path}")
    if not isinstance(payload.get("train"), str) or not isinstance(payload.get("val"), str):
        raise RuntimeError(f"data YAML must define train and val strings: {data_path}")
    return inferred


def _train_images(data_path: Path) -> dict[str, Path]:
    payload = _load_yaml_mapping(data_path)
    root = Path(payload.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    image_dir = (root / str(payload["train"])).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"training image directory is missing: {image_dir}")
    files = sorted(path.resolve() for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    result = {path.stem: path for path in files}
    if len(result) != len(files) or not result:
        raise RuntimeError(f"training images must be non-empty and have unique stems: {image_dir}")
    return result


def _validate_mapping(
    student_data: Path,
    privileged_data: Path,
    paired: dict[str, str],
    shuffled: dict[str, str],
) -> None:
    student_images = _train_images(student_data)
    privileged_images = _train_images(privileged_data)
    student_paths = {str(path) for path in student_images.values()}
    privileged_paths = {str(path) for path in privileged_images.values()}
    if set(paired) != student_paths or set(shuffled) != student_paths:
        raise RuntimeError("paired and shuffled mapping keys must cover exactly the student train images")
    for source, target in paired.items():
        if target not in privileged_paths or Path(source).stem != Path(target).stem:
            raise RuntimeError(f"paired mapping must use the exact privileged same-stem donor: {source}")
    for source, donor in shuffled.items():
        if donor not in privileged_paths or Path(source).stem == Path(donor).stem:
            raise RuntimeError(f"shuffled mapping must use a privileged different-stem donor: {source}")


def _validate_frozen_config(cfg: dict[str, Any]) -> None:
    expected = {
        "method_id": "RGBT-P3-CAUSAL-v1",
        "ultralytics_version": "8.4.115",
        "torch_version": "2.10.0+cu128",
        "imgsz": 640,
        "epochs": 200,
        "batch": 32,
        "nbs": 64,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "cos_lr": False,
        "close_mosaic": 0,
        "patience": 0,
        "amp": True,
        "deterministic": True,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise RuntimeError(f"frozen config requires {key}={value!r}, got {cfg.get(key)!r}")
    augmentation = cfg.get("augmentation")
    expected_augmentation = {
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "degrees": 0.0,
        "perspective": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "erasing": 0.0,
    }
    if not isinstance(augmentation, dict) or any(augmentation.get(key) != value for key, value in expected_augmentation.items()):
        raise RuntimeError("frozen RGB-T augmentation contract drifted")
    if cfg.get("evaluation_role") not in {"dev", "val"}:
        raise RuntimeError("evaluation_role must be dev or val; test is sealed")
    seeds = cfg.get("allowed_seeds")
    if seeds != [0, 42, 123]:
        raise RuntimeError("allowed_seeds must be [0, 42, 123]")


def validate_experiment(config_path: Path, *, arm: str) -> ExperimentInputs:
    if arm not in ARMS:
        raise ValueError(f"unknown RGBT arm {arm!r}")
    if not config_path.is_file():
        raise FileNotFoundError(f"config is missing: {config_path}")
    cfg = _load_yaml_mapping(config_path)
    _validate_frozen_config(cfg)
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeError("config paths must be a mapping")
    required_paths = ("student_data_yaml", "privileged_data_yaml", "paired_train_mapping", "shuffled_train_mapping", "preparation_receipt")
    if any(not isinstance(paths.get(key), str) for key in required_paths):
        raise RuntimeError("config paths are incomplete")
    weights = Path(cfg.get("model", ""))
    student_data = Path(paths["student_data_yaml"])
    privileged_data = Path(paths["privileged_data_yaml"])
    paired_path = Path(paths["paired_train_mapping"])
    shuffled_path = Path(paths["shuffled_train_mapping"])
    receipt_path = Path(paths["preparation_receipt"])
    required = (weights, student_data, privileged_data, paired_path, shuffled_path, receipt_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model, prepared data, mapping, or receipt is missing: {missing[0]}")
    expected_nc = int(cfg.get("expected_nc", -1))
    if _data_nc(student_data) != expected_nc or _data_nc(privileged_data) != expected_nc:
        raise RuntimeError("student/privileged data nc disagrees with the frozen config")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    direction_key = f"{cfg.get('student_modality')}_to_{cfg.get('privileged_modality')}"
    if receipt.get("schema") != "rgbt-experiment-prepare-v1" or receipt.get("directions", {}).get(direction_key) is None:
        raise RuntimeError("prepared-data receipt does not bind this RGBT direction")
    direction = receipt["directions"][direction_key]
    if not direction.get("student_native_gt_only") or direction.get("teacher_labels_used_by_kd"):
        raise RuntimeError("receipt does not preserve student-native GT-only KD semantics")
    paired = _load_mapping(paired_path)
    shuffled = _load_mapping(shuffled_path)
    _validate_mapping(student_data, privileged_data, paired, shuffled)
    return ExperimentInputs(cfg, config_path.resolve(), weights.resolve(), student_data.resolve(), privileged_data.resolve(), paired, shuffled)


def _freeze_teacher(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _ordinary_models(
    inputs: ExperimentInputs,
    *,
    arm: str,
    privileged_teacher: Path | None,
    student_teacher: Path | None,
    yolo_factory: Callable[..., Any] | None = None,
) -> dict[str, int | None]:
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    loaded = {"student": yolo_factory(str(inputs.weights), task="detect")}
    if arm in CROSS_MODAL_ARMS:
        if privileged_teacher is None or not privileged_teacher.is_file():
            raise FileNotFoundError(f"{arm} requires --privileged-teacher")
        loaded["privileged_teacher"] = yolo_factory(str(privileged_teacher), task="detect")
    if arm in SAME_MODAL_ARMS:
        if student_teacher is None or not student_teacher.is_file():
            raise FileNotFoundError(f"{arm} requires --student-teacher")
        loaded["student_teacher"] = yolo_factory(str(student_teacher), task="detect")

    def model_nc(yolo: Any) -> int | None:
        model = getattr(yolo, "model", yolo)
        value = getattr(model, "nc", None)
        return int(value) if value is not None else None

    return {name: model_nc(model) for name, model in loaded.items()}


def _mean_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(sum(float(row.get(key, 0.0)) for row in rows) / len(rows)) for key in keys} if rows else {}


def _package_versions() -> dict[str, str]:
    import ultralytics

    return {"torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__)}


def _validate_runtime_versions(cfg: dict[str, Any]) -> None:
    import ultralytics

    actual = _package_versions()
    if actual["torch"] != cfg["torch_version"] or actual["ultralytics"] != cfg["ultralytics_version"]:
        raise RuntimeError(
            "production package versions drifted: "
            f"torch={actual['torch']} ultralytics={actual['ultralytics']}"
        )


def build_trainer(
    *,
    inputs: ExperimentInputs,
    arm: str,
    overrides: dict[str, Any],
    privileged_teacher: Path | None,
    student_teacher: Path | None,
    max_steps: int | None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    expected_batch = int(overrides["batch"])

    class RGBTTrainer(DetectionTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.rgbt_started_at = time.time()
            super().__init__(*args, **kwargs)

        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            dataset = super().build_dataset(img_path, mode=mode, batch=batch)
            if mode != "train" or arm not in CROSS_MODAL_ARMS:
                return dataset
            donors = inputs.shuffled_mapping if arm in SHUFFLED_ARMS else None
            # The wrapper transforms the privileged image with student geometry only.
            # Its copied record intentionally keeps the student's labels; teacher labels never enter KD.
            return PairedDetectionDataset(dataset, inputs.paired_mapping, donors)

        def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
            batch = super().preprocess_batch(batch)
            for key in ("eo_img", "shuffled_eo_img"):
                if key in batch:
                    batch[key] = batch[key].to(self.device, non_blocking=True).float() / 255
            return batch

        def set_model_attributes(self) -> None:
            super().set_model_attributes()
            from ultralytics import YOLO

            privileged = None
            same_modal = None
            if arm in CROSS_MODAL_ARMS:
                assert privileged_teacher is not None
                privileged = YOLO(str(privileged_teacher), task="detect").model.to(self.device)
                _freeze_teacher(privileged)
            if arm in SAME_MODAL_ARMS:
                assert student_teacher is not None
                same_modal = YOLO(str(student_teacher), task="detect").model.to(self.device)
                _freeze_teacher(same_modal)
            self.model.criterion = FPSelectiveLoss(
                self.model.init_criterion(),
                arm=arm,
                optical_teacher=privileged,
                sar_teacher=same_modal,
            )

        def optimizer_step(self) -> None:
            super().optimizer_step()
            self.rgbt_optimizer_steps = int(getattr(self, "rgbt_optimizer_steps", 0)) + 1
            if max_steps is not None and self.rgbt_optimizer_steps >= max_steps:
                self.stop = True

        def require_frozen_batch(self) -> None:
            _require_frozen_batch(expected_batch, int(self.batch_size), int(self.args.batch))

        def save_rgbt_artifacts(self) -> None:
            criterion = getattr(self.model, "criterion", None)
            rows = getattr(criterion, "stats_rows", [])
            save_dir = Path(self.save_dir)
            method_stats = {
                "schema": "rgbt-p3-method-stats-v1",
                "arm": arm,
                "batch": int(self.batch_size),
                "nbs": int(self.args.nbs),
                "final_accumulate": int(getattr(self, "accumulate", 1)),
                "optimizer_steps": int(getattr(self, "rgbt_optimizer_steps", 0)),
                "training_batches": len(rows),
                **_mean_stats(rows),
            }
            completion = {
                "schema": "rgbt-p3-completion-receipt-v1",
                "receipt_id": f"rgbt-p3-{self.args.seed}-{arm}",
                "status": "completed",
                "method_id": inputs.config["method_id"],
                "dataset": inputs.config["dataset"],
                "student_modality": inputs.config["student_modality"],
                "privileged_modality": inputs.config["privileged_modality"],
                "ordinary_yolo_pretrained_initialization": True,
                "ssl_checkpoint_injected": False,
                "student_native_gt_assignment_only": True,
                "teacher_modality_labels_used_by_kd": False,
                "seed": int(self.args.seed),
                "epochs_configured": int(self.args.epochs),
                "max_steps": max_steps,
                "optimizer_steps": int(getattr(self, "rgbt_optimizer_steps", 0)),
                "last_checkpoint": str(save_dir / "weights" / "last.pt"),
                "last_checkpoint_exists_at_receipt": (save_dir / "weights" / "last.pt").is_file(),
                "walltime_seconds": float(time.time() - self.rgbt_started_at),
                "package_versions": _package_versions(),
                "method_stats": "method_stats.json",
            }
            (save_dir / "method_stats.json").write_text(json.dumps(method_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (save_dir / "completion_receipt.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    trainer = RGBTTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_start", lambda current: current.require_frozen_batch())
    trainer.add_callback("on_train_end", lambda current: current.save_rgbt_artifacts())
    return trainer


def _overrides(inputs: ExperimentInputs, args: argparse.Namespace, output: Path) -> dict[str, Any]:
    cfg = inputs.config
    augmentation = cfg["augmentation"]
    return {
        "model": str(inputs.weights),
        "data": str(inputs.student_data),
        "imgsz": int(cfg["imgsz"]),
        "epochs": int(cfg["epochs"]),
        "batch": int(args.batch),
        "workers": int(cfg["workers"]),
        "optimizer": str(cfg["optimizer"]),
        "lr0": float(cfg["lr0"]),
        "lrf": float(cfg["lrf"]),
        "momentum": float(cfg["momentum"]),
        "weight_decay": float(cfg["weight_decay"]),
        "warmup_epochs": float(cfg["warmup_epochs"]),
        "warmup_momentum": float(cfg["warmup_momentum"]),
        "warmup_bias_lr": float(cfg["warmup_bias_lr"]),
        "cos_lr": bool(cfg["cos_lr"]),
        "close_mosaic": int(cfg["close_mosaic"]),
        "nbs": int(cfg["nbs"]),
        "patience": int(cfg["patience"]),
        "amp": bool(cfg["amp"]),
        "deterministic": bool(cfg["deterministic"]),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--privileged-teacher", type=Path)
    parser.add_argument("--student-teacher", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_run_args(batch=args.batch, max_steps=args.max_steps)
    if not args.validate_only:
        require_bound_lease_from_environment()
    inputs = validate_experiment(args.config, arm=args.arm)
    if args.seed not in inputs.config["allowed_seeds"]:
        raise RuntimeError(f"seed must be one of {inputs.config['allowed_seeds']}")
    _validate_runtime_versions(inputs.config)
    loaded_ncs = _ordinary_models(
        inputs,
        arm=args.arm,
        privileged_teacher=args.privileged_teacher,
        student_teacher=args.student_teacher,
    )
    if args.validate_only:
        print(json.dumps({"status": "validated", "arm": args.arm, "ordinary_yolo_pretrained_initialization": True, "loaded_model_nc": loaded_ncs}, sort_keys=True))
        return 0
    output = args.output.resolve()
    trainer = build_trainer(
        inputs=inputs,
        arm=args.arm,
        overrides=_overrides(inputs, args, output),
        privileged_teacher=args.privileged_teacher.resolve() if args.privileged_teacher else None,
        student_teacher=args.student_teacher.resolve() if args.student_teacher else None,
        max_steps=args.max_steps,
    )
    trainer.train()
    save_dir = Path(trainer.save_dir).resolve()
    criterion = getattr(trainer.model, "criterion", None)
    paths = inputs.config["paths"]
    split_rosters = [Path(paths["paired_train_mapping"])]
    if args.arm in SHUFFLED_ARMS:
        split_rosters.append(Path(paths["shuffled_train_mapping"]))
    emit_bound_run_receipt(
        run_dir=save_dir / "run_evidence",
        method_identity="NATIVE" if args.arm == "native" else "HNEWA-INSPIRED",
        dataset=str(inputs.config["dataset"]),
        data_role="development_train",
        seed=args.seed,
        run_kind="train",
        trainers=[Path(__file__)],
        losses=implementation_files(criterion, getattr(criterion, "native", None)),
        configs=[inputs.config_path],
        split_rosters=split_rosters,
        metric_files=[save_dir / "method_stats.json", save_dir / "completion_receipt.json"],
        environment=_package_versions(),
        inputs={
            "initial_weights": str(inputs.weights),
            "privileged_teacher": str(args.privileged_teacher.resolve()) if args.privileged_teacher else None,
            "student_teacher": str(args.student_teacher.resolve()) if args.student_teacher else None,
            "arm": args.arm,
        },
    )
    print(json.dumps({"status": "completed", "arm": args.arm, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
