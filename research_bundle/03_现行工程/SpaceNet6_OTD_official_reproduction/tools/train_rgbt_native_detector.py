#!/usr/bin/env python3
"""Train a plain YOLO detector for an RGB-T student or privileged-modality anchor."""

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

from tools.project_resource_guard import require_bound_lease_from_environment  # noqa: E402
from tools.write_jstars_run_receipt import emit_bound_run_receipt, implementation_files  # noqa: E402


class NativeTrainerError(RuntimeError):
    pass


def _data_path(config: dict, modality: str) -> Path:
    key = "student_data_yaml" if modality == "student" else "privileged_data_yaml"
    value = config.get("paths", {}).get(key)
    if not isinstance(value, str) or not Path(value).is_file():
        raise NativeTrainerError(f"missing {modality} data YAML: {value}")
    return Path(value)


def build_trainer(*, overrides: dict, max_steps: int | None):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class NativeTrainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            self.native_steps = 0
            super().__init__(*args, **kwargs)

        def optimizer_step(self):
            super().optimizer_step()
            self.native_steps += 1
            if max_steps is not None and self.native_steps >= max_steps:
                self.stop = True

    return NativeTrainer(overrides=overrides)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=("student", "privileged"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise NativeTrainerError("config must be a YAML mapping")
    data_path = _data_path(config, args.modality)
    model_path = Path(str(config.get("model", "")))
    if not model_path.is_file():
        raise NativeTrainerError(f"initial model is missing: {model_path}")
    split_roster = Path(str(config.get("paths", {}).get("paired_train_mapping", "")))
    if not split_roster.is_file():
        raise NativeTrainerError(f"split roster is missing: {split_roster}")
    if args.max_steps is not None and args.max_steps <= 0:
        raise NativeTrainerError("max-steps must be positive")
    if args.validate_only:
        print(json.dumps({"status": "validated", "modality": args.modality}))
        return 0

    require_bound_lease_from_environment()
    batch = int(args.batch if args.batch is not None else config["batch"])
    workers = int(args.workers if args.workers is not None else config["workers"])
    augmentation = {key: float(value) for key, value in config["augmentation"].items()}
    overrides = {
        "model": str(model_path),
        "data": str(data_path),
        "imgsz": int(config["imgsz"]),
        "epochs": int(config["epochs"]),
        "batch": batch,
        "workers": workers,
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
        "val": True,
        "save": True,
        "save_period": -1,
        "pretrained": True,
        "plots": False,
        "project": str(args.output.parent),
        "name": args.output.name,
        "exist_ok": True,
        **augmentation,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(overrides=overrides, max_steps=args.max_steps)
    started = time.time()
    trainer.train()
    completion = {
        "schema": "rgbt-native-completion-v1",
        "status": "completed",
        "dataset": str(config["dataset"]),
        "modality": args.modality,
        "seed": int(args.seed),
        "batch": batch,
        "epochs_configured": int(config["epochs"]),
        "max_steps": args.max_steps,
        "optimizer_steps": int(getattr(trainer, "native_steps", 0)),
        "walltime_seconds": time.time() - started,
        "official_test_accessed": False,
    }
    completion_path = args.output / "completion_receipt.json"
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import ultralytics

    criterion = getattr(trainer.model, "criterion", None)
    emit_bound_run_receipt(
        run_dir=args.output.resolve() / "run_evidence",
        method_identity="NATIVE",
        dataset=str(config["dataset"]),
        data_role="development_train",
        seed=args.seed,
        run_kind="train",
        trainers=[Path(__file__)],
        losses=implementation_files(criterion),
        configs=[args.config],
        split_rosters=[split_roster],
        metric_files=[completion_path],
        environment={"torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__)},
        inputs={"modality": args.modality, "initial_weights": str(model_path.resolve())},
    )
    print(json.dumps({"status": "completed", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
