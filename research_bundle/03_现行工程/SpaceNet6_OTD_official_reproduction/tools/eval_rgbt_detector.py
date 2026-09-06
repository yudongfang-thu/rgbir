#!/usr/bin/env python3
"""Evaluate one completed RGBT detector on its frozen dev/val role and emit a gate record."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_yolo_detector import evaluate
from tools.project_resource_guard import require_bound_lease_from_environment
from tools.write_jstars_run_receipt import emit_bound_run_receipt


METRICS = ("AP50", "AP75", "mAP50_95")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class RGBTMetricAdapterError(RuntimeError):
    pass


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RGBTMetricAdapterError("config must be a YAML mapping")
    return payload


def _student_data_path(config: dict[str, Any]) -> Path:
    paths = config.get("paths")
    if not isinstance(paths, dict) or not isinstance(paths.get("student_data_yaml"), str):
        raise RGBTMetricAdapterError("config lacks paths.student_data_yaml")
    data = Path(paths["student_data_yaml"])
    if not data.is_file():
        raise FileNotFoundError(f"student data YAML is missing: {data}")
    payload = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "test" in payload:
        raise RGBTMetricAdapterError("RGB-T evaluator requires a train/val-only data YAML")
    if not isinstance(payload.get("val"), str):
        raise RGBTMetricAdapterError("RGB-T evaluator requires config data val")
    return data


def _evaluation_roster(data: Path, output: Path) -> Path:
    payload = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = Path(payload.get("path", data.parent))
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    val = Path(str(payload["val"]))
    val = val.resolve() if val.is_absolute() else (root / val).resolve()
    if val.is_file():
        return val
    if not val.is_dir():
        raise RGBTMetricAdapterError(f"evaluation split path is missing: {val}")
    images = sorted(
        path.resolve()
        for path in val.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RGBTMetricAdapterError(f"evaluation split contains no images: {val}")
    roster = output.with_name(f"{output.stem}_split_roster.txt")
    with roster.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(str(path) for path in images) + "\n")
    return roster


def _method_identity(arm: str) -> str:
    if arm == "native":
        return "NATIVE"
    if arm.startswith("cmdistill"):
        return "ADAPTED"
    if arm.startswith("cclkd"):
        return "ADAPTED-PARTIAL"
    return "HNEWA-INSPIRED"


def evaluate_record(
    *,
    config_path: Path,
    checkpoint: Path,
    arm: str,
    seed: int,
    output: Path,
    imgsz: int = 640,
    batch: int = 32,
    workers: int = 8,
    device: str = "0",
    evaluator: Callable[..., dict[str, float]] = evaluate,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite metric record: {output}")
    if not checkpoint.is_file() or checkpoint.name != "last.pt":
        raise FileNotFoundError("RGB-T evaluation requires an existing last.pt checkpoint")
    config = _load_config(config_path)
    role = str(config.get("evaluation_role", ""))
    if role not in {"dev", "val"}:
        raise RGBTMetricAdapterError("evaluation_role must be dev or val; test is sealed")
    data = _student_data_path(config)
    started_at = time.monotonic()
    metrics = evaluator(
        checkpoint,
        data,
        output,
        split="val",
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
    )
    required = {name: float(metrics[name]) for name in METRICS}
    record = {
        "schema": "rgbt-detector-metrics-record-v1",
        "dataset": str(config["dataset"]),
        "modality": str(config["student_modality"]),
        "arm": arm,
        "seed": int(seed),
        "evaluation_role": role,
        "split": "val",
        "data_yaml": str(data.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "walltime_seconds": float(time.monotonic() - started_at),
        "metrics": required,
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    require_bound_lease_from_environment()
    result = evaluate_record(
        config_path=args.config,
        checkpoint=args.checkpoint,
        arm=args.arm,
        seed=args.seed,
        output=args.output,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
    )
    config = _load_config(args.config)
    data = _student_data_path(config)
    roster = _evaluation_roster(data, args.output)
    emit_bound_run_receipt(
        run_dir=args.output.resolve().parent / "eval_run_evidence",
        method_identity=_method_identity(args.arm),
        dataset=str(config["dataset"]),
        data_role=str(config["evaluation_role"]),
        seed=args.seed,
        run_kind="eval",
        trainers=[Path(__file__)],
        losses=[],
        configs=[args.config],
        split_rosters=[roster],
        metric_files=[args.output],
        inputs={"checkpoint": str(args.checkpoint.resolve()), "arm": args.arm},
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
