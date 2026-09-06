#!/usr/bin/env python3
"""Evaluate a YOLO detector checkpoint and save detection metrics as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def evaluate(
    checkpoint: Path,
    data: Path,
    output: Path,
    *,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 64,
    workers: int = 8,
    device: str = "0",
    yolo_factory: Callable[..., Any] | None = None,
) -> dict[str, float]:
    if not checkpoint.is_file() or not data.is_file():
        raise FileNotFoundError("checkpoint or dataset YAML is missing")
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO
    metrics = yolo_factory(str(checkpoint), task="detect").val(
        data=str(data), split=split, imgsz=imgsz, batch=batch, workers=workers,
        device=device, plots=False, save_json=False, verbose=False,
    )
    result = {
        "AP50": float(metrics.box.map50),
        "AP75": float(metrics.box.map75),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint, args.data, args.output, split=args.split, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
