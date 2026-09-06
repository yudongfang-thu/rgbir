#!/usr/bin/env python3
"""Apply the frozen RGB-T native-anchor direction gate on dev/val metrics only."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml


METRICS = ("AP50", "AP75", "mAP50_95")
SEEDS = (0, 42, 123)


def _records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("metrics input must be a JSON record list or an object with records")
    return rows


def _metric(row: dict[str, Any], name: str) -> float:
    values = row.get("metrics", row)
    value = float(values[name])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {name}")
    return value


def _validate_scope(rows: list[dict[str, Any]], *, dataset: str, evaluation_role: str) -> None:
    for row in rows:
        split = str(row.get("split", "")).lower()
        role = str(row.get("evaluation_role", "")).lower()
        if split == "test" or role == "test":
            raise RuntimeError("RGB-T gates never access the sealed test split")
        if str(row.get("dataset", "")).lower() != dataset:
            raise RuntimeError("metrics dataset disagrees with the gate config")
        if role != evaluation_role:
            raise RuntimeError(f"metrics must be explicitly tagged evaluation_role={evaluation_role}")
        for name in METRICS:
            _metric(row, name)


def _native(rows: list[dict[str, Any]], modality: str, seed: int) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if row.get("arm") == "native" and str(row.get("modality")) == modality and int(row.get("seed")) == seed
    ]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate native metric for {modality} seed{seed}")
    return matches[0] if matches else None


def _direction_if_clear(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, str] | None:
    differences = {name: _metric(left, name) - _metric(right, name) for name in METRICS}
    if differences["mAP50_95"] >= 0.01 and differences["AP50"] > 0.0 and differences["AP75"] > 0.0:
        return str(left["modality"]), str(right["modality"])
    if differences["mAP50_95"] <= -0.01 and differences["AP50"] < 0.0 and differences["AP75"] < 0.0:
        return str(right["modality"]), str(left["modality"])
    return None


def decide_anchor_direction(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    evaluation_role: str,
    modality_a: str,
    modality_b: str,
) -> dict[str, Any]:
    _validate_scope(rows, dataset=dataset, evaluation_role=evaluation_role)
    seed42_a = _native(rows, modality_a, 42)
    seed42_b = _native(rows, modality_b, 42)
    if seed42_a is None or seed42_b is None:
        raise RuntimeError("seed42 native metrics for both modalities are required")
    clear = _direction_if_clear(seed42_a, seed42_b)
    if clear is not None:
        privileged, student = clear
        return {
            "status": "PASS_SEED42_DIRECTION",
            "dataset": dataset,
            "evaluation_role": evaluation_role,
            "privileged_modality": privileged,
            "student_modality": student,
            "basis": "seed42 mAP advantage >= 1.0 pp with AP50/AP75 same sign",
            "used_seeds": [42],
        }
    by_seed = {seed: (_native(rows, modality_a, seed), _native(rows, modality_b, seed)) for seed in SEEDS}
    missing = [seed for seed, pair in by_seed.items() if pair[0] is None or pair[1] is None]
    if missing:
        return {
            "status": "NEED_NATIVE_SEEDS_0_123",
            "dataset": dataset,
            "evaluation_role": evaluation_role,
            "missing_seeds": missing,
            "reason": "seed42 direction is ambiguous under the frozen 1.0 pp rule",
        }
    differences = {
        name: [_metric(by_seed[seed][0], name) - _metric(by_seed[seed][1], name) for seed in SEEDS]
        for name in METRICS
    }
    mean_map = statistics.mean(differences["mAP50_95"])
    sign = 1.0 if mean_map > 0.0 else -1.0 if mean_map < 0.0 else 0.0
    positive_count = sum(sign * value > 0.0 for value in differences["mAP50_95"]) if sign else 0
    ap50_consistent = sign * statistics.mean(differences["AP50"]) >= 0.0 if sign else False
    ap75_consistent = sign * statistics.mean(differences["AP75"]) >= 0.0 if sign else False
    if abs(mean_map) >= 0.005 and positive_count >= 2 and ap50_consistent and ap75_consistent:
        privileged, student = (modality_a, modality_b) if sign > 0 else (modality_b, modality_a)
        status = "PASS_THREE_SEED_DIRECTION"
    else:
        privileged = None
        student = None
        status = "STOP_NO_STABLE_WEAK_MODALITY"
    return {
        "status": status,
        "dataset": dataset,
        "evaluation_role": evaluation_role,
        "privileged_modality": privileged,
        "student_modality": student,
        "used_seeds": list(SEEDS),
        "mean_advantage_pp": {name: 100.0 * statistics.mean(values) for name, values in differences.items()},
        "positive_mAP_seeds": positive_count,
        "ap50_mean_does_not_reverse": ap50_consistent,
        "ap75_mean_does_not_reverse": ap75_consistent,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("config must be a YAML mapping")
    result = decide_anchor_direction(
        _records(args.metrics),
        dataset=str(cfg["dataset"]),
        evaluation_role=str(cfg["evaluation_role"]),
        modality_a=str(cfg["student_modality"]),
        modality_b=str(cfg["privileged_modality"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
