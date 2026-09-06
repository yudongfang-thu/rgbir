#!/usr/bin/env python3
"""Apply the seed42 RGBT-P3-CAUSAL-v1 causal gate on dev/val metrics only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


ARMS = ("native", "p3", "p3_same_modal", "p3_shuffled", "p3_random_dose")
ANCHORS = ("native", "p3_same_modal", "p3_shuffled", "p3_random_dose")
METRICS = ("AP50", "AP75", "mAP50_95")


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


def decide_core_gate(
    rows: list[dict[str, Any]], *, dataset: str, evaluation_role: str, student_modality: str
) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("split", "")).lower() == "test" or str(row.get("evaluation_role", "")).lower() == "test":
            raise RuntimeError("RGB-T core gate never accesses the sealed test split")
        if str(row.get("dataset", "")).lower() != dataset or str(row.get("evaluation_role", "")).lower() != evaluation_role:
            raise RuntimeError("metrics scope disagrees with the gate config")
        if str(row.get("modality")) != student_modality or int(row.get("seed")) != 42:
            continue
        arm = str(row.get("arm"))
        if arm not in ARMS:
            continue
        if arm in selected:
            raise RuntimeError(f"duplicate seed42 metric for arm {arm}")
        for metric in METRICS:
            _metric(row, metric)
        selected[arm] = row
    missing = [arm for arm in ARMS if arm not in selected]
    if missing:
        raise RuntimeError(f"seed42 core gate is missing arms: {', '.join(missing)}")
    comparisons = {
        anchor: {metric: 100.0 * (_metric(selected["p3"], metric) - _metric(selected[anchor], metric)) for metric in METRICS}
        for anchor in ANCHORS
    }
    passed = all(value > 0.0 for values in comparisons.values() for value in values.values())
    return {
        "status": "PASS_EXPAND_SEEDS_0_123" if passed else "KILL_RGBT-P3-CAUSAL-v1",
        "dataset": dataset,
        "evaluation_role": evaluation_role,
        "student_modality": student_modality,
        "seed": 42,
        "paired_p3_strictly_exceeds_all_anchors": passed,
        "paired_p3_minus_anchor_pp": comparisons,
        "next_action": "run seeds 0 and 123 for all five arms" if passed else "no tuning rescue; close this dataset v1",
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
    result = decide_core_gate(
        _records(args.metrics),
        dataset=str(cfg["dataset"]),
        evaluation_role=str(cfg["evaluation_role"]),
        student_modality=str(cfg["student_modality"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
