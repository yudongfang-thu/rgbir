#!/usr/bin/env python3
"""Apply the frozen three-seed final gate for RGBT-P3-CAUSAL-v1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


ARMS = ("native", "p3", "p3_same_modal", "p3_shuffled", "p3_random_dose")
CONTROLS = tuple(arm for arm in ARMS if arm != "p3")
SEEDS = (0, 42, 123)
METRICS = ("AP50", "AP75", "mAP50_95")


def _records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("metrics input must be a JSON record list or an object with records")
    return rows


def _metric(row: dict[str, Any], name: str) -> float:
    value = float(row.get("metrics", row)[name])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {name}")
    return value


def decide_multiseed_final_gate(
    rows: list[dict[str, Any]], *, dataset: str, evaluation_role: str, student_modality: str
) -> dict[str, Any]:
    """Judge the fixed final gate; incomplete metrics are a non-passing failure."""
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    unexpected: list[str] = []
    duplicates: list[str] = []
    for row in rows:
        if str(row.get("split", "")).lower() == "test" or str(row.get("evaluation_role", "")).lower() == "test":
            raise RuntimeError("RGB-T multiseed gate never accesses the sealed test split")
        if str(row.get("dataset", "")).lower() != dataset or str(row.get("evaluation_role", "")).lower() != evaluation_role:
            raise RuntimeError("metrics scope disagrees with the gate config")
        if str(row.get("modality")) != student_modality:
            continue
        arm = str(row.get("arm"))
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            unexpected.append(f"{arm}:invalid-seed")
            continue
        if arm not in ARMS or seed not in SEEDS:
            unexpected.append(f"{arm}:seed{seed}")
            continue
        key = (arm, seed)
        if key in selected:
            duplicates.append(f"{arm}:seed{seed}")
            continue
        for metric in METRICS:
            _metric(row, metric)
        selected[key] = row

    missing = [f"{arm}:seed{seed}" for arm in ARMS for seed in SEEDS if (arm, seed) not in selected]
    completeness = {
        "required_records": len(ARMS) * len(SEEDS),
        "observed_records": len(selected),
        "missing": missing,
        "duplicates": duplicates,
        "unexpected": unexpected,
    }
    result: dict[str, Any] = {
        "schema": "rgbt-p3-multiseed-final-gate-v1",
        "dataset": dataset,
        "evaluation_role": evaluation_role,
        "student_modality": student_modality,
        "seeds": list(SEEDS),
        "completeness": completeness,
    }
    if missing or duplicates or unexpected:
        result.update(
            {
                "status": "FAIL_INCOMPLETE_MULTISEED",
                "next_action": "do not declare a dataset terminal result; inspect the missing or invalid completed records",
            }
        )
        return result

    mean_map = {
        arm: sum(_metric(selected[(arm, seed)], "mAP50_95") for seed in SEEDS) / len(SEEDS)
        for arm in ARMS
    }
    p3_minus_control_mean_pp = {arm: 100.0 * (mean_map["p3"] - mean_map[arm]) for arm in CONTROLS}
    positive_seed_counts = {
        metric: sum(
            _metric(selected[("p3", seed)], metric) > _metric(selected[("native", seed)], metric)
            for seed in SEEDS
        )
        for metric in METRICS
    }
    pass_mean_map = all(value > 0.0 for value in p3_minus_control_mean_pp.values())
    pass_native_seed_majority = all(count >= 2 for count in positive_seed_counts.values())
    passed = pass_mean_map and pass_native_seed_majority
    result.update(
        {
            "status": "PASS_FINAL_MULTISEED" if passed else "KILL_RGBT-P3-CAUSAL-v1",
            "paired_p3_minus_control_mean_mAP50_95_pp": p3_minus_control_mean_pp,
            "paired_p3_positive_seed_count_vs_native": positive_seed_counts,
            "paired_p3_mean_mAP_strictly_exceeds_every_control": pass_mean_map,
            "paired_p3_beats_native_in_each_metric_at_least_two_of_three_seeds": pass_native_seed_majority,
            "next_action": "close dataset v1 as final pass" if passed else "no tuning rescue; close this dataset v1",
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("config must be a YAML mapping")
    result = decide_multiseed_final_gate(
        _records(args.metrics),
        dataset=str(config["dataset"]),
        evaluation_role=str(config["evaluation_role"]),
        student_modality=str(config["student_modality"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
