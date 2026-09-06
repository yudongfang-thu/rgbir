"""CPU-only primitives for the data-first paired-modality diagnosis.

The module intentionally stops at frozen predictions and feature-probe
outputs.  It does not load a model or start training.  Eligibility is decided
from three task-metric gates; representation similarities remain descriptive.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from yolo_osssl.error_complement import class_aware_greedy_match

QUADRANTS = ("both_correct", "source_only", "target_only", "both_wrong")
REQUIRED_GATES = ("TASK_HEADROOM", "PAIR_UTILITY", "TARGET_REACHABILITY")
DESCRIPTIVE_ONLY_METRICS = ("CKA", "RSA", "MSE")
PRIMARY_DATASETS = {"llvip", "dronevehicle"}


def normalize_dataset(dataset: str) -> str:
    value = str(dataset).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "llvip": "llvip",
        "drone": "dronevehicle",
        "dronevehicle": "dronevehicle",
        "vedai": "vedai",
        "spacenet6": "spacenet6",
    }
    if value not in aliases:
        raise ValueError(f"unsupported diagnostic dataset: {dataset}")
    return aliases[value]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _identity(row: Mapping) -> str:
    if row.get("pair_id") is not None:
        return str(row["pair_id"])
    if row.get("image_id") is not None:
        return str(row["image_id"])
    if row.get("image_name") is not None:
        return Path(str(row["image_name"])).stem
    raise ValueError("diagnostic rows require pair_id or image_id")


def inference_unit_metadata(row: Mapping, dataset: str) -> dict[str, str | bool]:
    """Return the frozen inference/bootstrap unit for one dataset row."""

    normalized = normalize_dataset(dataset)
    identity = _identity(row)
    if normalized == "llvip":
        prefix = str(row.get("sequence_prefix") or "").strip()
        if not prefix:
            raise ValueError("LLVIP rows require an explicit non-empty sequence_prefix")
        return {
            "unit_id": prefix,
            "unit_kind": "sequence_prefix",
            "inference_report": "leave_one_prefix_out",
        }
    return {
        "unit_id": identity,
        "unit_kind": "image",
        "inference_report": "image_bootstrap" if normalized == "dronevehicle" else "descriptive_only",
    }


def attach_inference_units(rows: Sequence[dict], dataset: str) -> list[dict]:
    output = []
    for row in rows:
        annotated = dict(row)
        annotated.update(inference_unit_metadata(row, dataset))
        output.append(annotated)
    return output


def _prediction_hits(
    predictions: Sequence[dict],
    targets: Sequence[dict],
    *,
    iou_threshold: float,
    min_confidence: float,
) -> dict[int, dict]:
    retained_pairs = [
        (index, dict(prediction))
        for index, prediction in enumerate(predictions)
        if float(prediction.get("confidence", 1.0)) >= float(min_confidence)
    ]
    retained = [prediction for _, prediction in retained_pairs]
    matched = class_aware_greedy_match(retained, targets, threshold=float(iou_threshold))
    return {
        target_index: {
            "prediction_index": retained_pairs[prediction_index][0],
            "iou": overlap,
            "confidence": float(retained[prediction_index].get("confidence", 1.0)),
        }
        for prediction_index, target_index, overlap in matched
    }


def _quadrant(target_hit: bool, source_hit: bool) -> str:
    if target_hit and source_hit:
        return "both_correct"
    if source_hit:
        return "source_only"
    if target_hit:
        return "target_only"
    return "both_wrong"


def annotate_pair_quadrants(
    row: Mapping,
    dataset: str,
    *,
    iou_threshold: float = 0.5,
    min_confidence: float = 0.0,
) -> tuple[list[dict], dict]:
    """Match both native predictors to one target-GT roster and label objects."""

    if not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    prediction_roles = dict(row.get("prediction_roles", {}))
    if prediction_roles != {"target": "native", "source": "native"}:
        raise ValueError(
            "pair rows must declare prediction_roles={target: native, source: native}"
        )
    pair_id = _identity(row)
    targets = [dict(target) for target in row.get("target_gt", [])]
    target_hits = _prediction_hits(
        row.get("target_predictions", []),
        targets,
        iou_threshold=iou_threshold,
        min_confidence=min_confidence,
    )
    source_hits = _prediction_hits(
        row.get("source_predictions", []),
        targets,
        iou_threshold=iou_threshold,
        min_confidence=min_confidence,
    )
    unit = inference_unit_metadata(row, dataset)
    objects = []
    counts = {name: 0 for name in QUADRANTS}
    for target_index, target in enumerate(targets):
        target_hit = target_index in target_hits
        source_hit = target_index in source_hits
        quadrant = _quadrant(target_hit, source_hit)
        counts[quadrant] += 1
        objects.append(
            {
                "schema": "data-first-object-quadrant-v1",
                "dataset": normalize_dataset(dataset),
                "pair_id": pair_id,
                **unit,
                "gt_index": target_index,
                "class_id": int(target["class_id"]),
                "box": [float(value) for value in target["box"]],
                "target_detected": target_hit,
                "source_detected": source_hit,
                "target_match": target_hits.get(target_index),
                "source_match": source_hits.get(target_index),
                "quadrant": quadrant,
            }
        )
    summary = {
        "schema": "data-first-pair-quadrants-v1",
        "dataset": normalize_dataset(dataset),
        "pair_id": pair_id,
        **unit,
        "target_path": row.get("target_path"),
        "source_path": row.get("source_path"),
        "no_data_mask_path": row.get("no_data_mask_path"),
        "data_attrs": dict(row.get("data_attrs", {})),
        "prediction_roles": prediction_roles,
        "gt_objects": len(targets),
        "quadrant_counts": counts,
        "quadrants_present": [name for name in QUADRANTS if counts[name] > 0],
        "match_protocol": {
            "target_gt_roster": "shared",
            "class_aware": True,
            "one_to_one": True,
            "iou_threshold": float(iou_threshold),
            "min_confidence": float(min_confidence),
        },
    }
    return objects, summary


def analyze_pair_records(
    rows: Sequence[dict],
    dataset: str,
    *,
    iou_threshold: float = 0.5,
    min_confidence: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    object_rows: list[dict] = []
    pair_rows: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        pair_id = _identity(row)
        if pair_id in seen:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        seen.add(pair_id)
        objects, summary = annotate_pair_quadrants(
            row,
            dataset,
            iou_threshold=iou_threshold,
            min_confidence=min_confidence,
        )
        object_rows.extend(objects)
        pair_rows.append(summary)
    return object_rows, sorted(pair_rows, key=lambda item: item["pair_id"])


def _sample_key(row: Mapping) -> tuple[str, str]:
    # Selection is deliberately limited to data attributes and stable identity.
    attributes = json.dumps(row.get("data_attrs", {}), ensure_ascii=False, sort_keys=True)
    return attributes, str(row["pair_id"])


def freeze_sample_selection(
    pair_rows: Sequence[dict],
    *,
    sample_count: int = 12,
) -> tuple[list[dict], dict]:
    """Select a deterministic quadrant-balanced sample without method outputs."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    ordered = sorted((dict(row) for row in pair_rows), key=_sample_key)
    available = [
        quadrant
        for quadrant in QUADRANTS
        if any(int(row["quadrant_counts"].get(quadrant, 0)) > 0 for row in ordered)
    ]
    selected: list[tuple[dict, str]] = []
    selected_ids: set[str] = set()
    while len(selected) < min(sample_count, len(ordered)):
        progressed = False
        for quadrant in available:
            candidate = next(
                (
                    row
                    for row in ordered
                    if row["pair_id"] not in selected_ids
                    and int(row["quadrant_counts"].get(quadrant, 0)) > 0
                ),
                None,
            )
            if candidate is not None:
                selected.append((candidate, quadrant))
                selected_ids.add(candidate["pair_id"])
                progressed = True
                if len(selected) == min(sample_count, len(ordered)):
                    break
        if not progressed:
            break
    for row in ordered:
        if len(selected) == min(sample_count, len(ordered)):
            break
        if row["pair_id"] not in selected_ids:
            selected.append((row, "coverage_fill"))
            selected_ids.add(row["pair_id"])

    output = []
    for rank, (row, reason) in enumerate(selected, start=1):
        output.append(
            {
                "schema": "data-first-frozen-sample-v1",
                "dataset": row["dataset"],
                "sample_rank": rank,
                "pair_id": row["pair_id"],
                "target_path": row.get("target_path"),
                "source_path": row.get("source_path"),
                "no_data_mask_path": row.get("no_data_mask_path"),
                "data_attrs": row.get("data_attrs", {}),
                "quadrant_counts": row["quadrant_counts"],
                "selection_reason": reason,
            }
        )
    selected_coverage = {
        quadrant: sum(int(row["quadrant_counts"].get(quadrant, 0)) for row, _ in selected)
        for quadrant in QUADRANTS
    }
    summary = {
        "schema": "data-first-sample-selection-v1",
        "requested": int(sample_count),
        "selected": len(output),
        "available_pairs": len(ordered),
        "selection_inputs": ["target_gt", "target_native", "source_native", "data_attrs"],
        "available_quadrants": available,
        "missing_quadrants": [name for name in QUADRANTS if name not in available],
        "selected_object_coverage": selected_coverage,
        "selected_coverage_missing": [name for name in available if selected_coverage[name] == 0],
    }
    return output, summary


GlobalMetric = Callable[[Sequence[dict]], float]


def _rows_by_unit(rows: Sequence[dict], unit_key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if unit_key not in row:
            raise ValueError(f"bootstrap row is missing {unit_key}")
        grouped[str(row[unit_key])].append(row)
    return dict(grouped)


def _unit_roster(indexed: Mapping[str, Sequence[dict]]) -> dict[str, list[str]]:
    """Describe paired observations inside every resampling unit."""

    roster = {}
    for unit, rows in indexed.items():
        identities = []
        for index, row in enumerate(rows):
            try:
                identity = _identity(row)
            except ValueError:
                identity = f"row_{index:08d}"
            identities.append(identity)
        roster[unit] = sorted(identities)
    return roster


def validate_paired_unit_rosters(
    arms: Mapping[str, Sequence[dict]],
    names: Sequence[str],
    *,
    unit_key: str = "unit_id",
) -> list[str]:
    """Require identical units and observation identities across paired arms."""

    if not names or any(name not in arms for name in names):
        raise KeyError("one or more paired arms are missing")
    indexed = {name: _rows_by_unit(list(arms[name]), unit_key) for name in names}
    units = sorted(indexed[names[0]])
    if len(units) < 2 or any(set(indexed[name]) != set(units) for name in names):
        raise ValueError("all arms must share at least two bootstrap units")
    first_roster = _unit_roster(indexed[names[0]])
    if any(_unit_roster(indexed[name]) != first_roster for name in names[1:]):
        raise ValueError("all arms must share the same paired observation roster")
    return units


def shared_global_metric_bootstrap(
    arms: Mapping[str, Sequence[dict]],
    candidate: str,
    references: Sequence[str],
    global_metric: GlobalMetric,
    *,
    unit_key: str = "unit_id",
    draws: int = 10_000,
    seed: int = 42,
) -> dict:
    """Recompute one global metric on shared whole-unit bootstrap draws.

    ``global_metric`` receives the full resampled dataset for one arm.  It is
    never called once per image, so callers cannot accidentally average
    per-image AP values.
    """

    if draws <= 0:
        raise ValueError("draws must be positive")
    names = (candidate, *references)
    if not references or any(name not in arms for name in names):
        raise KeyError("candidate or reference arm is missing")
    indexed = {name: _rows_by_unit(list(arms[name]), unit_key) for name in names}
    units = validate_paired_unit_rosters(arms, names, unit_key=unit_key)
    observed = {name: float(global_metric(list(arms[name]))) for name in names}
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        sampled = rng.integers(0, len(units), size=len(units))
        scores = {}
        for name in names:
            resampled = [
                row
                for unit_index in sampled
                for row in indexed[name][units[int(unit_index)]]
            ]
            scores[name] = float(global_metric(resampled))
        deltas[draw] = scores[candidate] - max(scores[name] for name in references)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "candidate": candidate,
        "references": list(references),
        "units": len(units),
        "unit_key": unit_key,
        "draws": int(draws),
        "seed": int(seed),
        "metric_protocol": "global_callback_recomputed_per_arm_per_draw",
        "shared_resampling": True,
        "point_metrics": observed,
        "effect": observed[candidate] - max(observed[name] for name in references),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_sd": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
    }


def leave_one_unit_out(
    arms: Mapping[str, Sequence[dict]],
    candidate: str,
    references: Sequence[str],
    global_metric: GlobalMetric,
    *,
    unit_key: str = "unit_id",
) -> list[dict]:
    """Report global candidate-minus-best-control after dropping each unit."""

    names = (candidate, *references)
    indexed = {name: _rows_by_unit(list(arms[name]), unit_key) for name in names}
    units = validate_paired_unit_rosters(arms, names, unit_key=unit_key)
    output = []
    for omitted in units:
        metrics = {
            name: float(
                global_metric(
                    [row for unit in units if unit != omitted for row in indexed[name][unit]]
                )
            )
            for name in names
        }
        output.append(
            {
                "omitted_unit": omitted,
                "candidate_metric": metrics[candidate],
                "best_reference_metric": max(metrics[name] for name in references),
                "effect": metrics[candidate] - max(metrics[name] for name in references),
            }
        )
    return output


def build_eligibility(
    dataset: str,
    gate_evidence: Mapping[str, Mapping],
    *,
    sample_selection: Mapping | None = None,
) -> dict:
    """Build the fixed three-gate eligibility contract."""

    unknown = sorted(set(gate_evidence) - set(REQUIRED_GATES))
    if unknown:
        raise ValueError(
            "only TASK_HEADROOM, PAIR_UTILITY, and TARGET_REACHABILITY may gate eligibility; "
            f"got {unknown}"
        )
    gates = {}
    normalized = normalize_dataset(dataset)
    for name in REQUIRED_GATES:
        evidence = dict(gate_evidence.get(name, {}))
        if not evidence:
            gates[name] = {"status": "NOT_EVALUATED"}
            continue
        if "ci95_low" not in evidence or "effect" not in evidence:
            raise ValueError(f"{name} lacks bootstrap effect or ci95_low")
        protocol_issues = []
        if int(evidence.get("draws", 0)) < 10_000:
            protocol_issues.append("bootstrap_draws_lt_10000")
        if evidence.get("shared_resampling") is not True:
            protocol_issues.append("resampling_not_shared")
        if not str(evidence.get("metric_protocol", "")).startswith("global_"):
            protocol_issues.append("metric_not_globally_recomputed")
        if evidence.get("cross_fitted") is not True:
            protocol_issues.append("not_cross_fitted")
        if normalized == "llvip" and not evidence.get("leave_one_prefix_out"):
            protocol_issues.append("missing_leave_one_prefix_out")
        if protocol_issues:
            evidence["status"] = "INVALID_PROTOCOL"
            evidence["protocol_issues"] = protocol_issues
        else:
            evidence["status"] = "PASS" if float(evidence["ci95_low"]) > 0.0 else "FAIL"
        gates[name] = evidence
    evaluated = all(gates[name]["status"] != "NOT_EVALUATED" for name in REQUIRED_GATES)
    valid_protocol = evaluated and all(
        gates[name]["status"] != "INVALID_PROTOCOL" for name in REQUIRED_GATES
    )
    passed = evaluated and all(gates[name]["status"] == "PASS" for name in REQUIRED_GATES)
    if normalized not in PRIMARY_DATASETS:
        status = "DIAGNOSTIC_ONLY"
    elif not evaluated:
        status = "INCOMPLETE"
    elif not valid_protocol:
        status = "INCOMPLETE_PROTOCOL"
    else:
        status = "ELIGIBLE" if passed else "NOT_ELIGIBLE"
    unit = (
        {"unit_kind": "sequence_prefix", "inference_report": "leave_one_prefix_out"}
        if normalized == "llvip"
        else {"unit_kind": "image", "inference_report": "image_bootstrap"}
        if normalized == "dronevehicle"
        else {"unit_kind": "image", "inference_report": "descriptive_only"}
    )
    return {
        "schema": "data-first-eligibility-v1",
        "dataset": normalized,
        "status": status,
        "eligible": bool(normalized in PRIMARY_DATASETS and passed),
        "primary_metric": "global_mAP50_95",
        "gate_rule": "all_three_ci95_low_gt_zero",
        "bootstrap_default_draws": 10_000,
        "shared_resampling": True,
        "inference_unit": unit,
        "descriptive_only_not_gates": list(DESCRIPTIVE_ONLY_METRICS),
        "gates": gates,
        "sample_selection": dict(sample_selection or {}),
    }
