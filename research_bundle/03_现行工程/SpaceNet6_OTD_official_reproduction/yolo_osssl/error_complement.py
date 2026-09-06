"""Small deterministic primitives for the D0+ error-complement audit.

This module deliberately contains no model loading.  It is shared by the split
builder, prediction audit, and result gate so that all three use the same
class-aware matching rule.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


ERROR_ROLES = ("teacher_labeled", "calibration", "discovery")
ERROR_DECISIONS = {
    "ALLOW_RESIDUAL_ADD_SPEC",
    "ALLOW_OPTICAL_VETO_SPEC",
    "KILL_NO_SHARED_ERROR_COMPLEMENT",
}


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """Return IoU for xyxy boxes without adding detector-specific policy."""

    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_greedy_match(
    predictions: Sequence[dict], targets: Sequence[dict], threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """Greedily match confidence-sorted predictions to one target of its class."""

    used: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    order = sorted(range(len(predictions)), key=lambda index: (-float(predictions[index]["confidence"]), index))
    for prediction_index in order:
        prediction = predictions[prediction_index]
        options = [
            (box_iou(prediction["box"], target["box"]), target_index)
            for target_index, target in enumerate(targets)
            if target_index not in used and int(target["class_id"]) == int(prediction["class_id"])
        ]
        if not options:
            continue
        overlap, target_index = max(options, key=lambda item: (item[0], -item[1]))
        if overlap >= threshold:
            used.add(target_index)
            matches.append((prediction_index, target_index, float(overlap)))
    return matches


def prediction_support(prediction: dict, optical_predictions: Sequence[dict]) -> float:
    """Maximum same-class EO confidence times overlap for one SAR prediction."""

    values = [
        float(item["confidence"]) * box_iou(prediction["box"], item["box"])
        for item in optical_predictions
        if int(item["class_id"]) == int(prediction["class_id"])
    ]
    return float(max(values, default=0.0))


def residual_candidates(
    sar_predictions: Sequence[dict], optical_predictions: Sequence[dict], targets: Sequence[dict]
) -> list[dict]:
    """Label EO additions which are not already covered by a SAR prediction."""

    sar_target_matches = class_aware_greedy_match(sar_predictions, targets)
    missed_targets = [target for index, target in enumerate(targets) if index not in {match[1] for match in sar_target_matches}]
    residual = []
    sar_by_class: dict[int, list[dict]] = defaultdict(list)
    for prediction in sar_predictions:
        sar_by_class[int(prediction["class_id"])].append(prediction)
    pending: list[tuple[int, dict]] = []
    for optical_index, prediction in enumerate(optical_predictions):
        if any(box_iou(prediction["box"], sar["box"]) >= 0.5 for sar in sar_by_class[int(prediction["class_id"])]):
            continue
        pending.append((optical_index, prediction))
    matched = {prediction: target for prediction, target, _ in class_aware_greedy_match([item[1] for item in pending], missed_targets)}
    for pending_index, (optical_index, prediction) in enumerate(pending):
        target_index = matched.get(pending_index)
        residual.append(
            {
                "prediction_index": optical_index,
                "class_id": int(prediction["class_id"]),
                "confidence": float(prediction["confidence"]),
                "box": [float(value) for value in prediction["box"]],
                "is_true": target_index is not None,
                "target_index": target_index,
            }
        )
    return residual


def p6_candidates(sar_predictions: Sequence[dict], targets: Sequence[dict], min_confidence: float = 0.0) -> list[dict]:
    """Attach one class-aware TP/FP match to an already selected SAR pool."""

    selected = [item for item in sar_predictions if float(item["confidence"]) >= min_confidence]
    matched = {prediction: target for prediction, target, _ in class_aware_greedy_match(selected, targets)}
    rows = []
    for index, prediction in enumerate(selected):
        x1, y1, x2, y2 = (float(value) for value in prediction["box"])
        rows.append(
            {
                "prediction_index": index,
                "class_id": int(prediction["class_id"]),
                "confidence": float(prediction["confidence"]),
                "box": [x1, y1, x2, y2],
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                "is_true": index in matched,
            }
        )
    return rows


def precision_threshold(rows: Sequence[dict], score_key: str, target_precision: float) -> dict:
    """Freeze the largest confidence prefix satisfying a calibrated precision target.

    Equal-score detections remain in the same prefix.  The lowest qualifying
    score is therefore the deterministic tie break and keeps source-image
    membership independent of input ordering.
    """

    ordered = sorted(rows, key=lambda row: (-float(row[score_key]), -float(row.get("confidence", 0.0))))
    best_count = 0
    best_threshold = 1.0
    for threshold in sorted({float(row[score_key]) for row in rows}, reverse=True):
        retained = [row for row in ordered if float(row[score_key]) >= threshold]
        precision = sum(bool(row["is_true"]) for row in retained) / len(retained)
        if precision >= target_precision and len(retained) >= best_count:
            best_count = len(retained)
            best_threshold = threshold
    if not best_count:
        return {"status": f"NO_PRECISION_{int(target_precision * 100)}", "retain_fraction": 0.0, "threshold": 1.0, "precision": 0.0}
    retained = [row for row in ordered if float(row[score_key]) >= best_threshold]
    return {
        "status": "FROZEN",
        "retain_fraction": best_count / len(ordered),
        "threshold": float(best_threshold),
        "precision": sum(bool(row["is_true"]) for row in retained) / len(retained),
    }


def frozen_retain_fraction(rows: Sequence[dict], target_precision: float = 0.90) -> dict:
    """Freeze the largest EO-support retain prefix satisfying 90% precision."""

    return precision_threshold(rows, "exact_support", target_precision)


def dose_match_by_image_class(rows: Sequence[dict], relations: Sequence[str]) -> list[dict]:
    """Keep equal counts per source image/class for all requested relations."""

    grouped: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(str(row["pair_id"]), int(row["class_id"]))][str(row["relation"])].append(row)
    selected: list[dict] = []
    for _, by_relation in sorted(grouped.items()):
        if any(not by_relation.get(relation) for relation in relations):
            continue
        count = min(len(by_relation[relation]) for relation in relations)
        for relation in relations:
            selected.extend(
                sorted(by_relation[relation], key=lambda row: (-float(row["confidence"]), int(row.get("prediction_index", 0))))[:count]
            )
    return sorted(selected, key=lambda row: (str(row["pair_id"]), int(row["class_id"]), str(row["relation"]), -float(row["confidence"])))


def area_bin(area: float, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(area), side="right"))


def matched_dose_flags(rows: Sequence[dict], score_key: str, retain_by_stratum: dict[str, int]) -> list[bool]:
    """Retain equal class-area counts for a control score.

    This is intentionally a simple ranking rule: it compares EO support with
    SAR confidence and wrong-EO support at the same class-area dose, rather
    than introducing a fitted calibration model.
    """

    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["stratum"])].append((index, row))
    flags = [False] * len(rows)
    for stratum, values in grouped.items():
        count = min(int(retain_by_stratum.get(stratum, 0)), len(values))
        for index, _ in sorted(values, key=lambda item: (-float(item[1][score_key]), item[0]))[:count]:
            flags[index] = True
    return flags


def binary_metrics(rows: Sequence[dict], keep_key: str, score_key: str) -> dict[str, float]:
    """Compute finite pseudo-label metrics from an already matched candidate pool."""

    if not rows:
        return {"precision": 0.0, "tp_retain": 0.0, "logloss": 0.0, "auprc": 0.0}
    labels = np.asarray([float(bool(row["is_true"])) for row in rows], dtype=np.float64)
    keep = np.asarray([bool(row[keep_key]) for row in rows], dtype=bool)
    scores = np.clip(np.asarray([float(row[score_key]) for row in rows], dtype=np.float64), 1e-6, 1 - 1e-6)
    retained = labels[keep]
    precision = float(retained.mean()) if len(retained) else 0.0
    positives = max(int(labels.sum()), 1)
    tp_retain = float(labels[keep].sum() / positives)
    logloss = float(-np.mean(labels * np.log(scores) + (1.0 - labels) * np.log(1.0 - scores)))
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1, dtype=np.float64)
    auprc = float(np.sum((cumulative / ranks) * sorted_labels) / positives)
    return {"precision": precision, "tp_retain": tp_retain, "logloss": logloss, "auprc": auprc}


def unit_metric_delta(rows: Iterable[dict], exact_key: str, control_key: str) -> dict[str, list[float]]:
    """Return one paired precision delta per unit for grouped bootstrap callers."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["unit_id"])].append(row)
    result: dict[str, list[float]] = {}
    for unit, items in grouped.items():
        exact = binary_metrics(items, exact_key, "exact_support")["precision"]
        control = binary_metrics(items, control_key, "sar_confidence")["precision"]
        if math.isfinite(exact) and math.isfinite(control):
            result[unit] = [exact - control]
    return result


def paired_best_null_bootstrap(values: dict[str, dict[str, tuple[float, float]]], draws: int, seed: int) -> dict[str, float]:
    """Bootstrap paired sufficient counts and select the strongest null per draw."""

    units = sorted(
        unit
        for unit, row in values.items()
        if all(
            len(row.get(key, ())) == 2 and all(math.isfinite(float(value)) for value in row[key])
            for key in ("exact", "relaxed", "wrong")
        )
    )
    if not units:
        return {"effect": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "sample_sd": 0.0, "groups": 0}
    matrix = np.asarray([[values[unit]["exact"], values[unit]["relaxed"], values[unit]["wrong"]] for unit in units], dtype=np.float64)

    def delta(counts: np.ndarray) -> np.ndarray:
        ratios = counts[..., 0] / np.maximum(counts[..., 1], 1.0)
        return ratios[..., 0] - np.maximum(ratios[..., 1], ratios[..., 2])

    observed = float(delta(matrix.sum(axis=0)))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, max(1, min(512, draws))):
        stop = min(start + max(1, min(512, draws)), draws)
        indices = rng.integers(0, len(units), size=(stop - start, len(units)))
        sampled = matrix[indices].sum(axis=1)
        samples[start:stop] = delta(sampled)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"effect": observed, "ci95_low": float(low), "ci95_high": float(high), "sample_sd": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0, "groups": len(units)}
