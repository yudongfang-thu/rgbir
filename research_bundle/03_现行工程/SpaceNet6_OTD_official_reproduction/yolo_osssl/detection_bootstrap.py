"""Detection AP recomputation and paired cluster bootstrap primitives."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Sequence

import numpy as np


IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Ultralytics-compatible 101-point interpolated AP."""

    tail = recall[-1] if len(recall) else 1.0
    mrec = np.concatenate(([0.0], recall, [tail], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(np.interp(x, mrec, mpre), x))


def detection_map(rows: Sequence[dict]) -> dict[str, float]:
    """Recompute AP50/AP75/mAP50-95 from saved per-image validator stats."""

    target_cls = np.concatenate(
        [np.asarray(row["target_cls"], dtype=np.int64) for row in rows], axis=0
    ) if rows else np.empty(0, dtype=np.int64)
    classes, counts = np.unique(target_cls, return_counts=True)
    if not len(classes):
        return {"AP50": 0.0, "AP75": 0.0, "mAP50_95": 0.0}

    confidences = []
    pred_classes = []
    true_positive = []
    for row in rows:
        conf = np.asarray(row["conf"], dtype=np.float64)
        pred = np.asarray(row["pred_cls"], dtype=np.int64)
        tp = np.asarray(row["tp"], dtype=bool)
        if tp.size == 0:
            tp = np.empty((0, len(IOU_THRESHOLDS)), dtype=bool)
        elif tp.ndim != 2 or tp.shape[1] != len(IOU_THRESHOLDS):
            raise ValueError("each tp array must have shape [detections, 10]")
        if len(conf) != len(pred) or len(conf) != len(tp):
            raise ValueError("conf, pred_cls and tp lengths differ")
        confidences.append(conf)
        pred_classes.append(pred)
        true_positive.append(tp)

    conf = np.concatenate(confidences, axis=0)
    pred_cls = np.concatenate(pred_classes, axis=0)
    tp = np.concatenate(true_positive, axis=0)
    order = np.argsort(-conf, kind="stable")
    conf, pred_cls, tp = conf[order], pred_cls[order], tp[order]
    ap = np.zeros((len(classes), len(IOU_THRESHOLDS)), dtype=np.float64)
    for class_index, (class_id, target_count) in enumerate(zip(classes, counts)):
        selected = pred_cls == class_id
        if not np.any(selected):
            continue
        class_tp = tp[selected]
        false_positive = (1 - class_tp).cumsum(axis=0)
        true_positive_cumulative = class_tp.cumsum(axis=0)
        recall = true_positive_cumulative / max(int(target_count), 1)
        precision = true_positive_cumulative / np.maximum(
            true_positive_cumulative + false_positive, 1e-16
        )
        for threshold_index in range(len(IOU_THRESHOLDS)):
            ap[class_index, threshold_index] = compute_ap(
                recall[:, threshold_index], precision[:, threshold_index]
            )
    return {
        "AP50": float(ap[:, 0].mean()),
        "AP75": float(ap[:, 5].mean()),
        "mAP50_95": float(ap.mean()),
    }


def index_groups(rows: Sequence[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    return dict(grouped)


def sample_group_rows(indexed: dict[str, list[dict]], sampled: Sequence[str]) -> list[dict]:
    return [row for group in sampled for row in indexed[str(group)]]


def prepare_weighted_stats(rows: Sequence[dict], groups: Sequence[str]) -> dict:
    """Pre-sort detections once so bootstrap draws only change cluster weights."""

    group_index = {str(group): index for index, group in enumerate(groups)}
    target_by_group: list[np.ndarray] = []
    conf, pred_cls, tp, pred_group = [], [], [], []
    max_class = -1
    for row in rows:
        targets = np.asarray(row["target_cls"], dtype=np.int64)
        predictions = np.asarray(row["pred_cls"], dtype=np.int64)
        if len(targets):
            max_class = max(max_class, int(targets.max()))
        if len(predictions):
            max_class = max(max_class, int(predictions.max()))
    nc = max_class + 1
    target_by_group = [np.zeros(nc, dtype=np.float64) for _ in groups]
    for row in rows:
        index = group_index[str(row["group_id"])]
        targets = np.asarray(row["target_cls"], dtype=np.int64)
        if len(targets):
            target_by_group[index] += np.bincount(targets, minlength=nc)
        row_conf = np.asarray(row["conf"], dtype=np.float64)
        row_pred = np.asarray(row["pred_cls"], dtype=np.int64)
        row_tp = np.asarray(row["tp"], dtype=bool)
        if row_tp.size == 0:
            row_tp = np.empty((0, len(IOU_THRESHOLDS)), dtype=bool)
        conf.append(row_conf)
        pred_cls.append(row_pred)
        tp.append(row_tp)
        pred_group.append(np.full(len(row_conf), index, dtype=np.int64))
    confidence = np.concatenate(conf)
    order = np.argsort(-confidence, kind="stable")
    return {
        "target_by_group": np.asarray(target_by_group, dtype=np.float64),
        "conf": confidence[order],
        "pred_cls": np.concatenate(pred_cls)[order],
        "tp": np.concatenate(tp, axis=0)[order],
        "pred_group": np.concatenate(pred_group)[order],
        "nc": nc,
    }


def weighted_detection_map(prepared: dict, group_weights: np.ndarray) -> float:
    """Recompute mAP after duplicating clusters according to integer weights."""

    target_counts = group_weights @ prepared["target_by_group"]
    detection_weights = group_weights[prepared["pred_group"]]
    ap_rows = []
    for class_id in np.flatnonzero(target_counts > 0):
        selected = (prepared["pred_cls"] == class_id) & (detection_weights > 0)
        if not np.any(selected):
            ap_rows.append(np.zeros(len(IOU_THRESHOLDS), dtype=np.float64))
            continue
        weights = detection_weights[selected, None]
        class_tp = prepared["tp"][selected].astype(np.float64)
        true_positive = np.cumsum(class_tp * weights, axis=0)
        false_positive = np.cumsum((1.0 - class_tp) * weights, axis=0)
        recall = true_positive / target_counts[class_id]
        precision = true_positive / np.maximum(true_positive + false_positive, 1e-16)
        ap_rows.append(np.asarray([
            compute_ap(recall[:, index], precision[:, index])
            for index in range(len(IOU_THRESHOLDS))
        ]))
    return float(np.asarray(ap_rows).mean()) if ap_rows else 0.0


def paired_cluster_bootstrap(
    arms: dict[str, Sequence[dict]],
    candidate: str,
    anchors: Sequence[str],
    nulls: Sequence[str],
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bootstrap cluster identities and choose best anchor/null inside each draw."""

    if candidate not in arms or any(name not in arms for name in (*anchors, *nulls)):
        raise KeyError("candidate, anchor, or null arm is missing")
    indexed = {arm: index_groups(rows) for arm, rows in arms.items()}
    groups = sorted(indexed[candidate])
    if not groups or any(set(indexed[name]) != set(groups) for name in arms):
        raise ValueError("all arms must contain the same non-empty group roster")

    observed = {name: detection_map(list(arms[name]))["mAP50_95"] for name in arms}
    prepared = {arm: prepare_weighted_stats(rows, groups) for arm, rows in arms.items()}
    rng = np.random.default_rng(seed)
    candidate_anchor = np.empty(draws, dtype=np.float64)
    candidate_null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        weights = np.bincount(sampled, minlength=len(groups)).astype(np.float64)
        scores = {arm: weighted_detection_map(prepared[arm], weights) for arm in arms}
        candidate_anchor[draw] = scores[candidate] - max(scores[name] for name in anchors)
        candidate_null[draw] = scores[candidate] - max(scores[name] for name in nulls)

    def summarize(values: np.ndarray, observed_delta: float) -> dict[str, float]:
        return {
            "effect": float(observed_delta),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_sd": float(values.std(ddof=1)),
        }

    return {
        "groups": len(groups),
        "draws": draws,
        "candidate": candidate,
        "anchors": list(anchors),
        "nulls": list(nulls),
        "metrics": observed,
        "candidate_minus_best_anchor": summarize(
            candidate_anchor, observed[candidate] - max(observed[name] for name in anchors)
        ),
        "candidate_minus_best_null": summarize(
            candidate_null, observed[candidate] - max(observed[name] for name in nulls)
        ),
    }


def paired_cluster_bootstrap_multiseed(
    arms: dict[str, dict[int, Sequence[dict]]],
    candidate: str,
    anchors: Sequence[str],
    nulls: Sequence[str],
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bootstrap shared cluster draws and average detector scores over seeds."""

    if candidate not in arms or any(name not in arms for name in (*anchors, *nulls)):
        raise KeyError("candidate, anchor, or null arm is missing")
    seeds = sorted(arms[candidate])
    if not seeds:
        raise ValueError("multiseed bootstrap requires at least one seed")
    if any(set(arms[name]) != set(seeds) for name in arms):
        raise ValueError("all arms must contain the same seed roster")

    indexed: dict[str, dict[int, dict[str, list[dict]]]] = {}
    shared_groups: list[str] | None = None
    for arm, by_seed in arms.items():
        indexed[arm] = {}
        for run_seed, rows in by_seed.items():
            grouped = index_groups(rows)
            indexed[arm][run_seed] = grouped
            groups = sorted(grouped)
            if not groups:
                raise ValueError("all seed arms must contain at least one bootstrap group")
            if shared_groups is None:
                shared_groups = groups
            elif shared_groups != groups:
                raise ValueError("all arms and seeds must share one group roster")

    metrics_by_seed = {
        arm: {
            str(run_seed): detection_map(list(arms[arm][run_seed]))["mAP50_95"] for run_seed in seeds
        }
        for arm in arms
    }
    observed = {
        arm: float(np.mean([metrics_by_seed[arm][str(run_seed)] for run_seed in seeds])) for arm in arms
    }
    prepared = {
        arm: {
            run_seed: prepare_weighted_stats(arms[arm][run_seed], shared_groups)
            for run_seed in seeds
        }
        for arm in arms
    }
    rng = np.random.default_rng(seed)
    candidate_anchor = np.empty(draws, dtype=np.float64)
    candidate_null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(shared_groups), size=len(shared_groups))
        weights = np.bincount(sampled, minlength=len(shared_groups)).astype(np.float64)
        per_arm_scores = {arm: [] for arm in arms}
        for run_seed in seeds:
            for arm in arms:
                per_arm_scores[arm].append(weighted_detection_map(prepared[arm][run_seed], weights))
        mean_scores = {
            arm: float(np.mean(values)) if values else float("nan") for arm, values in per_arm_scores.items()
        }
        if any(not math.isfinite(value) for value in mean_scores.values()):
            raise RuntimeError("multiseed bootstrap produced a non-finite score")
        candidate_anchor[draw] = mean_scores[candidate] - max(mean_scores[name] for name in anchors)
        candidate_null[draw] = mean_scores[candidate] - max(mean_scores[name] for name in nulls)

    def summarize(values: np.ndarray, observed_delta: float) -> dict[str, float]:
        return {
            "effect": float(observed_delta),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_sd": float(values.std(ddof=1)),
        }

    return {
        "groups": len(shared_groups),
        "draws": draws,
        "candidate": candidate,
        "anchors": list(anchors),
        "nulls": list(nulls),
        "seeds": seeds,
        "metrics_by_seed": metrics_by_seed,
        "metrics_mean": observed,
        "candidate_minus_best_anchor": summarize(
            candidate_anchor, observed[candidate] - max(observed[name] for name in anchors)
        ),
        "candidate_minus_best_null": summarize(
            candidate_null, observed[candidate] - max(observed[name] for name in nulls)
        ),
    }


def flatten_replicates(rows: Iterable[dict]) -> list[dict]:
    """Materialize an iterable for callers that build bootstrap samples lazily."""

    return list(rows)
