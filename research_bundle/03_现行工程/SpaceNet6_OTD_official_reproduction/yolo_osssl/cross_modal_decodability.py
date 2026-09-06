"""Object-level primitives for the cross-modal decodability pilot."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional


class CrossModalDecodabilityError(RuntimeError):
    """Raised when a probe comparison would violate the frozen protocol."""


@dataclass(frozen=True)
class SkillResult:
    skill: float
    model_loss: float
    null_loss: float


def fixed_center_patch_vectors(feature: torch.Tensor, centers_xy: torch.Tensor, *, patch_size: int = 3) -> torch.Tensor:
    """Sample fixed spatial patches without using target width or height.

    ``feature`` is ``[B,C,H,W]`` and ``centers_xy`` is ``[N,3]`` containing
    ``(batch_index, normalized_x, normalized_y)``.  Bilinear grid sampling
    keeps the extraction rule identical across modalities and feature levels.
    """

    if feature.ndim != 4 or centers_xy.ndim != 2 or centers_xy.shape[1] != 3:
        raise CrossModalDecodabilityError("feature or center tensor has an invalid shape")
    if patch_size <= 0 or patch_size % 2 != 1:
        raise CrossModalDecodabilityError("patch_size must be a positive odd integer")
    if centers_xy.numel() == 0:
        return feature.new_zeros((0, feature.shape[1] * patch_size * patch_size))
    batch_indices = centers_xy[:, 0].long()
    if int(batch_indices.min()) < 0 or int(batch_indices.max()) >= feature.shape[0]:
        raise CrossModalDecodabilityError("center batch index exceeds feature batch")
    half = patch_size // 2
    offsets_x = torch.arange(-half, half + 1, device=feature.device, dtype=feature.dtype) * (2.0 / feature.shape[-1])
    offsets_y = torch.arange(-half, half + 1, device=feature.device, dtype=feature.dtype) * (2.0 / feature.shape[-2])
    yy, xx = torch.meshgrid(offsets_y, offsets_x, indexing="ij")
    base_x = centers_xy[:, 1].to(feature.dtype) * 2.0 - 1.0
    base_y = centers_xy[:, 2].to(feature.dtype) * 2.0 - 1.0
    grid = torch.stack(
        (
            base_x[:, None, None] + xx[None],
            base_y[:, None, None] + yy[None],
        ),
        dim=-1,
    )
    sampled = functional.grid_sample(
        feature[batch_indices],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled.flatten(1)


def geometry_target(cx: float, cy: float, width: float, height: float, *, image_size: int, p3_stride: int = 8) -> tuple[float, ...]:
    """Return the frozen centre-offset and log-size geometry target."""

    x_grid = float(cx) * float(image_size) / float(p3_stride)
    y_grid = float(cy) * float(image_size) / float(p3_stride)
    return (
        x_grid - np.floor(x_grid) - 0.5,
        y_grid - np.floor(y_grid) - 0.5,
        float(np.log(max(float(width), 1e-8))),
        float(np.log(max(float(height), 1e-8))),
    )


def deterministic_background_centers(
    boxes_xywh: Sequence[Sequence[float]],
    *,
    image_index: int,
) -> list[tuple[float, float]]:
    """Choose one fixed background point per object from a 16x16 grid."""

    boxes = [tuple(float(value) for value in box) for box in boxes_xywh]
    grid = [((x + 0.5) / 16.0, (y + 0.5) / 16.0) for y in range(16) for x in range(16)]
    output: list[tuple[float, float]] = []
    used: set[int] = set()
    for object_index in range(len(boxes)):
        start = (17 * int(image_index) + 37 * object_index) % len(grid)
        chosen = None
        for delta in range(len(grid)):
            candidate_index = (start + delta) % len(grid)
            if candidate_index in used:
                continue
            x, y = grid[candidate_index]
            inside = any(abs(x - cx) <= width / 2 and abs(y - cy) <= height / 2 for _, cx, cy, width, height in boxes)
            if not inside:
                chosen = candidate_index
                break
        if chosen is None:
            raise CrossModalDecodabilityError("image has no available fixed-grid background point")
        used.add(chosen)
        output.append(grid[chosen])
    return output


def orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit a label-free orthogonal row-vector mapping ``source @ A``."""

    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise CrossModalDecodabilityError("Procrustes inputs must be equal-shaped matrices")
    if source.shape[0] < 2:
        raise CrossModalDecodabilityError("Procrustes needs at least two paired rows")
    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    return u @ vt


def macro_log_loss(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: Sequence[int],
    sample_weight: np.ndarray | None = None,
) -> float:
    """Class-balanced negative log likelihood."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = np.ones(labels.shape[0], dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    losses: list[float] = []
    for column, class_id in enumerate(classes):
        positions = labels == int(class_id)
        if not bool(positions.any()):
            continue
        selected_weights = weights[positions]
        selected = -np.log(np.clip(probabilities[positions, column], 1e-12, 1.0))
        losses.append(float(np.average(selected, weights=selected_weights)))
    if not losses:
        raise CrossModalDecodabilityError("macro log loss received no represented class")
    return float(np.mean(losses))


def classification_skill(
    labels: np.ndarray,
    probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    *,
    classes: Sequence[int],
    sample_weight: np.ndarray | None = None,
) -> SkillResult:
    model_loss = macro_log_loss(labels, probabilities, classes=classes, sample_weight=sample_weight)
    null_loss = macro_log_loss(labels, null_probabilities, classes=classes, sample_weight=sample_weight)
    return SkillResult(1.0 - model_loss / null_loss, model_loss, null_loss)


def geometry_skill(target: np.ndarray, prediction: np.ndarray, *, sample_weight: np.ndarray | None = None) -> SkillResult:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2:
        raise CrossModalDecodabilityError("geometry target and prediction must be equal-shaped matrices")
    weights = np.ones(target.shape[0], dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    model_per_row = np.square(target - prediction).mean(axis=1)
    null_per_row = np.square(target).mean(axis=1)
    model_loss = float(np.average(model_per_row, weights=weights))
    null_loss = float(np.average(null_per_row, weights=weights))
    return SkillResult(1.0 - model_loss / max(null_loss, 1e-12), model_loss, null_loss)


def shuffled_pair_indices(
    group_ids: Sequence[str],
    *,
    labels: Sequence[int] | None = None,
    bins: Sequence[int] | None = None,
    shift: int = 1,
) -> np.ndarray:
    """Build a deterministic cross-group cyclic donor permutation."""

    groups = np.asarray(group_ids, dtype=str)
    count = len(groups)
    if count < 2:
        raise CrossModalDecodabilityError("shuffling needs at least two rows")
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    label_values = np.zeros(count, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
    bin_values = np.zeros(count, dtype=np.int64) if bins is None else np.asarray(bins, dtype=np.int64)
    for index, key in enumerate(zip(label_values.tolist(), bin_values.tolist())):
        strata[key].append(index)
    result = np.full(count, -1, dtype=np.int64)
    for positions in strata.values():
        ordered = sorted(positions, key=lambda index: (groups[index], index))
        for offset, source in enumerate(ordered):
            donor = None
            for step in range(1, len(ordered) + 1):
                candidate = ordered[(offset + int(shift) + step - 1) % len(ordered)]
                if groups[candidate] != groups[source]:
                    donor = candidate
                    break
            if donor is not None:
                result[source] = donor
    if bool((result < 0).any()):
        raise CrossModalDecodabilityError("a shuffled stratum lacks a cross-group donor")
    return result


def paired_group_bootstrap(
    group_ids: Sequence[str],
    statistic,
    *,
    repeats: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap an arbitrary paired statistic by whole group."""

    groups = np.asarray(group_ids, dtype=str)
    unique = np.unique(groups)
    if unique.size < 2:
        raise CrossModalDecodabilityError("group bootstrap needs at least two groups")
    point = float(statistic(np.arange(groups.shape[0], dtype=np.int64)))
    rng = np.random.default_rng(seed)
    values = np.empty(int(repeats), dtype=np.float64)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    for repeat in range(int(repeats)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([rows_by_group[group] for group in sampled])
        values[repeat] = float(statistic(rows))
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    """Centered linear CKA for paired object rows."""

    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise CrossModalDecodabilityError("CKA inputs must share paired rows")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / max(float(denominator), 1e-12))


def semantic_status(class_count: int) -> str:
    return "available" if int(class_count) > 1 else "N/A_single_class"

