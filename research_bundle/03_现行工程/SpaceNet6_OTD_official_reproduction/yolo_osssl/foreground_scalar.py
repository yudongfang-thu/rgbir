"""Independent foreground-any scalar transfer diagnostic.

This module is deliberately separate from the frozen component-v2 arm table.
It implements a fit-only kill test for one low-dimensional target: the
probability that at least one detector class is foreground at an anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

from .b0_distillation import frozen_teacher_prediction, native_student_loss
from .component_kd import ObjectMetadata, _scores, object_metadata


FOREGROUND_SCALAR_SOURCES = (
    "exact_optical",
    "same_sar",
    "permuted_optical",
    "hard_wrong_optical",
)


class ForegroundScalarError(RuntimeError):
    pass


def foreground_any_logit(logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Collapse class logits to the Bernoulli logit for any foreground class."""

    if logits.ndim < 1 or logits.shape[-1] < 1 or temperature <= 0:
        raise ForegroundScalarError("foreground-any requires at least one class logit")
    scaled = logits.float() / float(temperature)
    if scaled.shape[-1] == 1:
        return scaled
    log_background = functional.logsigmoid(-scaled).sum(dim=-1, keepdim=True)
    return torch.log(-torch.expm1(log_background)) - log_background


def object_scalar_targets(
    teacher_scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    *,
    temperature: float = 2.0,
) -> dict[tuple[int, int], torch.Tensor]:
    """Average class logits within an object, then collapse to foreground-any."""

    return {
        (item.image_index, item.gt_index): foreground_any_logit(
            teacher_scores[item.image_index, anchors[(item.image_index, item.gt_index)]].mean(0, keepdim=True),
            temperature,
        )
        for item in metadata
    }


def deranged_object_targets(
    targets: Mapping[tuple[int, int], torch.Tensor],
    metadata: Sequence[ObjectMetadata],
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], tuple[int, int]]]:
    """Cyclically permute object targets with no self mapping.

    The operation preserves the exact scalar multiset.  Object identity is the
    only information destroyed by this matched null.
    """

    keys = [(item.image_index, item.gt_index) for item in metadata if (item.image_index, item.gt_index) in targets]
    if len(keys) < 2:
        raise ForegroundScalarError("object permutation requires at least two assigned objects")
    donors = {key: keys[(position + 1) % len(keys)] for position, key in enumerate(keys)}
    return {key: targets[donors[key]] for key in keys}, donors


def _bernoulli_kl(student_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    student_logits = student_logits.float()
    target_probability = torch.sigmoid(target.detach().float())
    return target_probability * (
        target_probability.clamp_min(torch.finfo(torch.float32).tiny).log()
        - functional.logsigmoid(student_logits)
    ) + (1.0 - target_probability) * (
        (1.0 - target_probability).clamp_min(torch.finfo(torch.float32).tiny).log()
        - functional.logsigmoid(-student_logits)
    )


def object_balanced_foreground_kd(
    student_scores: torch.Tensor,
    metadata: Sequence[ObjectMetadata],
    anchors: Mapping[tuple[int, int], torch.Tensor],
    targets: Mapping[tuple[int, int], torch.Tensor],
    *,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, int]:
    """Average anchors within objects, then average objects within a batch."""

    losses: list[torch.Tensor] = []
    for item in metadata:
        key = (item.image_index, item.gt_index)
        target = targets.get(key)
        if target is None:
            continue
        student = foreground_any_logit(student_scores[item.image_index, anchors[key]], temperature)
        losses.append(_bernoulli_kl(student, target).mean())
    if not losses:
        return student_scores.float().sum() * 0.0, 0
    return float(temperature) ** 2 * torch.stack(losses).mean() * student_scores.shape[0], len(losses)


@dataclass(frozen=True)
class ForegroundScalarResult:
    native_loss: torch.Tensor
    kd_loss: torch.Tensor
    total_loss: torch.Tensor
    detached_loss: Any
    objects: int
    used: int
    target_pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    target_identities: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    target_multiset_preserved: bool | None


def foreground_scalar_from_native(
    native: Any,
    batch: Mapping[str, Any],
    *,
    source: str,
    teacher: torch.nn.Module,
    kd_lambda: float,
    area_quartiles: Sequence[float],
    temperature: float = 2.0,
) -> ForegroundScalarResult:
    """Add foreground-scalar KD to an already computed native loss."""

    if source not in FOREGROUND_SCALAR_SOURCES:
        raise ForegroundScalarError(f"unknown foreground scalar source: {source}")
    zero = native.native_loss * 0.0
    if not bool(native.assignment.fg_mask.any()):
        return ForegroundScalarResult(
            native.native_loss, zero, native.native_loss, native.detached_loss, 0, 0, (), (), None
        )
    metadata, anchors = object_metadata(
        native.assignment,
        native.parsed_prediction,
        batch,
        area_quartiles=area_quartiles,
    )
    image_key = "img" if source == "same_sar" else "eo_img"
    images = batch.get(image_key)
    if not isinstance(images, torch.Tensor):
        raise ForegroundScalarError(f"{source} requires batch tensor {image_key}")
    teacher_scores = _scores(frozen_teacher_prediction(teacher, images))
    targets = object_scalar_targets(teacher_scores, metadata, anchors, temperature=temperature)
    original_targets = targets
    identities = {key: key for key in targets}
    multiset_preserved: bool | None = None
    if source == "permuted_optical":
        targets, identities = deranged_object_targets(targets, metadata)
        original_values = torch.sort(torch.stack([value.reshape(()) for value in original_targets.values()]))[0]
        permuted_values = torch.sort(torch.stack([value.reshape(()) for value in targets.values()]))[0]
        multiset_preserved = bool(torch.equal(original_values, permuted_values))
    student_scores = _scores(native.parsed_prediction)
    kd_loss, used = object_balanced_foreground_kd(
        student_scores,
        metadata,
        anchors,
        targets,
        temperature=temperature,
    )
    pairs = tuple(
        (
            foreground_any_logit(student_scores[item.image_index, anchors[key]], temperature).mean(0).detach(),
            targets[key].reshape(-1).mean().detach(),
        )
        for item in metadata
        for key in [(item.image_index, item.gt_index)]
        if key in targets
    )
    total = native.native_loss + float(kd_lambda) * kd_loss
    if not bool(torch.isfinite(total).detach().cpu()):
        raise ForegroundScalarError(f"{source} produced a nonfinite loss")
    return ForegroundScalarResult(
        native.native_loss,
        kd_loss,
        total,
        native.detached_loss,
        len(metadata),
        used,
        pairs,
        tuple((key, identities[key]) for key in targets),
        multiset_preserved,
    )


def foreground_scalar_criterion(
    criterion: Any,
    prediction: Any,
    batch: Mapping[str, Any],
    *,
    source: str,
    teacher: torch.nn.Module,
    kd_lambda: float,
    area_quartiles: Sequence[float],
    temperature: float = 2.0,
) -> ForegroundScalarResult:
    native = native_student_loss(criterion, prediction, batch)
    return foreground_scalar_from_native(
        native,
        batch,
        source=source,
        teacher=teacher,
        kd_lambda=kd_lambda,
        area_quartiles=area_quartiles,
        temperature=temperature,
    )
