"""Selective optical-to-SAR DFL distillation for the FP-v2 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as functional

from .b0_distillation import (
    B0Assignment,
    criterion_reg_max,
    dfl_logits,
    frozen_teacher_prediction,
    native_student_loss,
)
from .stage2 import masked_dfl_kl, per_anchor_localization_error


FP_ARMS = (
    "native",
    "all_fg",
    "legacy_conf",
    "p3",
    "p4p5",
    "p3_shuffled",
    "p3_same_modal",
    "p3_gt",
    "p3_random_dose",
    "p3_fp_hard_dose",
    "p3_conf_dose",
    "mo",
    "mo_shuffled",
    "mo_fp_hard",
    "mo_gt",
    "p3_mo",
    "apr_dfl",
)

OPTICAL_TEACHER_ARMS = frozenset(
    {
        "all_fg",
        "legacy_conf",
        "p3",
        "p4p5",
        "p3_shuffled",
        "p3_random_dose",
        "p3_fp_hard_dose",
        "p3_conf_dose",
        "mo",
        "mo_shuffled",
        "mo_fp_hard",
        "mo_gt",
        "p3_mo",
        "apr_dfl",
    }
)
SAR_TEACHER_ARMS = frozenset({"legacy_conf", "apr_dfl", "p3_same_modal", "p3_conf_dose"})
PAIRED_DATA_ARMS = OPTICAL_TEACHER_ARMS
SHUFFLED_ARMS = frozenset({"mo_shuffled", "p3_shuffled"})

_MO_ARMS = frozenset({"legacy_conf", "mo", "mo_shuffled", "mo_fp_hard", "mo_gt", "p3_mo", "apr_dfl"})


class FPSelectiveDistillationError(RuntimeError):
    """Raised when an FP-v2 arm cannot preserve the shared detector scaffold."""


@dataclass(frozen=True)
class FPSelectiveResult:
    native_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    total_loss: torch.Tensor
    detached_loss: Any
    stats: dict[str, float]


def _foreground_indices(assignment: B0Assignment) -> torch.Tensor:
    return assignment.fg_mask.nonzero(as_tuple=False)


def optical_advantage_mask(student_error: torch.Tensor, optical_error: torch.Tensor) -> torch.Tensor:
    if student_error.shape != optical_error.shape or student_error.ndim != 1:
        raise FPSelectiveDistillationError("student/optical localization errors must be matching vectors")
    return optical_error.detach().float() < student_error.detach().float()


def p3_foreground_mask(prediction: Mapping[str, Any], assignment: B0Assignment) -> torch.Tensor:
    """Return a foreground-vector mask for the first native raw-head level."""

    features = prediction.get("feats")
    if not isinstance(features, (tuple, list)) or len(features) != 3:
        raise FPSelectiveDistillationError("FP-v2 requires native P3/P4/P5 feature maps")
    if not all(isinstance(level, torch.Tensor) and level.ndim == 4 for level in features):
        raise FPSelectiveDistillationError("raw-head levels must be [B,C,H,W] tensors")
    level_sizes = [int(level.shape[-2] * level.shape[-1]) for level in features]
    if sum(level_sizes) != int(assignment.fg_mask.shape[1]):
        raise FPSelectiveDistillationError("raw-head levels disagree with assignment anchor count")
    return _foreground_indices(assignment)[:, 1] < level_sizes[0]


def per_image_matched_topk_mask(
    values: torch.Tensor,
    assignment: B0Assignment,
    reference_mask: torch.Tensor,
) -> torch.Tensor:
    """Select per-image top-k FG values with k copied from a reference mask."""

    foreground = _foreground_indices(assignment)
    if values.ndim != 1 or reference_mask.ndim != 1 or values.shape != reference_mask.shape:
        raise FPSelectiveDistillationError("matched top-k inputs must match the foreground vector")
    if values.shape[0] != foreground.shape[0]:
        raise FPSelectiveDistillationError("matched top-k inputs disagree with foreground anchors")
    selected = torch.zeros_like(reference_mask, dtype=torch.bool)
    for image_index in range(int(assignment.fg_mask.shape[0])):
        positions = (foreground[:, 0] == image_index).nonzero(as_tuple=False).flatten()
        count = int(reference_mask[positions].sum().item())
        if count == 0:
            continue
        order = torch.argsort(values[positions].detach().float(), descending=True, stable=True)
        selected[positions[order[:count]]] = True
    return selected


def per_image_random_matched_mask(
    assignment: B0Assignment,
    reference_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Randomly select each image's P3 dose without touching the global RNG."""

    values = torch.rand(reference_mask.shape, device=reference_mask.device, generator=generator)
    return per_image_matched_topk_mask(values, assignment, reference_mask)


def teacher_confidence_on_foreground(
    teacher_prediction: Mapping[str, Any],
    assignment: B0Assignment,
) -> torch.Tensor:
    scores = teacher_prediction.get("scores")
    if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
        raise FPSelectiveDistillationError("teacher raw scores must be [B,nc,A]")
    if tuple(scores.shape[::2]) != (assignment.fg_mask.shape[0], assignment.fg_mask.shape[1]):
        raise FPSelectiveDistillationError("teacher scores disagree with assignment anchors")
    confidence = scores.detach().float().sigmoid().amax(dim=1)
    return confidence[assignment.fg_mask]


def probability_target_dfl_kl(
    target_probability: torch.Tensor,
    student_dfl: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Target-score-weighted DFL KL when the target is already a distribution."""

    if target_probability.shape != student_dfl.shape or target_probability.ndim != 3:
        raise FPSelectiveDistillationError("DFL probability/student shapes must match [FG,4,R]")
    if weights.shape != mask.shape or weights.ndim != 1 or weights.shape[0] != student_dfl.shape[0]:
        raise FPSelectiveDistillationError("DFL probability weight/mask shape mismatch")
    if not bool(mask.any()):
        return student_dfl.float().sum() * 0.0
    target = target_probability.detach().float()
    student_log = functional.log_softmax(student_dfl.float(), dim=-1)
    per_anchor = (target * (target.clamp_min(torch.finfo(torch.float32).tiny).log() - student_log)).sum(-1).mean(-1)
    selected_weight = weights.float() * mask.float()
    return float(batch_size) * (per_anchor * selected_weight).sum() / selected_weight.sum().clamp_min(
        torch.finfo(torch.float32).eps
    )


def gt_dfl_probability(assignment: B0Assignment, *, reg_max: int) -> torch.Tensor:
    """Build Ultralytics' two-neighbour-bin GT DFL target on student FG anchors."""

    from ultralytics.utils.tal import bbox2dist

    target_grid = assignment.target_bboxes.float() / assignment.stride_tensor
    distances = bbox2dist(assignment.anchor_points, target_grid, reg_max - 1)[assignment.fg_mask].float()
    lower = distances.long()
    upper = lower + 1
    upper_weight = distances - lower.float()
    lower_weight = 1.0 - upper_weight
    target = torch.zeros((*distances.shape, reg_max), device=distances.device, dtype=torch.float32)
    target.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return target


def combine_fp_loss(native_loss: torch.Tensor, auxiliary_loss: torch.Tensor, *, coefficient: float = 0.1) -> torch.Tensor:
    if coefficient < 0:
        raise FPSelectiveDistillationError("auxiliary coefficient must be non-negative")
    return native_loss + float(coefficient) * auxiliary_loss


def _density(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item()) if mask.numel() else 0.0


def selective_fp_criterion(
    criterion: Any,
    student_prediction: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    arm: str,
    optical_teacher: torch.nn.Module | None = None,
    sar_teacher: torch.nn.Module | None = None,
    coefficient: float = 0.1,
    random_generator: torch.Generator | None = None,
) -> FPSelectiveResult:
    """Run one FP-v2 loss using the student's sole native assignment."""

    if arm not in FP_ARMS:
        raise FPSelectiveDistillationError(f"unknown FP-v2 arm {arm!r}")
    native = native_student_loss(criterion, student_prediction, batch)
    zero = native.native_loss * 0.0
    empty_stats = {"mO_density": 0.0, "selected_density": 0.0, "p3_density": 0.0, "empty_fg": 1.0}
    if arm == "native" or not bool(native.assignment.fg_mask.any()):
        return FPSelectiveResult(native.native_loss, zero, native.native_loss, native.detached_loss, empty_stats)
    parsed_s = native.parsed_prediction
    reg_max = criterion_reg_max(criterion)
    student_dfl = dfl_logits(parsed_s, reg_max=reg_max)[native.assignment.fg_mask]
    weights = native.assignment.target_score_weights
    batch_size = int(batch["img"].shape[0])
    with torch.no_grad():
        p3 = p3_foreground_mask(parsed_s, native.assignment)

    if arm == "p3_gt":
        target_probability = gt_dfl_probability(native.assignment, reg_max=reg_max)
        auxiliary = probability_target_dfl_kl(
            target_probability, student_dfl, weights, p3, batch_size=batch_size
        )
        total = combine_fp_loss(native.native_loss, auxiliary, coefficient=coefficient)
        return FPSelectiveResult(
            native.native_loss,
            auxiliary,
            total,
            native.detached_loss,
            {"mO_density": 0.0, "selected_density": _density(p3), "p3_density": _density(p3), "empty_fg": 0.0},
        )

    parsed_o = None
    optical_dfl = None
    if arm in OPTICAL_TEACHER_ARMS and arm != "p3_shuffled":
        if optical_teacher is None or "eo_img" not in batch:
            raise FPSelectiveDistillationError(f"{arm} requires a paired optical teacher batch")
        parsed_o = frozen_teacher_prediction(optical_teacher, batch["eo_img"])
        optical_dfl = dfl_logits(parsed_o, reg_max=reg_max)[native.assignment.fg_mask]

    if arm == "p3_shuffled":
        if optical_teacher is None or "shuffled_eo_img" not in batch:
            raise FPSelectiveDistillationError("p3_shuffled requires a shuffled optical teacher batch")
        parsed_shuffled = frozen_teacher_prediction(optical_teacher, batch["shuffled_eo_img"])
        target_dfl = dfl_logits(parsed_shuffled, reg_max=reg_max)[native.assignment.fg_mask]
    elif arm == "p3_same_modal":
        if sar_teacher is None:
            raise FPSelectiveDistillationError("p3_same_modal requires the frozen SAR teacher")
        parsed_sar_teacher = frozen_teacher_prediction(sar_teacher, batch["img"])
        target_dfl = dfl_logits(parsed_sar_teacher, reg_max=reg_max)[native.assignment.fg_mask]
    else:
        if optical_dfl is None:
            raise FPSelectiveDistillationError(f"{arm} has no optical DFL target")
        target_dfl = optical_dfl

    loc_s_mask = None
    m_o = None
    if arm in _MO_ARMS or arm == "p3_fp_hard_dose":
        with torch.no_grad():
            loc_s_mask = per_anchor_localization_error(criterion, parsed_s, native.assignment)
    if arm in _MO_ARMS:
        if parsed_o is None or loc_s_mask is None:
            raise FPSelectiveDistillationError(f"{arm} requires paired optical localization")
        with torch.no_grad():
            loc_o = per_anchor_localization_error(criterion, parsed_o, native.assignment)
            m_o = optical_advantage_mask(loc_s_mask, loc_o)

    if arm == "all_fg":
        selected = torch.ones_like(p3)
    elif arm in {"p3", "p3_shuffled", "p3_same_modal"}:
        selected = p3
    elif arm == "p4p5":
        selected = ~p3
    elif arm == "p3_random_dose":
        if random_generator is None:
            raise FPSelectiveDistillationError("p3_random_dose requires an isolated random generator")
        selected = per_image_random_matched_mask(native.assignment, p3, generator=random_generator)
    elif arm == "p3_fp_hard_dose":
        assert loc_s_mask is not None
        selected = per_image_matched_topk_mask(loc_s_mask, native.assignment, p3)
    elif arm == "p3_conf_dose":
        if sar_teacher is None:
            raise FPSelectiveDistillationError("p3_conf_dose requires the frozen SAR teacher")
        parsed_sar_teacher = frozen_teacher_prediction(sar_teacher, batch["img"])
        confidence = teacher_confidence_on_foreground(parsed_sar_teacher, native.assignment)
        selected = per_image_matched_topk_mask(confidence, native.assignment, p3)
    elif arm == "mo":
        assert m_o is not None
        selected = m_o
    elif arm == "p3_mo":
        assert m_o is not None
        selected = p3 & m_o
    elif arm == "legacy_conf":
        assert m_o is not None
        if sar_teacher is None:
            raise FPSelectiveDistillationError("legacy_conf requires the frozen SAR teacher")
        parsed_sar_teacher = frozen_teacher_prediction(sar_teacher, batch["img"])
        confidence = teacher_confidence_on_foreground(parsed_sar_teacher, native.assignment)
        selected = per_image_matched_topk_mask(confidence, native.assignment, m_o)
    elif arm == "mo_fp_hard":
        assert loc_s_mask is not None and m_o is not None
        selected = per_image_matched_topk_mask(loc_s_mask, native.assignment, m_o)
    elif arm == "mo_shuffled":
        assert m_o is not None
        if "shuffled_eo_img" not in batch:
            raise FPSelectiveDistillationError("mo_shuffled requires shuffled EO images")
        parsed_shuffled = frozen_teacher_prediction(optical_teacher, batch["shuffled_eo_img"])
        target_dfl = dfl_logits(parsed_shuffled, reg_max=reg_max)[native.assignment.fg_mask]
        selected = m_o
    elif arm == "mo_gt":
        assert loc_s_mask is not None and m_o is not None
        selected = m_o
        loc_s_grad = per_anchor_localization_error(criterion, parsed_s, native.assignment)
        selected_weight = weights.float() * selected.float()
        auxiliary = float(batch_size) * (loc_s_grad * selected.float()).sum() / selected_weight.sum().clamp_min(
            torch.finfo(torch.float32).eps
        )
        total = combine_fp_loss(native.native_loss, auxiliary, coefficient=coefficient)
        return FPSelectiveResult(
            native.native_loss,
            auxiliary,
            total,
            native.detached_loss,
            {"mO_density": _density(m_o), "selected_density": _density(selected), "p3_density": _density(p3), "empty_fg": 0.0},
        )
    elif arm == "apr_dfl":
        assert m_o is not None
        if sar_teacher is None:
            raise FPSelectiveDistillationError("apr_dfl requires the frozen SAR teacher")
        parsed_sar_teacher = frozen_teacher_prediction(sar_teacher, batch["img"])
        sar_dfl = dfl_logits(parsed_sar_teacher, reg_max=reg_max)[native.assignment.fg_mask]
        target_probability = 0.5 * functional.softmax(sar_dfl.detach().float(), dim=-1) + 0.5 * functional.softmax(
            optical_dfl.detach().float(), dim=-1
        )
        selected = m_o
        auxiliary = probability_target_dfl_kl(target_probability, student_dfl, weights, selected, batch_size=batch_size)
        total = combine_fp_loss(native.native_loss, auxiliary, coefficient=coefficient)
        return FPSelectiveResult(
            native.native_loss,
            auxiliary,
            total,
            native.detached_loss,
            {"mO_density": _density(m_o), "selected_density": _density(selected), "p3_density": _density(p3), "empty_fg": 0.0},
        )
    else:  # pragma: no cover - arm exhaustiveness is checked above
        raise FPSelectiveDistillationError(f"unhandled FP-v2 arm {arm!r}")

    auxiliary = masked_dfl_kl(target_dfl, student_dfl, weights, selected, batch_size=batch_size)
    total = combine_fp_loss(native.native_loss, auxiliary, coefficient=coefficient)
    return FPSelectiveResult(
        native.native_loss,
        auxiliary,
        total,
        native.detached_loss,
        {"mO_density": _density(m_o) if m_o is not None else 0.0, "selected_density": _density(selected), "p3_density": _density(p3), "empty_fg": 0.0},
    )


class FPSelectiveLoss:
    """Ultralytics criterion façade with compact selector statistics."""

    def __init__(
        self,
        native: Any,
        *,
        arm: str,
        optical_teacher: torch.nn.Module | None,
        sar_teacher: torch.nn.Module | None,
    ) -> None:
        self.native = native
        self.arm = arm
        self.optical_teacher = optical_teacher
        self.sar_teacher = sar_teacher
        self.stats_rows: list[dict[str, float]] = []
        self.random_generators: dict[str, torch.Generator] = {}

    def __call__(self, prediction: Any, batch: dict[str, torch.Tensor]):
        if self.arm in PAIRED_DATA_ARMS and "eo_img" not in batch:
            return self.native(prediction, batch)
        random_generator = None
        if self.arm == "p3_random_dose":
            device = batch["img"].device
            key = str(device)
            if key not in self.random_generators:
                self.random_generators[key] = torch.Generator(device=device).manual_seed(42)
            random_generator = self.random_generators[key]
        result = selective_fp_criterion(
            self.native,
            prediction,
            batch,
            arm=self.arm,
            optical_teacher=self.optical_teacher,
            sar_teacher=self.sar_teacher,
            random_generator=random_generator,
        )
        self.stats_rows.append(result.stats)
        detached = dict(result.detached_loss)
        detached["fp_aux"] = result.auxiliary_loss.detach()
        return result.total_loss, detached
