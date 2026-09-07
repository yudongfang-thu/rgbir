"""Object/scale/class relative-logit KD; independent C1 and C1_y only.

This does not turn relative evidence into calibrated detector confidence and
does not assert that non-target class content is safe to transfer. Selection is
fixed by C0. Native loss and actual batch-size scaling belong to the trainer.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    from .selection_adapter import ClassificationSelection, build_classification_selection, pool_relative_logits, evidence_config, SELECTION_VERSION, LEGACY_SOURCE
except ImportError:
    from selection_adapter import ClassificationSelection, build_classification_selection, pool_relative_logits, evidence_config, SELECTION_VERSION, LEGACY_SOURCE


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")


def _classification_terms(student_delta, teacher_delta, valid_levels, selected, labels,
                          temperature=2., raw_teacher_clip=16., off_target_weight=.25):
    _positive(temperature, "temperature")
    _positive(raw_teacher_clip, "raw_teacher_clip")
    if not math.isfinite(off_target_weight) or off_target_weight < 0:
        raise ValueError("off_target_weight must be finite and nonnegative")
    for name, tensor in (("student_delta", student_delta), ("teacher_delta", teacher_delta)):
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(name + " must be finite floating point")
    if student_delta.ndim != 3 or student_delta.shape != teacher_delta.shape:
        raise ValueError("deltas must have the same [M,L,C] shape")
    m, levels, classes = student_delta.shape
    if levels < 1 or classes < 1 or student_delta.device != teacher_delta.device:
        raise ValueError("Invalid delta dimensions or device")
    if valid_levels.shape != (m, levels) or valid_levels.dtype != torch.bool:
        raise ValueError("valid_levels must be bool[M,L]")
    if selected.shape != (m,) or selected.dtype != torch.bool:
        raise ValueError("selected must be bool[M]")
    if labels.shape != (m,) or labels.dtype != torch.long:
        raise ValueError("labels must be long[M]")
    if any(t.device != student_delta.device for t in (valid_levels, selected, labels)):
        raise ValueError("Selection and delta devices differ")
    if m and not bool(((labels >= 0) & (labels < classes)).all()):
        raise ValueError("Class index out of range")
    if bool((selected & ~valid_levels.any(-1)).any()):
        raise ValueError("Selected object has no valid level")
    s = student_delta.float() / temperature
    t = teacher_delta.detach().float().clamp(-raw_teacher_clip, raw_teacher_clip) / temperature
    lp, lq = F.logsigmoid(t), F.logsigmoid(-t)
    p, q = lp.exp(), lq.exp()
    kl = p * (lp - F.logsigmoid(s)) + q * (lq - F.logsigmoid(-s))
    if m:
        target = kl.gather(-1, labels[:, None, None].expand(-1, levels, 1)).squeeze(-1)
    else:
        target = kl.sum(-1)
    if classes > 1:
        # Summing the explicit other-channel mask avoids cancellation from
        # subtracting a large target term from an almost-equal total.
        mask = F.one_hot(labels, classes).to(kl.dtype)[:, None, :]
        non_target = (kl * (1. - mask)).sum(-1) / (classes-1)
    else:
        non_target = kl.sum(-1) * 0.
    reduction = valid_levels.float() / valid_levels.sum(-1).clamp_min(1)[:, None]
    reduction = reduction * selected[:, None] * (temperature ** 2 / max(1, m))
    target_loss = (target * reduction).sum()
    off_loss_unit = (non_target * reduction).sum()
    loss = target_loss + off_target_weight * off_loss_unit
    # Keep the full raw-score path attached when M==0, not a constant Tensor.
    if m == 0:
        loss = student_delta.float().sum() * 0.
        target_loss = loss
        off_loss_unit = loss
    return dict(loss=loss, target_loss=target_loss, off_target_loss_unit=off_loss_unit,
                kl=kl, target=target, non_target=non_target, reduction=reduction,
                teacher_probability=p, teacher_log_probability=lp, teacher_log_negative=lq)


def classification_relative_kd(student_delta: Tensor, teacher_delta: Tensor,
                               valid_levels: Tensor, selected: Tensor, labels: Tensor,
                               temperature: float = 2., raw_teacher_clip: float = 16.,
                               off_target_weight: float = .25) -> Tensor:
    """Pure C1 kernel. First dimension MUST contain all pre-teacher E_C rows."""
    return _classification_terms(student_delta, teacher_delta, valid_levels, selected,
                                 labels, temperature, raw_teacher_clip, off_target_weight)["loss"]


def classification_loss_components(selection: ClassificationSelection, temperature=2.,
                                   raw_teacher_clip=16., off_target_weight=.25):
    """Differentiable target/non-target components for fixed-dose diagnostics.

    Shared-parameter component gradients may cancel. Removing the non-target
    term does not necessarily reduce the norm of their sum.
    """
    terms = _classification_terms(selection.student_delta, selection.teacher_delta,
        selection.valid_levels, selection.selected, selection.labels,
        temperature, raw_teacher_clip, off_target_weight)
    return dict(loss=terms["loss"], target_loss=terms["target_loss"],
                off_target_loss_unit=terms["off_target_loss_unit"],
                off_target_loss=off_target_weight * terms["off_target_loss_unit"],
                c0_loss=selection.c0_loss)


def classification_loss_from_selection(selection: ClassificationSelection, *,
                                       temperature=2., raw_teacher_clip=16., off_target_weight=.25,
                                       return_records=False):
    terms = _classification_terms(selection.student_delta, selection.teacher_delta,
        selection.valid_levels, selection.selected, selection.labels,
        temperature, raw_teacher_clip, off_target_weight)
    loss = terms["loss"]
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite relative class KD")
    m, levels, nc = selection.student_delta.shape
    mask = selection.valid_levels & selection.selected[:, None]
    stats = dict(selection.c0_stats)
    stats.update(arm="C1_y" if off_target_weight == 0 else "C1",
                 selection_version=SELECTION_VERSION, selection_source=str(LEGACY_SOURCE),
                 base_object_ids=list(selection.base_object_ids),
                 base_to_matched=selection.base_to_matched.tolist(),
                 matched_to_base=selection.matched_to_base.tolist(),
                 valid_levels=selection.valid_levels.tolist(),
                 temperature=temperature, raw_teacher_clip=raw_teacher_clip,
                 off_target_weight=off_target_weight, class_count=nc,
                 supervised_class_channels=nc if off_target_weight > 0 else 1,
                 payload_shape=[m, levels, nc], loss_unweighted=float(loss.detach()),
                 c0_loss_unweighted=float(selection.c0_loss.detach()),
                 target_loss_unweighted=float(terms["target_loss"].detach()),
                 off_target_loss_unit=float(terms["off_target_loss_unit"].detach()),
                 off_target_loss_unweighted=float((off_target_weight * terms["off_target_loss_unit"]).detach()),
                 normalizer=max(1, m), selected_valid_level_count=int(mask.sum()),
                 relative_evidence_not_detector_confidence=True,
                 student_background_has_gradient=True,
                 teacher_gradient_enabled=False,
                 target_binary_entropy_semantics="relative_evidence")
    # Replace inherited C0 clipping/entropy with this arm's actual target fields.
    with torch.no_grad():
        p, lp, lq = (terms[key].detach() for key in ("teacher_probability", "teacher_log_probability", "teacher_log_negative"))
        entropy = -(p * lp + (1-p) * lq)
        target_indices = selection.labels[:, None, None].expand(-1, levels, 1)
        target_raw = selection.teacher_delta.gather(-1, target_indices).squeeze(-1) if m else selection.teacher_delta.sum(-1)
        selected_values = selection.teacher_delta[mask] if off_target_weight > 0 else target_raw[mask]
        stats["target_clipped_count"] = int((selected_values.abs() > raw_teacher_clip).sum())
        stats["target_clipping_population"] = int(selected_values.numel())
        stats["target_clipped_fraction"] = (stats["target_clipped_count"] / selected_values.numel()) if selected_values.numel() else None
        target_entropy = entropy.gather(-1, target_indices).squeeze(-1) if m else entropy.sum(-1)
        active_entropy = entropy[mask] if off_target_weight > 0 else target_entropy[mask]
        stats["target_binary_entropy_selected"] = float(active_entropy.mean()) if bool(mask.any()) else None
        stats["gt_channel_relative_entropy_selected"] = float(target_entropy[mask].mean()) if bool(mask.any()) else None
        target_p = p.gather(-1, target_indices).squeeze(-1) if m else p.sum(-1)
        stats["target_gap_to_positive_selected"] = float((1-target_p[mask]).mean()) if bool(mask.any()) else None
        stats["per_level"] = []
        for li, level in enumerate(selection.config.levels):
            kmask = mask[:, li]
            stats["per_level"].append(dict(level=int(level), selected_count=int(kmask.sum()),
                target_loss_contribution=float((terms["target"][:, li] * terms["reduction"][:, li]).sum()),
                non_target_loss_contribution=float((terms["non_target"][:, li] * terms["reduction"][:, li]).sum() * off_target_weight),
                target_kl_mean=float(terms["target"][kmask, li].mean()) if bool(kmask.any()) else None,
                non_target_kl_mean=float(terms["non_target"][kmask, li].mean()) if bool(kmask.any()) else None))
        stats["per_class"] = [dict(class_id=c, base_count=int((selection.labels == c).sum()),
            eligible_count=int(((selection.labels == c) & selection.eligible).sum()),
            selected_count=int(((selection.labels == c) & selection.selected).sum())) for c in range(nc)]
        for name, tensor in (("student", selection.student_delta.detach()), ("teacher", selection.teacher_delta),
                             ("reference", selection.reference_delta)):
            stats[name + "_delta_class_mean_base_valid"] = tensor[selection.valid_levels].mean(0).tolist() if bool(selection.valid_levels.any()) else None
            stats[name + "_delta_class_mean_selected_valid"] = tensor[mask].mean(0).tolist() if bool(mask.any()) else None
        level_positions = {level: i for i, level in enumerate(selection.config.levels)}
        region_active = [mask[region.base_rows, level_positions[region.level]] for region in selection.regions]
        overlaps = [region.rgb_other_gt_fraction[active] for region, active in zip(selection.regions, region_active)]
        toverlaps = [region.teacher_other_gt_fraction[active] for region, active in zip(selection.regions, region_active)]
        rgb_overlap = torch.cat(overlaps) if overlaps else selection.teacher_delta.new_empty(0)
        teacher_overlap = torch.cat(toverlaps) if toverlaps else selection.teacher_delta.new_empty(0)
        stats["rgb_other_gt_foreground_fraction_selected_level_mean"] = float(rgb_overlap.mean()) if len(rgb_overlap) else None
        stats["teacher_other_gt_foreground_fraction_selected_level_mean"] = float(teacher_overlap.mean()) if len(teacher_overlap) else None
        if return_records:
            stats["base_records"] = [dict(base_index=i, object_id=selection.base_object_ids[i],
                matched_index=int(selection.base_to_matched[i]), class_id=int(selection.labels[i]),
                quality=float(selection.quality[i]), eligible=bool(selection.eligible[i]), selected=bool(selection.selected[i]),
                valid_levels=selection.valid_levels[i].tolist(), student_delta=selection.student_delta[i].detach().tolist(),
                teacher_delta=selection.teacher_delta[i].tolist(), reference_delta=selection.reference_delta[i].tolist()) for i in range(m)]
    return loss, stats


def classification_logit_loss(student, teacher, reference, batch, *, strides=(8, 16, 32),
                               config=None, selection_seed=0, temperature=2.,
                               raw_teacher_clip=16., off_target_weight=.25,
                               selection: Optional[ClassificationSelection] = None,
                               return_records=False):
    if selection is None:
        selection = build_classification_selection(student, teacher, reference, batch,
            strides=strides, config=config, selection_seed=selection_seed)
    else:
        selection.assert_source(student, teacher, reference, batch)
        if config is not None and asdict(evidence_config(config)) != asdict(selection.config):
            raise ValueError("Cached classification selection uses a different evidence configuration")
    return classification_loss_from_selection(selection, temperature=temperature,
        raw_teacher_clip=raw_teacher_clip, off_target_weight=off_target_weight,
        return_records=return_records)
