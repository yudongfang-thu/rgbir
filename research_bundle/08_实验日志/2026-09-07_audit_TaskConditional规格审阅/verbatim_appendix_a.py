"""Minimal DFL-KD kernel for already aligned teacher/student anchor grids.

This is NOT a complete RGB-IR trainer or a new published method. The caller
must establish identical physical coordinates, anchor locations/order, strides,
bin semantics and class/GT identities. Equal tensor shapes alone are not enough.

Object-balanced use: give each of object i's eligible anchors weight
w_i / number_of_eligible_anchors_i, and use the pre-selection base-object count
as normalizer. Each anchor must have one declared object assignment. Do not
normalize by the sum of the gates unless that is a separately declared design.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F


def aligned_dfl_kd(
    student_raw: Tensor,
    teacher_raw: Tensor,
    anchor_weight: Tensor,
    *,
    normalizer: float,
    temperature: float = 2.0,
) -> Tensor:
    """Return a scalar, without detector batch scaling or a global KD lambda.

    Args:
        student_raw, teacher_raw: raw DFL logits [B, 4 * reg_max, A].
        anchor_weight: nonnegative detached weights [B, A], already containing
            any per-object anchor-count correction and reliability gate.
        normalizer: positive, predeclared denominator (e.g. max(1, |E_loc|)).
        temperature: shared softmax temperature; use a fixed value per protocol.
    """
    if student_raw.ndim != 3 or student_raw.shape != teacher_raw.shape:
        raise ValueError('Expected equal teacher/student shapes [B, 4*R, A].')
    b, channels, a = student_raw.shape
    if channels % 4 or channels // 4 < 2:
        raise ValueError('DFL requires 4 * reg_max channels and reg_max >= 2.')
    if anchor_weight.shape != (b, a):
        raise ValueError('anchor_weight must be [B, A].')
    if teacher_raw.device != student_raw.device or anchor_weight.device != student_raw.device:
        raise ValueError('All tensors must be on the same device.')
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('normalizer must be finite and positive.')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive.')
    for tensor in (student_raw, teacher_raw, anchor_weight):
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Inputs must be finite floating-point tensors.')
    if bool((anchor_weight < 0).any()):
        raise ValueError('anchor_weight must be nonnegative.')

    r = channels // 4
    # [B, 4R, A] -> [B, A, 4, R]; compare bins on the last axis, in FP32.
    s = student_raw.float().reshape(b, 4, r, a).permute(0, 3, 1, 2)
    t = teacher_raw.detach().float().reshape(b, 4, r, a).permute(0, 3, 1, 2)
    log_s = F.log_softmax(s / temperature, dim=-1)
    p_t = F.softmax(t / temperature, dim=-1)
    per_anchor = F.kl_div(log_s, p_t, reduction='none').sum(-1).mean(-1)
    per_anchor = per_anchor * temperature**2
    return (per_anchor * anchor_weight.detach().float()).sum() / normalizer


def combine_detection_loss(
    native_total: Tensor,
    classification_kd: Tensor,
    localization_kd: Tensor,
    *,
    batch_size: int,
    classification_weight: float,
    localization_weight: float,
) -> Tensor:
    """For the repo's pinned native-loss convention, add each scalar only once."""
    if classification_kd.numel() != 1 or localization_kd.numel() != 1:
        raise ValueError('KD terms must each contain one scalar.')
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    if any(not math.isfinite(v) or v < 0 for v in (classification_weight, localization_weight)):
        raise ValueError('KD weights must be finite and nonnegative.')
    return native_total.sum() + batch_size * (
        classification_weight * classification_kd.reshape(())
        + localization_weight * localization_kd.reshape(())
    )
