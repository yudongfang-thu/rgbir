"""CPU-testable kernels for an implementation specification, not a trainer.

C1 inputs are per-object/per-level pooled raw logit differences (before T).
L1 inputs must ALREADY have verified physical anchor/stride/side correspondence.
This file cannot establish correspondence, select samples or validate AP gains.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _finite_float(tensor: Tensor, name: str) -> None:
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f'{name} must be finite floating point')


def pool_relative_logits(scores: Tensor, foreground: Tensor, background: Tensor,
                         minimum_foreground: int = 1,
                         minimum_background: int = 4) -> tuple[Tensor, Tensor]:
    """One scale: [C,A], bool[M,A], bool[M,A] -> delta[M,C], valid[M].

    A different modality may have different region masks/counts. The caller
    supplies matched OBJECTS and the exact frozen OEv1 background exclusions.
    No temperature is applied here. Student foreground AND background get grads.
    """
    _finite_float(scores, 'scores')
    if scores.ndim != 2 or scores.shape[0] < 1:
        raise ValueError('scores must be [C,A], C>=1')
    if foreground.dtype != torch.bool or background.dtype != torch.bool:
        raise ValueError('region masks must be bool')
    if foreground.ndim != 2 or foreground.shape != background.shape or foreground.shape[1] != scores.shape[1]:
        raise ValueError('masks must have matching [M,A] shapes')
    if foreground.device != scores.device or background.device != scores.device:
        raise ValueError('masks and scores must share device')
    if minimum_foreground < 1 or minimum_background < 1:
        raise ValueError('minimum region counts must be positive')
    if bool((foreground & background).any()):
        raise ValueError('foreground and background overlap')
    valid = (foreground.sum(-1) >= minimum_foreground) & (background.sum(-1) >= minimum_background)
    z = scores.float()
    values = []
    for fg, bg, ok in zip(foreground, background, valid):
        if bool(ok):
            f, b = z[:, fg], z[:, bg]
            values.append(f.logsumexp(-1) - math.log(f.shape[-1])
                          - b.logsumexp(-1) + math.log(b.shape[-1]))
        else:
            values.append(z.sum(-1) * 0.0)
    if not values:
        return z[:, :0].T, valid
    return torch.stack(values), valid


def classification_relative_kd(student_delta: Tensor, teacher_delta: Tensor,
                               valid_levels: Tensor, selected: Tensor,
                               labels: Tensor, temperature: float = 2.0,
                               raw_teacher_clip: float = 16.0,
                               off_target_weight: float = 0.25) -> Tensor:
    """C1: [M,L,C] deltas for the entire pre-quality base E_C.

    Per-object: mean(valid levels)[KL_y + eta*mean(KL_non_target)].
    Sum selected objects / M. Target coefficient is one even if eta changes.
    M must be the ORIGINAL base count, not the count after quality selection.
    Teacher target is clipped before division by temperature; student is not.
    """
    _positive(temperature, 'temperature')
    _positive(raw_teacher_clip, 'raw_teacher_clip')
    if not math.isfinite(off_target_weight) or off_target_weight < 0:
        raise ValueError('off_target_weight must be finite and nonnegative')
    _finite_float(student_delta, 'student_delta')
    _finite_float(teacher_delta, 'teacher_delta')
    if student_delta.ndim != 3 or student_delta.shape != teacher_delta.shape:
        raise ValueError('deltas must share [M,L,C] shape')
    m, levels, classes = student_delta.shape
    if levels < 1 or classes < 1 or teacher_delta.device != student_delta.device:
        raise ValueError('invalid dimensions/device')
    if valid_levels.shape != (m, levels) or valid_levels.dtype != torch.bool:
        raise ValueError('valid_levels must be bool[M,L]')
    if selected.shape != (m,) or selected.dtype != torch.bool:
        raise ValueError('selected must be bool[M]')
    if labels.shape != (m,) or labels.dtype != torch.long:
        raise ValueError('labels must be long[M]')
    if any(t.device != student_delta.device for t in (valid_levels, selected, labels)):
        raise ValueError('selection, labels and deltas must share device')
    if m and not bool(((labels >= 0) & (labels < classes)).all()):
        raise ValueError('class index out of range')
    if bool((selected & ~valid_levels.any(-1)).any()):
        raise ValueError('selected object has no valid levels')
    if m == 0:
        return student_delta.float().sum() * 0.0
    s = student_delta.float() / temperature
    t = teacher_delta.detach().float().clamp(-raw_teacher_clip, raw_teacher_clip) / temperature
    lp, lq = F.logsigmoid(t), F.logsigmoid(-t)
    p, q = lp.exp(), lq.exp()
    binary_kl = p * (lp - F.logsigmoid(s)) + q * (lq - F.logsigmoid(-s))
    target_mask = F.one_hot(labels, classes).to(binary_kl.dtype)[:, None, :]
    target = (binary_kl * target_mask).sum(-1)
    if classes > 1:
        non_target = (binary_kl * (1.0 - target_mask)).sum(-1) / (classes - 1)
        point_loss = target + off_target_weight * non_target
    else:
        point_loss = target  # No nonexistent non-target classes for LLVIP.
    per_object = (point_loss * valid_levels).sum(-1) / valid_levels.sum(-1).clamp_min(1)
    return (per_object * selected).sum() * (temperature ** 2 / m)


def localization_from_target(student_logits: Tensor, target_probability: Tensor,
                             base_count: int, temperature: float = 2.0) -> Tensor:
    """[K,4,R] selected single-anchor objects; target already at temperature T.

    The enclosing trainer, not this kernel, must verify geometry and base_count.
    """
    _positive(temperature, 'temperature')
    if not isinstance(base_count, int) or base_count < 0:
        raise ValueError('base_count must be a nonnegative integer')
    if student_logits.ndim != 3 or student_logits.shape[1] != 4 or student_logits.shape[-1] < 2:
        raise ValueError('expected [K,4,R], R>=2')
    if student_logits.shape != target_probability.shape or student_logits.device != target_probability.device:
        raise ValueError('student and target shape/device mismatch')
    if student_logits.shape[0] > base_count:
        raise ValueError('selected objects exceed original base_count')
    _finite_float(student_logits, 'student_logits')
    _finite_float(target_probability, 'target_probability')
    target = target_probability.detach().float()
    if bool((target < 0).any()) or not torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1)), atol=1e-6, rtol=1e-6):
        raise ValueError('target must be normalized and nonnegative')
    return F.kl_div(F.log_softmax(student_logits.float() / temperature, -1), target,
                    reduction='sum') * temperature ** 2 / (4 * max(1, base_count))


def aligned_dfl_kd(student_logits: Tensor, teacher_logits: Tensor,
                   base_count: int, temperature: float = 2.0) -> Tensor:
    _positive(temperature, 'temperature')
    _finite_float(teacher_logits, 'teacher_logits')
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('teacher/student localization shapes differ')
    target = F.softmax(teacher_logits.detach().float() / temperature, -1)
    return localization_from_target(student_logits, target, base_count, temperature)


def gt_dfl_target(distances: Tensor, reg_max: int, temperature: float = 2.0,
                  support_epsilon: float = 0.01) -> Tensor:
    """RGB GT distances in bins, BEFORE clamp. Two-bin q, then q**(1/T).

    This supplies the same-mask GT content control, not ordinary native DFL.
    """
    _positive(temperature, 'temperature')
    _finite_float(distances, 'distances')
    if distances.ndim != 2 or distances.shape[-1] != 4 or reg_max < 2:
        raise ValueError('expected distances[K,4] and reg_max>=2')
    if not 0 < support_epsilon < 1:
        raise ValueError('support_epsilon must be in (0,1)')
    d = distances.detach().float()
    if not bool(((d >= 0) & (d <= reg_max - 1 - support_epsilon)).all()):
        raise ValueError('GT distances outside unclamped support')
    lo = d.floor().long()
    fraction = d - lo
    target = d.new_zeros((*d.shape, reg_max))
    target.scatter_add_(-1, lo.unsqueeze(-1), (1 - fraction).unsqueeze(-1))
    target.scatter_add_(-1, (lo + 1).unsqueeze(-1), fraction.unsqueeze(-1))
    softened = target.pow(1 / temperature)
    return softened / softened.sum(-1, keepdim=True)


def combine_single_task(native_total: Tensor, kd: Tensor, actual_batch: int,
                        weight: float, task: str) -> Tensor:
    """API deliberately has only ONE KD scalar: no C+L path in this phase."""
    allowed = {'N', 'C0', 'C1', 'C1_y', 'L1', 'L_GT'}
    if task not in allowed:
        raise ValueError('unsupported task; fusion/routing not implemented')
    if not isinstance(actual_batch, int) or actual_batch < 1:
        raise ValueError('actual_batch must be positive integer')
    if not math.isfinite(weight) or weight < 0 or kd.ndim != 0:
        raise ValueError('weight must be nonnegative; KD must be scalar')
    if task == 'N' and weight != 0:
        raise ValueError('N must have exactly zero KD weight')
    _finite_float(native_total, 'native_total')
    _finite_float(kd, 'kd')
    if native_total.device != kd.device:
        raise ValueError('native and KD device mismatch')
    return native_total.sum() + actual_batch * weight * kd
