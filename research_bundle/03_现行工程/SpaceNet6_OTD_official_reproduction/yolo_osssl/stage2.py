"""The small Stage-II DCODFL-QAT-v1 loss core.

All tensor logic is independent of a trainer.  The entrypoint supplies the
native criterion and captures its one canonical q-student assigner invocation.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

from .raw_head_abi import RawHeadABIError, raw_box_logits


class Stage2Error(RuntimeError):
    """Raised when Stage-II would silently change native assignment semantics."""


@dataclass(frozen=True)
class Assignment:
    target_bboxes: torch.Tensor
    target_scores: torch.Tensor
    fg_mask: torch.Tensor
    target_gt_idx: torch.Tensor
    anchor_points: torch.Tensor
    stride_tensor: torch.Tensor

    @property
    def fg_indices(self) -> torch.Tensor:
        return self.fg_mask.nonzero(as_tuple=False)

    @property
    def target_score_weights(self) -> torch.Tensor:
        return self.target_scores.sum(-1)[self.fg_mask]


@dataclass(frozen=True)
class NativeQResult:
    assignment: Assignment
    loss: torch.Tensor
    detached_loss: Any


@dataclass(frozen=True)
class ThresholdResult:
    tau_det: float
    n_fg: int
    n_positive: int
    quantile: float
    interpolation: str
    calibration_set: str = "train"


class AssignmentCapture(AbstractContextManager["AssignmentCapture"]):
    """Observe a native assigner call without substituting its computation."""

    def __init__(self, assigner: Any) -> None:
        self.assigner = assigner
        self.calls: list[tuple[tuple[Any, ...], Mapping[str, Any], Any]] = []
        self._owner: type[Any] | None = None
        self._original: Any = None

    def __enter__(self) -> "AssignmentCapture":
        self._owner = type(self.assigner)
        self._original = self._owner.forward

        def observed(instance: Any, *args: Any, **kwargs: Any) -> Any:
            output = self._original(instance, *args, **kwargs)
            if instance is self.assigner:
                self.calls.append((args, kwargs, output))
            return output

        self._owner.forward = observed
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        assert self._owner is not None
        self._owner.forward = self._original
        return False

    def assignment(self) -> Assignment:
        if len(self.calls) != 1:
            raise Stage2Error(f"expected exactly one q-student native assigner call, got {len(self.calls)}")
        output = self.calls[0][2]
        if not isinstance(output, (tuple, list)) or len(output) != 5:
            raise Stage2Error("native assigner output no longer has five canonical tensors")
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = output
        # Anchor geometry originates from get_assigned_targets_and_loss, not
        # from a second assigner call; it is attached by native_q_loss below.
        return Assignment(target_bboxes, target_scores, fg_mask, target_gt_idx, torch.empty(0), torch.empty(0))


def _native_result_parts(value: Any) -> tuple[tuple[Any, ...], Any, Any]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise Stage2Error("native get_assigned_targets_and_loss return ABI drift")
    assignment, losses, detached = value
    if not isinstance(assignment, (tuple, list)) or len(assignment) != 5:
        raise Stage2Error("native assigned-target ABI drift")
    return tuple(assignment), losses, detached


def native_q_loss(criterion: Any, prediction: Any, batch: Mapping[str, Any]) -> NativeQResult:
    """Run q native loss once and bind captured assigner outputs to its anchors."""

    method = getattr(criterion, "get_assigned_targets_and_loss", None)
    if not callable(method) or not hasattr(criterion, "assigner"):
        raise Stage2Error("criterion lacks Stage-II native loss/assigner seam")
    with AssignmentCapture(criterion.assigner) as capture:
        assigned, losses, detached = _native_result_parts(method(prediction, batch))
    fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned
    captured = capture.assignment()
    if (
        captured.fg_mask.shape != fg_mask.shape
        or not torch.equal(captured.fg_mask, fg_mask)
        or not torch.equal(captured.target_bboxes, target_bboxes)
    ):
        raise Stage2Error("captured assigner output disagrees with native criterion return")
    if not isinstance(losses, torch.Tensor):
        raise Stage2Error("native loss is not a Tensor")
    if isinstance(prediction, Mapping) and isinstance(prediction.get("boxes"), torch.Tensor):
        batch_size = int(prediction["boxes"].shape[0])
    else:
        image = batch.get("img")
        if not isinstance(image, torch.Tensor):
            raise Stage2Error("cannot recover native batch size from prediction or batch")
        batch_size = int(image.shape[0])
    return NativeQResult(
        Assignment(target_bboxes, captured.target_scores, fg_mask, target_gt_idx, anchor_points, stride_tensor),
        # v8DetectionLoss.loss applies this factor only after
        # get_assigned_targets_and_loss; preserve QAT-only native scale.
        losses.sum() * batch_size,
        detached,
    )


def raw_dfl_logits(prediction: Any, *, reg_max: int) -> torch.Tensor:
    raw = raw_box_logits(prediction, reg_max=reg_max)
    return raw.reshape(raw.shape[0], raw.shape[1], 4, reg_max)


def _bbox_iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    try:
        from ultralytics.utils.metrics import bbox_iou
    except ImportError as exc:  # pragma: no cover - production runtime only
        raise Stage2Error("Ultralytics bbox_iou is unavailable") from exc
    return bbox_iou(prediction, target, xywh=False, CIoU=True)


def _bbox2dist(anchor_points: torch.Tensor, target_bboxes: torch.Tensor, reg_max: int) -> torch.Tensor:
    try:
        from ultralytics.utils.tal import bbox2dist
    except ImportError as exc:  # pragma: no cover - production runtime only
        raise Stage2Error("Ultralytics bbox2dist is unavailable") from exc
    return bbox2dist(anchor_points, target_bboxes, reg_max - 1)


def per_anchor_localization_error(
    criterion: Any,
    prediction: Any,
    assignment: Assignment,
) -> torch.Tensor:
    """Return unnormalized native-weighted box+DFL scalar for every q FG anchor."""

    dfl_loss = getattr(getattr(criterion, "bbox_loss", None), "dfl_loss", None)
    reg_max = int(getattr(dfl_loss, "reg_max", 0))
    decoder = getattr(criterion, "bbox_decode", None)
    hyp = getattr(criterion, "hyp", None)
    if reg_max <= 0 or dfl_loss is None or not callable(decoder) or hyp is None:
        raise Stage2Error("criterion lost canonical DFL localization components")
    dfl = raw_dfl_logits(prediction, reg_max=reg_max)
    pred_distri = raw_box_logits(prediction, reg_max=reg_max)
    if assignment.fg_mask.sum().item() == 0:
        return torch.zeros(0, device=pred_distri.device, dtype=torch.float32)
    decoded = decoder(assignment.anchor_points, pred_distri.float())
    weights = assignment.target_score_weights.float().unsqueeze(-1)
    target_grid = assignment.target_bboxes.float() / assignment.stride_tensor
    iou = _bbox_iou(decoded[assignment.fg_mask], target_grid[assignment.fg_mask])
    box = ((1.0 - iou.float()) * weights).squeeze(-1)
    target_ltrb = _bbox2dist(assignment.anchor_points, target_grid, reg_max)
    raw_dfl = dfl_loss(
        dfl[assignment.fg_mask].reshape(-1, reg_max).float(), target_ltrb[assignment.fg_mask].reshape(-1, 4).float()
    )
    # Pinned DFLoss has already averaged the four LTRB sides and returns
    # [N_fg, 1], exactly as BboxLoss consumes before score normalization.
    dfl_per_fg = (raw_dfl * weights).squeeze(-1)
    return float(hyp.box) * box + float(hyp.dfl) * dfl_per_fg


def masks_from_localization(
    loc_q: torch.Tensor,
    loc_f: torch.Tensor,
    loc_o: torch.Tensor,
    *,
    tau_det: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (loc_q.shape == loc_f.shape == loc_o.shape):
        raise Stage2Error("q/f/o per-anchor localization vectors disagree")
    m_q = (loc_q.detach().float() - loc_f.detach().float()) > float(tau_det)
    m_o = loc_o.detach().float() < loc_f.detach().float()
    return m_q, m_o, m_q & m_o


def calibrate_tau_det(deltas: Sequence[torch.Tensor], *, calibration_set: str = "train") -> ThresholdResult:
    """Frozen positive-tail Q80 threshold; callers provide all 498 occurrences."""

    flat = [delta.detach().float().cpu().reshape(-1) for delta in deltas]
    values = torch.cat(flat) if flat else torch.zeros(0)
    positive = values[values > 0]
    if positive.numel() < 256:
        raise Stage2Error(f"Q80 calibration has insufficient positive delta anchors: {positive.numel()} < 256")
    tau = torch.quantile(positive, 0.80, interpolation="linear")
    return ThresholdResult(float(tau.item()), int(values.numel()), int(positive.numel()), 0.80, "linear", calibration_set)


def p3_calibration_delta(delta: torch.Tensor, p3_mask: torch.Tensor) -> torch.Tensor:
    """Restrict threshold calibration to foreground anchors from P3."""

    if delta.ndim != 1 or p3_mask.shape != delta.shape or p3_mask.dtype != torch.bool:
        raise Stage2Error("P3 calibration delta/mask mismatch")
    return delta[p3_mask]


def fp_hard_matched_mask(loc_f: torch.Tensor, m_o: torch.Tensor, *, count: int) -> torch.Tensor:
    """S2-H: top-k FP localization error within mO, stable on ties by index."""

    if loc_f.shape != m_o.shape or count < 0:
        raise Stage2Error("invalid S2-H inputs")
    output = torch.zeros_like(m_o, dtype=torch.bool)
    candidates = m_o.nonzero(as_tuple=False).flatten()
    if count > int(candidates.numel()):
        raise Stage2Error("S2-H count exceeds mO candidate count")
    if count == 0:
        return output
    order = torch.argsort(loc_f[candidates], descending=True, stable=True)
    output[candidates[order[:count]]] = True
    return output


def fp_hard_matched_mask_per_image(
    loc_f: torch.Tensor,
    candidates: torch.Tensor,
    dose: torch.Tensor,
    fg_batch_idx: torch.Tensor,
) -> torch.Tensor:
    """Select per-image FP-hard anchors with the exact selector dose."""

    if not (loc_f.shape == candidates.shape == dose.shape == fg_batch_idx.shape):
        raise Stage2Error("per-image matched-dose vectors disagree")
    output = torch.zeros_like(candidates, dtype=torch.bool)
    for image_index in torch.unique(fg_batch_idx, sorted=True):
        image = fg_batch_idx == image_index
        image_candidates = (candidates & image).nonzero(as_tuple=False).flatten()
        count = int((dose & image).sum().item())
        if count > int(image_candidates.numel()):
            raise Stage2Error("per-image hard dose exceeds candidate count")
        if count:
            order = torch.argsort(loc_f[image_candidates], descending=True, stable=True)
            output[image_candidates[order[:count]]] = True
    return output


def ranked_matched_mask_per_image(
    values: torch.Tensor,
    candidates: torch.Tensor,
    dose: torch.Tensor,
    fg_batch_idx: torch.Tensor,
    *,
    largest: bool,
) -> torch.Tensor:
    """Select the largest or smallest values with the exact per-image dose."""

    if not (values.shape == candidates.shape == dose.shape == fg_batch_idx.shape):
        raise Stage2Error("per-image ranked-dose vectors disagree")
    output = torch.zeros_like(candidates, dtype=torch.bool)
    for image_index in torch.unique(fg_batch_idx, sorted=True):
        image = fg_batch_idx == image_index
        image_candidates = (candidates & image).nonzero(as_tuple=False).flatten()
        count = int((dose & image).sum().item())
        if count:
            order = torch.argsort(values[image_candidates], descending=largest, stable=True)
            output[image_candidates[order[:count]]] = True
    return output


def random_matched_mask_per_image(
    candidates: torch.Tensor,
    dose: torch.Tensor,
    fg_batch_idx: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample an independent fixed-seed random support with the exact per-image dose."""

    if not (candidates.shape == dose.shape == fg_batch_idx.shape):
        raise Stage2Error("per-image random-dose vectors disagree")
    output = torch.zeros_like(candidates, dtype=torch.bool)
    for image_index in torch.unique(fg_batch_idx, sorted=True):
        image = fg_batch_idx == image_index
        image_candidates = (candidates & image).nonzero(as_tuple=False).flatten()
        count = int((dose & image).sum().item())
        if count:
            order = torch.randperm(int(image_candidates.numel()), generator=generator)[:count]
            output[image_candidates[order.to(image_candidates.device)]] = True
    return output


def p3_foreground_mask(prediction: Mapping[str, Any], assignment: Assignment) -> torch.Tensor:
    """Return the foreground-vector mask for the first raw-head level."""

    features = prediction.get("feats")
    if not isinstance(features, (tuple, list)) or len(features) != 3:
        raise Stage2Error("Stage-II requires native P3/P4/P5 feature maps")
    if not all(isinstance(level, torch.Tensor) and level.ndim == 4 for level in features):
        raise Stage2Error("raw-head levels must be [B,C,H,W] tensors")
    level_sizes = [int(level.shape[-2] * level.shape[-1]) for level in features]
    if sum(level_sizes) != int(assignment.fg_mask.shape[1]):
        raise Stage2Error("raw-head levels disagree with assignment anchor count")
    foreground = assignment.fg_mask.nonzero(as_tuple=False)
    return foreground[:, 1] < level_sizes[0]


def masked_dfl_kl(
    teacher_dfl: torch.Tensor,
    student_dfl: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """FP32, four-side averaged, target-score-weighted DFL KL with finite zero."""

    if temperature != 1.0:
        raise Stage2Error("DCODFL-QAT-v1 freezes DFL temperature at 1")
    if teacher_dfl.shape != student_dfl.shape or teacher_dfl.ndim != 3 or teacher_dfl.shape[1] != 4:
        raise Stage2Error("DFL tensor ABI must be matching [FG,4,R]")
    if weights.ndim != 1 or mask.ndim != 1 or weights.shape != mask.shape or weights.shape[0] != teacher_dfl.shape[0]:
        raise Stage2Error("DFL weight/mask shape mismatch")
    if not bool(mask.any()):
        return student_dfl.float().sum() * 0.0
    teacher = functional.softmax(teacher_dfl.detach().float(), dim=-1)
    student_log = functional.log_softmax(student_dfl.float(), dim=-1)
    kl = (teacher * (teacher.clamp_min(torch.finfo(torch.float32).tiny).log() - student_log)).sum(-1).mean(-1)
    selected_weight = weights.float() * mask.float()
    return float(batch_size) * (kl * selected_weight).sum() / selected_weight.sum().clamp_min(torch.finfo(torch.float32).eps)


def stage2_selected_mask(
    arm: str,
    *,
    m_q: torch.Tensor,
    m_o: torch.Tensor,
    loc_f: torch.Tensor,
    p3_mask: torch.Tensor | None = None,
    fg_batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the exact foreground-vector support used by a Stage-II arm."""

    if not (m_q.shape == m_o.shape == loc_f.shape):
        raise Stage2Error("Stage-II selector vectors disagree")
    if arm == "S2-0":
        return torch.zeros_like(m_q)
    if arm == "S2-1":
        return torch.ones_like(m_q)
    proposed = m_q & m_o
    if arm in {"S2-4", "S2-5", "S2-7"}:
        return proposed
    if arm == "S2-H":
        return fp_hard_matched_mask(loc_f, m_o, count=int(proposed.sum().item()))
    if p3_mask is None or p3_mask.shape != m_q.shape:
        raise Stage2Error(f"{arm} requires a P3 foreground mask")
    if arm == "S2-P3":
        return p3_mask
    if arm == "S2-Q":
        return m_q
    if arm in {"S2-P3Q", "S2-P3Q-SD"}:
        return p3_mask & m_q
    if arm == "S2-P3O":
        return p3_mask & m_o
    if arm == "S2-P3QO":
        return p3_mask & m_q & m_o
    if arm == "S2-P3H":
        if fg_batch_idx is None:
            raise Stage2Error("S2-P3H requires foreground batch indices")
        return fp_hard_matched_mask_per_image(loc_f, p3_mask, p3_mask & m_q, fg_batch_idx)
    if arm == "S2-P3OH":
        if fg_batch_idx is None:
            raise Stage2Error("S2-P3OH requires foreground batch indices")
        candidates = p3_mask & m_o
        return fp_hard_matched_mask_per_image(loc_f, candidates, candidates & m_q, fg_batch_idx)
    raise Stage2Error(f"unsupported frozen Stage-II arm {arm!r}")


def stage2_kd_for_arm(
    arm: str,
    *,
    q_dfl: torch.Tensor,
    optical_dfl: torch.Tensor,
    same_modal_dfl: torch.Tensor,
    shuffled_dfl: torch.Tensor,
    weights: torch.Tensor,
    m_q: torch.Tensor,
    m_o: torch.Tensor,
    loc_f: torch.Tensor,
    batch_size: int,
    p3_mask: torch.Tensor | None = None,
    fg_batch_idx: torch.Tensor | None = None,
    target_kind: str | None = None,
    selected_mask: torch.Tensor | None = None,
    quant_off_dfl: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stage-II objective selector; all teacher targets are caller detached."""

    if arm == "S2-0":
        return q_dfl.float().sum() * 0.0
    selected = selected_mask
    if selected is None:
        selected = stage2_selected_mask(
            arm,
            m_q=m_q,
            m_o=m_o,
            loc_f=loc_f,
            p3_mask=p3_mask,
            fg_batch_idx=fg_batch_idx,
        )
    if arm == "S2-P3Q-SD":
        if quant_off_dfl is None:
            raise Stage2Error("S2-P3Q-SD requires quant-off shadow DFL logits")
        target = quant_off_dfl
    elif arm == "S2-4" or target_kind == "same-modal":
        target = same_modal_dfl
    elif arm == "S2-5" or target_kind == "shuffled":
        target = shuffled_dfl
    else:
        target = optical_dfl
    return masked_dfl_kl(target, q_dfl, weights, selected, batch_size=batch_size)


def combine_stage2_loss(native_q_loss: torch.Tensor, kd_loss: torch.Tensor, *, lambda_kd: float = 0.1) -> torch.Tensor:
    """Combine native q loss and DFL KD; production fixes ``lambda_kd`` to 0.1."""

    if lambda_kd < 0:
        raise Stage2Error("KD lambda must be non-negative")
    return native_q_loss + float(lambda_kd) * kd_loss
