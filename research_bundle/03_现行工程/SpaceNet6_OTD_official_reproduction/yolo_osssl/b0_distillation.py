"""Pure-FP B0 F3/F4 native-loss and all-foreground DFL-KD primitives.

This module intentionally has no QAT, selector, threshold, or Stage-II
dependency.  F3 is the unmodified native student detector loss.  F4 keeps the
same one student assignment and adds only an optical-teacher DFL KL term over
every student foreground anchor.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as functional

from .raw_head_abi import RawHeadABIError, raw_box_logits


class B0DistillationError(RuntimeError):
    """Raised when F3/F4 would drift from the native detector-loss ABI."""


@dataclass(frozen=True)
class B0Assignment:
    """The sole student-native assignment, including target-score weights."""

    fg_mask: torch.Tensor
    target_gt_idx: torch.Tensor
    target_bboxes: torch.Tensor
    target_scores: torch.Tensor
    anchor_points: torch.Tensor
    stride_tensor: torch.Tensor

    @property
    def target_score_weights(self) -> torch.Tensor:
        return self.target_scores.sum(-1)[self.fg_mask]


@dataclass(frozen=True)
class NativeStudentResult:
    assignment: B0Assignment
    native_loss: torch.Tensor
    detached_loss: Any
    parsed_prediction: Mapping[str, Any]
    loss_parts: torch.Tensor | None = None


@dataclass(frozen=True)
class B0LossResult:
    native_loss: torch.Tensor
    kd_loss: torch.Tensor
    total_loss: torch.Tensor
    assignment: B0Assignment
    detached_loss: Any


class _StudentAssignmentCapture(AbstractContextManager["_StudentAssignmentCapture"]):
    """Observe exactly one existing student assigner call; never replace it."""

    def __init__(self, assigner: Any) -> None:
        self.assigner = assigner
        self.calls: list[Any] = []
        self._owner: type[Any] | None = None
        self._original: Any = None

    def __enter__(self) -> "_StudentAssignmentCapture":
        self._owner = type(self.assigner)
        self._original = self._owner.forward

        def observed(instance: Any, *args: Any, **kwargs: Any) -> Any:
            output = self._original(instance, *args, **kwargs)
            if instance is self.assigner:
                self.calls.append(output)
            return output

        self._owner.forward = observed
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        assert self._owner is not None
        self._owner.forward = self._original
        return False

    def result(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.calls) != 1:
            raise B0DistillationError(f"F3/F4 requires exactly one student native assigner call, got {len(self.calls)}")
        value = self.calls[0]
        if not isinstance(value, (tuple, list)) or len(value) != 5 or not all(isinstance(item, torch.Tensor) for item in value):
            raise B0DistillationError("student assigner output ABI drifted")
        labels, boxes, scores, foreground, gt_idx = value
        del labels
        return boxes, scores, foreground, gt_idx, torch.empty(0, device=boxes.device)


def _native_parts(value: Any) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, Any]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise B0DistillationError("native get_assigned_targets_and_loss return ABI drifted")
    assignment, losses, detached = value
    if not isinstance(assignment, (tuple, list)) or len(assignment) != 5 or not all(isinstance(item, torch.Tensor) for item in assignment):
        raise B0DistillationError("native assigned-target tensors ABI drifted")
    if not isinstance(losses, torch.Tensor):
        raise B0DistillationError("native loss parts are not a tensor")
    return tuple(assignment), losses, detached


def _batch_size(prediction: Any, batch: Mapping[str, Any]) -> int:
    raw = _raw_prediction(prediction)
    boxes = raw.get("boxes")
    if isinstance(boxes, torch.Tensor) and boxes.ndim == 3:
        return int(boxes.shape[0])
    image = batch.get("img")
    if isinstance(image, torch.Tensor) and image.ndim >= 1:
        return int(image.shape[0])
    raise B0DistillationError("cannot recover native detector batch size")


def _raw_prediction(value: Any) -> Mapping[str, Any]:
    """Return the raw native dict from train output or eval ``(decoded, raw)``."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, (tuple, list)):
        dictionaries = [item for item in value if isinstance(item, Mapping)]
        if len(dictionaries) == 1:
            return dictionaries[0]
    raise B0DistillationError("detector forward did not expose one native raw prediction mapping")


def native_student_loss(criterion: Any, prediction: Any, batch: Mapping[str, Any]) -> NativeStudentResult:
    """Call the native student criterion once and preserve its ``loss * B`` scale."""

    method = getattr(criterion, "get_assigned_targets_and_loss", None)
    assigner = getattr(criterion, "assigner", None)
    parse_output = getattr(criterion, "parse_output", None)
    if not callable(method) or assigner is None or not callable(parse_output):
        raise B0DistillationError("student criterion lacks native parse/assignment/loss seams")
    parsed_prediction = _raw_prediction(parse_output(prediction))
    with _StudentAssignmentCapture(assigner) as capture:
        assigned, loss_parts, detached = _native_parts(method(parsed_prediction, batch))
    fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned
    captured_bboxes, captured_scores, captured_foreground, captured_gt_idx, _ = capture.result()
    if not (
        torch.equal(captured_foreground, fg_mask)
        and torch.equal(captured_bboxes, target_bboxes)
        and torch.equal(captured_gt_idx, target_gt_idx)
    ):
        raise B0DistillationError("captured student assignment disagrees with native criterion return")
    return NativeStudentResult(
        assignment=B0Assignment(
            fg_mask=fg_mask,
            target_gt_idx=target_gt_idx,
            target_bboxes=target_bboxes,
            target_scores=captured_scores,
            anchor_points=anchor_points,
            stride_tensor=stride_tensor,
        ),
        native_loss=loss_parts.sum() * _batch_size(parsed_prediction, batch),
        detached_loss=detached,
        parsed_prediction=parsed_prediction,
        loss_parts=loss_parts,
    )


def dfl_logits(prediction: Any, *, reg_max: int) -> torch.Tensor:
    """Return raw native DFL logits as ``[B,A,4,R]`` without using features."""

    try:
        boxes = raw_box_logits(_raw_prediction(prediction), reg_max=reg_max)
    except RawHeadABIError as exc:
        raise B0DistillationError(str(exc)) from exc
    return boxes.reshape(boxes.shape[0], boxes.shape[1], 4, reg_max)


def criterion_reg_max(criterion: Any) -> int:
    value = getattr(getattr(getattr(criterion, "bbox_loss", None), "dfl_loss", None), "reg_max", 0)
    if not isinstance(value, int) or value <= 0:
        raise B0DistillationError("student criterion lacks a positive native DFL reg_max")
    return value


def freeze_optical_teacher(teacher: torch.nn.Module) -> None:
    """Freeze the optical detector before any F4 teacher forward."""

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise B0DistillationError("F4 optical teacher was not frozen in eval/no-grad state")


def frozen_teacher_prediction(teacher: torch.nn.Module, eo_images: torch.Tensor) -> Mapping[str, Any]:
    """Forward only the frozen teacher model; criterion/assigner is never touched."""

    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise B0DistillationError("F4 requires an eval teacher with every parameter frozen")
    with torch.no_grad():
        prediction = teacher(eo_images)
    return _raw_prediction(prediction)


def all_foreground_dfl_kl(
    *,
    teacher_prediction: Any,
    student_prediction: Any,
    assignment: B0Assignment,
    reg_max: int,
    batch_size: int,
) -> torch.Tensor:
    """T=1 FP32 target-score-weighted DFL KL over every student FG anchor."""

    teacher_dfl = dfl_logits(teacher_prediction, reg_max=reg_max)
    student_dfl = dfl_logits(student_prediction, reg_max=reg_max)
    if teacher_dfl.shape != student_dfl.shape:
        raise B0DistillationError("F4 teacher/student raw DFL ABI mismatch")
    if assignment.fg_mask.shape != teacher_dfl.shape[:2]:
        raise B0DistillationError("student foreground assignment disagrees with raw DFL anchors")
    if not bool(assignment.fg_mask.any()):
        return student_dfl.float().sum() * 0.0
    teacher_fg = teacher_dfl[assignment.fg_mask].detach().float()
    student_fg = student_dfl[assignment.fg_mask].float()
    weights = assignment.target_score_weights.float()
    if weights.ndim != 1 or weights.shape[0] != teacher_fg.shape[0]:
        raise B0DistillationError("student target-score weights disagree with foreground anchors")
    teacher_probability = functional.softmax(teacher_fg, dim=-1)
    student_log_probability = functional.log_softmax(student_fg, dim=-1)
    per_anchor = (
        teacher_probability
        * (teacher_probability.clamp_min(torch.finfo(torch.float32).tiny).log() - student_log_probability)
    ).sum(-1).mean(-1)
    return float(batch_size) * (per_anchor * weights).sum() / weights.sum().clamp_min(torch.finfo(torch.float32).eps)


def f3_criterion(criterion: Any, student_prediction: Any, batch: Mapping[str, Any]) -> B0LossResult:
    """F3: exactly native student loss; it has no teacher parameter or forward."""

    native = native_student_loss(criterion, student_prediction, batch)
    zero = native.native_loss * 0.0
    return B0LossResult(native.native_loss, zero, native.native_loss, native.assignment, native.detached_loss)


def f4_criterion(
    criterion: Any,
    student_prediction: Any,
    batch: Mapping[str, Any],
    *,
    teacher: torch.nn.Module,
    eo_images: torch.Tensor,
    lambda_kd: float = 0.1,
) -> B0LossResult:
    """F4: native student objective plus all-FG frozen-optical DFL KL, lambda=.1."""

    if float(lambda_kd) != 0.1:
        raise B0DistillationError("F4 freezes the optical DFL-KD coefficient at lambda=0.1")
    native = native_student_loss(criterion, student_prediction, batch)
    # Empty student foreground has no DFL targets.  Preserve native loss and
    # do not spend a teacher forward or touch teacher buffers at all.
    if not bool(native.assignment.fg_mask.any()):
        zero = native.native_loss * 0.0
        return B0LossResult(native.native_loss, zero, native.native_loss, native.assignment, native.detached_loss)
    teacher_prediction = frozen_teacher_prediction(teacher, eo_images)
    kd_loss = all_foreground_dfl_kl(
        teacher_prediction=teacher_prediction,
        student_prediction=native.parsed_prediction,
        assignment=native.assignment,
        reg_max=criterion_reg_max(criterion),
        batch_size=_batch_size(native.parsed_prediction, batch),
    )
    return B0LossResult(native.native_loss, kd_loss, native.native_loss + 0.1 * kd_loss, native.assignment, native.detached_loss)
