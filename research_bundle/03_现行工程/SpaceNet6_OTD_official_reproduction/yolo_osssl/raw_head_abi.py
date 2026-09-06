"""Raw-head ABI layout for the Stage-II optical teacher/student boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


class RawHeadABIError(RuntimeError):
    """Raised when teacher and student cannot index the same raw DFL logit."""


def prediction_from_qat_wrapper_output(value: Any) -> dict[str, Any]:
    """Rebuild the pinned native mapping from ``YoloQATWrapper``'s fixed tuple."""

    if not isinstance(value, tuple) or len(value) != 5 or not all(isinstance(item, torch.Tensor) for item in value):
        raise RawHeadABIError("QAT wrapper output must be (boxes, scores, P3, P4, P5) tensors")
    boxes, scores, p3, p4, p5 = value
    return {"boxes": boxes, "scores": scores, "feats": [p3, p4, p5]}


@dataclass(frozen=True)
class RawHeadLayout:
    input_shape: tuple[int, int, int, int]
    level_shapes: tuple[tuple[int, int], ...]
    raw_shapes: tuple[tuple[int, ...], ...]
    strides: tuple[float, ...]
    nc: int
    reg_max: int
    candidate_count: int
    flatten_order: str
    anchor_coordinate_system: str
    dfl_bin_order: str
    class_order: tuple[str, ...]
    dtype: str

def _as_levels(prediction: Any) -> list[torch.Tensor]:
    """Extract feature-map levels only; never mistake decoded output for raw head data."""

    if isinstance(prediction, dict):
        features = prediction.get("feats")
        if features is None:
            raise RawHeadABIError("Ultralytics prediction dict lacks feats")
        return _as_levels(features)
    if isinstance(prediction, torch.Tensor):
        if prediction.ndim != 4:
            raise RawHeadABIError("raw feature level must be [B,C,H,W]")
        return [prediction]
    if isinstance(prediction, (list, tuple)):
        tensors = [item for item in prediction if isinstance(item, torch.Tensor) and item.ndim == 4]
        if tensors:
            return tensors
        for item in prediction:
            try:
                return _as_levels(item)
            except RawHeadABIError:
                pass
    raise RawHeadABIError("raw prediction must contain a non-empty sequence of [B,C,H,W] tensors")


def raw_box_logits(prediction: Any, *, reg_max: int) -> torch.Tensor:
    """Return the native DFL tensor as ``[B,A,4*reg_max]``.

    Pinned Ultralytics 8.4.115 exposes ``preds['boxes']`` as ``[B,4R,A]``;
    feature maps are retained solely for anchor/ABI accounting.
    """

    required = 4 * int(reg_max)
    if isinstance(prediction, dict):
        boxes = prediction.get("boxes")
        if not isinstance(boxes, torch.Tensor) or boxes.ndim != 3:
            raise RawHeadABIError("prediction['boxes'] must be [B,4R,A]")
        if boxes.shape[1] != required:
            raise RawHeadABIError(f"prediction['boxes'] channels {boxes.shape[1]} != 4*reg_max ({required})")
        return boxes.permute(0, 2, 1).contiguous()
    flat, _ = flatten_raw_heads(prediction)
    if flat.shape[-1] < required:
        raise RawHeadABIError("raw prediction lacks 4*reg_max DFL channels")
    return flat[..., :required]


def validate_native_prediction(prediction: Any, *, nc: int, reg_max: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """Validate the complete pinned ``boxes/scores/feats`` prediction ABI."""

    if not isinstance(prediction, dict):
        raise RawHeadABIError("Ultralytics 8.4.115 ABI requires a prediction dict")
    boxes = raw_box_logits(prediction, reg_max=reg_max)
    scores = prediction.get("scores")
    if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
        raise RawHeadABIError("prediction['scores'] must be [B,nc,A]")
    batch_size, anchors, _ = boxes.shape
    if tuple(scores.shape) != (batch_size, int(nc), anchors):
        raise RawHeadABIError("prediction boxes/scores disagree in B, A, or nc")
    level_shapes = _feature_level_shapes(_as_levels(prediction))
    if sum(height * width for height, width in level_shapes) != anchors:
        raise RawHeadABIError("prediction boxes/features disagree in native make_anchors order")
    return batch_size, anchors, level_shapes


def _feature_level_shapes(levels: Sequence[torch.Tensor]) -> tuple[tuple[int, int], ...]:
    """Validate feature levels without assuming equal P3/P4/P5 channels."""

    if not levels:
        raise RawHeadABIError("raw prediction has no feature levels")
    batch_sizes: set[int] = set()
    shapes: list[tuple[int, int]] = []
    for level in levels:
        if level.ndim != 4:
            raise RawHeadABIError(f"raw head must be [B,C,H,W], got {tuple(level.shape)}")
        batch_sizes.add(int(level.shape[0]))
        shapes.append((int(level.shape[2]), int(level.shape[3])))
    if len(batch_sizes) != 1:
        raise RawHeadABIError("raw head levels disagree in batch size")
    return tuple(shapes)


def flatten_raw_heads(prediction: Any) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
    """Flatten P3→P5 in level-major, row-major order to ``[B,A,C]``."""

    levels = _as_levels(prediction)
    shapes = _feature_level_shapes(levels)
    flattened: list[torch.Tensor] = []
    for level in levels:
        batch, channels, height, width = level.shape
        flattened.append(level.permute(0, 2, 3, 1).reshape(batch, height * width, channels))
    if len({value.shape[-1] for value in flattened}) != 1:
        raise RawHeadABIError("cannot flatten feature levels with unequal channels")
    return torch.cat(flattened, dim=1), shapes


def make_raw_head_layout(
    prediction: Any,
    *,
    input_shape: Sequence[int],
    strides: Iterable[float],
    nc: int,
    reg_max: int,
    class_order: Iterable[str],
) -> RawHeadLayout:
    levels = _as_levels(prediction)
    level_shapes = _feature_level_shapes(levels)
    stride_tuple = tuple(float(value) for value in strides)
    if len(level_shapes) != len(stride_tuple):
        raise RawHeadABIError("raw-head level count disagrees with Detect strides")
    if isinstance(prediction, dict):
        _, boxes_count, _ = validate_native_prediction(prediction, nc=nc, reg_max=reg_max)
        candidate_count = boxes_count
        dtype = str(raw_box_logits(prediction, reg_max=reg_max).dtype)
    else:
        flat, _ = flatten_raw_heads(prediction)
        expected_channels = 4 * int(reg_max) + int(nc)
        if flat.shape[-1] != expected_channels:
            raise RawHeadABIError(f"raw head channel count {flat.shape[-1]} != 4*reg_max+nc ({expected_channels})")
        candidate_count = int(flat.shape[1])
        dtype = str(flat.dtype)
    return RawHeadLayout(
        input_shape=tuple(int(value) for value in input_shape),
        level_shapes=level_shapes,
        raw_shapes=tuple(tuple(int(value) for value in level.shape) for level in levels),
        strides=stride_tuple,
        nc=int(nc),
        reg_max=int(reg_max),
        candidate_count=candidate_count,
        flatten_order="P3_to_P5_level_major_row_major",
        anchor_coordinate_system="grid_units_make_anchors_offset_0.5",
        dfl_bin_order="LTRB_each_low_to_high",
        class_order=tuple(class_order),
        dtype=dtype,
    )


def assert_raw_head_compatible(student: RawHeadLayout, teacher: RawHeadLayout) -> None:
    shared = (
        "input_shape",
        "level_shapes",
        "strides",
        "nc",
        "reg_max",
        "candidate_count",
        "flatten_order",
        "anchor_coordinate_system",
        "dfl_bin_order",
        "class_order",
    )
    mismatched = [name for name in shared if getattr(student, name) != getattr(teacher, name)]
    if mismatched:
        raise RawHeadABIError("student/teacher raw-head ABI mismatch: " + ", ".join(mismatched))


def anchor_points_grid(layout: RawHeadLayout, *, device: torch.device | None = None) -> torch.Tensor:
    """Return native ``make_anchors(..., offset=0.5)`` grid-unit points."""

    points: list[torch.Tensor] = []
    for height, width in layout.level_shapes:
        yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        points.append(torch.stack((xx.reshape(-1) + 0.5, yy.reshape(-1) + 0.5), dim=-1))
    return torch.cat(points, dim=0)


def anchor_centers_pixel(layout: RawHeadLayout, *, device: torch.device | None = None) -> torch.Tensor:
    """Return pixel centers only when a caller explicitly needs them."""

    grid = anchor_points_grid(layout, device=device)
    strides = torch.cat(
        [torch.full((height * width, 1), float(stride), device=device) for (height, width), stride in zip(layout.level_shapes, layout.strides, strict=True)]
    )
    return grid * strides


def anchor_centers(layout: RawHeadLayout, *, device: torch.device | None = None) -> torch.Tensor:
    """Backward-compatible alias for grid-unit native anchors."""

    return anchor_points_grid(layout, device=device)
