"""RCFMD training-only losses for low-resolution SAR YOLO11n.

The detector keeps its native loss and assignment.  This module adds the
resource-counterfactual DFL transport and, for the FM arms, a frozen MaRS RGB
ROI target.  None of the teachers or the adapter belongs in the deployable
detector checkpoint; the trainer removes ``sn6_fm_aux`` from ``last.pt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as functional

from .b0_distillation import (
    B0Assignment,
    criterion_reg_max,
    dfl_logits,
    frozen_teacher_prediction,
    native_student_loss,
)


METHODS = (
    "lr_native",
    "same_res_optical",
    "hr_sar_uniform",
    "hr_optical_uniform",
    "teacher_quality_soft",
    "dfl_discrepancy_object",
    "resource_gate",
    "optical_gate",
    "rc_dfl",
    "rc_dfl_hard",
    "rc_dfl_same_modal",
    "rc_dfl_shuffled",
    "mars_roi",
    "rcfmd",
    "rcfmd_imagenet",
    "rcfmd_shuffled",
    "safe_sar",
    "cmv_kd",
    "cmv_kd_shuffled",
    "cmv_random_veto",
    "cmv_arbitration",
    "cmv_optical_target",
    "sar_discrepancy_dose",
    "resource_arbitration",
)

CMV_METHODS = frozenset(
    {
        "safe_sar",
        "cmv_kd",
        "cmv_kd_shuffled",
        "cmv_random_veto",
        "cmv_arbitration",
        "cmv_optical_target",
        "sar_discrepancy_dose",
        "resource_arbitration",
    }
)
OPTICAL_TEACHER_METHODS = frozenset(
    {
        "same_res_optical",
        "hr_optical_uniform",
        "teacher_quality_soft",
        "dfl_discrepancy_object",
        "resource_gate",
        "optical_gate",
        "rc_dfl",
        "rc_dfl_hard",
        "rc_dfl_same_modal",
        "rc_dfl_shuffled",
        "mars_roi",
        "rcfmd",
        "rcfmd_imagenet",
        "rcfmd_shuffled",
        "cmv_kd",
        "cmv_kd_shuffled",
        "cmv_random_veto",
        "cmv_arbitration",
        "cmv_optical_target",
        "sar_discrepancy_dose",
        "resource_arbitration",
    }
)
PAIRED_DATA_METHODS = frozenset(
    method
    for method in METHODS
    if method not in {"lr_native", "hr_sar_uniform", "safe_sar"}
)
SHUFFLED_DATA_METHODS = frozenset({"rc_dfl_shuffled", "rcfmd_shuffled", "cmv_kd_shuffled"})
MARS_METHODS = frozenset({"mars_roi", "rcfmd", "rcfmd_imagenet", "rcfmd_shuffled"})
SAR_TEACHER_METHODS = frozenset({"hr_sar_uniform", "rc_dfl_same_modal", *CMV_METHODS})
GATED_METHODS = frozenset(
    {
        "resource_gate",
        "optical_gate",
        "rc_dfl",
        "rc_dfl_hard",
        "rc_dfl_same_modal",
        "rc_dfl_shuffled",
        "mars_roi",
        "rcfmd",
        "rcfmd_imagenet",
        "rcfmd_shuffled",
        "dfl_discrepancy_object",
    }
)

# Values published in MaRS' mars_dataset.py.  MaRS receives RGB samples in
# the original 0--255 radiometric range, not ImageNet-normalized tensors.
MARS_RGB_MEAN = (87.01, 91.52, 83.51)
MARS_RGB_STD = (62.66, 54.58, 53.11)
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)


class FMResourceDistillationError(RuntimeError):
    """Raised when an RCFMD training-only contract is not available."""


@dataclass(frozen=True)
class ObjectGate:
    """Object decisions broadcast onto the corresponding P3 foreground rows."""

    resource: torch.Tensor
    optical: torch.Tensor
    selected: torch.Tensor
    object_rows: torch.Tensor
    resource_fraction: float
    optical_fraction: float
    selected_fraction: float


@dataclass(frozen=True)
class CMVGate:
    """Resource/SAR-safe support plus an optical reliability conflict veto."""

    resource: torch.Tensor
    sar_safe: torch.Tensor
    safe: torch.Tensor
    optical_reliable: torch.Tensor
    agreement: torch.Tensor
    conflict: torch.Tensor
    keep: torch.Tensor
    selected: torch.Tensor
    object_rows: torch.Tensor
    resource_fraction: float
    sar_safe_fraction: float
    safe_fraction: float
    optical_reliable_fraction: float
    conflict_fraction: float
    vetoed_fraction: float
    selected_fraction: float


@dataclass(frozen=True)
class CMVState:
    """Detached counterfactual targets, errors and decisions for one batch."""

    gate: CMVGate
    sar_target: torch.Tensor
    optical_target: torch.Tensor
    low_error: torch.Tensor
    full_error: torch.Tensor
    sar_error: torch.Tensor
    optical_error: torch.Tensor


@dataclass(frozen=True)
class FMResourceResult:
    native_loss: torch.Tensor
    dfl_loss: torch.Tensor
    mars_loss: torch.Tensor
    total_loss: torch.Tensor
    detached_loss: Any
    stats: dict[str, float]


def mars_rgb_preprocess(eo_images: torch.Tensor) -> torch.Tensor:
    """Resize transformed EO RGB to 512 then apply the official MaRS scale."""

    if eo_images.ndim != 4 or eo_images.shape[1] != 3:
        raise FMResourceDistillationError("MaRS requires RGB tensors shaped [B,3,H,W]")
    rgb = functional.interpolate(eo_images.float(), size=(512, 512), mode="bilinear", align_corners=False)
    # PairedDetectionDataset follows Ultralytics preprocessing, which supplies
    # CHW RGB in [0, 1].  MaRS' official mean/std are expressed in [0, 255].
    rgb = rgb * 255.0
    mean = rgb.new_tensor(MARS_RGB_MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(MARS_RGB_STD).view(1, 3, 1, 1)
    return (rgb - mean) / std


def imagenet_rgb_preprocess(eo_images: torch.Tensor) -> torch.Tensor:
    """Resize transformed RGB then apply the native ImageNet SwinV2 scale."""

    if eo_images.ndim != 4 or eo_images.shape[1] != 3:
        raise FMResourceDistillationError("ImageNet SwinV2 requires RGB tensors shaped [B,3,H,W]")
    rgb = functional.interpolate(eo_images.float(), size=(512, 512), mode="bilinear", align_corners=False)
    mean = rgb.new_tensor(IMAGENET_RGB_MEAN).view(1, 3, 1, 1)
    std = rgb.new_tensor(IMAGENET_RGB_STD).view(1, 3, 1, 1)
    return (rgb - mean) / std


class FrozenMaRSRGB(nn.Module):
    """Official MaRS-Base RGB SwinV2-B encoder with reduction-8 features."""

    feature_channels: int = 256

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        backbone: nn.Module | None = None,
        normalization: str = "mars",
    ) -> None:
        super().__init__()
        if normalization not in {"mars", "imagenet"}:
            raise FMResourceDistillationError("Frozen RGB SwinV2 normalization must be mars or imagenet")
        if backbone is None:
            if checkpoint is None:
                raise FMResourceDistillationError("FrozenMaRSRGB requires a MaRS RGB checkpoint")
            try:
                import timm
            except ImportError as exc:  # pragma: no cover - production environment dependency
                raise FMResourceDistillationError("RCFMD MaRS arms require timm==1.0.15") from exc
            backbone = timm.create_model(
                "swinv2_base_window8_256",
                pretrained=False,
                features_only=True,
                out_indices=(1, 2, 3),
                in_chans=3,
                img_size=512,
                checkpoint_path=str(checkpoint),
            )
        self.backbone = backbone
        self.normalization = normalization
        channels = getattr(backbone, "feature_info", None)
        if channels is not None and hasattr(channels, "channels"):
            values = channels.channels()
            if values:
                self.feature_channels = int(values[0])
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, eo_images: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
        if self.training or any(parameter.requires_grad for parameter in self.parameters()):
            raise FMResourceDistillationError("MaRS must remain frozen in eval mode")
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            prepared = mars_rgb_preprocess(eo_images) if self.normalization == "mars" else imagenet_rgb_preprocess(eo_images)
            for chunk in prepared.split(int(chunk_size), dim=0):
                features = self.backbone(chunk)
                if not isinstance(features, (tuple, list)) or not features:
                    raise FMResourceDistillationError("MaRS features_only backbone lacks reduction-8 output")
                feature = features[0]
                if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                    raise FMResourceDistillationError("MaRS reduction-8 output must be [B,C,H,W]")
                # timm SwinV2 features_only emits NHWC for this exact official
                # backbone. Normalize at the boundary so ROIAlign always sees
                # NCHW, including if a future timm build emits NCHW instead.
                if feature.shape[-1] == self.feature_channels and feature.shape[1] != self.feature_channels:
                    feature = feature.permute(0, 3, 1, 2).contiguous()
                if feature.shape[1] != self.feature_channels:
                    raise FMResourceDistillationError("MaRS reduction-8 channel layout is not NCHW or NHWC")
                chunks.append(feature)
        return torch.cat(chunks, dim=0)


class P3MaRSAdapter(nn.Module):
    """The only trainable RCFMD auxiliary module: student P3 to MaRS channels."""

    def __init__(self, student_channels: int, mars_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(int(student_channels), int(mars_channels), kernel_size=1, bias=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.proj(feature)


def _raw_prediction(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (tuple, list)):
        mappings = [item for item in value if isinstance(item, Mapping)]
        if len(mappings) == 1:
            return mappings[0]
    raise FMResourceDistillationError("detector forward did not expose one raw prediction mapping")


def _p3_shape(prediction: Mapping[str, Any]) -> tuple[int, int]:
    features = prediction.get("feats")
    if not isinstance(features, (tuple, list)) or len(features) != 3:
        raise FMResourceDistillationError("RCFMD requires native P3/P4/P5 feature maps")
    p3 = features[0]
    if not isinstance(p3, torch.Tensor) or p3.ndim != 4:
        raise FMResourceDistillationError("student P3 must be [B,C,H,W]")
    return int(p3.shape[-2]), int(p3.shape[-1])


def p3_foreground_mask(prediction: Mapping[str, Any], assignment: B0Assignment) -> torch.Tensor:
    """Return a full-anchor mask restricted to native P3 foreground anchors."""

    height, width = _p3_shape(prediction)
    p3_count = height * width
    if p3_count > assignment.fg_mask.shape[1]:
        raise FMResourceDistillationError("P3 extent exceeds native assignment anchors")
    result = torch.zeros_like(assignment.fg_mask, dtype=torch.bool)
    result[:, :p3_count] = assignment.fg_mask[:, :p3_count]
    return result


def _p3_dfl_logits(prediction: Mapping[str, Any], *, reg_max: int) -> torch.Tensor:
    height, width = _p3_shape(prediction)
    return dfl_logits(prediction, reg_max=reg_max)[:, : height * width]


def cross_resolution_dfl_transport(
    teacher_p3_logits: torch.Tensor,
    *,
    teacher_p3_shape: tuple[int, int],
    student_p3_shape: tuple[int, int],
    canonical_imgsz: int,
    student_imgsz: int,
    student_reg_max: int,
) -> torch.Tensor:
    """Interpolate HR DFL probabilities at LR centres and re-bin distances.

    Each HR anchor first becomes a DFL probability distribution.  Bilinear
    interpolation is then applied to that probability field (rather than to
    logits), before the resulting boundary distances are expressed in the LR
    anchor coordinate system and linearly re-binned.
    """

    if teacher_p3_logits.ndim != 4:
        raise FMResourceDistillationError("teacher P3 DFL must be [B,N,4,R]")
    batch, teacher_count, sides, teacher_reg_max = teacher_p3_logits.shape
    teacher_h, teacher_w = teacher_p3_shape
    student_h, student_w = student_p3_shape
    if sides != 4 or teacher_count != teacher_h * teacher_w:
        raise FMResourceDistillationError("teacher P3 DFL shape disagrees with its feature map")
    source = (
        functional.softmax(teacher_p3_logits.float(), dim=-1)
        .reshape(batch, teacher_h, teacher_w, 4, teacher_reg_max)
        .permute(0, 3, 4, 1, 2)
        .reshape(batch, 4 * teacher_reg_max, teacher_h, teacher_w)
    )
    yy, xx = torch.meshgrid(
        torch.arange(student_h, device=source.device, dtype=source.dtype),
        torch.arange(student_w, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    grid = torch.stack(((xx + 0.5) * (2.0 / student_w) - 1.0, (yy + 0.5) * (2.0 / student_h) - 1.0), dim=-1)
    sampled = functional.grid_sample(
        source,
        grid.unsqueeze(0).expand(batch, -1, -1, -1),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    probability = sampled.permute(0, 2, 3, 1).reshape(batch, student_h * student_w, 4, teacher_reg_max)

    # One teacher bin equals teacher_stride canonical pixels.  Convert it to
    # LR input pixels then divide by the LR P3 stride to obtain LR DFL bins.
    teacher_stride = float(canonical_imgsz) / float(teacher_w)
    student_stride = float(student_imgsz) / float(student_w)
    distance_scale = teacher_stride * float(student_imgsz) / (float(canonical_imgsz) * student_stride)
    source_bins = torch.arange(teacher_reg_max, device=probability.device, dtype=probability.dtype) * distance_scale
    lower = source_bins.floor().long().clamp(max=int(student_reg_max) - 1)
    upper = (lower + 1).clamp(max=int(student_reg_max) - 1)
    upper_weight = (source_bins - lower.to(source_bins.dtype)).clamp(0.0, 1.0)
    lower_weight = 1.0 - upper_weight
    transported = probability.new_zeros((*probability.shape[:-1], int(student_reg_max)))
    transported.scatter_add_(-1, lower.view(1, 1, 1, -1).expand_as(probability), probability * lower_weight)
    transported.scatter_add_(-1, upper.view(1, 1, 1, -1).expand_as(probability), probability * upper_weight)
    return transported / transported.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(transported.dtype).eps)


def _p3_localization_error(
    probability: torch.Tensor,
    assignment: B0Assignment,
    p3_mask: torch.Tensor,
    *,
    reg_max: int,
) -> torch.Tensor:
    """A common GT DFL+IoU localization error for low/HR teacher comparisons."""

    if probability.ndim != 3 or probability.shape[1:] != (4, int(reg_max)):
        raise FMResourceDistillationError("P3 probabilities must be [N,4,R]")
    rows = p3_mask.nonzero(as_tuple=False)
    if probability.shape[0] != rows.shape[0]:
        raise FMResourceDistillationError("P3 probability rows disagree with foreground mask")
    anchors = assignment.anchor_points[rows[:, 1]].float()
    strides = assignment.stride_tensor[rows[:, 1]].float()
    anchor_pixels = anchors * strides
    target = assignment.target_bboxes[rows[:, 0], rows[:, 1]].float()
    target_ltrb = torch.stack(
        (
            (anchor_pixels[:, 0] - target[:, 0]) / strides[:, 0],
            (anchor_pixels[:, 1] - target[:, 1]) / strides[:, 0],
            (target[:, 2] - anchor_pixels[:, 0]) / strides[:, 0],
            (target[:, 3] - anchor_pixels[:, 1]) / strides[:, 0],
        ),
        dim=-1,
    ).clamp(0.0, float(reg_max - 1))
    bins = torch.arange(reg_max, device=probability.device, dtype=probability.dtype)
    expected = (probability * bins).sum(-1) * strides
    predicted = torch.stack(
        (
            anchor_pixels[:, 0] - expected[:, 0],
            anchor_pixels[:, 1] - expected[:, 1],
            anchor_pixels[:, 0] + expected[:, 2],
            anchor_pixels[:, 1] + expected[:, 3],
        ),
        dim=-1,
    )
    left_top = torch.maximum(predicted[:, :2], target[:, :2])
    right_bottom = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection_size = (right_bottom - left_top).clamp_min(0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    prediction_area = (predicted[:, 2] - predicted[:, 0]).clamp_min(0) * (predicted[:, 3] - predicted[:, 1]).clamp_min(0)
    target_area = (target[:, 2] - target[:, 0]).clamp_min(0) * (target[:, 3] - target[:, 1]).clamp_min(0)
    iou = intersection / (prediction_area + target_area - intersection).clamp_min(1e-6)
    lower = target_ltrb.long()
    upper = (lower + 1).clamp(max=reg_max - 1)
    upper_weight = target_ltrb - lower.float()
    lower_weight = 1.0 - upper_weight
    log_probability = probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
    dfl = -(
        lower_weight * log_probability.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
        + upper_weight * log_probability.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    ).mean(-1)
    return (1.0 - iou + dfl).detach()


def anchor_correction_cosine(
    low_probability: torch.Tensor,
    sar_probability: torch.Tensor,
    optical_probability: torch.Tensor,
    *,
    reg_max: int,
    stride_tensor: torch.Tensor,
    anchor_indices: torch.Tensor,
    student_imgsz: int,
) -> torch.Tensor:
    """Cosine between SAR and optical four-side corrections in LR coordinates.

    All three probability tensors are already expressed in the student's DFL
    bins.  The decoded correction uses normalized LR pixels; anchors cancel in
    the teacher-minus-student difference, so only the foreground strides are
    needed here.  A zero correction provides no directional evidence and has
    cosine zero.
    """

    expected_shape = (int(anchor_indices.numel()), 4, int(reg_max))
    if not (
        tuple(low_probability.shape)
        == tuple(sar_probability.shape)
        == tuple(optical_probability.shape)
        == expected_shape
    ):
        raise FMResourceDistillationError("CMV correction probabilities must be matching [N,4,R] tensors")
    bins = torch.arange(reg_max, device=low_probability.device, dtype=torch.float32)
    stride = stride_tensor[anchor_indices.long()].float().reshape(-1, 1)
    direction = low_probability.new_tensor((-1.0, -1.0, 1.0, 1.0), dtype=torch.float32).reshape(1, 4)

    def relative_box(probability: torch.Tensor) -> torch.Tensor:
        distance = (probability.detach().float() * bins).sum(-1) * stride
        return distance * direction / float(student_imgsz)

    low_box = relative_box(low_probability)
    sar_correction = relative_box(sar_probability) - low_box
    optical_correction = relative_box(optical_probability) - low_box
    denominator = sar_correction.norm(dim=-1) * optical_correction.norm(dim=-1)
    nonzero = denominator > torch.finfo(torch.float32).eps
    cosine = torch.zeros_like(denominator)
    cosine[nonzero] = (sar_correction[nonzero] * optical_correction[nonzero]).sum(-1) / denominator[nonzero]
    return cosine.clamp(-1.0, 1.0).detach()


def cmv_object_gate(
    low_error: torch.Tensor,
    full_error: torch.Tensor,
    sar_error: torch.Tensor,
    optical_error: torch.Tensor,
    anchor_cosine: torch.Tensor,
    weights: torch.Tensor,
    *,
    batch_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> CMVGate:
    """Build resource/SAR-safe support and the reliability-conditioned veto."""

    vectors = (
        low_error,
        full_error,
        sar_error,
        optical_error,
        anchor_cosine,
        weights,
        batch_indices,
        target_indices,
    )
    if any(value.ndim != 1 or value.shape != low_error.shape for value in vectors):
        raise FMResourceDistillationError("CMV gate inputs must be matching foreground vectors")
    empty = torch.zeros_like(low_error, dtype=torch.bool)
    if not low_error.numel():
        return CMVGate(
            empty,
            empty,
            empty,
            empty,
            low_error.detach(),
            empty,
            empty,
            empty,
            torch.empty((0, 2), device=low_error.device, dtype=torch.long),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    pairs = torch.stack((batch_indices.long(), target_indices.long()), dim=1)
    objects, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    resource = empty.clone()
    sar_safe = empty.clone()
    safe = empty.clone()
    optical_reliable = empty.clone()
    agreement = torch.zeros_like(anchor_cosine, dtype=torch.float32)
    conflict = empty.clone()
    keep = empty.clone()
    selected = empty.clone()
    object_decisions: list[tuple[bool, bool, bool, bool, bool, bool, bool]] = []
    for object_id in range(int(objects.shape[0])):
        members = inverse == object_id
        damaged = bool((low_error[members].mean() > full_error[members].mean()).item())
        sar_is_safe = bool((low_error[members].mean() > sar_error[members].mean()).item())
        base_safe = damaged and sar_is_safe
        optical_is_reliable = bool((optical_error[members].mean() < sar_error[members].mean()).item())
        member_weights = weights[members].detach().float().clamp_min(0.0)
        denominator = member_weights.sum()
        mean_cosine = (
            (anchor_cosine[members].float() * member_weights).sum() / denominator
            if bool(denominator > 0)
            else anchor_cosine[members].float().mean()
        )
        is_conflict = bool((mean_cosine < 0).item())
        is_kept = (not optical_is_reliable) or (not is_conflict)
        is_selected = base_safe and is_kept
        resource[members] = damaged
        sar_safe[members] = sar_is_safe
        safe[members] = base_safe
        optical_reliable[members] = optical_is_reliable
        agreement[members] = mean_cosine
        conflict[members] = is_conflict
        keep[members] = is_kept
        selected[members] = is_selected
        object_decisions.append(
            (damaged, sar_is_safe, base_safe, optical_is_reliable, is_conflict, base_safe and not is_kept, is_selected)
        )

    count = float(len(object_decisions))
    fractions = [sum(int(values[index]) for values in object_decisions) / count for index in range(7)]
    return CMVGate(
        resource,
        sar_safe,
        safe,
        optical_reliable,
        agreement.detach(),
        conflict,
        keep,
        selected,
        objects[torch.tensor([values[-1] for values in object_decisions], device=objects.device, dtype=torch.bool)],
        *fractions,
    )


def matched_random_object_mask(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    batch_indices: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Randomly retain the reference object dose per image from candidates."""

    if not (candidate.shape == reference.shape == batch_indices.shape == target_indices.shape):
        raise FMResourceDistillationError("matched random selection requires matching foreground vectors")
    selected = torch.zeros_like(candidate, dtype=torch.bool)
    pairs = torch.stack((batch_indices.long(), target_indices.long()), dim=1)
    objects, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    for image in torch.unique(batch_indices, sorted=True):
        image_objects = (objects[:, 0] == image).nonzero(as_tuple=False).flatten()
        candidates = torch.stack(
            [object_id for object_id in image_objects if bool(candidate[inverse == object_id].any())]
        ) if any(bool(candidate[inverse == object_id].any()) for object_id in image_objects) else image_objects[:0]
        dose = sum(bool(reference[inverse == object_id].any()) for object_id in image_objects)
        if dose and candidates.numel():
            order = torch.randperm(int(candidates.numel()), generator=generator, device="cpu").to(candidates.device)
            for object_id in candidates[order[: min(dose, int(candidates.numel()))]]:
                selected[inverse == object_id] = True
    return selected


def matched_object_score_mask(
    score: torch.Tensor,
    batch_indices: torch.Tensor,
    target_indices: torch.Tensor,
    reference: torch.Tensor,
    *,
    candidate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the highest-scoring objects with the reference dose per image."""

    if not (score.shape == batch_indices.shape == target_indices.shape == reference.shape):
        raise FMResourceDistillationError("matched object score selection requires matching foreground vectors")
    if candidate is None:
        candidate = torch.ones_like(reference, dtype=torch.bool)
    if candidate.shape != reference.shape:
        raise FMResourceDistillationError("matched object score candidate mask has the wrong shape")
    selected = torch.zeros_like(reference, dtype=torch.bool)
    pairs = torch.stack((batch_indices.long(), target_indices.long()), dim=1)
    objects, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    for image in torch.unique(batch_indices, sorted=True):
        image_objects = (objects[:, 0] == image).nonzero(as_tuple=False).flatten()
        candidates = [object_id for object_id in image_objects if bool(candidate[inverse == object_id].any())]
        dose = sum(bool(reference[inverse == object_id].any()) for object_id in image_objects)
        if dose and candidates:
            object_scores = torch.stack([score[inverse == object_id].detach().float().mean() for object_id in candidates])
            order = torch.argsort(object_scores, descending=True, stable=True)
            for index in order[: min(dose, len(candidates))]:
                selected[inverse == candidates[int(index)]] = True
    return selected


def mixed_teacher_target(
    sar_target: torch.Tensor,
    optical_target: torch.Tensor,
    use_optical: torch.Tensor,
) -> torch.Tensor:
    """Choose one detached DFL teacher per foreground anchor."""

    if sar_target.shape != optical_target.shape or use_optical.shape != sar_target.shape[:1]:
        raise FMResourceDistillationError("teacher arbitration target shapes disagree")
    return torch.where(use_optical[:, None, None], optical_target, sar_target).detach()


def resource_optical_object_gate(
    low_error: torch.Tensor,
    full_error: torch.Tensor,
    optical_error: torch.Tensor,
    *,
    batch_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> ObjectGate:
    """Implement m_j = 1[r_j>0]1[o_j>0] without a tuned threshold."""

    if not (
        low_error.shape == full_error.shape == optical_error.shape == batch_indices.shape == target_indices.shape
    ):
        raise FMResourceDistillationError("resource gate inputs must be matching P3 foreground vectors")
    count = int(low_error.numel())
    empty = torch.zeros_like(low_error, dtype=torch.bool)
    if count == 0:
        return ObjectGate(empty, empty, empty, torch.empty((0, 2), device=low_error.device, dtype=torch.long), 0.0, 0.0, 0.0)
    pairs = torch.stack((batch_indices.long(), target_indices.long()), dim=1)
    unique, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    resource = torch.zeros_like(empty)
    optical = torch.zeros_like(empty)
    selected = torch.zeros_like(empty)
    for object_id in range(int(unique.shape[0])):
        members = inverse == object_id
        damaged = bool((low_error[members].mean() > full_error[members].mean()).item())
        beneficial = bool((full_error[members].mean() > optical_error[members].mean()).item())
        resource[members] = damaged
        optical[members] = beneficial
        selected[members] = damaged and beneficial
    resource_objects = []
    optical_objects = []
    selected_objects = []
    for object_id in range(int(unique.shape[0])):
        members = inverse == object_id
        resource_objects.append(bool(resource[members].any()))
        optical_objects.append(bool(optical[members].any()))
        selected_objects.append(bool(selected[members].any()))
    return ObjectGate(
        resource,
        optical,
        selected,
        unique[torch.tensor(selected_objects, device=unique.device, dtype=torch.bool)],
        float(sum(resource_objects) / len(resource_objects)),
        float(sum(optical_objects) / len(optical_objects)),
        float(sum(selected_objects) / len(selected_objects)),
    )


def selected_gate_object_rows(gate: ObjectGate, batch_indices: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    """Return one [batch, target_gt_idx] row for each selected target."""

    if not bool(gate.selected.any()):
        return torch.empty((0, 2), dtype=torch.long, device=batch_indices.device)
    return torch.unique(torch.stack((batch_indices[gate.selected], target_indices[gate.selected]), dim=1), dim=0, sorted=True)


def matched_hard_mask(
    low_error: torch.Tensor,
    batch_indices: torch.Tensor,
    target_indices: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Select hard objects, matching the gate's selected-object dose per image."""

    if not (low_error.shape == batch_indices.shape == target_indices.shape == reference.shape):
        raise FMResourceDistillationError("matched hard selection requires matching foreground vectors")
    selected = torch.zeros_like(reference, dtype=torch.bool)
    pairs = torch.stack((batch_indices.long(), target_indices.long()), dim=1)
    objects, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    for image in torch.unique(batch_indices, sorted=True):
        object_ids = (objects[:, 0] == image).nonzero(as_tuple=False).flatten()
        count = sum(bool(reference[inverse == object_id].any()) for object_id in object_ids)
        if count:
            object_error = torch.stack([low_error[inverse == object_id].mean() for object_id in object_ids])
            order = torch.argsort(object_error, descending=True, stable=True)
            chosen = object_ids[order[:count]]
            for object_id in chosen:
                selected[inverse == object_id] = True
    return selected


def probability_dfl_kl(
    target: torch.Tensor,
    student_logits: torch.Tensor,
    weights: torch.Tensor,
    selected: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """FP32 T=1 target-score-weighted KL over a selected P3 foreground set."""

    if not bool(selected.any()):
        return student_logits.float().sum() * 0.0
    target_probability = target.detach().float()
    student_log_probability = functional.log_softmax(student_logits.float(), dim=-1)
    per_anchor = (
        target_probability
        * (target_probability.clamp_min(torch.finfo(torch.float32).tiny).log() - student_log_probability)
    ).sum(-1).mean(-1)
    selected_weight = weights.float() * selected.float()
    return float(batch_size) * (per_anchor * selected_weight).sum() / selected_weight.sum().clamp_min(torch.finfo(torch.float32).eps)


def mars_roi_cosine_loss(student_roi: torch.Tensor, mars_roi: torch.Tensor) -> torch.Tensor:
    """Spatially pool 3x3 ROI features and compare their L2-normalized vectors."""

    if student_roi.shape != mars_roi.shape:
        raise FMResourceDistillationError("student and MaRS ROI features must match")
    if student_roi.numel() == 0:
        return student_roi.sum() * 0.0
    student_vector = functional.normalize(student_roi.mean(dim=(-2, -1)), dim=-1)
    mars_vector = functional.normalize(mars_roi.detach().mean(dim=(-2, -1)), dim=-1)
    return (1.0 - (student_vector * mars_vector).sum(-1)).mean()


def combine_rcfmd_loss(native_loss: torch.Tensor, dfl_loss: torch.Tensor, mars_loss: torch.Tensor, *, lambda_mars: float = 0.05) -> torch.Tensor:
    """Fixed RCFMD objective; ``lambda_mars=0`` is the exact rc_dfl ablation."""

    return native_loss + 0.1 * dfl_loss + float(lambda_mars) * mars_loss


def _concatenate_raw_predictions(predictions: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(predictions) == 1:
        return predictions[0]
    features = [prediction.get("feats") for prediction in predictions]
    if not all(isinstance(value, (tuple, list)) and len(value) == 3 for value in features):
        raise FMResourceDistillationError("chunked teacher predictions lack native P3/P4/P5 features")
    return {
        "boxes": torch.cat([prediction["boxes"] for prediction in predictions], dim=0),
        "scores": torch.cat([prediction["scores"] for prediction in predictions], dim=0),
        "feats": [torch.cat([value[level] for value in features], dim=0) for level in range(3)],
    }


def _chunked_frozen_teacher_prediction(teacher: nn.Module, images: torch.Tensor, *, chunk_size: int) -> Mapping[str, Any]:
    predictions = [frozen_teacher_prediction(teacher, chunk) for chunk in images.split(int(chunk_size), dim=0)]
    return _concatenate_raw_predictions(predictions)


def _student_shadow_prediction(
    student_model: nn.Module,
    canonical_sar: torch.Tensor,
    *,
    chunk_size: int,
) -> Mapping[str, Any]:
    was_training = student_model.training
    student_model.eval()
    try:
        with torch.no_grad():
            predictions = [_raw_prediction(student_model(chunk)) for chunk in canonical_sar.split(int(chunk_size), dim=0)]
            return _concatenate_raw_predictions(predictions)
    finally:
        student_model.train(was_training)


def _object_rois(
    object_rows: torch.Tensor,
    assignment: B0Assignment,
    *,
    scale: float,
) -> torch.Tensor:
    if object_rows.numel() == 0:
        return assignment.target_bboxes.new_zeros((0, 5))
    boxes: list[torch.Tensor] = []
    for batch_index, target_index in object_rows.tolist():
        candidates = ((assignment.target_gt_idx[int(batch_index)] == int(target_index)) & assignment.fg_mask[int(batch_index)]).nonzero(as_tuple=False)
        if candidates.numel():
            box = assignment.target_bboxes[int(batch_index), int(candidates[0, 0])].float() * float(scale)
            boxes.append(torch.cat((box.new_tensor([float(batch_index)]), box)))
    return torch.stack(boxes) if boxes else assignment.target_bboxes.new_zeros((0, 5))


def mars_roi_loss(
    *,
    adapter: P3MaRSAdapter,
    student_p3: torch.Tensor,
    mars_p3: torch.Tensor,
    object_rows: torch.Tensor,
    assignment: B0Assignment,
    student_imgsz: int,
    canonical_imgsz: int,
) -> torch.Tensor:
    if object_rows.numel() == 0:
        return student_p3.sum() * 0.0
    try:
        from torchvision.ops import roi_align
    except ImportError as exc:  # pragma: no cover - production environment dependency
        raise FMResourceDistillationError("RCFMD MaRS ROI loss requires torchvision") from exc
    student_rois = _object_rois(object_rows, assignment, scale=1.0)
    mars_rois = _object_rois(object_rows, assignment, scale=float(canonical_imgsz) / float(student_imgsz))
    student_scale = float(student_p3.shape[-1]) / float(student_imgsz)
    mars_scale = float(mars_p3.shape[-1]) / float(canonical_imgsz)
    student_feature = adapter(student_p3)
    student_roi = roi_align(student_feature, student_rois, output_size=(3, 3), spatial_scale=student_scale, aligned=True, sampling_ratio=2)
    mars_roi = roi_align(mars_p3, mars_rois, output_size=(3, 3), spatial_scale=mars_scale, aligned=True, sampling_ratio=2)
    return mars_roi_cosine_loss(student_roi, mars_roi)


class FMResourceDistillationLoss:
    """Ultralytics criterion wrapper for all frozen RCFMD method arms."""

    def __init__(
        self,
        native: Any,
        *,
        method: str,
        student_model: nn.Module,
        canonical_imgsz: int,
        student_imgsz: int,
        optical_teacher: nn.Module | None = None,
        sar_teacher: nn.Module | None = None,
        mars_teacher: FrozenMaRSRGB | None = None,
        adapter: P3MaRSAdapter | None = None,
        mars_chunk: int = 4,
        teacher_chunk: int = 8,
        lambda_mars: float = 0.05,
        detector_seed: int = 42,
    ) -> None:
        if method not in METHODS:
            raise FMResourceDistillationError(f"unknown RCFMD method {method!r}")
        self.native = native
        self.method = method
        self.student_model = student_model
        self.canonical_imgsz = int(canonical_imgsz)
        self.student_imgsz = int(student_imgsz)
        self.optical_teacher = optical_teacher
        self.sar_teacher = sar_teacher
        self.mars_teacher = mars_teacher
        self.adapter = adapter
        self.mars_chunk = int(mars_chunk)
        self.teacher_chunk = int(teacher_chunk)
        self.lambda_mars = float(lambda_mars)
        self.detector_seed = int(detector_seed)
        self._selection_generator = torch.Generator(device="cpu")
        self._selection_generator.manual_seed(self.detector_seed)
        self.stats_rows: list[dict[str, float]] = []

    def _require_optical(self) -> nn.Module:
        if self.optical_teacher is None:
            raise FMResourceDistillationError(f"{self.method} requires --optical-final")
        return self.optical_teacher

    def _require_sar(self) -> nn.Module:
        if self.sar_teacher is None:
            raise FMResourceDistillationError(f"{self.method} requires --sar-teacher-final")
        return self.sar_teacher

    def _gate(
        self,
        *,
        native: Any,
        p3_mask: torch.Tensor,
        reg_max: int,
        low_shadow: Mapping[str, Any],
        full_shadow: Mapping[str, Any],
        optical: Mapping[str, Any],
    ) -> tuple[ObjectGate, torch.Tensor, torch.Tensor]:
        """Compute the counterfactual gate from eval-mode low/full shadows.

        ``native.parsed_prediction`` is intentionally absent from all three
        counterfactual errors: it is the train-mode prediction that receives
        the DFL gradient, while both SAR comparisons must share eval behavior.
        """
        student_p3_logits = _p3_dfl_logits(native.parsed_prediction, reg_max=reg_max)
        low_shadow_probability = functional.softmax(
            _p3_dfl_logits(low_shadow, reg_max=reg_max).float(), dim=-1
        )[p3_mask[:, : student_p3_logits.shape[1]]]
        full_shadow_probability = cross_resolution_dfl_transport(
            _p3_dfl_logits(full_shadow, reg_max=reg_max),
            teacher_p3_shape=_p3_shape(full_shadow),
            student_p3_shape=_p3_shape(native.parsed_prediction),
            canonical_imgsz=self.canonical_imgsz,
            student_imgsz=self.student_imgsz,
            student_reg_max=reg_max,
        )[p3_mask[:, : student_p3_logits.shape[1]]]
        optical_probability = cross_resolution_dfl_transport(
            _p3_dfl_logits(optical, reg_max=reg_max),
            teacher_p3_shape=_p3_shape(optical),
            student_p3_shape=_p3_shape(native.parsed_prediction),
            canonical_imgsz=self.canonical_imgsz,
            student_imgsz=self.student_imgsz,
            student_reg_max=reg_max,
        )[p3_mask[:, : student_p3_logits.shape[1]]]
        low_error = _p3_localization_error(low_shadow_probability, native.assignment, p3_mask, reg_max=reg_max)
        full_error = _p3_localization_error(full_shadow_probability, native.assignment, p3_mask, reg_max=reg_max)
        optical_error = _p3_localization_error(optical_probability, native.assignment, p3_mask, reg_max=reg_max)
        rows = p3_mask.nonzero(as_tuple=False)
        gate = resource_optical_object_gate(
            low_error,
            full_error,
            optical_error,
            batch_indices=rows[:, 0],
            target_indices=native.assignment.target_gt_idx[p3_mask],
        )
        return gate, optical_probability, low_error

    def _cmv_state(
        self,
        *,
        native: Any,
        p3_mask: torch.Tensor,
        p3_weights: torch.Tensor,
        reg_max: int,
        low_shadow: Mapping[str, Any],
        full_shadow: Mapping[str, Any],
        sar_high: Mapping[str, Any],
        optical_high: Mapping[str, Any] | None,
    ) -> CMVState:
        """Compute detached CMV counterfactual state from one native assignment."""

        student_shape = _p3_shape(native.parsed_prediction)
        student_p3_count = student_shape[0] * student_shape[1]
        foreground = p3_mask[:, :student_p3_count]
        low_probability = functional.softmax(
            _p3_dfl_logits(low_shadow, reg_max=reg_max).float(), dim=-1
        )[foreground]

        def transported(prediction: Mapping[str, Any]) -> torch.Tensor:
            return cross_resolution_dfl_transport(
                _p3_dfl_logits(prediction, reg_max=reg_max),
                teacher_p3_shape=_p3_shape(prediction),
                student_p3_shape=student_shape,
                canonical_imgsz=self.canonical_imgsz,
                student_imgsz=self.student_imgsz,
                student_reg_max=reg_max,
            )[foreground]

        full_probability = transported(full_shadow)
        sar_target = transported(sar_high)
        optical_target = transported(optical_high) if optical_high is not None else sar_target
        low_error = _p3_localization_error(
            low_probability, native.assignment, p3_mask, reg_max=reg_max
        )
        full_error = _p3_localization_error(
            full_probability, native.assignment, p3_mask, reg_max=reg_max
        )
        sar_error = _p3_localization_error(
            sar_target, native.assignment, p3_mask, reg_max=reg_max
        )
        optical_error = (
            _p3_localization_error(optical_target, native.assignment, p3_mask, reg_max=reg_max)
            if optical_high is not None
            else sar_error
        )
        rows = p3_mask.nonzero(as_tuple=False)
        cosine = anchor_correction_cosine(
            low_probability,
            sar_target,
            optical_target,
            reg_max=reg_max,
            stride_tensor=native.assignment.stride_tensor,
            anchor_indices=rows[:, 1],
            student_imgsz=self.student_imgsz,
        )
        gate = cmv_object_gate(
            low_error,
            full_error,
            sar_error,
            optical_error,
            cosine,
            p3_weights,
            batch_indices=rows[:, 0],
            target_indices=native.assignment.target_gt_idx[p3_mask],
        )
        return CMVState(
            gate=gate,
            sar_target=sar_target.detach(),
            optical_target=optical_target.detach(),
            low_error=low_error,
            full_error=full_error,
            sar_error=sar_error,
            optical_error=optical_error,
        )

    def __call__(self, prediction: Any, batch: dict[str, torch.Tensor]):
        if not self.student_model.training:
            return self.native(prediction, batch)
        native = native_student_loss(self.native, prediction, batch)
        zero = native.native_loss * 0.0
        if self.method == "lr_native" or not bool(native.assignment.fg_mask.any()):
            result = FMResourceResult(native.native_loss, zero, zero, native.native_loss, native.detached_loss, _zero_stats())
            self.stats_rows.append(result.stats)
            return result.total_loss, _detached(result)
        if self.method in PAIRED_DATA_METHODS and "eo_img" not in batch:
            raise FMResourceDistillationError(f"{self.method} requires synchronized paired EO")
        reg_max = criterion_reg_max(self.native)
        p3_mask = p3_foreground_mask(native.parsed_prediction, native.assignment)
        if not bool(p3_mask.any()):
            result = FMResourceResult(native.native_loss, zero, zero, native.native_loss, native.detached_loss, _zero_stats())
            self.stats_rows.append(result.stats)
            return result.total_loss, _detached(result)
        student_p3_logits = _p3_dfl_logits(native.parsed_prediction, reg_max=reg_max)
        p3_student = student_p3_logits[p3_mask[:, : student_p3_logits.shape[1]]]
        p3_weights = native.assignment.target_scores[:, : student_p3_logits.shape[1]].sum(-1)[p3_mask[:, : student_p3_logits.shape[1]]]
        p3_rows = p3_mask.nonzero(as_tuple=False)
        all_selected = torch.ones(p3_student.shape[0], device=p3_student.device, dtype=torch.bool)
        dfl_loss = zero
        mars_loss = zero
        gate: ObjectGate | CMVGate | None = None
        selected_for_stats: torch.Tensor | None = None
        teacher_forwards = 0.0

        if self.method == "same_res_optical":
            teacher = self._require_optical()
            optical_low = _chunked_frozen_teacher_prediction(teacher, batch["eo_img_low"], chunk_size=self.teacher_chunk)
            target = functional.softmax(_p3_dfl_logits(optical_low, reg_max=reg_max).float(), dim=-1)[p3_mask[:, : student_p3_logits.shape[1]]]
            dfl_loss = probability_dfl_kl(target, p3_student, p3_weights, all_selected, batch_size=int(batch["img"].shape[0]))
            teacher_forwards = 1.0
        elif self.method == "hr_sar_uniform":
            sar_high = _chunked_frozen_teacher_prediction(
                self._require_sar(), batch["canonical_sar_img"], chunk_size=self.teacher_chunk
            )
            teacher_forwards = 1.0
            target = cross_resolution_dfl_transport(
                _p3_dfl_logits(sar_high, reg_max=reg_max),
                teacher_p3_shape=_p3_shape(sar_high),
                student_p3_shape=_p3_shape(native.parsed_prediction),
                canonical_imgsz=self.canonical_imgsz,
                student_imgsz=self.student_imgsz,
                student_reg_max=reg_max,
            )[p3_mask[:, : student_p3_logits.shape[1]]]
            dfl_loss = probability_dfl_kl(target, p3_student, p3_weights, all_selected, batch_size=int(batch["img"].shape[0]))
        elif self.method in CMV_METHODS:
            low_shadow = _student_shadow_prediction(
                self.student_model, batch["img"], chunk_size=self.teacher_chunk
            )
            full_shadow = _student_shadow_prediction(
                self.student_model, batch["canonical_sar_img"], chunk_size=self.teacher_chunk
            )
            sar_high = _chunked_frozen_teacher_prediction(
                self._require_sar(), batch["canonical_sar_img"], chunk_size=self.teacher_chunk
            )
            teacher_forwards = 3.0
            optical_high = None
            if self.method != "safe_sar":
                optical_key = "shuffled_eo_img" if self.method == "cmv_kd_shuffled" else "eo_img"
                if optical_key not in batch:
                    raise FMResourceDistillationError(f"{self.method} requires {optical_key}")
                optical_high = _chunked_frozen_teacher_prediction(
                    self._require_optical(), batch[optical_key], chunk_size=self.teacher_chunk
                )
                teacher_forwards += 1.0
            state = self._cmv_state(
                native=native,
                p3_mask=p3_mask,
                p3_weights=p3_weights,
                reg_max=reg_max,
                low_shadow=low_shadow,
                full_shadow=full_shadow,
                sar_high=sar_high,
                optical_high=optical_high,
            )
            gate = state.gate
            selected = gate.selected
            target = state.sar_target
            target_indices = native.assignment.target_gt_idx[p3_mask]
            if self.method == "safe_sar":
                selected = gate.safe
            elif self.method == "cmv_random_veto":
                selected = matched_random_object_mask(
                    gate.safe,
                    gate.selected,
                    p3_rows[:, 0],
                    target_indices,
                    generator=self._selection_generator,
                )
            elif self.method == "cmv_arbitration":
                target = mixed_teacher_target(
                    state.sar_target, state.optical_target, gate.optical_reliable
                )
            elif self.method == "cmv_optical_target":
                target = state.optical_target
            elif self.method == "sar_discrepancy_dose":
                sar_discrepancy = (
                    state.sar_target
                    * (
                        state.sar_target.clamp_min(torch.finfo(torch.float32).tiny).log()
                        - functional.log_softmax(p3_student.detach().float(), dim=-1)
                    )
                ).sum(-1).mean(-1)
                selected = matched_object_score_mask(
                    sar_discrepancy,
                    p3_rows[:, 0],
                    target_indices,
                    gate.selected,
                )
            elif self.method == "resource_arbitration":
                selected = gate.resource
                target = mixed_teacher_target(
                    state.sar_target, state.optical_target, gate.optical_reliable
                )
            selected_for_stats = selected
            dfl_loss = probability_dfl_kl(
                target,
                p3_student,
                p3_weights,
                selected,
                batch_size=int(batch["img"].shape[0]),
            )
        else:
            optical_high = _chunked_frozen_teacher_prediction(
                self._require_optical(), batch["eo_img"], chunk_size=self.teacher_chunk
            )
            teacher_forwards = 1.0
            optical_target = cross_resolution_dfl_transport(
                _p3_dfl_logits(optical_high, reg_max=reg_max),
                teacher_p3_shape=_p3_shape(optical_high),
                student_p3_shape=_p3_shape(native.parsed_prediction),
                canonical_imgsz=self.canonical_imgsz,
                student_imgsz=self.student_imgsz,
                student_reg_max=reg_max,
            )[p3_mask[:, : student_p3_logits.shape[1]]]
            if self.method == "hr_optical_uniform":
                dfl_loss = probability_dfl_kl(
                    optical_target, p3_student, p3_weights, all_selected, batch_size=int(batch["img"].shape[0])
                )
            elif self.method == "teacher_quality_soft":
                quality = torch.exp(-_p3_localization_error(optical_target, native.assignment, p3_mask, reg_max=reg_max))
                quality_weights = p3_weights * quality / quality.mean().clamp_min(1e-6)
                dfl_loss = probability_dfl_kl(
                    optical_target, p3_student, quality_weights, all_selected, batch_size=int(batch["img"].shape[0])
                )
            else:
                if self.method not in GATED_METHODS:
                    raise FMResourceDistillationError(f"{self.method} has no RCFMD loss route")
                low_shadow = _student_shadow_prediction(self.student_model, batch["img"], chunk_size=self.teacher_chunk)
                full_shadow = _student_shadow_prediction(
                    self.student_model, batch["canonical_sar_img"], chunk_size=self.teacher_chunk
                )
                teacher_forwards += 2.0
                gate, optical_target, low_error = self._gate(
                    native=native,
                    p3_mask=p3_mask,
                    reg_max=reg_max,
                    low_shadow=low_shadow,
                    full_shadow=full_shadow,
                    optical=optical_high,
                )
                if self.method == "resource_gate":
                    selected = gate.resource
                    target = optical_target
                elif self.method == "optical_gate":
                    selected = gate.optical
                    target = optical_target
                elif self.method == "rc_dfl_hard":
                    selected = matched_hard_mask(
                        low_error, p3_rows[:, 0], native.assignment.target_gt_idx[p3_mask], gate.selected
                    )
                    target = optical_target
                elif self.method == "dfl_discrepancy_object":
                    discrepancy = (
                        optical_target
                        * (
                            optical_target.clamp_min(torch.finfo(torch.float32).tiny).log()
                            - functional.log_softmax(p3_student.float(), dim=-1)
                        )
                    ).sum(-1).mean(-1)
                    selected = matched_hard_mask(
                        discrepancy, p3_rows[:, 0], native.assignment.target_gt_idx[p3_mask], gate.selected
                    )
                    target = optical_target
                elif self.method == "rc_dfl_same_modal":
                    sar_high = _chunked_frozen_teacher_prediction(
                        self._require_sar(), batch["canonical_sar_img"], chunk_size=self.teacher_chunk
                    )
                    teacher_forwards += 1.0
                    selected = gate.selected
                    target = cross_resolution_dfl_transport(
                        _p3_dfl_logits(sar_high, reg_max=reg_max),
                        teacher_p3_shape=_p3_shape(sar_high),
                        student_p3_shape=_p3_shape(native.parsed_prediction),
                        canonical_imgsz=self.canonical_imgsz,
                        student_imgsz=self.student_imgsz,
                        student_reg_max=reg_max,
                    )[p3_mask[:, : student_p3_logits.shape[1]]]
                elif self.method == "rc_dfl_shuffled":
                    if "shuffled_eo_img" not in batch:
                        raise FMResourceDistillationError("rc_dfl_shuffled requires shuffled paired EO")
                    shuffled = _chunked_frozen_teacher_prediction(
                        self._require_optical(), batch["shuffled_eo_img"], chunk_size=self.teacher_chunk
                    )
                    teacher_forwards += 1.0
                    target = cross_resolution_dfl_transport(
                        _p3_dfl_logits(shuffled, reg_max=reg_max),
                        teacher_p3_shape=_p3_shape(shuffled),
                        student_p3_shape=_p3_shape(native.parsed_prediction),
                        canonical_imgsz=self.canonical_imgsz,
                        student_imgsz=self.student_imgsz,
                        student_reg_max=reg_max,
                    )[p3_mask[:, : student_p3_logits.shape[1]]]
                    selected = gate.selected
                else:
                    selected = gate.selected
                    target = optical_target
                if self.method != "mars_roi":
                    dfl_loss = probability_dfl_kl(target, p3_student, p3_weights, selected, batch_size=int(batch["img"].shape[0]))

                if self.method in MARS_METHODS:
                    if self.mars_teacher is None or self.adapter is None:
                        raise FMResourceDistillationError(f"{self.method} requires frozen MaRS and a P3 adapter")
                    # RCFMD's shuffled control keeps paired optical DFL and
                    # changes only the MaRS RGB input.
                    mars_key = "shuffled_eo_img" if self.method == "rcfmd_shuffled" else "eo_img"
                    if mars_key not in batch:
                        raise FMResourceDistillationError(f"{self.method} requires {mars_key}")
                    mars_p3 = self.mars_teacher(batch[mars_key], chunk_size=self.mars_chunk)
                    object_rows = selected_gate_object_rows(gate, p3_rows[:, 0], native.assignment.target_gt_idx[p3_mask])
                    student_features = native.parsed_prediction["feats"][0]
                    mars_loss = mars_roi_loss(
                        adapter=self.adapter,
                        student_p3=student_features,
                        mars_p3=mars_p3,
                        object_rows=object_rows,
                        assignment=native.assignment,
                        student_imgsz=self.student_imgsz,
                        canonical_imgsz=self.canonical_imgsz,
                    )
                    teacher_forwards += 1.0

        total = combine_rcfmd_loss(native.native_loss, dfl_loss, mars_loss, lambda_mars=self.lambda_mars if self.method in MARS_METHODS else 0.0)
        stats = _gate_stats(
            gate,
            dfl_loss=dfl_loss,
            mars_loss=mars_loss,
            p3_count=p3_student.shape[0],
            teacher_forwards=teacher_forwards,
            selected=selected_for_stats,
            batch_indices=p3_rows[:, 0],
            target_indices=native.assignment.target_gt_idx[p3_mask],
        )
        result = FMResourceResult(native.native_loss, dfl_loss, mars_loss, total, native.detached_loss, stats)
        self.stats_rows.append(stats)
        return result.total_loss, _detached(result)


def _zero_stats() -> dict[str, float]:
    return {
        "resource_damaged_object_fraction": 0.0,
        "optical_beneficial_object_fraction": 0.0,
        "sar_safe_object_fraction": 0.0,
        "safe_support_object_fraction": 0.0,
        "optical_reliable_object_fraction": 0.0,
        "optical_conflict_object_fraction": 0.0,
        "optical_vetoed_object_fraction": 0.0,
        "cmv_selected_object_fraction": 0.0,
        "mean_optical_sar_correction_cosine": 0.0,
        "selected_object_fraction": 0.0,
        "selected_p3_anchor_fraction": 0.0,
        "p3_foreground_count": 0.0,
        "dfl_loss": 0.0,
        "mars_loss": 0.0,
        "teacher_forwards": 0.0,
    }


def _gate_stats(
    gate: ObjectGate | CMVGate | None,
    *,
    dfl_loss: torch.Tensor,
    mars_loss: torch.Tensor,
    p3_count: int,
    teacher_forwards: float,
    selected: torch.Tensor | None = None,
    batch_indices: torch.Tensor | None = None,
    target_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    stats = _zero_stats()
    stats.update(
        {
            "p3_foreground_count": float(p3_count),
            "dfl_loss": float(dfl_loss.detach().item()),
            "mars_loss": float(mars_loss.detach().item()),
            "teacher_forwards": float(teacher_forwards),
        }
    )
    if isinstance(gate, ObjectGate):
        stats.update(
            {
                "resource_damaged_object_fraction": gate.resource_fraction,
                "optical_beneficial_object_fraction": gate.optical_fraction,
                "selected_object_fraction": gate.selected_fraction,
                "selected_p3_anchor_fraction": float(gate.selected.float().mean().item()) if gate.selected.numel() else 0.0,
            }
        )
    elif isinstance(gate, CMVGate):
        actual = gate.selected if selected is None else selected
        if actual.numel() and batch_indices is not None and target_indices is not None:
            pairs = torch.stack((batch_indices.long(), target_indices.long()), dim=1)
            objects, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
            selected_count = sum(bool(actual[inverse == object_id].any()) for object_id in range(int(objects.shape[0])))
            selected_fraction = selected_count / max(int(objects.shape[0]), 1)
        else:
            selected_fraction = 0.0
        stats.update(
            {
                "resource_damaged_object_fraction": gate.resource_fraction,
                "sar_safe_object_fraction": gate.sar_safe_fraction,
                "safe_support_object_fraction": gate.safe_fraction,
                "optical_reliable_object_fraction": gate.optical_reliable_fraction,
                "optical_conflict_object_fraction": gate.conflict_fraction,
                "optical_vetoed_object_fraction": gate.vetoed_fraction,
                "cmv_selected_object_fraction": gate.selected_fraction,
                "mean_optical_sar_correction_cosine": float(gate.agreement.float().mean().item())
                if gate.agreement.numel()
                else 0.0,
                "selected_object_fraction": float(selected_fraction),
                "selected_p3_anchor_fraction": float(actual.float().mean().item()) if actual.numel() else 0.0,
            }
        )
    return stats


def _detached(result: FMResourceResult) -> dict[str, Any]:
    detached = dict(result.detached_loss)
    detached["xres_dfl"] = result.dfl_loss.detach()
    detached["mars_roi"] = result.mars_loss.detach()
    return detached
