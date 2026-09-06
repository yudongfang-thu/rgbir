"""P3 cross-modal method zoo built on one native YOLO assignment."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .b0_distillation import (
    B0Assignment,
    criterion_reg_max,
    dfl_logits,
    frozen_teacher_prediction,
    native_student_loss,
)
from .fp_selective_distillation import gt_dfl_probability, p3_foreground_mask, probability_target_dfl_kl
from .p3_ladd import P3LADDModules, P3RawFeatureAdapter, p3_ladd_loss
from .stage2 import masked_dfl_kl, per_anchor_localization_error


METHODS = (
    "native",
    "p3_dfl",
    "p3_same_modal",
    "p3_ladd",
    "p3_apr_dfl",
    "p3_prw_loc",
    "p3_rankkd",
    "p3_hscard",
    "p3_popa",
    "p3_pemt",
    "p3_gbrd_v2",
    "p3_raw_feature_kd",
    "p3_apr_shuffled",
    "p3_prw_sar",
    "p3_rankkd_shuffled",
    "p3_hscard_shuffled",
    "p3_popa_sar",
    "p3_pemt_consistency",
    "p3_gbrd_shuffled",
    "p3_dfl_rankkd_002",
    "p3_dfl_rankkd_005",
    "p3_apr_dfl_005",
    "p3_dfl_prw_05",
    "p3_pemt_shuffled",
    "p3_pemt_005",
    "p3_dfl_pemt_005",
    "p3_pemt_002",
    "p3_dfl_pemt_002",
    "p3_dfl_pemt_shuffled_005",
    "p3_dfl_pemt_shuffled_002",
    "p3_dfl_pemt_consistency_005",
    "p3_dfl_pemt_consistency_002",
)

OPTICAL_METHODS = frozenset(
    {
        "p3_dfl",
        "p3_ladd",
        "p3_apr_dfl",
        "p3_prw_loc",
        "p3_rankkd",
        "p3_hscard",
        "p3_popa",
        "p3_pemt",
        "p3_gbrd_v2",
        "p3_raw_feature_kd",
        "p3_apr_shuffled",
        "p3_rankkd_shuffled",
        "p3_hscard_shuffled",
        "p3_gbrd_shuffled",
        "p3_dfl_rankkd_002",
        "p3_dfl_rankkd_005",
        "p3_apr_dfl_005",
        "p3_dfl_prw_05",
        "p3_pemt_shuffled",
        "p3_pemt_005",
        "p3_dfl_pemt_005",
        "p3_pemt_002",
        "p3_dfl_pemt_002",
        "p3_dfl_pemt_shuffled_005",
        "p3_dfl_pemt_shuffled_002",
        "p3_dfl_pemt_consistency_005",
        "p3_dfl_pemt_consistency_002",
    }
)
SAR_METHODS = frozenset(
    {"p3_same_modal", "p3_apr_dfl", "p3_apr_dfl_005", "p3_apr_shuffled", "p3_prw_sar", "p3_popa_sar"}
)
PAIRED_METHODS = OPTICAL_METHODS
SHUFFLED_METHODS = frozenset(
    {
        "p3_apr_shuffled",
        "p3_rankkd_shuffled",
        "p3_hscard_shuffled",
        "p3_gbrd_shuffled",
        "p3_pemt_shuffled",
        "p3_dfl_pemt_shuffled_005",
        "p3_dfl_pemt_shuffled_002",
    }
)
AUXILIARY_MODULE_METHODS = frozenset({"p3_ladd", "p3_raw_feature_kd"})
_P3_DFL_RANK_COEFFICIENTS = {"p3_dfl_rankkd_002": 0.02, "p3_dfl_rankkd_005": 0.05}
_APR_DFL_COEFFICIENTS = {"p3_apr_dfl": 0.1, "p3_apr_shuffled": 0.1, "p3_apr_dfl_005": 0.05}
_PEMT_COEFFICIENTS = {
    "p3_pemt_005": 0.05,
    "p3_pemt_002": 0.02,
    "p3_dfl_pemt_005": 0.05,
    "p3_dfl_pemt_002": 0.02,
    "p3_dfl_pemt_shuffled_005": 0.05,
    "p3_dfl_pemt_shuffled_002": 0.02,
    "p3_dfl_pemt_consistency_005": 0.05,
    "p3_dfl_pemt_consistency_002": 0.02,
}
_PEMT_ONLY_METHODS = frozenset(
    {"p3_pemt", "p3_pemt_shuffled", "p3_pemt_consistency", "p3_pemt_005", "p3_pemt_002"}
)
_DFL_PEMT_METHODS = frozenset(
    {
        "p3_dfl_pemt_005",
        "p3_dfl_pemt_002",
        "p3_dfl_pemt_shuffled_005",
        "p3_dfl_pemt_shuffled_002",
        "p3_dfl_pemt_consistency_005",
        "p3_dfl_pemt_consistency_002",
    }
)
_DFL_PEMT_SHUFFLED_METHODS = frozenset(
    {"p3_dfl_pemt_shuffled_005", "p3_dfl_pemt_shuffled_002"}
)
_DFL_PEMT_CONSISTENCY_METHODS = frozenset(
    {"p3_dfl_pemt_consistency_005", "p3_dfl_pemt_consistency_002"}
)
_PEMT_VIEW_CHUNK_SIZE = 16


class FPMethodZooError(RuntimeError):
    pass


@dataclass(frozen=True)
class MethodZooResult:
    native_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    total_loss: torch.Tensor
    detached_loss: Any
    stats: dict[str, float]


def _zero_stats(method: str) -> dict[str, float]:
    return {"p3_density": 0.0, "aux_loss": 0.0, "teacher_forwards": 0.0, "method_code": float(METHODS.index(method))}


def _p3_extent(prediction: Mapping[str, Any]) -> int:
    features = prediction.get("feats")
    if not isinstance(features, (tuple, list)) or len(features) != 3:
        raise FPMethodZooError("method zoo requires native P3/P4/P5 feature maps")
    return int(features[0].shape[-2] * features[0].shape[-1])


def _foreground_rows(assignment: B0Assignment) -> torch.Tensor:
    return assignment.fg_mask.nonzero(as_tuple=False)


def _density(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item()) if mask.numel() else 0.0


def _decoded_boxes(criterion: Any, prediction: Mapping[str, Any], assignment: B0Assignment) -> torch.Tensor:
    raw = prediction["boxes"].permute(0, 2, 1).contiguous()
    return criterion.bbox_decode(assignment.anchor_points, raw.float())


def _aligned_iou(boxes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(boxes[..., :2], target[..., :2])
    right_bottom = torch.minimum(boxes[..., 2:], target[..., 2:])
    size = (right_bottom - left_top).clamp_min(0)
    intersection = size[..., 0] * size[..., 1]
    box_area = (boxes[..., 2] - boxes[..., 0]).clamp_min(0) * (boxes[..., 3] - boxes[..., 1]).clamp_min(0)
    target_area = (target[..., 2] - target[..., 0]).clamp_min(0) * (target[..., 3] - target[..., 1]).clamp_min(0)
    return intersection / (box_area + target_area - intersection).clamp_min(1e-7)


def p3_apr_loss(
    student_dfl: torch.Tensor,
    optical_dfl: torch.Tensor,
    sar_dfl: torch.Tensor,
    assignment: B0Assignment,
    p3: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    target = 0.5 * functional.softmax(optical_dfl.detach().float(), -1)
    target = target + 0.5 * functional.softmax(sar_dfl.detach().float(), -1)
    return probability_target_dfl_kl(target, student_dfl, assignment.target_score_weights, p3, batch_size=batch_size)


def p3_prw_residual(
    criterion: Any,
    prediction: Mapping[str, Any],
    teacher_prediction: Mapping[str, Any],
    assignment: B0Assignment,
    p3: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = teacher_prediction["scores"].detach().float().sigmoid().amax(1)[assignment.fg_mask]
    gt_indices = assignment.target_gt_idx[assignment.fg_mask]
    rows = _foreground_rows(assignment)
    weights = torch.ones_like(scores)
    for image_index in range(int(assignment.fg_mask.shape[0])):
        image_rows = rows[:, 0] == image_index
        for gt_index in torch.unique(gt_indices[image_rows & p3]):
            selected = image_rows & p3 & (gt_indices == gt_index)
            weights[selected] = scores[selected].max()
    active = p3 & (weights > 0)
    mean = weights[active].mean() if bool(active.any()) else weights.new_tensor(1.0)
    weights = (weights / mean.clamp_min(1e-6)).clamp(0.25, 2.0)
    per_anchor = per_anchor_localization_error(criterion, prediction, assignment)
    score_sum = assignment.target_scores.sum().float().clamp_min(1.0)
    residual = float(batch_size) * (((weights - 1.0) * per_anchor) * p3.float()).sum() / score_sum
    return residual, weights[p3]


def p3_rank_loss(
    student_prediction: Mapping[str, Any],
    teacher_prediction: Mapping[str, Any],
    assignment: B0Assignment,
    p3: torch.Tensor,
    *,
    alpha: float = 2.0,
    nearest: int = 8,
) -> torch.Tensor:
    student_scores = student_prediction["scores"].permute(0, 2, 1).float()
    teacher_scores = teacher_prediction["scores"].permute(0, 2, 1).detach().float()
    rows = _foreground_rows(assignment)
    gt_indices = assignment.target_gt_idx[assignment.fg_mask]
    losses: list[torch.Tensor] = []
    for image_index in range(int(assignment.fg_mask.shape[0])):
        image_rows = rows[:, 0] == image_index
        for gt_index in torch.unique(gt_indices[image_rows & p3]):
            positions = (image_rows & p3 & (gt_indices == gt_index)).nonzero(as_tuple=False).flatten()
            if positions.numel() < 2:
                continue
            anchors = rows[positions, 1]
            box = assignment.target_bboxes[image_index, anchors[0]]
            center = 0.5 * (box[:2] + box[2:])
            anchor_pixels = assignment.anchor_points[anchors] * assignment.stride_tensor[anchors]
            order = (anchor_pixels - center).pow(2).sum(-1).argsort(stable=True)[:nearest]
            anchors = anchors[order]
            s = student_scores[image_index, anchors, 0]
            t = teacher_scores[image_index, anchors, 0]
            pair = torch.triu_indices(s.numel(), s.numel(), offset=1, device=s.device)
            teacher_logit = float(alpha) * (t[pair[0]] - t[pair[1]])
            target = teacher_logit.sigmoid()
            student_logit = float(alpha) * (s[pair[0]] - s[pair[1]])
            bce = functional.binary_cross_entropy_with_logits(student_logit, target, reduction="none")
            entropy = functional.binary_cross_entropy_with_logits(teacher_logit, target, reduction="none")
            losses.append((bce - entropy).clamp_min(0).mean())
    return torch.stack(losses).mean() if losses else student_scores.sum() * 0.0


def p3_hscard_loss(
    student_dfl: torch.Tensor,
    teacher_dfl: torch.Tensor,
    assignment: B0Assignment,
    p3: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, float, float]:
    gt_probability = gt_dfl_probability(assignment, reg_max=student_dfl.shape[-1]).float()
    student_probability = functional.softmax(student_dfl.float(), -1)
    teacher_probability = functional.softmax(teacher_dfl.detach().float(), -1)
    gt_gradient = student_probability - gt_probability
    teacher_gradient = student_probability - teacher_probability
    dot = (teacher_gradient * gt_gradient).sum(-1, keepdim=True)
    projected = project_nonconflicting_gradient(teacher_gradient, gt_gradient)
    surrogate = ((student_dfl.float() - student_dfl.detach().float()) * projected.detach()).sum(-1).mean(-1)
    selected_weight = assignment.target_score_weights.float() * p3.float()
    loss = float(batch_size) * (surrogate * selected_weight).sum() / selected_weight.sum().clamp_min(1e-6)
    conflict = float((dot.squeeze(-1)[p3] < 0).float().mean().item()) if bool(p3.any()) else 0.0
    projected_norm = float(projected[p3].norm(dim=-1).mean().item()) if bool(p3.any()) else 0.0
    return loss, conflict, projected_norm


def project_nonconflicting_gradient(teacher_gradient: torch.Tensor, gt_gradient: torch.Tensor) -> torch.Tensor:
    """Remove only the component of the teacher direction that conflicts with GT."""

    if teacher_gradient.shape != gt_gradient.shape:
        raise FPMethodZooError("teacher and GT gradient tensors must match")
    dot = (teacher_gradient * gt_gradient).sum(-1, keepdim=True)
    return teacher_gradient - dot.clamp_max(0) * gt_gradient / gt_gradient.pow(2).sum(-1, keepdim=True).clamp_min(1e-8)


def p3_gbrd_loss(
    criterion: Any,
    student_prediction: Mapping[str, Any],
    teacher_prediction: Mapping[str, Any],
    assignment: B0Assignment,
    p3: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    student = _decoded_boxes(criterion, student_prediction, assignment)[assignment.fg_mask]
    teacher = _decoded_boxes(criterion, teacher_prediction, assignment)[assignment.fg_mask].detach()
    residual = functional.smooth_l1_loss(student.float(), teacher.float(), reduction="none").mean(-1)
    selected_weight = assignment.target_score_weights.float() * p3.float()
    return float(batch_size) * (residual * selected_weight).sum() / selected_weight.sum().clamp_min(1e-6)


def _midranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float32)
    _, inverse, counts = torch.unique_consecutive(
        sorted_values, return_inverse=True, return_counts=True
    )
    starts = counts.cumsum(0) - counts
    midpoints = starts.float() + 0.5 * (counts.float() - 1.0)
    ranks[order] = midpoints[inverse]
    return ranks / max(values.numel() - 1, 1)


def popa_assignment(
    criterion: Any,
    student_prediction: Mapping[str, Any],
    teacher_prediction: Mapping[str, Any],
    assignment: B0Assignment,
    *,
    p3_extent: int,
    optical_mix: float = 0.5,
) -> B0Assignment:
    if optical_mix == 0:
        return assignment
    from scipy.optimize import linear_sum_assignment

    student_boxes = _decoded_boxes(criterion, student_prediction, assignment) * assignment.stride_tensor
    teacher_boxes = _decoded_boxes(criterion, teacher_prediction, assignment).detach() * assignment.stride_tensor
    foreground = assignment.fg_mask.clone()
    target_gt_idx = assignment.target_gt_idx.clone()
    target_bboxes = assignment.target_bboxes.clone()
    target_scores = assignment.target_scores.clone()
    p3_native = assignment.fg_mask[:, :p3_extent].clone()
    foreground[:, :p3_extent] = False
    target_scores[:, :p3_extent] = 0
    p3_anchors = assignment.anchor_points[:p3_extent] * assignment.stride_tensor[:p3_extent]
    for image_index in range(int(assignment.fg_mask.shape[0])):
        native_indices = p3_native[image_index].nonzero(as_tuple=False).flatten()
        if native_indices.numel() == 0:
            continue
        native_gt = assignment.target_gt_idx[image_index, native_indices]
        row_specs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        row_candidates: list[torch.Tensor] = []
        row_utilities: list[torch.Tensor] = []
        for gt_index in torch.unique(native_gt):
            old_indices = native_indices[native_gt == gt_index]
            quota = int(old_indices.numel())
            gt_box = assignment.target_bboxes[image_index, old_indices[0]]
            inside = (
                (p3_anchors[:, 0] >= gt_box[0])
                & (p3_anchors[:, 0] <= gt_box[2])
                & (p3_anchors[:, 1] >= gt_box[1])
                & (p3_anchors[:, 1] <= gt_box[3])
            ).nonzero(as_tuple=False).flatten()
            # The native positives keep the LSAP feasible; every newly eligible
            # anchor still belongs to the native GT-inside candidate geometry.
            candidates = torch.unique(torch.cat((old_indices, inside)), sorted=True)
            target = gt_box.expand(candidates.numel(), 4)
            student_quality = _aligned_iou(student_boxes[image_index, candidates], target)
            teacher_quality = _aligned_iou(teacher_boxes[image_index, candidates], target)
            utility = (1.0 - float(optical_mix)) * _midranks(student_quality) + float(optical_mix) * _midranks(teacher_quality)
            old_scores = assignment.target_scores[image_index, old_indices].clone()
            for slot in range(quota):
                row_specs.append((int(gt_index.item()), gt_box, old_scores[slot]))
                row_candidates.append(candidates)
                row_utilities.append(utility)
        candidate_columns = torch.unique(torch.cat(row_candidates), sorted=True)
        utility_tensor = student_boxes.new_full((len(row_specs), candidate_columns.numel()), -1e6)
        for row_index, (candidates, utility) in enumerate(zip(row_candidates, row_utilities, strict=True)):
            utility_tensor[row_index, torch.searchsorted(candidate_columns, candidates)] = utility
        matrix = utility_tensor.detach().cpu().numpy()
        rows_out, columns_out = linear_sum_assignment(matrix, maximize=True)
        selected_by_gt: dict[int, list[tuple[int, float]]] = {}
        scores_by_gt: dict[int, list[torch.Tensor]] = {}
        boxes_by_gt: dict[int, torch.Tensor] = {}
        for row_index, compact_column in zip(rows_out.tolist(), columns_out.tolist()):
            if matrix[row_index, compact_column] < -1e5:
                raise FPMethodZooError("POPA could not satisfy unique P3 assignment")
            anchor_index = int(candidate_columns[compact_column].item())
            gt_index, gt_box, score = row_specs[row_index]
            selected_by_gt.setdefault(gt_index, []).append((anchor_index, float(matrix[row_index, compact_column])))
            scores_by_gt.setdefault(gt_index, []).append(score)
            boxes_by_gt[gt_index] = gt_box
        for gt_index, selected in selected_by_gt.items():
            selected.sort(key=lambda item: item[1], reverse=True)
            scores = sorted(scores_by_gt[gt_index], key=lambda value: float(value.sum().item()), reverse=True)
            gt_box = boxes_by_gt[gt_index]
            for (anchor_index, _), score in zip(selected, scores, strict=True):
                foreground[image_index, anchor_index] = True
                target_gt_idx[image_index, anchor_index] = gt_index
                target_bboxes[image_index, anchor_index] = gt_box
                target_scores[image_index, anchor_index] = score
    return B0Assignment(
        fg_mask=foreground,
        target_gt_idx=target_gt_idx,
        target_bboxes=target_bboxes,
        target_scores=target_scores,
        anchor_points=assignment.anchor_points,
        stride_tensor=assignment.stride_tensor,
    )


def native_loss_for_assignment(
    criterion: Any,
    prediction: Mapping[str, Any],
    assignment: B0Assignment,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_distri = prediction["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = prediction["scores"].permute(0, 2, 1).contiguous()
    pred_bboxes = criterion.bbox_decode(assignment.anchor_points, pred_distri)
    score_sum = assignment.target_scores.sum().clamp_min(1.0)
    losses = torch.zeros(3, device=pred_scores.device)
    bce = criterion.bce(pred_scores, assignment.target_scores.to(pred_scores.dtype))
    if criterion.class_weights is not None:
        bce *= criterion.class_weights
    losses[1] = bce.sum() / score_sum
    if bool(assignment.fg_mask.any()):
        image_size = torch.tensor(prediction["feats"][0].shape[2:], device=pred_scores.device, dtype=pred_scores.dtype)
        image_size = image_size * criterion.stride[0]
        losses[0], losses[2] = criterion.bbox_loss(
            pred_distri,
            pred_bboxes,
            assignment.anchor_points,
            assignment.target_bboxes / assignment.stride_tensor,
            assignment.target_scores,
            score_sum,
            assignment.fg_mask,
            image_size,
            assignment.stride_tensor,
        )
    losses[0] *= criterion.hyp.box
    losses[1] *= criterion.hyp.cls
    losses[2] *= criterion.hyp.dfl
    detached = dict(zip(criterion.loss_names, losses.detach()))
    return losses.sum() * pred_scores.shape[0], detached


def _view_image(image: torch.Tensor, view: str) -> torch.Tensor:
    if view == "hflip":
        return image.flip(-1)
    if view == "r90":
        return torch.rot90(image, 1, (-2, -1))
    if view == "r180":
        return torch.rot90(image, 2, (-2, -1))
    raise ValueError(view)


def _view_anchor_indices(indices: torch.Tensor, height: int, width: int, view: str) -> torch.Tensor:
    y, x = torch.div(indices, width, rounding_mode="floor"), indices % width
    if view == "hflip":
        yy, xx = y, width - 1 - x
    elif view == "r90":
        yy, xx = width - 1 - x, y
    elif view == "r180":
        yy, xx = height - 1 - y, width - 1 - x
    else:
        raise ValueError(view)
    return yy * width + xx


def _inverse_view_boxes(boxes: torch.Tensor, size: float, view: str) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    if view == "hflip":
        return torch.stack((size - x2, y1, size - x1, y2), -1)
    if view == "r90":
        return torch.stack((size - y2, x1, size - y1, x2), -1)
    if view == "r180":
        return torch.stack((size - x2, size - y2, size - x1, size - y1), -1)
    raise ValueError(view)


@contextmanager
def _batchnorm_eval(model: nn.Module):
    modules = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    states = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        yield
    finally:
        for module, state in zip(modules, states):
            module.train(state)


def _view_decoded_boxes(
    model: nn.Module,
    criterion: Any,
    image: torch.Tensor,
    assignment: B0Assignment,
    *,
    grad: bool,
) -> torch.Tensor:
    if grad:
        parsed = criterion.parse_output(model(image))
    else:
        parsed = frozen_teacher_prediction(model, image)
    raw = parsed["boxes"].permute(0, 2, 1).contiguous()
    return criterion.bbox_decode(assignment.anchor_points, raw.float()) * assignment.stride_tensor


def _checkpointed_student_view_boxes(
    model: nn.Module,
    criterion: Any,
    image: torch.Tensor,
    assignment: B0Assignment,
    rows: torch.Tensor,
    mapped: torch.Tensor,
    *,
    image_size: float,
    view: str,
) -> torch.Tensor:
    """Return one transformed student response without retaining its activations."""

    def checkpointed_chunk(
        value: torch.Tensor,
        chunk_rows: torch.Tensor,
        chunk_mapped: torch.Tensor,
    ) -> torch.Tensor:
        def forward(chunk_value: torch.Tensor) -> torch.Tensor:
            # Recompute happens after p3_pemt_loss returns, so BN eval belongs
            # inside the replayed closure.
            with _batchnorm_eval(model):
                boxes = _view_decoded_boxes(model, criterion, chunk_value, assignment, grad=True)
                selected = boxes[chunk_rows[:, 0], chunk_mapped]
                return _inverse_view_boxes(selected, image_size, view)

        return activation_checkpoint(forward, _view_image(value, view), use_reentrant=False)

    selected_chunks: list[torch.Tensor] = []
    for start in range(0, int(image.shape[0]), _PEMT_VIEW_CHUNK_SIZE):
        end = min(start + _PEMT_VIEW_CHUNK_SIZE, int(image.shape[0]))
        selected = (rows[:, 0] >= start) & (rows[:, 0] < end)
        if not bool(selected.any()):
            continue
        chunk_rows = rows[selected].clone()
        chunk_rows[:, 0] -= start
        selected_chunks.append(checkpointed_chunk(image[start:end], chunk_rows, mapped[selected]))
    return torch.cat(selected_chunks, 0)


def _chunked_teacher_view_boxes(
    model: nn.Module,
    criterion: Any,
    image: torch.Tensor,
    assignment: B0Assignment,
    rows: torch.Tensor,
    mapped: torch.Tensor,
    *,
    image_size: float,
    view: str | None,
) -> torch.Tensor:
    """Build one stopped teacher response in B16 chunks, preserving row order."""

    selected_chunks: list[torch.Tensor] = []
    for start in range(0, int(image.shape[0]), _PEMT_VIEW_CHUNK_SIZE):
        end = min(start + _PEMT_VIEW_CHUNK_SIZE, int(image.shape[0]))
        selected = (rows[:, 0] >= start) & (rows[:, 0] < end)
        if not bool(selected.any()):
            continue
        chunk_rows = rows[selected].clone()
        chunk_rows[:, 0] -= start
        value = image[start:end] if view is None else _view_image(image[start:end], view)
        boxes = _view_decoded_boxes(model, criterion, value, assignment, grad=False)
        chosen = boxes[chunk_rows[:, 0], mapped[selected]]
        if view is not None:
            chosen = _inverse_view_boxes(chosen, image_size, view)
        selected_chunks.append(chosen)
    return torch.cat(selected_chunks, 0)


def _chunked_optical_identity_targets(
    model: nn.Module,
    criterion: Any,
    image: torch.Tensor,
    assignment: B0Assignment,
    p3: torch.Tensor,
    *,
    reg_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paired DFL and PEMT identity targets from the same B16 forwards."""

    rows = _foreground_rows(assignment)[p3]
    dfl_chunks: list[torch.Tensor] = []
    identity_chunks: list[torch.Tensor] = []
    for start in range(0, int(image.shape[0]), _PEMT_VIEW_CHUNK_SIZE):
        end = min(start + _PEMT_VIEW_CHUNK_SIZE, int(image.shape[0]))
        prediction = frozen_teacher_prediction(model, image[start:end])
        dfl_chunks.append(dfl_logits(prediction, reg_max=reg_max))
        selected = (rows[:, 0] >= start) & (rows[:, 0] < end)
        if not bool(selected.any()):
            continue
        chunk_rows = rows[selected].clone()
        chunk_rows[:, 0] -= start
        boxes = _decoded_boxes(criterion, prediction, assignment) * assignment.stride_tensor
        identity_chunks.append(boxes[chunk_rows[:, 0], chunk_rows[:, 1]])
    identity = torch.cat(identity_chunks, 0) if identity_chunks else image.new_empty((0, 4))
    return torch.cat(dfl_chunks, 0), identity


def p3_pemt_loss(
    criterion: Any,
    student_model: nn.Module,
    optical_teacher: nn.Module | None,
    student_prediction: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    assignment: B0Assignment,
    p3: torch.Tensor,
    *,
    consistency_only: bool,
    optical_image_key: str = "eo_img",
    identity_teacher_boxes: torch.Tensor | None = None,
) -> torch.Tensor:
    rows = _foreground_rows(assignment)[p3]
    if rows.numel() == 0:
        return student_prediction["boxes"].float().sum() * 0.0
    feature = student_prediction["feats"][0]
    height, width = int(feature.shape[-2]), int(feature.shape[-1])
    image_size = float(batch["img"].shape[-1])
    student_changes: list[torch.Tensor] = []
    teacher_changes: list[torch.Tensor] = []
    if not consistency_only:
        if optical_teacher is None or optical_image_key not in batch:
            raise FPMethodZooError("p3_pemt requires paired optical images")
        identity_teacher = identity_teacher_boxes
        if identity_teacher is None:
            identity_teacher = _chunked_teacher_view_boxes(
                optical_teacher,
                criterion,
                batch[optical_image_key],
                assignment,
                rows,
                rows[:, 1],
                image_size=image_size,
                view=None,
            )
    with _batchnorm_eval(student_model):
        with torch.no_grad():
            identity_prediction = criterion.parse_output(student_model(batch["img"]))
            identity_student_all = _decoded_boxes(criterion, identity_prediction, assignment)
            identity_student_all = identity_student_all * assignment.stride_tensor
            identity_student = identity_student_all[rows[:, 0], rows[:, 1]]
    for view in ("hflip", "r90", "r180"):
        mapped = _view_anchor_indices(rows[:, 1], height, width, view)
        student_back = _checkpointed_student_view_boxes(
            student_model,
            criterion,
            batch["img"],
            assignment,
            rows,
            mapped,
            image_size=image_size,
            view=view,
        )
        student_changes.append((student_back - identity_student) / image_size)
        if not consistency_only:
            teacher_back = _chunked_teacher_view_boxes(
                optical_teacher,
                criterion,
                batch[optical_image_key],
                assignment,
                rows,
                mapped,
                image_size=image_size,
                view=view,
            )
            teacher_changes.append((teacher_back - identity_teacher) / image_size)
    student_matrix = torch.stack(student_changes, 0).reshape(3, -1).float()
    if consistency_only:
        return student_matrix.pow(2).mean()
    teacher_matrix = torch.stack(teacher_changes, 0).reshape(3, -1).detach().float()
    student_gram = student_matrix @ student_matrix.t() / max(student_matrix.shape[1], 1)
    teacher_gram = teacher_matrix @ teacher_matrix.t() / max(teacher_matrix.shape[1], 1)
    student_gram = student_gram / student_gram.trace().clamp_min(1e-8)
    teacher_gram = teacher_gram / teacher_gram.trace().clamp_min(1e-8)
    return functional.mse_loss(student_gram, teacher_gram)


def _dfl_pemt_total(
    native_loss: torch.Tensor,
    dfl_loss: torch.Tensor,
    pemt_loss: torch.Tensor,
    *,
    lambda_pemt: float,
) -> torch.Tensor:
    """Frozen DFL+PEMT objective; the PEMT coefficient is method-selected only."""

    return native_loss + 0.1 * dfl_loss + float(lambda_pemt) * pemt_loss


def method_zoo_criterion(
    criterion: Any,
    student_prediction: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    method: str,
    student_model: nn.Module,
    optical_teacher: nn.Module | None = None,
    sar_teacher: nn.Module | None = None,
    auxiliary_modules: nn.Module | None = None,
    ladd_stage: str = "b",
) -> MethodZooResult:
    if method not in METHODS:
        raise FPMethodZooError(f"unknown method {method!r}")
    native = native_student_loss(criterion, student_prediction, batch)
    zero = native.native_loss * 0.0
    if method == "native" or not bool(native.assignment.fg_mask.any()):
        return MethodZooResult(native.native_loss, zero, native.native_loss, native.detached_loss, _zero_stats(method))
    if method in PAIRED_METHODS and "eo_img" not in batch:
        raise FPMethodZooError(f"{method} requires paired EO input")
    parsed_s = native.parsed_prediction
    p3 = p3_foreground_mask(parsed_s, native.assignment)
    reg_max = criterion_reg_max(criterion)
    student_dfl = dfl_logits(parsed_s, reg_max=reg_max)[native.assignment.fg_mask]
    batch_size = int(batch["img"].shape[0])
    teacher_forwards = 0

    optical_prediction = None
    if method in OPTICAL_METHODS and method not in (_PEMT_ONLY_METHODS | _DFL_PEMT_METHODS):
        if optical_teacher is None:
            raise FPMethodZooError(f"{method} requires optical teacher")
        image_key = "shuffled_eo_img" if method in SHUFFLED_METHODS else "eo_img"
        if image_key not in batch:
            raise FPMethodZooError(f"{method} requires {image_key}")
        optical_prediction = frozen_teacher_prediction(optical_teacher, batch[image_key])
        teacher_forwards += 1
    sar_prediction = None
    if method in SAR_METHODS:
        if sar_teacher is None:
            raise FPMethodZooError(f"{method} requires SAR teacher")
        sar_prediction = frozen_teacher_prediction(sar_teacher, batch["img"])
        teacher_forwards += 1

    coefficient = 0.1
    frozen_total: torch.Tensor | None = None
    conflict_ratio = 0.0
    projected_gradient_norm = 0.0
    if method == "p3_dfl":
        target = dfl_logits(optical_prediction, reg_max=reg_max)[native.assignment.fg_mask]
        auxiliary = masked_dfl_kl(target, student_dfl, native.assignment.target_score_weights, p3, batch_size=batch_size)
    elif method == "p3_same_modal":
        target = dfl_logits(sar_prediction, reg_max=reg_max)[native.assignment.fg_mask]
        auxiliary = masked_dfl_kl(target, student_dfl, native.assignment.target_score_weights, p3, batch_size=batch_size)
    elif method in _APR_DFL_COEFFICIENTS:
        optical_dfl = dfl_logits(optical_prediction, reg_max=reg_max)[native.assignment.fg_mask]
        sar_dfl = dfl_logits(sar_prediction, reg_max=reg_max)[native.assignment.fg_mask]
        auxiliary = p3_apr_loss(student_dfl, optical_dfl, sar_dfl, native.assignment, p3, batch_size=batch_size)
        coefficient = _APR_DFL_COEFFICIENTS[method]
    elif method in {"p3_prw_loc", "p3_prw_sar"}:
        source = optical_prediction if method == "p3_prw_loc" else sar_prediction
        auxiliary, prw_weights = p3_prw_residual(
            criterion, parsed_s, source, native.assignment, p3, batch_size=batch_size
        )
        coefficient = 1.0
    elif method == "p3_dfl_prw_05":
        target = dfl_logits(optical_prediction, reg_max=reg_max)[native.assignment.fg_mask]
        dfl_auxiliary = masked_dfl_kl(
            target, student_dfl, native.assignment.target_score_weights, p3, batch_size=batch_size
        )
        prw_auxiliary, prw_weights = p3_prw_residual(
            criterion, parsed_s, optical_prediction, native.assignment, p3, batch_size=batch_size
        )
        auxiliary = 0.1 * dfl_auxiliary + 0.5 * prw_auxiliary
        coefficient = 1.0
    elif method in {"p3_rankkd", "p3_rankkd_shuffled"}:
        auxiliary = float(batch_size) * p3_rank_loss(parsed_s, optical_prediction, native.assignment, p3)
        coefficient = 0.05
    elif method in _P3_DFL_RANK_COEFFICIENTS:
        target = dfl_logits(optical_prediction, reg_max=reg_max)[native.assignment.fg_mask]
        dfl_auxiliary = masked_dfl_kl(
            target, student_dfl, native.assignment.target_score_weights, p3, batch_size=batch_size
        )
        rank_auxiliary = float(batch_size) * p3_rank_loss(parsed_s, optical_prediction, native.assignment, p3)
        auxiliary = 0.1 * dfl_auxiliary + _P3_DFL_RANK_COEFFICIENTS[method] * rank_auxiliary
        coefficient = 1.0
    elif method in {"p3_hscard", "p3_hscard_shuffled"}:
        target = dfl_logits(optical_prediction, reg_max=reg_max)[native.assignment.fg_mask]
        auxiliary, conflict_ratio, projected_gradient_norm = p3_hscard_loss(
            student_dfl, target, native.assignment, p3, batch_size=batch_size
        )
    elif method in {"p3_popa", "p3_popa_sar"}:
        source = optical_prediction if method == "p3_popa" else sar_prediction
        modified = popa_assignment(
            criterion, parsed_s, source, native.assignment, p3_extent=_p3_extent(parsed_s), optical_mix=0.5
        )
        modified_loss, detached = native_loss_for_assignment(criterion, parsed_s, modified)
        auxiliary = modified_loss - native.native_loss
        stats = _zero_stats(method)
        stats.update({"p3_density": _density(p3), "aux_loss": float(auxiliary.detach().item()), "teacher_forwards": float(teacher_forwards)})
        return MethodZooResult(native.native_loss, auxiliary, modified_loss, detached, stats)
    elif method in _PEMT_ONLY_METHODS:
        auxiliary = p3_pemt_loss(
            criterion,
            student_model,
            optical_teacher,
            parsed_s,
            batch,
            native.assignment,
            p3,
            consistency_only=method == "p3_pemt_consistency",
            optical_image_key="shuffled_eo_img" if method == "p3_pemt_shuffled" else "eo_img",
        )
        auxiliary = float(batch_size) * auxiliary
        teacher_forwards += 4 if method in _PEMT_ONLY_METHODS - {"p3_pemt_consistency"} else 0
        coefficient = _PEMT_COEFFICIENTS.get(method, coefficient)
    elif method in _DFL_PEMT_METHODS:
        if optical_teacher is None:
            raise FPMethodZooError(f"{method} requires optical teacher")
        paired_dfl, paired_identity = _chunked_optical_identity_targets(
            optical_teacher,
            criterion,
            batch["eo_img"],
            native.assignment,
            p3,
            reg_max=reg_max,
        )
        dfl_auxiliary = masked_dfl_kl(
            paired_dfl[native.assignment.fg_mask],
            student_dfl,
            native.assignment.target_score_weights,
            p3,
            batch_size=batch_size,
        )
        pemt_shuffled = method in _DFL_PEMT_SHUFFLED_METHODS
        pemt_consistency = method in _DFL_PEMT_CONSISTENCY_METHODS
        pemt_auxiliary = p3_pemt_loss(
            criterion,
            student_model,
            optical_teacher,
            parsed_s,
            batch,
            native.assignment,
            p3,
            consistency_only=pemt_consistency,
            optical_image_key="shuffled_eo_img" if pemt_shuffled else "eo_img",
            identity_teacher_boxes=None if pemt_shuffled or pemt_consistency else paired_identity,
        )
        pemt_auxiliary = float(batch_size) * pemt_auxiliary
        frozen_total = _dfl_pemt_total(
            native.native_loss,
            dfl_auxiliary,
            pemt_auxiliary,
            lambda_pemt=_PEMT_COEFFICIENTS[method],
        )
        auxiliary = frozen_total - native.native_loss
        coefficient = 1.0
        teacher_forwards += 1 if pemt_consistency else 5 if pemt_shuffled else 4
    elif method in {"p3_gbrd_v2", "p3_gbrd_shuffled"}:
        auxiliary = p3_gbrd_loss(
            criterion, parsed_s, optical_prediction, native.assignment, p3, batch_size=batch_size
        )
    elif method == "p3_raw_feature_kd":
        if not isinstance(auxiliary_modules, P3RawFeatureAdapter):
            raise FPMethodZooError("p3_raw_feature_kd requires its training adapter")
        student_feature = parsed_s["feats"][0]
        teacher_feature = optical_prediction["feats"][0].detach()
        extent = _p3_extent(parsed_s)
        weights = native.assignment.target_scores[:, :extent].sum(-1).reshape(
            student_feature.shape[0], 1, student_feature.shape[-2], student_feature.shape[-1]
        )
        mask = native.assignment.fg_mask[:, :extent].reshape_as(weights).to(weights.dtype)
        diff = (auxiliary_modules(student_feature).float() - teacher_feature.float()).pow(2).mean(1, keepdim=True)
        auxiliary = float(batch_size) * (diff * weights.float() * mask.float()).sum() / (
            weights.float() * mask.float()
        ).sum().clamp_min(1e-6)
    elif method == "p3_ladd":
        if not isinstance(auxiliary_modules, P3LADDModules):
            raise FPMethodZooError("p3_ladd requires training-only decomposition modules")
        ladd = p3_ladd_loss(
            parsed_s["feats"][0], optical_prediction["feats"][0], native.assignment, auxiliary_modules, stage=ladd_stage
        )
        auxiliary = float(batch_size) * ladd.total
        coefficient = 1.0
        total = auxiliary if ladd_stage == "a" else native.native_loss + auxiliary
        stats = _zero_stats(method)
        stats.update({"p3_density": _density(p3), "aux_loss": float(auxiliary.detach().item()), "teacher_forwards": 1.0})
        stats.update({key: float(value.detach().item()) for key, value in ladd.components.items()})
        return MethodZooResult(native.native_loss, auxiliary, total, native.detached_loss, stats)
    else:  # pragma: no cover
        raise FPMethodZooError(f"unhandled method {method}")

    total = frozen_total if frozen_total is not None else native.native_loss + float(coefficient) * auxiliary
    stats = _zero_stats(method)
    stats.update(
        {
            "p3_density": _density(p3),
            "aux_loss": float(auxiliary.detach().item()),
            "teacher_forwards": float(teacher_forwards),
            "hscard_conflict_ratio": float(conflict_ratio),
            "hscard_projected_gradient_norm": float(projected_gradient_norm),
        }
    )
    if method in {"p3_prw_loc", "p3_prw_sar", "p3_dfl_prw_05"}:
        stats["prw_weight_mean"] = float(prw_weights.detach().float().mean().item()) if prw_weights.numel() else 1.0
    return MethodZooResult(native.native_loss, auxiliary, total, native.detached_loss, stats)


class FPMethodZooLoss:
    def __init__(
        self,
        native: Any,
        *,
        method: str,
        student_model: nn.Module,
        optical_teacher: nn.Module | None,
        sar_teacher: nn.Module | None,
        auxiliary_modules: nn.Module | None,
        ladd_stage: str = "b",
    ) -> None:
        self.native = native
        self.method = method
        self.student_model = student_model
        self.optical_teacher = optical_teacher
        self.sar_teacher = sar_teacher
        self.auxiliary_modules = auxiliary_modules
        self.ladd_stage = ladd_stage
        self.stats_rows: list[dict[str, float]] = []

    def __call__(self, prediction: Any, batch: dict[str, torch.Tensor]):
        # Validation has SAR-only batches. The auxiliary method is training-only;
        # evaluation must use the unchanged detector criterion.
        if not self.student_model.training:
            return self.native(prediction, batch)
        result = method_zoo_criterion(
            self.native,
            prediction,
            batch,
            method=self.method,
            student_model=self.student_model,
            optical_teacher=self.optical_teacher,
            sar_teacher=self.sar_teacher,
            auxiliary_modules=self.auxiliary_modules,
            ladd_stage=self.ladd_stage,
        )
        self.stats_rows.append(result.stats)
        detached = dict(result.detached_loss)
        detached["method_aux"] = result.auxiliary_loss.detach()
        return result.total_loss, detached
