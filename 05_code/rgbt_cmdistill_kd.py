"""CMDistill-adapted (JSTARS 2025, DOI 10.1109/JSTARS.2024.3479717) loss components.

CPU-testable reimplementation of the published equations. No official code
exists; every non-unique choice is declared in DECLARED_ADAPTATIONS below and
must be reported as ``CMDistill-adapted``, never as an exact reproduction.

Published structure (equation numbers from the paper):
- PCCFD (Eq 1): per-scale Pearson correlation ``r`` between flattened
  teacher/student FPN feature maps. The printed objective minimizes ``r``;
  the corrected runnable identity minimizes ``1 - r`` (identical to half the
  MSE between z-scored maps, which the paper also describes).
- SLRD (Eq 2-3): cosine affinity matrices ``A_ij`` between per-location
  feature vectors of the deepest semantic feature, matched by mean L1.
- IBCLD (Eq 4-7): teacher/student predicted-box IoU plus per-anchor/per-class
  binary cross-entropy with teacher probabilities as soft targets;
  ``L_log = L_IoU + L_cls``.
- Total (Eq 8): ``L = L_det + lambda1*L_fea + lambda2*L_rela + lambda3*L_log``.

DECLARED_ADAPTATIONS (paper does not uniquely define these for YOLO11):
1. The printed ``r`` (Eq 1) and IoU (Eq 4) minimization directions are
   sign-inconsistent with their stated intent; ``literal`` keeps the printed
   form (CPU oracle/appendix only), ``corrected`` uses ``1 - r`` and
   ``1 - IoU`` and is the only runnable identity.
2. PCCFD is computed per image over all C*H*W map elements, equal-weight
   averaged over the three scales (P3/P4/P5), then batch-averaged.
3. SLRD feature vectors are per spatial location (K = H*W, C-dim) of the
   deepest scale only; the paper's unspecified "adaptive layer" is omitted
   (Eq 2-3 as printed contain no adaptive module).
4. Logic loss operates on the shared dense anchor grid: teacher and student
   raw DFL boxes decode into pixel xyxy with identical anchors/strides, the
   IoU term is restricted to teacher-foreground anchors (teacher max class
   probability >= 0.5), and BCE covers all anchors x classes. Both sub-losses
   are mean-reduced before the unweighted Eq 7 sum.
5. lambda1 = lambda2 = lambda3 = 1.0 (paper values unpublished; CCLKD's
   published convention is used), and the KD batch mean is scaled by B to
   match the Ultralytics native ``loss * B`` dose.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

CMDISTILL_EPS = 1e-8
CMDISTILL_FG_THRESHOLD = 0.5
DFL_BINS = 16


class CMDistillError(RuntimeError):
    """Raised when the CMDistill-adapted contract is violated."""


def _variant_direction(loss: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "literal":
        return loss
    if variant == "corrected":
        return 1.0 - loss
    raise CMDistillError(f"unknown variant {variant}")


def pcc_loss(student_map: torch.Tensor, teacher_map: torch.Tensor, variant: str) -> torch.Tensor:
    """Eq 1 on one feature map pair: batch-mean Pearson ``r`` over all elements."""

    if student_map.shape != teacher_map.shape or student_map.ndim != 4:
        raise CMDistillError("PCCFD maps must share [B,C,H,W] shapes")
    s = student_map.reshape(student_map.shape[0], -1)
    t = teacher_map.detach().reshape(teacher_map.shape[0], -1)
    s = s - s.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    denom = (s.norm(dim=1) * t.norm(dim=1)).clamp(min=CMDISTILL_EPS)
    r = (s * t).sum(dim=1) / denom
    degenerate = (s.norm(dim=1) < CMDISTILL_EPS) | (t.norm(dim=1) < CMDISTILL_EPS)
    r = torch.where(degenerate, torch.zeros_like(r), r)
    loss = _variant_direction(r, variant)
    return torch.where(degenerate, torch.zeros_like(loss), loss).mean()


def pccfd_loss(
    student_feats: tuple[torch.Tensor, ...],
    teacher_feats: tuple[torch.Tensor, ...],
    variant: str,
) -> torch.Tensor:
    """Eq 1 equal-weight mean over the multiscale feature maps."""

    if len(student_feats) != len(teacher_feats) or not student_feats:
        raise CMDistillError("PCCFD requires matching non-empty scale lists")
    per_scale = [pcc_loss(s, t, variant) for s, t in zip(student_feats, teacher_feats)]
    return torch.stack(per_scale).mean()


def affinity_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Eq 2: cosine similarity between per-location C-dim vectors, (B, K, K)."""

    if feat.ndim != 4:
        raise CMDistillError("affinity expects [B,C,H,W] features")
    b, c, h, w = feat.shape
    vectors = feat.reshape(b, c, h * w).transpose(1, 2)  # (B, K, C)
    normed = functional.normalize(vectors, p=2, dim=2, eps=CMDISTILL_EPS)
    return torch.bmm(normed, normed.transpose(1, 2))


def slrd_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """Eq 3: batch-mean L1 distance between deepest-scale affinity matrices."""

    if student_feat.shape != teacher_feat.shape or student_feat.ndim != 4:
        raise CMDistillError("SLRD maps must share [B,C,H,W] shapes")
    a_t = affinity_matrix(teacher_feat.detach())
    a_s = affinity_matrix(student_feat)
    return (a_t - a_s).abs().mean(dim=(1, 2)).mean()


def dfl_project(box_raw: torch.Tensor) -> torch.Tensor:
    """(B, 4*reg_max, N) raw DFL logits -> (B, N, 4) expected offsets."""

    b, _, n = box_raw.shape
    dist = box_raw.reshape(b, 4, DFL_BINS, n).softmax(2)
    proj = dist.new_tensor(list(range(DFL_BINS))).view(1, 1, DFL_BINS, 1)
    return (dist * proj).sum(2).permute(0, 2, 1)  # (B, N, 4)


def build_anchor_points(
    feats: tuple[torch.Tensor, ...], strides: tuple[float, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor grid replicating Ultralytics ``make_anchors(feats, strides, 0.5)``.

    Returns feature-unit anchor points (N, 2) and the matching stride tensor
    (N, 1); anchors concatenate level-by-level, matching the raw prediction
    anchor order.
    """

    if len(feats) != len(strides):
        raise CMDistillError("feature/stride mismatch")
    points: list[torch.Tensor] = []
    stride_rows: list[torch.Tensor] = []
    for feature, stride in zip(feats, strides):
        h, w = feature.shape[-2], feature.shape[-1]
        xs = torch.arange(w, device=feature.device, dtype=feature.dtype) + 0.5
        ys = torch.arange(h, device=feature.device, dtype=feature.dtype) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        points.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1))
        stride_rows.append(torch.full((h * w, 1), float(stride), device=feature.device, dtype=feature.dtype))
    return torch.cat(points), torch.cat(stride_rows)


def decode_boxes(
    box_raw: torch.Tensor,
    feats: tuple[torch.Tensor, ...],
    strides: tuple[float, ...],
) -> torch.Tensor:
    """Raw DFL logits -> pixel-space xyxy boxes (B, N, 4) on the shared grid."""

    anchor_points, stride_tensor = build_anchor_points(feats, strides)
    dist = dfl_project(box_raw) * stride_tensor.unsqueeze(0)  # (B, N, 4) pixels
    lt, rb = dist.chunk(2, dim=-1)
    anchors = anchor_points.unsqueeze(0) * stride_tensor.unsqueeze(0)
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    return torch.cat([x1y1, x2y2], dim=-1)


def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Plain IoU between two (N, 4) xyxy box sets."""

    lt = torch.maximum(box1[:, :2], box2[:, :2])
    rb = torch.minimum(box1[:, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=CMDISTILL_EPS)


def iou_logic_loss(
    student_boxes: torch.Tensor,
    teacher_boxes: torch.Tensor,
    fg_mask: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    """Eq 4 restricted to teacher-foreground anchors; batch-mean over images."""

    if student_boxes.shape != teacher_boxes.shape or student_boxes.ndim != 3:
        raise CMDistillError("logic boxes must share (B, N, 4) shapes")
    per_image = student_boxes.new_zeros(student_boxes.shape[0])
    for i in range(student_boxes.shape[0]):
        mask = fg_mask[i]
        if int(mask.sum()) == 0:
            continue
        iou = box_iou_xyxy(student_boxes[i][mask], teacher_boxes[i].detach()[mask])
        per_image[i] = _variant_direction(iou, variant).mean()
    return per_image.mean()


def cls_logic_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
) -> torch.Tensor:
    """Eq 5-6: BCE of student logits against teacher probabilities, mean-reduced."""

    if student_scores.shape != teacher_scores.shape or student_scores.ndim != 3:
        raise CMDistillError(
            f"logic scores must share (B, N, C) shapes, got {tuple(student_scores.shape)} "
            f"vs {tuple(teacher_scores.shape)}"
        )
    targets = teacher_scores.detach().sigmoid()
    return functional.binary_cross_entropy_with_logits(student_scores, targets)


@dataclass
class CMDistillTerms:
    """Batch-mean KD terms before lambda weighting and the dose scale."""

    pccfd: torch.Tensor
    slrd: torch.Tensor
    iou: torch.Tensor
    cls: torch.Tensor
    fg_ratio: float

    @property
    def log(self) -> torch.Tensor:
        """Eq 7: unweighted logic sum."""

        return self.iou + self.cls


def cmdistill_terms(
    student_preds: dict,
    teacher_preds: dict,
    strides: tuple[float, ...],
    *,
    variant: str = "corrected",
    fg_threshold: float = CMDISTILL_FG_THRESHOLD,
) -> CMDistillTerms:
    """All CMDistill KD terms for one batch from raw Detect output mappings.

    ``student_preds``/``teacher_preds`` expose ``boxes`` (B, 4*reg_max, N),
    ``scores`` (B, nc, N) and ``feats`` (per-scale maps). The student side
    keeps gradients; every teacher tensor is detached here.
    """

    student_feats = tuple(student_preds["feats"])
    teacher_feats = tuple(f.detach() for f in teacher_preds["feats"])
    pccfd = pccfd_loss(student_feats, teacher_feats, variant)
    slrd = slrd_loss(student_feats[-1], teacher_feats[-1])
    student_boxes = decode_boxes(student_preds["boxes"], student_feats, strides)
    teacher_boxes = decode_boxes(teacher_preds["boxes"].detach(), teacher_feats, strides)
    teacher_cls = teacher_preds["scores"].detach().sigmoid()
    fg_mask = teacher_cls.max(dim=1).values >= fg_threshold
    iou = iou_logic_loss(student_boxes, teacher_boxes, fg_mask, variant)
    cls = cls_logic_loss(
        student_preds["scores"].transpose(1, 2), teacher_preds["scores"].detach().transpose(1, 2)
    )
    return CMDistillTerms(
        pccfd=pccfd, slrd=slrd, iou=iou, cls=cls,
        fg_ratio=float(fg_mask.float().mean().detach()),
    )
