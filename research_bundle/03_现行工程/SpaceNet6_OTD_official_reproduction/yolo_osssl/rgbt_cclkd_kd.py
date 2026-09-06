"""CCLKD-adapted (GIS 2026, DOI 10.1080/10095020.2026.2633014) loss components.

CPU-testable reimplementation of the published equations. No official code
exists; every non-unique choice is declared in DECLARED_ADAPTATIONS below and
must be reported as ``CCLKD-adapted``, never as an exact reproduction.

Published structure (equation numbers from the paper):
- COP (Eq 1-4): dominant-category mask M from teacher class scores; sigmoided
  box outputs p_box are masked and softmaxed into target/non-target
  distributions b_j / b_hat_j over candidate boxes.
- PATM (Eq 5-7): per-class adaptive temperature
  T_j = Tmin + (alpha - Tmin) * sigmoid(H_j), H_j = mean positive-score
  entropy, [Tmin, Tmax] = [0.5, 5.0], alpha learnable init 5.0.
- LKD (Eq 8-15): L_kd = L_kd_logit + L_kd_feat + L_kd_corr, each scaled by
  T_j^2 / B.
- CCL (Eq 16-18): class-balanced weights w_j, InfoNCE-style loss on cosine
  similarity of teacher/student masked distributions.
- Total (Eq 29): L = L_det + lambda_kd * L_kd + lambda_cc * L_cc with
  lambda_kd = lambda_cc = 1.0.

DECLARED_ADAPTATIONS (paper does not uniquely define these for YOLO11):
1. p_cls: per-anchor sigmoid classification scores (B, N, C).
2. p_box: per-anchor decoded 4-vector DFL box offsets passed through sigmoid
   (B, N, 4); the paper's z_box -> sigmoid mapping is applied to the decoded
   ltrb offsets (the 16-bin DFL logits are collapsed by the standard DFL
   projection before sigmoid).
3. Positive set: the COP mask M is the one-hot dominant teacher category per
   anchor (Eq 2); anchors whose dominant category is background-only are not
   created because YOLO11 has no objectness channel — every anchor has a
   dominant class. To keep the positive sets meaningful, only anchors whose
   dominant-class score exceeds ``min_positive_score`` (default 0.05) enter M.
4. FLD features: k x k (default 7) bilinear grid samples from the Detect-input
   pyramid level matching each anchor stride, then a training-only 1x1 conv
   projection (per network) to ``proj_dim`` (default 64).
5. CCL literal (Eq 17) keeps the published denominator (positive term plus one
   negative term, no temperature); ``infonce`` is the corrected variant with
   temperature tau over per-class positive/negative distribution pairs and is
   reported separately, never silently as the paper's loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

CCLKD_T_MIN = 0.5
CCLKD_ALPHA_INIT = 5.0
CCLKD_EPS = 1e-6


class CCLKDError(RuntimeError):
    """Raised when the CCLKD-adapted contract is violated."""


@dataclass
class COPPartitions:
    """Category-oriented partition of candidate boxes (Eq 1-4)."""

    mask: torch.Tensor  # (N, C) one-hot dominant category, float
    dominant: torch.Tensor  # (N,) long, dominant class index
    dominant_score: torch.Tensor  # (N,) teacher dominant-class score
    class_counts: torch.Tensor  # (C,) positive counts per class


def cop_partition(teacher_cls_scores: torch.Tensor, min_positive_score: float = 0.05) -> COPPartitions:
    """Build the one-hot dominant-category mask from teacher sigmoid scores."""

    scores = teacher_cls_scores.detach()
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise CCLKDError("teacher_cls_scores must be (N, C)")
    dominant = scores.argmax(dim=1)
    dominant_score = scores.gather(1, dominant.unsqueeze(1)).squeeze(1)
    keep = dominant_score >= min_positive_score
    mask = torch.zeros_like(scores)
    mask[keep, dominant[keep]] = 1.0
    counts = mask.sum(dim=0)
    return COPPartitions(mask, dominant, dominant_score, counts)


def masked_box_distributions(
    teacher_box: torch.Tensor,
    student_box: torch.Tensor,
    mask: torch.Tensor,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target/non-target box distributions (Eq 3-4, 9-10), per coordinate.

    Softmax runs over the N candidate boxes for each class j and each of the
    four box coordinates, at the class temperature T_j.
    """

    if teacher_box.shape != student_box.shape or teacher_box.ndim != 2:
        raise CCLKDError("box tensors must share (N, 4)")
    n, d = teacher_box.shape
    c = mask.shape[1]
    if mask.shape[0] != n or temperature.shape[0] != c:
        raise CCLKDError("mask/temperature shapes disagree with boxes")
    # v3: operate on the logit domain so narrow sigmoid bands cannot flatten
    # the distribution; eps keeps (0,1) values finite in log space.
    eps = CCLKD_EPS
    tb = teacher_box.clamp(eps, 1.0 - eps)
    sb = student_box.clamp(eps, 1.0 - eps)
    logit_t = ((tb + eps).log() - (1.0 - tb + eps).log()).unsqueeze(1)
    logit_s = ((sb + eps).log() - (1.0 - sb + eps).log()).unsqueeze(1)
    scaled_t = logit_t / temperature.view(1, c, 1)  # (N,C,4)
    scaled_s = logit_s / temperature.view(1, c, 1)
    log_t = functional.log_softmax(scaled_t, dim=0)
    log_s = functional.log_softmax(scaled_s, dim=0)
    m = mask.unsqueeze(-1)  # (N,C,1)
    return (
        (m * log_t.exp()).sum(dim=0),      # b^t_j: (C,4)
        (m * log_s.exp()).sum(dim=0),
        ((1 - m) * log_t.exp()).sum(dim=0),  # b_hat^j: (C,4)
        ((1 - m) * log_s.exp()).sum(dim=0),
    )


def patm_temperature(
    teacher_cls_scores: torch.Tensor, mask: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Entropy-mapped class-wise temperature (Eq 5-7)."""

    counts = mask.sum(dim=0)
    pos = teacher_cls_scores.detach() * mask
    entropy = -(pos * (pos + CCLKD_EPS).log()).sum(dim=0)
    safe = torch.where(counts > 0, counts, torch.ones_like(counts))
    entropy = entropy / safe
    temperature = CCLKD_T_MIN + (alpha - CCLKD_T_MIN) * torch.sigmoid(entropy)
    return temperature


def lkd_logit(
    teacher_cls_scores: torch.Tensor,
    teacher_box: torch.Tensor,
    student_box: torch.Tensor,
    mask: torch.Tensor,
    temperature: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """LLD (Eq 8): class-wise KL between masked box distributions."""

    b_t, b_s, bhat_t, bhat_s = masked_box_distributions(teacher_box, student_box, mask, temperature)
    eps = CCLKD_EPS
    kl_pos = (b_t * ((b_t + eps).log() - (b_s + eps).log())).sum(dim=-1)      # (C,)
    kl_neg = (bhat_t * ((bhat_t + eps).log() - (bhat_s + eps).log())).sum(dim=-1)
    counts = mask.sum(dim=0)
    active = ((counts > 0) & (counts < mask.shape[0])).float()
    loss = (temperature.pow(2) * (kl_pos + kl_neg) * active).sum()
    return loss / max(batch_size, 1)


def gather_anchor_features(
    feats: tuple[torch.Tensor, ...],
    strides: tuple[int, ...],
    image_size: tuple[int, int],
    k: int = 7,
) -> list[torch.Tensor]:
    """k x k bilinear regional samples per pyramid level (Eq 11).

    Returns one (n_i, C_i, k*k) tensor per level; levels have different
    channel counts so a per-level 1x1 projection must run before concat
    (owned by lkd_feature). Ultralytics anchors are laid out level-by-level,
    so level blocks concatenated in order match the raw anchor order.
    """

    if len(feats) != len(strides):
        raise CCLKDError("feature/stride mismatch")
    grid = torch.linspace(-0.5 + 1.0 / (2 * k), 0.5 - 1.0 / (2 * k), k)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")
    grid_offsets = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).to(feats[0].device)
    rows: list[torch.Tensor] = []
    for feature, stride in zip(feats, strides):
        if feature.shape[0] != 1:
            raise CCLKDError("FLD gathers per image; feature batch must be 1")
        h, w = feature.shape[-2], feature.shape[-1]
        ys = (torch.arange(h, device=feature.device, dtype=feature.dtype) + 0.5) / h
        xs = (torch.arange(w, device=feature.device, dtype=feature.dtype) + 0.5) / w
        centres = torch.stack([xs.repeat(h), ys.repeat_interleave(w)], dim=1)
        pts = centres.unsqueeze(1) + grid_offsets.unsqueeze(0) * (stride / max(image_size))
        rows.append(
            functional.grid_sample(
                feature, pts.view(1, 1, -1, 2), mode="bilinear", align_corners=False
            ).reshape(feature.shape[1], -1, k * k).permute(1, 0, 2)
        )
    return rows


def lkd_feature(
    teacher_feats: tuple[torch.Tensor, ...],
    student_feats: tuple[torch.Tensor, ...],
    strides: tuple[int, ...],
    image_size: tuple[int, int],
    mask: torch.Tensor,
    temperature: torch.Tensor,
    batch_size: int,
    proj: torch.nn.ModuleList | None = None,
    k: int = 7,
) -> torch.Tensor:
    """FLD (Eq 11-12): masked KL between softmaxed regional features.

    ``proj`` holds one training-only 1x1 conv per level, applied identically
    to teacher and student maps before sampling so levels share one channel
    dimension. Without ``proj`` the levels must already agree in channels.
    """

    t_levels = gather_anchor_features(teacher_feats, strides, image_size, k)
    s_levels = gather_anchor_features(student_feats, strides, image_size, k)
    if proj is not None and len(proj) != len(t_levels):
        raise CCLKDError("one projection per level required")
    prepared_t: list[torch.Tensor] = []
    prepared_s: list[torch.Tensor] = []
    for idx, (tl, sl) in enumerate(zip(t_levels, s_levels)):
        if proj is not None:
            tl = proj[idx](tl)
            sl = proj[idx](sl)
        prepared_t.append(tl)
        prepared_s.append(sl)
    t = torch.cat(prepared_t, dim=0)
    s = torch.cat(prepared_s, dim=0)
    if t.shape[0] != mask.shape[0]:
        raise CCLKDError("feature anchors disagree with mask rows")
    log_t = functional.log_softmax(t, dim=-1)
    log_s = functional.log_softmax(s, dim=-1)
    kl = (log_t.exp() * (log_t - log_s)).sum(dim=-1).mean(dim=-1)  # (N,)
    weighted = mask.t() * kl.unsqueeze(0)  # (C, N)
    active = (mask.sum(dim=0) > 0).float()
    loss = (temperature.pow(2).unsqueeze(1) * weighted).sum() / active.sum().clamp(min=1)
    return loss / max(batch_size, 1)


def lkd_relation(
    teacher_feats_flat: torch.Tensor,
    student_feats_flat: torch.Tensor,
    mask: torch.Tensor,
    temperature: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """RLD (Eq 13-14): per-class self-correlation matching."""

    if teacher_feats_flat.shape != student_feats_flat.shape:
        raise CCLKDError("teacher/student feature matrices disagree")
    loss = teacher_feats_flat.new_zeros(())
    active_classes = 0
    for j in range(mask.shape[1]):
        idx = mask[:, j] > 0
        if int(idx.sum()) < 2:
            continue
        ft = teacher_feats_flat[idx]
        fs = student_feats_flat[idx]
        rt = ft @ ft.t()
        rs = fs @ fs.t()
        loss = loss + temperature[j].pow(2) * (rt - rs).norm(p="fro")
        active_classes += 1
    if active_classes == 0:
        return teacher_feats_flat.new_zeros(())
    return loss / active_classes / max(batch_size, 1)


def ccl_class_weights(mask: torch.Tensor) -> torch.Tensor:
    """Class-balanced weights (Eq 16)."""

    counts = mask.sum(dim=0)
    inv = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts))
    total = inv.sum()
    if float(total) <= 0:
        return torch.zeros_like(counts)
    return inv / total


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return functional.cosine_similarity(a, b, dim=0).clamp(-1 + CCLKD_EPS, 1 - CCLKD_EPS)


def ccl_literal(
    b_pos_t: torch.Tensor,
    b_pos_s: torch.Tensor,
    b_neg_t: torch.Tensor,
    b_neg_s: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """CCL exactly as published (Eq 17-18); can go negative by construction."""

    weights = ccl_class_weights(mask)
    loss = b_pos_t.new_zeros(())
    for j in range(mask.shape[1]):
        if float(weights[j]) <= 0:
            continue
        pos = _cosine(b_pos_t[j], b_pos_s[j])
        neg = _cosine(b_neg_t[j], b_neg_s[j])
        loss = loss + weights[j] * -(pos.exp() / (pos.exp() + neg.exp())).log()
    return loss / max(batch_size, 1)


def ccl_infonce(
    b_pos_t: torch.Tensor,
    b_pos_s: torch.Tensor,
    b_neg_t: torch.Tensor,
    b_neg_s: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    tau: float = 0.2,
) -> torch.Tensor:
    """Corrected variant: temperature-scaled InfoNCE with in-batch negatives."""

    weights = ccl_class_weights(mask)
    loss = b_pos_t.new_zeros(())
    active = 0
    for j in range(mask.shape[1]):
        if float(weights[j]) <= 0:
            continue
        anchor = b_pos_s[j] / tau
        positive = (anchor * b_pos_t[j]).sum() / tau
        negatives = [(anchor * b_neg_t[j]).sum() / tau, (anchor * b_neg_s[j]).sum() / tau]
        logits = torch.stack([positive, *negatives])
        loss = loss + weights[j] * (-functional.log_softmax(logits, dim=0)[0])
        active += 1
    if active == 0:
        return b_pos_t.new_zeros(())
    return loss / max(batch_size, 1)


class CCLKDLoss:
    """Ultralytics criterion facade adding the CCLKD-adapted KD and CCL terms."""

    def __init__(
        self,
        native: Any,
        student: nn.Module,
        teacher: nn.Module,
        *,
        lambda_kd: float = 1.0,
        lambda_cc: float = 1.0,
        ccl_variant: str = "literal",
        min_positive_score: float = 0.05,
    ) -> None:
        self.native = native
        self.student = student
        self.teacher = teacher
        self.lambda_kd = lambda_kd
        self.lambda_cc = lambda_cc
        if ccl_variant not in ("literal", "infonce"):
            raise CCLKDError(f"unknown ccl variant {ccl_variant}")
        self.ccl_variant = ccl_variant
        self.min_positive_score = min_positive_score
        self.alpha = nn.Parameter(torch.tensor(float(CCLKD_ALPHA_INIT)))
        self.stats_rows: list[dict[str, float]] = []

    def kd_terms(
        self,
        teacher_cls_scores: torch.Tensor,
        teacher_box: torch.Tensor,
        student_box: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(L_lld, L_ccl_value, temperature, mask) for one image's anchors."""

        parts = cop_partition(teacher_cls_scores, self.min_positive_score)
        temperature = patm_temperature(teacher_cls_scores, parts.mask, self.alpha)
        lld = lkd_logit(teacher_cls_scores, teacher_box, student_box, parts.mask, temperature, batch_size)
        b_pos_t, b_pos_s, b_neg_t, b_neg_s = masked_box_distributions(
            teacher_box, student_box, parts.mask, temperature
        )
        if self.ccl_variant == "literal":
            ccl = ccl_literal(b_pos_t, b_pos_s, b_neg_t, b_neg_s, parts.mask, batch_size)
        else:
            ccl = ccl_infonce(b_pos_t, b_pos_s, b_neg_t, b_neg_s, parts.mask, batch_size)
        return lld, ccl, temperature, parts.mask

    def __call__(self, prediction: Any, batch: dict[str, torch.Tensor]):
        raise CCLKDError(
            "CCLKDLoss.kd_terms operates per image on raw anchors; the trainer "
            "adapter owns the native-loss facade (see tools/train_rgbt_cclkd.py)"
        )
