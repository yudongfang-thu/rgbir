"""Frozen-reference localization selection and DFL KD for OEv1-plus-L v1.

This module does not implement C, registration, training or resource admission.
All coordinates below are augmented input pixels on an externally verified
approximately identical RGB/IR grid. A geometry mask is evidence supplied by
the caller, never inferred from label IoU. Without a verified mask the function
is explicitly diagnostic; the enclosing trainer must reject formal training.

``boxes`` contains raw DFL logits [B,4*R,A], not decoded xyxy. Selection never
reads student values. Loss has neither lambda nor batch-size scaling. Objects
are normalized BEFORE teacher-quality filtering; random selection uses the
same K from that base population with a private CPU generator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy_oev1"))
from object_evidence_loss import (
    _decode_boxes, _iou, _labels, _layout, _match_objects,
)


@dataclass(frozen=True)
class LocalizationConfig:
    input_size: Union[int, Tuple[int, int]] = 640
    levels: Tuple[int, ...] = (0, 1)
    temperature: float = 2.0
    match_iou: float = 0.5
    pair_iou: float = 0.8
    reference_conf: float = 0.05
    reference_iou: float = 0.1
    reliable_conf: float = 0.25
    reference_iou_max: float = 0.70
    teacher_rgb_iou_min: float = 0.60
    teacher_ir_iou_min: float = 0.50
    localization_margin: float = 0.05
    support_epsilon: float = 0.01
    native_candidate_epsilon: float = 1e-9

    def __post_init__(self):
        size = (self.input_size, self.input_size) if isinstance(self.input_size, int) else self.input_size
        if len(size) != 2 or any(int(v) != v or v <= 0 for v in size):
            raise ValueError("input_size must be positive (height, width)")
        if tuple(self.levels) != (0, 1):
            raise ValueError("v1 is frozen to P3/P4 levels (0, 1)")
        if self.temperature <= 0 or not 0 < self.support_epsilon < 1:
            raise ValueError("invalid temperature or support epsilon")
        for key in ("match_iou", "pair_iou", "reference_conf", "reference_iou", "reliable_conf",
                    "reference_iou_max", "teacher_rgb_iou_min", "teacher_ir_iou_min", "localization_margin"):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError("{} must lie in (0,1]".format(key))
        if self.pair_iou < self.match_iou or self.native_candidate_epsilon < 0:
            raise ValueError("invalid pairing or native-candidate threshold")


def _config(value):
    return value if isinstance(value, LocalizationConfig) else LocalizationConfig(**dict(value or {}))


def dfl_view(raw_logits: Tensor) -> Tensor:
    """[B,4*R,A] -> [B,A,4,R], preserving side/bin/anchor meaning."""
    if raw_logits.ndim != 3 or raw_logits.shape[1] % 4 or raw_logits.shape[1] < 8:
        raise ValueError("expected DFL logits [B,4*R,A] with R >= 2")
    b, channels, a = raw_logits.shape
    return raw_logits.float().reshape(b, 4, channels // 4, a).permute(0, 3, 1, 2)


def native_candidate_mask(boxes: Tensor, centers: Tensor, strides: Sequence[int], eps=1e-9) -> Tensor:
    """Pinned TAL geometry, including its small-GT expansion to stride[1].

    All real RGB GT must be passed, including unpaired objects/other classes.
    This reproduces select_candidates_in_gts of pinned Ultralytics 8.4.115;
    the trainer must verify the live assigner still has this behavior.
    """
    xy = (boxes[:, :2] + boxes[:, 2:]) / 2
    wh = boxes[:, 2:] - boxes[:, :2]
    expanded_to = strides[1] if len(strides) > 1 else strides[0]
    wh = torch.where(wh < strides[0], torch.full_like(wh, float(expanded_to)), wh)
    lower, upper = xy - wh / 2, xy + wh / 2
    return ((centers[None] - lower[:, None] > eps) & (upper[:, None] - centers[None] > eps)).all(-1)


def unclamped_distances(boxes: Tensor, centers: Tensor, stride_values: Tensor) -> Tensor:
    """Aligned boxes[N,4], centers[N,2], strides[N,1] -> l/t/r/b bins."""
    return torch.cat((centers - boxes[:, :2], boxes[:, 2:] - centers), -1) / stride_values


def gt_dfl_distribution(distances: Tensor, reg_max: int, temperature: float = 2.0,
                        support_epsilon: float = 0.01) -> Tensor:
    """Exact two-bin GT q, softened to q**(1/T), with zero support preserved.

    Checking *unclamped* distances is intentional: native DFL's in-place clamp
    must not turn unsupported targets into apparently valid KD evidence.
    """
    if distances.ndim != 2 or distances.shape[1] != 4 or reg_max < 2 or temperature <= 0:
        raise ValueError("expected distances[N,4], R >= 2 and positive temperature")
    d = distances.detach().float()
    if not bool(torch.isfinite(d).all()) or not bool(((d >= 0) & (d <= reg_max - 1 - support_epsilon)).all()):
        raise ValueError("unclamped GT distances outside native DFL support")
    lo = d.floor().long()
    upper_mass = d - lo.float()
    q = d.new_zeros(tuple(d.shape) + (reg_max,))
    q.scatter_add_(-1, lo.unsqueeze(-1), (1 - upper_mass).unsqueeze(-1))
    q.scatter_add_(-1, (lo + 1).unsqueeze(-1), upper_mass.unsqueeze(-1))
    softened = q.pow(1.0 / temperature)
    return softened / softened.sum(-1, keepdim=True)


def localization_kd(student_logits: Tensor, target_probability: Tensor, normalizer: int,
                    temperature: float = 2.0) -> Tensor:
    """Selected logits[N,4,R]; KL(target||student)*T^2/4 / fixed base N."""
    if student_logits.ndim != 3 or student_logits.shape[1] != 4 or student_logits.shape != target_probability.shape:
        raise ValueError("student and target must have identical [N,4,R] shape")
    if normalizer < 1 or temperature <= 0:
        raise ValueError("normalizer >= 1 and temperature > 0 required")
    if not bool(torch.isfinite(student_logits).all()) or not bool(torch.isfinite(target_probability).all()):
        raise FloatingPointError("non-finite localization distribution")
    target = target_probability.detach().float()
    if bool((target < 0).any()) or not torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1)), atol=1e-6, rtol=1e-6):
        raise ValueError("target must be a normalized nonnegative distribution")
    # PyTorch >= 1.8 treats target zero as zero contribution, without epsilon.
    return F.kl_div(F.log_softmax(student_logits.float() / temperature, -1), target,
                    reduction="sum") * (temperature * temperature / (4 * normalizer))


@dataclass
class LocalizationSelection:
    batch_indices: Tensor
    anchor_indices: Tensor
    rgb_gt_indices: Tensor
    ir_gt_indices: Tensor
    rgb_distances: Tensor
    quality_gate: Tensor
    selected_indices: Tensor
    stats: Dict[str, Any]


def _check_raw(raw, cfg, strides):
    result = _layout(raw, cfg, strides)
    if raw["boxes"].device != raw["scores"].device:
        raise ValueError("scores and DFL boxes must share a device")
    if not raw["boxes"].is_floating_point() or not raw["scores"].is_floating_point():
        raise ValueError("raw logits must have floating dtype")
    dfl_view(raw["boxes"])
    return result


def _check_aligned(raw, template, cfg, strides):
    result = _check_raw(raw, cfg, strides)
    for key in ("boxes", "scores"):
        if raw[key].shape != template[key].shape or raw[key].device != template[key].device:
            raise ValueError("all models require matching classes, DFL bins, anchors and device")
    return result


@torch.no_grad()
def build_localization_selection(teacher: Mapping[str, Any], reference: Mapping[str, Any],
                                 batch: Mapping[str, Any], strides: Sequence[int] = (8, 16, 32),
                                 config=None, mode: str = "teacher", seed: int = 0,
                                 geometry_eligible: Optional[Tensor] = None,
                                 geometry_verified: bool = False, return_records: bool = False) -> LocalizationSelection:
    """Build immutable-in-value base and selected rows using frozen T/R only.

    geometry_eligible is bool[N_rgb_gt] or bool[N_rgb_gt,A] in augmented batch
    GT order. None means an UNVERIFIED all-true diagnostic, never formal proof.
    Object indices explicitly refer to this augmented batch, not raw-instance
    IDs; the loader/receipt owns any mapping to pre-augmentation identities.
    """
    if mode not in ("teacher", "gt", "random", "weight0"):
        raise ValueError("unknown localization mode: {}".format(mode))
    cfg = _config(config)
    centers, stride_values, ranges, size = _check_raw(reference, cfg, strides)
    _check_aligned(teacher, reference, cfg, strides)
    device = centers.device
    batch_size, nc, anchor_count = reference["scores"].shape
    reg_max = reference["boxes"].shape[1] // 4
    rgb_idx, rgb_cls, rgb_boxes = _labels(batch, batch_size, nc, device, size)
    teacher_batch = batch.get("teacher_batch", batch.get("ir_batch"))
    if teacher_batch is None:
        raise ValueError("independent teacher labels required as teacher_batch")
    ir_idx, ir_cls, ir_boxes = _labels(teacher_batch, batch_size, nc, device, size)
    if geometry_eligible is None:
        if geometry_verified:
            raise ValueError("verified geometry requires an explicit evidence mask")
        geometry = torch.ones(len(rgb_boxes), dtype=torch.bool, device=device)
    else:
        geometry = torch.as_tensor(geometry_eligible, device=device).detach()
        if geometry.dtype != torch.bool or tuple(geometry.shape) not in ((len(rgb_boxes),), (len(rgb_boxes), anchor_count)):
            raise ValueError("geometry_eligible must be bool[N_rgb_gt] or bool[N_rgb_gt,A]")
    rb_pred = _decode_boxes(reference, centers, stride_values)
    tb_pred = _decode_boxes(teacher, centers, stride_values)
    rconf, rclass = reference["scores"].detach().float().sigmoid().max(1)
    tconf, tclass = teacher["scores"].detach().float().sigmoid().max(1)
    active = torch.zeros(anchor_count, dtype=torch.bool, device=device)
    for level in cfg.levels:
        start, end = ranges[level]
        active[start:end] = True
    stats = {"mode": mode, "geometry_verified": bool(geometry_verified),
             "geometry_status": "caller_verified_mask" if geometry_verified else "UNVERIFIED_DIAGNOSTIC",
             "geometry_mask_supplied": geometry_eligible is not None,
             "rgb_gt_count": len(rgb_boxes), "teacher_gt_count": len(ir_boxes),
             "common_count": 0, "pair_iou_count": 0, "geometry_count": 0,
             "inside_count": 0, "support_count": 0, "unique_owner_count": 0,
             "reference_candidate_count": 0, "base_count": 0, "eligible_count": 0,
             "reference_reliable_count": 0, "both_reliable_count": 0,
             "reference_localization_gap_count": 0, "teacher_rgb_quality_count": 0,
             "teacher_own_quality_count": 0,
             "selected_count": 0, "normalizer": 1, "nominal_dose": 0.0,
             "loss_unweighted": 0.0, "selected_anchors": [], "selected_object_ids": [],
             "object_index_space": "augmented_batch_global_gt_rows", "seed": int(seed), "config": asdict(cfg)}
    rows, distances, gates, records = [], [], [], []
    for bi in range(batch_size):
        rg = torch.nonzero(rgb_idx == bi, as_tuple=False).flatten()
        tg = torch.nonzero(ir_idx == bi, as_tuple=False).flatten()
        rb, tb = rgb_boxes[rg], ir_boxes[tg]
        ri, ti = _match_objects(rb, rgb_cls[rg], tb, ir_cls[tg], cfg.match_iou)
        stats["common_count"] += len(ri)
        native_inside = native_candidate_mask(rb, centers, strides, cfg.native_candidate_epsilon)
        unique = native_inside.sum(0) == 1
        for ril, til in zip(ri.tolist(), ti.tolist()):
            rgi, tgi = int(rg[ril]), int(tg[til])
            pair_iou = float(_iou(rb[ril:ril+1], tb[til:til+1])[0, 0])
            if pair_iou < cfg.pair_iou:
                continue
            stats["pair_iou_count"] += 1
            candidate = active & geometry[rgi]
            if not bool(candidate.any()):
                continue
            stats["geometry_count"] += 1
            rd = unclamped_distances(rb[ril].expand(anchor_count, -1), centers, stride_values)
            td = unclamped_distances(tb[til].expand(anchor_count, -1), centers, stride_values)
            candidate = candidate & (rd >= 0).all(1) & (td >= 0).all(1)
            if not bool(candidate.any()):
                continue
            stats["inside_count"] += 1
            candidate = candidate & (rd <= reg_max - 1 - cfg.support_epsilon).all(1) & (td <= reg_max - 1 - cfg.support_epsilon).all(1)
            if not bool(candidate.any()):
                continue
            stats["support_count"] += 1
            candidate = candidate & unique & native_inside[ril]
            if not bool(candidate.any()):
                continue
            stats["unique_owner_count"] += 1
            overlaps = _iou(rb[ril:ril+1], rb_pred[bi])[0]
            candidate = candidate & (rconf[bi] >= cfg.reference_conf) & (overlaps >= cfg.reference_iou)
            if not bool(candidate.any()):
                continue
            stats["reference_candidate_count"] += 1
            # Confidence first, then IoU; torch.nonzero has ascending global
            # anchor order, so exact ties deterministically use the first ID.
            population = torch.nonzero(candidate, as_tuple=False).flatten()
            best_conf = rconf[bi, population].max()
            population = population[rconf[bi, population] == best_conf]
            best_iou = overlaps[population].max()
            ai = int(population[overlaps[population] == best_iou][0])
            rc = int(rgb_cls[rgi])
            r_iou = float(overlaps[ai])
            t_rgb_iou = float(_iou(rb[ril:ril+1], tb_pred[bi, ai:ai+1])[0, 0])
            t_ir_iou = float(_iou(tb[til:til+1], tb_pred[bi, ai:ai+1])[0, 0])
            r_correct = int(rclass[bi, ai]) == rc and float(rconf[bi, ai]) >= cfg.reliable_conf
            t_correct = int(tclass[bi, ai]) == rc and float(tconf[bi, ai]) >= cfg.reliable_conf
            # Cumulative filter counts expose attrition without changing E.
            gate = r_correct
            stats["reference_reliable_count"] += int(gate)
            gate = gate and t_correct
            stats["both_reliable_count"] += int(gate)
            gate = gate and r_iou < cfg.reference_iou_max
            stats["reference_localization_gap_count"] += int(gate)
            gate = gate and t_rgb_iou >= cfg.teacher_rgb_iou_min
            stats["teacher_rgb_quality_count"] += int(gate)
            gate = gate and t_ir_iou >= cfg.teacher_ir_iou_min
            stats["teacher_own_quality_count"] += int(gate)
            gate = gate and t_rgb_iou - r_iou > cfg.localization_margin
            rows.append((bi, ai, rgi, tgi))
            distances.append(rd[ai])
            gates.append(gate)
            if return_records:
                image_ids = batch.get("im_file", batch.get("image_id"))
                image_id = str(image_ids[bi]) if image_ids is not None else str(bi)
                records.append({"image_id": image_id, "batch_index": bi, "anchor_index": ai,
                                "rgb_gt_index": rgi, "rgb_gt_local_index": ril, "ir_gt_index": tgi,
                                "class": rc, "stride": float(stride_values[ai, 0]),
                                "center": centers[ai].tolist(), "pair_iou": pair_iou,
                                "reference_iou": r_iou, "teacher_rgb_iou": t_rgb_iou,
                                "teacher_ir_iou": t_ir_iou, "reference_conf": float(rconf[bi, ai]),
                                "teacher_conf": float(tconf[bi, ai]), "reference_correct": r_correct,
                                "teacher_correct": t_correct, "quality_gate": bool(gate),
                                "rgb_distances": rd[ai].tolist(), "ir_distances": td[ai].tolist()})
    row_tensor = torch.tensor(rows, dtype=torch.long, device=device).reshape(-1, 4)
    dists = torch.stack(distances) if distances else centers.new_empty((0, 4))
    gate_tensor = torch.tensor(gates, dtype=torch.bool, device=device)
    eligible = torch.nonzero(gate_tensor, as_tuple=False).flatten()
    if mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        chosen = torch.randperm(len(rows), generator=generator)[:len(eligible)].to(device)
    else:
        chosen = eligible
    normalizer = max(1, len(rows))
    stats.update(base_count=len(rows), eligible_count=len(eligible), selected_count=len(chosen),
                 normalizer=normalizer, nominal_dose=len(chosen) / normalizer,
                 selected_anchors=[tuple(v[:3]) for v in row_tensor[chosen].tolist()],
                 selected_object_ids=[(v[0], v[2], v[3]) for v in row_tensor[chosen].tolist()],
                 selected_quality_count=int(gate_tensor[chosen].sum()))
    if return_records:
        chosen_set = set(chosen.tolist())
        if rows:
            tlog = dfl_view(teacher["boxes"])[row_tensor[:, 0], row_tensor[:, 1]].detach()
            rlog = dfl_view(reference["boxes"])[row_tensor[:, 0], row_tensor[:, 1]].detach()
            q = gt_dfl_distribution(dists, reg_max, 1.0, cfg.support_epsilon)
            tce = -(q * F.log_softmax(tlog, -1)).sum(-1).mean(-1)
            rce = -(q * F.log_softmax(rlog, -1)).sum(-1).mean(-1)
            tp, rp = F.softmax(tlog, -1), F.softmax(rlog, -1)
            tent = -(tp * F.log_softmax(tlog, -1)).sum(-1).mean(-1)
            rent = -(rp * F.log_softmax(rlog, -1)).sum(-1).mean(-1)
            # This is a read-only local logit-gradient proxy, not a shared-
            # parameter gradient claim. Native training gradients are external.
            grad_native = (rp - q).reshape(len(rows), -1)
            grad_kd = (cfg.temperature * (F.softmax(rlog / cfg.temperature, -1) - F.softmax(tlog / cfg.temperature, -1))).reshape(len(rows), -1)
            dot = (grad_native * grad_kd).sum(-1)
            norms = grad_native.norm(dim=-1) * grad_kd.norm(dim=-1)
            for i, record in enumerate(records):
                record.update(selected=i in chosen_set, teacher_gt_dfl_ce=float(tce[i]),
                              reference_gt_dfl_ce=float(rce[i]), teacher_dfl_entropy=float(tent[i]),
                              reference_dfl_entropy=float(rent[i]), reference_logit_kd_native_dot=float(dot[i]),
                              reference_logit_kd_native_cosine=float(dot[i] / norms[i]) if float(norms[i]) > 0 else None)
        stats["base_records"] = records
    return LocalizationSelection(row_tensor[:, 0], row_tensor[:, 1], row_tensor[:, 2], row_tensor[:, 3],
                                 dists, gate_tensor, chosen, stats)


def localization_loss(student: Mapping[str, Any], teacher: Mapping[str, Any],
                      reference: Mapping[str, Any], batch: Mapping[str, Any],
                      strides: Sequence[int] = (8, 16, 32), config=None,
                      mode: str = "teacher", seed: int = 0,
                      geometry_eligible: Optional[Tensor] = None,
                      geometry_verified: bool = False, return_records: bool = False):
    """Return unweighted scalar and JSON-friendly metrics/optional base records."""
    cfg = _config(config)
    _check_aligned(student, reference, cfg, strides)
    selection = build_localization_selection(teacher, reference, batch, strides, cfg, mode, seed,
                                             geometry_eligible, geometry_verified, return_records)
    chosen = selection.selected_indices
    if len(chosen):
        bi, ai = selection.batch_indices[chosen], selection.anchor_indices[chosen]
        s = dfl_view(student["boxes"])[bi, ai]
        if mode == "gt":
            target = gt_dfl_distribution(selection.rgb_distances[chosen], s.shape[-1],
                                         cfg.temperature, cfg.support_epsilon)
        else:
            target = F.softmax(dfl_view(teacher["boxes"])[bi, ai].detach() / cfg.temperature, -1)
        loss = localization_kd(s, target, selection.stats["normalizer"], cfg.temperature)
    else:
        loss = student["boxes"].float().sum() * 0.0
    if mode == "weight0":
        loss = loss * 0.0
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite localization loss")
    selection.stats["loss_unweighted"] = float(loss.detach())
    return loss, selection.stats
