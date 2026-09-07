"""Object-versus-local-background class-evidence KD, pilot v1.

This is a hypothesis implementation, not a validated gain or novelty claim.
Inputs are YOLO raw ``scores`` logits [B,C,A], DFL ``boxes`` [B,4R,A],
and P3/P4/P5 ``feats`` with corresponding spatial sizes.  Student RGB labels
are ``batch_idx/cls/bboxes`` (normalized xywh); independent teacher labels
are ``batch['teacher_batch']``.  No image, feature, or GT-coordinate copying
across modalities occurs. ``same_modal`` explicitly uses RGB labels only.

For each common object, each valid level pools the correct-class logits by
log-mean-exp inside its own GT box and inside a 2x concentric annulus excluding
ALL GT of that modality. Its evidence is the foreground-minus-background
logit difference / T, averaged over levels valid in BOTH modalities. The RGB
frozen reference uses precisely the student's regions. No DFL/feature loss,
learnable gate, NMS, optimizer scaling, batch-size scaling, or lambda occurs
inside this module. Candidate checks use decoded *pre-NMS* DFL boxes.

The base set is common class/IoU-matched objects with valid regions and a
class-independent frozen-RGB candidate. Teacher correctness is measured
against its own GT. q = max(softplus(-reference evidence) -
softplus(-teacher evidence), 0) is a ranking surrogate, NOT calibrated error
or semantic information. Paired selects ceil(rho * eligible_count), where
eligible means base AND teacher-correct AND q>0. SmoothL1 targets are clipped
teacher evidence, normalized by base_count BEFORE these teacher filters.

paired_random selects the identical K uniformly from the entire base set;
this tests the complete quality/correctness selection versus random selection.
paired_uniform uses the whole base (higher nominal dose). weight0 computes
the same diagnostics but returns differentiable zero. same_modal is paired
selection using a supplied RGB teacher and RGB GT. Shuffled / GT controls
are deliberately unavailable until their content policies are frozen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Mapping, Sequence

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class EvidenceConfig:
    input_size: int | tuple[int, int] = 640
    levels: tuple[int, ...] = (0, 1)
    temperature: float = 2.0
    background_scale: float = 2.0
    minimum_foreground: int = 1
    minimum_background: int = 4
    match_iou: float = 0.5
    reference_conf: float = 0.05
    reference_iou: float = 0.1
    teacher_conf: float = 0.25
    teacher_iou: float = 0.5
    rho: float = 0.5
    target_clip: float = 8.0
    smooth_l1_beta: float = 1.0

    def __post_init__(self):
        size = (self.input_size, self.input_size) if isinstance(self.input_size, int) else self.input_size
        if len(size) != 2 or any(int(v) != v or v <= 0 for v in size):
            raise ValueError("input_size must be positive (height, width)")
        if not self.levels or len(set(self.levels)) != len(self.levels) or any(v < 0 or v > 2 for v in self.levels):
            raise ValueError("levels must be unique indices from 0,1,2")
        if not (self.temperature > 0 and self.background_scale > 1 and self.target_clip > 0 and self.smooth_l1_beta > 0):
            raise ValueError("invalid evidence scale/temperature/target/beta")
        if self.minimum_foreground < 1 or self.minimum_background < 1:
            raise ValueError("region minimum counts must be positive")
        for name in ("match_iou", "reference_conf", "reference_iou", "teacher_conf", "teacher_iou", "rho"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")


def _config(config: EvidenceConfig | Mapping[str, Any] | None) -> EvidenceConfig:
    if config is None:
        return EvidenceConfig()
    return config if isinstance(config, EvidenceConfig) else EvidenceConfig(**dict(config))


def _iou(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise xyxy IoU, including empty inputs."""
    overlap = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
               torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return overlap / (area_a[:, None] + area_b[None, :] - overlap).clamp_min(1e-9)


def _match_objects(rgb_boxes: Tensor, rgb_classes: Tensor, teacher_boxes: Tensor,
                   teacher_classes: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
    """Maximum-cardinality eligible matching, then maximum total IoU.

    A raw IoU-sum Hungarian assignment followed by thresholding is incorrect:
    subthreshold edges can steal a valid match. Each eligible edge receives a
    cardinality bonus larger than the total possible IoU tie-breaker.
    """
    empty = torch.empty(0, dtype=torch.long, device=rgb_boxes.device)
    if not len(rgb_boxes) or not len(teacher_boxes):
        return empty, empty
    overlaps = _iou(rgb_boxes, teacher_boxes)
    eligible = (rgb_classes[:, None] == teacher_classes[None, :]) & (overlaps >= threshold)
    bonus = min(len(rgb_boxes), len(teacher_boxes)) + 1.0
    cost = (eligible * (bonus + overlaps)).detach().cpu().numpy()
    rows, cols = linear_sum_assignment(-cost)
    rows = torch.as_tensor(rows, device=rgb_boxes.device, dtype=torch.long)
    cols = torch.as_tensor(cols, device=rgb_boxes.device, dtype=torch.long)
    keep = eligible[rows, cols]
    return rows[keep], cols[keep]


def _layout(raw: Mapping[str, Any], cfg: EvidenceConfig, strides: Sequence[int]):
    if not all(k in raw for k in ("scores", "boxes", "feats")):
        raise ValueError("raw predictions require scores, boxes, feats")
    scores, boxes, feats = raw["scores"], raw["boxes"], raw["feats"]
    if scores.ndim != 3 or boxes.ndim != 3 or boxes.shape[1] % 4:
        raise ValueError("expected scores[B,C,A] and DFL boxes[B,4R,A]")
    if scores.shape[0] != boxes.shape[0] or scores.shape[2] != boxes.shape[2]:
        raise ValueError("scores and boxes batch/anchor dimensions differ")
    if len(feats) != 3 or len(strides) != 3:
        raise ValueError("expected exactly P3/P4/P5 and three strides")
    size = (cfg.input_size, cfg.input_size) if isinstance(cfg.input_size, int) else tuple(cfg.input_size)
    centers, ranges, stride_values, offset = [], [], [], 0
    for feature, stride in zip(feats, strides):
        h, w = feature.shape[-2:]
        if feature.shape[0] != scores.shape[0] or (h * stride, w * stride) != size:
            raise ValueError("feature grids/strides do not match batch or input_size")
        x = torch.arange(w, device=scores.device, dtype=torch.float32).repeat(h)
        y = torch.arange(h, device=scores.device, dtype=torch.float32).repeat_interleave(w)
        centers.append(torch.stack((x + .5, y + .5), -1) * stride)
        stride_values.append(torch.full((h * w, 1), stride, device=scores.device, dtype=torch.float32))
        ranges.append((offset, offset + h * w))
        offset += h * w
    if offset != scores.shape[2] or scores.shape[1] < 1 or boxes.shape[1] < 4:
        raise ValueError("anchor count or channel count is invalid")
    if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(boxes).all()):
        raise FloatingPointError("non-finite raw scores or DFL logits")
    return torch.cat(centers), torch.cat(stride_values), ranges, size


def _decode_boxes(raw: Mapping[str, Any], centers: Tensor, stride_values: Tensor) -> Tensor:
    logits = raw["boxes"].detach().float()
    regmax = logits.shape[1] // 4
    distances = (logits.reshape(logits.shape[0], 4, regmax, -1).softmax(2) *
                 torch.arange(regmax, device=logits.device, dtype=torch.float32)[None, None, :, None]).sum(2)
    distances = distances.transpose(1, 2) * stride_values[None]
    return torch.cat((centers[None] - distances[..., :2], centers[None] + distances[..., 2:]), -1)


def _labels(batch: Mapping[str, Any], batch_size: int, nc: int, device: torch.device, size):
    values = [torch.as_tensor(batch[k], device=device).detach() for k in ("batch_idx", "cls", "bboxes")]
    idx, cls, boxes = values[0].reshape(-1), values[1].reshape(-1), values[2]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(idx) != len(cls) or len(idx) != len(boxes):
        raise ValueError("GT fields must have consistent lengths and bboxes[N,4]")
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise ValueError("GT contains non-finite values")
    if len(idx) and (not bool((idx == idx.long()).all()) or not bool(((idx >= 0) & (idx < batch_size)).all())):
        raise ValueError("GT batch indices must be integers in range")
    if len(cls) and (not bool((cls == cls.long()).all()) or not bool(((cls >= 0) & (cls < nc)).all())):
        raise ValueError("GT class indices must be integers in range")
    boxes = boxes.float()
    if len(boxes) and not bool(((boxes >= 0) & (boxes <= 1)).all() & (boxes[:, 2:] > 0).all()):
        raise ValueError("GT boxes must be normalized xywh with positive sizes")
    scale = boxes.new_tensor((size[1], size[0]))
    xyxy = torch.cat(((boxes[:, :2] - boxes[:, 2:] / 2) * scale,
                       (boxes[:, :2] + boxes[:, 2:] / 2) * scale), -1)
    return idx.long(), cls.long(), xyxy


def _inside(boxes: Tensor, centers: Tensor) -> Tensor:
    return ((centers[None, :, :] >= boxes[:, None, :2]) &
            (centers[None, :, :] < boxes[:, None, 2:])).all(-1)


def _evidence(scores: Tensor, classes: Tensor, boxes: Tensor, all_boxes: Tensor,
              centers: Tensor, ranges, cfg: EvidenceConfig) -> tuple[Tensor, Tensor]:
    values, valid = [], []
    midpoint = (boxes[:, :2] + boxes[:, 2:]) / 2
    half_size = (boxes[:, 2:] - boxes[:, :2]) * (cfg.background_scale / 2)
    outer = torch.cat((midpoint - half_size, midpoint + half_size), -1)
    for level in cfg.levels:
        start, end = ranges[level]
        xy = centers[start:end]
        fg = _inside(boxes, xy)
        bg = _inside(outer, xy) & ~_inside(all_boxes, xy).any(0)[None]
        nfg, nbg = fg.sum(1), bg.sum(1)
        ok = (nfg >= cfg.minimum_foreground) & (nbg >= cfg.minimum_background)
        z = scores[classes, start:end].float()
        # Empty regions use a finite sentinel; their value is excluded by ok.
        zfg = z.masked_fill(~fg, -1e30).logsumexp(1) - nfg.clamp_min(1).float().log()
        zbg = z.masked_fill(~bg, -1e30).logsumexp(1) - nbg.clamp_min(1).float().log()
        values.append(torch.where(ok, (zfg - zbg) / cfg.temperature, torch.zeros_like(zfg)))
        valid.append(ok)
    return torch.stack(values, 1), torch.stack(valid, 1)


def _has_candidate(pred_boxes: Tensor, probabilities: Tensor, gt_boxes: Tensor,
                   conf: float, iou: float, classes: Tensor | None = None) -> Tensor:
    if classes is None:
        confidence = probabilities.max(0).values[None].expand(len(gt_boxes), -1)
    else:
        confidence = probabilities[classes]
    class_ok = torch.ones_like(confidence, dtype=torch.bool) if classes is None else probabilities.argmax(0)[None] == classes[:, None]
    return ((confidence >= conf) & class_ok & (_iou(gt_boxes, pred_boxes) >= iou)).any(1)


def _choose(q: Tensor, eligible: Tensor, base: Tensor, arm: str, rho: float, seed: int) -> Tensor:
    k = ceil(rho * int(eligible.sum().item()))
    if arm == "paired_uniform":
        return torch.nonzero(base, as_tuple=False).flatten()
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=q.device)
    if arm == "paired_random":
        population = torch.nonzero(base, as_tuple=False).flatten()
        # A private CPU generator never advances training/augmentation RNG.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        order = torch.randperm(len(population), generator=generator)[:k].to(q.device)
        return population[order]
    population = torch.nonzero(eligible, as_tuple=False).flatten()
    values = q[population].detach().cpu().tolist()
    # Python sorting is stable; exact quality ties retain batch/GT order.
    order = sorted(range(len(values)), key=lambda i: -values[i])[:k]
    return population[torch.tensor(order, dtype=torch.long, device=q.device)]


def object_evidence_loss(student: Mapping[str, Any], teacher: Mapping[str, Any],
                         reference: Mapping[str, Any], batch: Mapping[str, Any],
                         strides: Sequence[int] = (8, 16, 32),
                         config: EvidenceConfig | Mapping[str, Any] | None = None,
                         arm: str = "paired", seed: int = 0) -> tuple[Tensor, dict[str, Any]]:
    """Return unweighted scalar KD and detached JSON-friendly batch diagnostics.

    ``seed`` must change with the registered global batch index for random
    controls; equal seed/input gives equal selection without global RNG use.
    Caller adds ``lambda * KD`` exactly once in its detector loss convention.
    """
    supported = {"paired", "weight0", "paired_random", "paired_uniform", "same_modal"}
    if arm not in supported:
        raise NotImplementedError(f"arm {arm!r} has no frozen content policy; supported: {sorted(supported)}")
    cfg = _config(config)
    centers, stride_values, ranges, size = _layout(student, cfg, strides)
    for name, raw in (("teacher", teacher), ("reference", reference)):
        _, _, their_ranges, their_size = _layout(raw, cfg, strides)
        if raw["scores"].shape != student["scores"].shape or their_ranges != ranges or their_size != size:
            raise ValueError(f"{name} layout/classes must match student")
        if raw["scores"].device != student["scores"].device:
            raise ValueError("all models must be on the same device")
    scores = student["scores"].float()
    batch_size, nc, _ = scores.shape
    zero = scores.sum() * 0.0
    rgb_idx, rgb_cls, rgb_boxes = _labels(batch, batch_size, nc, scores.device, size)
    teacher_batch = batch if arm == "same_modal" else batch.get("teacher_batch", batch.get("ir_batch"))
    if teacher_batch is None:
        raise ValueError("independent teacher GT required as batch['teacher_batch']")
    ir_idx, ir_cls, ir_boxes = _labels(teacher_batch, batch_size, nc, scores.device, size)
    with torch.no_grad():
        teacher_scores, reference_scores = teacher["scores"].detach().float(), reference["scores"].detach().float()
        teacher_boxes = _decode_boxes(teacher, centers, stride_values)
        reference_boxes = _decode_boxes(reference, centers, stride_values)
        teacher_probs, reference_probs = teacher_scores.sigmoid(), reference_scores.sigmoid()
    collected_s, collected_t, collected_r, valid_regions, candidates, correctness, object_ids = [], [], [], [], [], [], []
    for bi in range(batch_size):
        rgb_global = torch.nonzero(rgb_idx == bi, as_tuple=False).flatten()
        ir_global = torch.nonzero(ir_idx == bi, as_tuple=False).flatten()
        rb, rc, tb, tc = rgb_boxes[rgb_global], rgb_cls[rgb_global], ir_boxes[ir_global], ir_cls[ir_global]
        ri, ti = _match_objects(rb, rc, tb, tc, cfg.match_iou)
        if not len(ri):
            continue
        es, vs = _evidence(scores[bi], rc[ri], rb[ri], rb, centers, ranges, cfg)
        with torch.no_grad():
            et, vt = _evidence(teacher_scores[bi], tc[ti], tb[ti], tb, centers, ranges, cfg)
            er, vr = _evidence(reference_scores[bi], rc[ri], rb[ri], rb, centers, ranges, cfg)
            common_valid = vs & vt & vr
            denominator = common_valid.sum(1).clamp_min(1)
            ref_ok = _has_candidate(reference_boxes[bi], reference_probs[bi], rb[ri], cfg.reference_conf, cfg.reference_iou)
            teacher_ok = _has_candidate(teacher_boxes[bi], teacher_probs[bi], tb[ti], cfg.teacher_conf, cfg.teacher_iou, tc[ti])
        collected_s.append((es * common_valid).sum(1) / denominator)
        collected_t.append((et * common_valid).sum(1) / denominator)
        collected_r.append((er * common_valid).sum(1) / denominator)
        valid_regions.append(common_valid.any(1))
        candidates.append(ref_ok)
        correctness.append(teacher_ok)
        object_ids.extend((int(bi), int(a), int(b)) for a, b in zip(rgb_global[ri].tolist(), ir_global[ti].tolist()))
    stats: dict[str, Any] = {"arm": arm, "rgb_gt_count": len(rgb_boxes), "teacher_gt_count": len(ir_boxes),
                             "common_count": 0, "valid_region_count": 0, "reference_candidate_count": 0,
                             "base_count": 0, "teacher_correct_base_count": 0, "eligible_count": 0,
                             "selected_count": 0, "normalizer": 1, "nominal_dose": 0.0,
                             "loss_unweighted": 0.0, "selected_object_ids": [], "config": asdict(cfg)}
    if not collected_s:
        return zero, stats
    es, et, er = torch.cat(collected_s), torch.cat(collected_t).detach(), torch.cat(collected_r).detach()
    region_ok, reference_ok, teacher_ok = torch.cat(valid_regions), torch.cat(candidates), torch.cat(correctness)
    base = region_ok & reference_ok
    q = (F.softplus(-er) - F.softplus(-et)).clamp_min(0).detach()
    eligible = base & teacher_ok & (q > 0)
    selected = _choose(q, eligible, base, arm, cfg.rho, seed)
    normalizer = max(1, int(base.sum().item()))
    target = et.clamp(-cfg.target_clip, cfg.target_clip)
    loss = F.smooth_l1_loss(es[selected], target[selected], reduction="sum", beta=cfg.smooth_l1_beta) / normalizer if len(selected) else zero
    if arm == "weight0":
        loss = loss * 0.0
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite object evidence loss")
    stats.update(common_count=len(es), valid_region_count=int(region_ok.sum()), reference_candidate_count=int(reference_ok.sum()),
                 base_count=int(base.sum()), teacher_correct_base_count=int((base & teacher_ok).sum()),
                 eligible_count=int(eligible.sum()), selected_count=len(selected), normalizer=normalizer,
                 nominal_dose=len(selected) / normalizer, loss_unweighted=float(loss.detach()),
                 selected_object_ids=[object_ids[i] for i in selected.tolist()])
    for name, values in (("student_evidence", es.detach()), ("teacher_evidence", et), ("reference_evidence", er), ("quality", q)):
        subset = values[base]
        stats[name + "_base_mean"] = float(subset.mean()) if len(subset) else None
        stats[name + "_base_min"] = float(subset.min()) if len(subset) else None
        stats[name + "_base_max"] = float(subset.max()) if len(subset) else None
        stats[name + "_base_sd"] = float(subset.std(unbiased=False)) if len(subset) else None
        stats[name + "_selected_mean"] = float(values[selected].mean()) if len(selected) else None
        stats[name + "_selected_sd"] = float(values[selected].std(unbiased=False)) if len(selected) else None
    p = target[selected].sigmoid()
    stats["target_binary_entropy_selected"] = float(-(p * p.clamp_min(1e-9).log() + (1-p) * (1-p).clamp_min(1e-9).log()).mean()) if len(p) else None
    stats["target_gap_to_positive_selected"] = float((1-p).mean()) if len(p) else None
    stats["target_clipped_count"] = int((et[selected].abs() > cfg.target_clip).sum())
    return loss, stats
