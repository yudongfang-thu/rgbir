"""Content controls with frozen paired selection; original OEv1 stays unchanged.

object_evidence_content_loss is a minimal extension of the reviewed, vendored
legacy_oev1/object_evidence_loss.py function. Every original paired computation
is retained. Only the final target evidence can come from another forward pass.
The wrong-image content is pooled in ORIGINAL paired-IR GT foreground/rings,
excluding ORIGINAL paired-IR GT, with the ORIGINAL common-level mask. Wrong-
image labels/boxes/correctness never determine objects, regions, E, q or K.

Selection diagnostics named teacher_evidence/quality describe the paired
selector teacher; target entropy/gap/clipping describe the actual content.
For identical content, the complete returned dictionary matches legacy C.
No teacher forward, loader, augmentation, lambda, B scaling or RNG call lives
here. A caller must freeze train-only donors and replay paired geometry without
advancing training RNG before supplying content_teacher's raw predictions.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "legacy_oev1"))
sys.path.insert(0, str(HERE))
from object_evidence_loss import (EvidenceConfig, object_evidence_loss,
    _config, _layout, _labels, _decode_boxes, _match_objects, _evidence, _has_candidate, _choose)
from localization_loss import (LocalizationConfig, LocalizationSelection,
    _check_aligned, dfl_view, localization_kd)


def object_evidence_content_loss(student: Mapping[str, Any], teacher: Mapping[str, Any],
                         reference: Mapping[str, Any], batch: Mapping[str, Any],
                         strides: Sequence[int] = (8, 16, 32),
                         config: EvidenceConfig | Mapping[str, Any] | None = None,
                         arm: str = "paired", seed: int = 0, content_teacher=None) -> tuple[Tensor, dict[str, Any]]:
    """Return unweighted scalar KD and detached JSON-friendly batch diagnostics.

    ``seed`` must change with the registered global batch index for random
    controls; equal seed/input gives equal selection without global RNG use.
    Caller adds ``lambda * KD`` exactly once in its detector loss convention.
    """
    content_teacher = teacher if content_teacher is None else content_teacher
    supported = {"paired", "weight0", "paired_random", "paired_uniform", "same_modal"}
    if arm not in supported:
        raise NotImplementedError(f"arm {arm!r} has no frozen content policy; supported: {sorted(supported)}")
    cfg = _config(config)
    centers, stride_values, ranges, size = _layout(student, cfg, strides)
    for name, raw in (("teacher", teacher), ("reference", reference), ("content_teacher", content_teacher)):
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
    collected_content = []
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
            content_evidence, _ = _evidence(content_teacher["scores"][bi].detach().float(), tc[ti], tb[ti], tb, centers, ranges, cfg)
            er, vr = _evidence(reference_scores[bi], rc[ri], rb[ri], rb, centers, ranges, cfg)
            common_valid = vs & vt & vr
            denominator = common_valid.sum(1).clamp_min(1)
            ref_ok = _has_candidate(reference_boxes[bi], reference_probs[bi], rb[ri], cfg.reference_conf, cfg.reference_iou)
            teacher_ok = _has_candidate(teacher_boxes[bi], teacher_probs[bi], tb[ti], cfg.teacher_conf, cfg.teacher_iou, tc[ti])
        collected_s.append((es * common_valid).sum(1) / denominator)
        collected_t.append((et * common_valid).sum(1) / denominator)
        collected_content.append((content_evidence * common_valid).sum(1) / denominator)
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
    content_values = torch.cat(collected_content).detach()
    target = content_values.clamp(-cfg.target_clip, cfg.target_clip)
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
    stats["target_clipped_count"] = int((content_values[selected].abs() > cfg.target_clip).sum())
    return loss, stats


def localization_content_loss(student, content_teacher, selection: LocalizationSelection,
                              temperature=2.0, weight0=False, strides=(8, 16, 32)):
    """Use a previously frozen L selection and replace ONLY its DFL target.

    Build selection once using paired T/R. The content teacher's class scores
    are validated as part of the raw contract, but never read for selection.
    Returned stats are a shallow copy; input selection and its gate stay intact.
    """
    cfg = LocalizationConfig(**selection.stats["config"])
    if temperature != cfg.temperature:
        raise ValueError("Content control must retain the frozen temperature")
    _check_aligned(content_teacher, student, cfg, strides)
    _check_aligned(student, content_teacher, cfg, strides)
    chosen = selection.selected_indices
    if len(chosen):
        bi, ai = selection.batch_indices[chosen], selection.anchor_indices[chosen]
        student_logits = dfl_view(student["boxes"])[bi, ai]
        target = F.softmax(dfl_view(content_teacher["boxes"])[bi, ai].detach() / temperature, -1)
        loss = localization_kd(student_logits, target, selection.stats["normalizer"], temperature)
    else:
        loss = student["boxes"].float().sum() * 0.0
    if weight0:
        loss = loss * 0.0
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite localization content control")
    stats = dict(selection.stats)
    stats["loss_unweighted"] = float(loss.detach())
    return loss, stats


def rgb_only_labels(batch):
    """Explicit allowlist: a genuine same-modal control never reads IR labels."""
    return {key: batch[key] for key in ("batch_idx", "cls", "bboxes")}


def same_modal_evidence_loss(student, rgb_teacher, reference, batch,
                             strides=(8, 16, 32), config=None, seed=0):
    """Existing same_modal C using an independent frozen RGB teacher.

    The campaign must supply a distinct trained teacher (approved seed0) and
    check nonzero signal; this function neither substitutes R nor updates it.
    """
    return object_evidence_loss(student, rgb_teacher, reference, rgb_only_labels(batch),
                                strides=strides, config=config, arm="same_modal", seed=seed)
