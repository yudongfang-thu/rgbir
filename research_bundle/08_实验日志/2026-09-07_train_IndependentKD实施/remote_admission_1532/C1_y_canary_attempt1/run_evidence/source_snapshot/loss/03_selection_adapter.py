"""C0-compatible object selection for independent class KD.

The frozen OEv1 helpers remain the selection oracle.  This adapter exposes the
matched -> pre-teacher-base mapping and per-modality regions that the original
public function does not return.  It never changes a running legacy module.
No teacher/reference tensor contributes gradients.  A selection is local to one
prediction/batch, not a reusable raw-image or cross-augmentation cache.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
REFERENCE = (HERE / "task_conditional_reference" if (HERE / "task_conditional_reference").is_dir()
             else HERE.parent / "rgbir_task_conditional_v1")
LEGACY_SOURCE = REFERENCE / "legacy_oev1" / "object_evidence_loss.py"
_NAME = "_rgbir_independent_pinned_object_evidence_loss"
if _NAME in sys.modules:
    original = sys.modules[_NAME]
    if Path(original.__file__).resolve() != LEGACY_SOURCE:
        raise RuntimeError("Pinned evidence module resolved to another source")
else:
    _spec = importlib.util.spec_from_file_location(_NAME, LEGACY_SOURCE)
    if _spec is None or _spec.loader is None:
        raise RuntimeError("Missing frozen OEv1 evidence module")
    original = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = original
    _spec.loader.exec_module(original)

EvidenceConfig = original.EvidenceConfig
SELECTION_VERSION = "oev1_paired_selection_adapter_v1"


def evidence_config(config=None):
    # A dataclass from the original trainer's separately imported snapshot is
    # semantically the same config but has a different Python class identity.
    if config is not None and is_dataclass(config):
        config = asdict(config)
    return original._config(config)


@dataclass(frozen=True)
class RegionLevel:
    image_index: int
    level: int
    anchor_start: int
    anchor_end: int
    base_rows: Tensor
    rgb_foreground: Tensor
    rgb_background: Tensor
    teacher_foreground: Tensor
    teacher_background: Tensor
    rgb_other_gt_fraction: Tensor
    teacher_other_gt_fraction: Tensor


@dataclass(frozen=True)
class ClassificationSelection:
    student_delta: Tensor                  # [M,L,C], raw difference, before T
    teacher_delta: Tensor                  # detached
    reference_delta: Tensor                # detached
    valid_levels: Tensor                   # [M,L], original common validity
    selected: Tensor                       # bool[M], C0 selected object set
    labels: Tensor                         # long[M]
    base_to_matched: Tensor                 # original matched row for each base
    matched_to_base: Tensor                 # -1 for non-base matched rows
    selected_matched_indices: Tensor        # original C0 q order
    matched_object_ids: tuple
    base_object_ids: tuple                  # (batch index, RGB GT row, IR GT row)
    quality: Tensor                        # q in base order
    eligible: Tensor                       # teacher-correct and q>0, in base
    regions: tuple
    c0_loss: Tensor                         # unweighted exact original scalar
    c0_stats: dict
    config: Any
    source_tensors: tuple
    source_versions: tuple
    source_batch: Any

    @property
    def base_count(self):
        return int(self.student_delta.shape[0])

    @property
    def rgb_gt_indices(self):
        return torch.tensor([row[1] for row in self.base_object_ids], dtype=torch.long,
                            device=self.student_delta.device)

    @property
    def ir_gt_indices(self):
        return torch.tensor([row[2] for row in self.base_object_ids], dtype=torch.long,
                            device=self.student_delta.device)

    @property
    def raw_delta_s(self):
        return self.student_delta

    @property
    def raw_delta_t(self):
        return self.teacher_delta

    @property
    def raw_delta_r(self):
        return self.reference_delta

    def assert_source(self, student, teacher, reference, batch):
        tensors = _source_tensors(student, teacher, reference, batch)
        if batch is not self.source_batch or len(tensors) != len(self.source_tensors):
            raise ValueError("Classification selection belongs to another batch")
        if any(a is not b for a, b in zip(tensors, self.source_tensors)):
            raise ValueError("Classification selection belongs to another prediction/augmentation")
        if tuple(_tensor_version(t) for t in tensors) != self.source_versions:
            raise ValueError("Classification selection source was mutated after construction")


def _teacher_batch(batch):
    value = batch.get("teacher_batch", batch.get("ir_batch"))
    if value is None:
        raise ValueError("Independent teacher labels are required")
    return value


def _tensor_version(t):
    try:
        return t._version
    except RuntimeError:  # inference tensors do not expose a version counter
        return None


def _source_tensors(student, teacher, reference, batch):
    other = _teacher_batch(batch)
    tensors = tuple(raw[key] for raw in (student, teacher, reference) for key in ("scores", "boxes"))
    tensors += tuple(raw["feats"][i] for raw in (student, teacher, reference) for i in range(3))
    tensors += tuple(labels[key] for labels in (batch, other) for key in ("batch_idx", "cls", "bboxes"))
    if not all(isinstance(value, Tensor) for value in tensors):
        raise ValueError("Adapter requires preprocessed tensor labels and raw predictions")
    return tensors


def region_masks(boxes, all_boxes, centers, cfg):
    """Exact original foreground/annulus geometry; reference shares RGB masks."""
    midpoint = (boxes[:, :2] + boxes[:, 2:]) / 2
    half_size = (boxes[:, 2:] - boxes[:, :2]) * (cfg.background_scale / 2)
    outer = torch.cat((midpoint - half_size, midpoint + half_size), -1)
    fg = original._inside(boxes, centers)
    all_inside = original._inside(all_boxes, centers)
    bg = original._inside(outer, centers) & ~all_inside.any(0)[None]
    # A foreground box can contain another GT. This is a diagnostic, not a gate.
    overlap = (fg & (all_inside.sum(0) > 1)[None]).sum(1).float() / fg.sum(1).clamp_min(1)
    return fg, bg, overlap


def pool_relative_logits(scores, foreground, background, minimum_foreground=1, minimum_background=4):
    """[C,A] and bool[M,A] -> raw [M,C] LME differences, bool[M].

    Student foreground and background both retain gradient.  Invalid regions
    have finite differentiable zeros.  No temperature or clipping occurs here.
    """
    if scores.ndim != 2 or scores.shape[0] < 1 or not scores.is_floating_point():
        raise ValueError("scores must be floating point [C,A], C>=1")
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Nonfinite pooling scores")
    if (foreground.dtype != torch.bool or background.dtype != torch.bool
            or foreground.ndim != 2 or foreground.shape != background.shape
            or foreground.shape[1] != scores.shape[1]):
        raise ValueError("region masks must be matching bool[M,A]")
    if foreground.device != scores.device or background.device != scores.device:
        raise ValueError("region masks must share score device")
    if minimum_foreground < 1 or minimum_background < 1:
        raise ValueError("Minimum region sizes must be positive")
    if bool((foreground & background).any()):
        raise ValueError("Foreground and background must not overlap")
    nfg, nbg = foreground.sum(-1), background.sum(-1)
    valid = (nfg >= minimum_foreground) & (nbg >= minimum_background)
    z = scores.float()
    if foreground.shape[0] == 0:
        return z[:, :0].T, valid
    # Match legacy masked-LME arithmetic, extended across class channels.  The
    # per-object loop bounds peak temporary allocation instead of [M,C,A].
    values = []
    for i in range(foreground.shape[0]):
        fg_value = z.masked_fill(~foreground[i][None], -1e30).logsumexp(1) - nfg[i].clamp_min(1).float().log()
        bg_value = z.masked_fill(~background[i][None], -1e30).logsumexp(1) - nbg[i].clamp_min(1).float().log()
        values.append(torch.where(valid[i], fg_value - bg_value, torch.zeros_like(fg_value)))
    return torch.stack(values), valid


def build_classification_selection(student: Mapping[str, Any], teacher: Mapping[str, Any],
                                   reference: Mapping[str, Any], batch: Mapping[str, Any],
                                   strides: Sequence[int] = (8, 16, 32), config=None,
                                   selection_seed: int = 0) -> ClassificationSelection:
    """Expose the unchanged paired-C0 gate and independent full-class content.

    Teacher/reference correctness, reference coarse candidate, q, rho and stable
    ties call the frozen implementation.  C1 content NEVER feeds selection.
    `c0_loss` retains the old GT-only arithmetic graph for calibration/regression.
    """
    cfg = evidence_config(config)
    centers, stride_values, ranges, size = original._layout(student, cfg, strides)
    for name, raw in (("teacher", teacher), ("reference", reference)):
        _, _, their_ranges, their_size = original._layout(raw, cfg, strides)
        if raw["scores"].shape != student["scores"].shape or their_ranges != ranges or their_size != size:
            raise ValueError(name + " layout/classes differ from student")
        if raw["scores"].device != student["scores"].device:
            raise ValueError("Models must share device")
    sources = _source_tensors(student, teacher, reference, batch)
    scores = student["scores"].float()
    batch_size, nc, _ = scores.shape
    zero = scores.sum() * 0.0
    rgb_idx, rgb_cls, rgb_boxes = original._labels(batch, batch_size, nc, scores.device, size)
    ir_idx, ir_cls, ir_boxes = original._labels(_teacher_batch(batch), batch_size, nc, scores.device, size)
    with torch.no_grad():
        ts, rs = teacher["scores"].detach().float(), reference["scores"].detach().float()
        tboxes = original._decode_boxes(teacher, centers, stride_values)
        rboxes = original._decode_boxes(reference, centers, stride_values)
        tprobs, rprobs = ts.sigmoid(), rs.sigmoid()
    es_rows, et_rows, er_rows = [], [], []
    sd_rows, td_rows, rd_rows, common_rows, ref_rows, correct_rows, class_rows = [], [], [], [], [], [], []
    object_ids, region_rows = [], []
    for bi in range(batch_size):
        rgb_global = torch.nonzero(rgb_idx == bi, as_tuple=False).flatten()
        ir_global = torch.nonzero(ir_idx == bi, as_tuple=False).flatten()
        rb, rc, tb, tc = rgb_boxes[rgb_global], rgb_cls[rgb_global], ir_boxes[ir_global], ir_cls[ir_global]
        ri, ti = original._match_objects(rb, rc, tb, tc, cfg.match_iou)
        if not len(ri):
            continue
        # C0 arithmetic is unchanged, including its order of scale reduction.
        es, vs = original._evidence(scores[bi], rc[ri], rb[ri], rb, centers, ranges, cfg)
        with torch.no_grad():
            et, vt = original._evidence(ts[bi], tc[ti], tb[ti], tb, centers, ranges, cfg)
            er, vr = original._evidence(rs[bi], rc[ri], rb[ri], rb, centers, ranges, cfg)
            common = vs & vt & vr
            denom = common.sum(1).clamp_min(1)
            ref_ok = original._has_candidate(rboxes[bi], rprobs[bi], rb[ri], cfg.reference_conf, cfg.reference_iou)
            teacher_ok = original._has_candidate(tboxes[bi], tprobs[bi], tb[ti], cfg.teacher_conf, cfg.teacher_iou, tc[ti])
        es_rows.append((es * common).sum(1) / denom)
        et_rows.append((et * common).sum(1) / denom)
        er_rows.append((er * common).sum(1) / denom)
        common_rows.append(common)
        ref_rows.append(ref_ok)
        correct_rows.append(teacher_ok)
        class_rows.append(rc[ri])
        first = len(object_ids)
        object_ids.extend((int(bi), int(a), int(b)) for a, b in zip(rgb_global[ri].tolist(), ir_global[ti].tolist()))
        sd_levels, td_levels, rd_levels = [], [], []
        for li, level in enumerate(cfg.levels):
            start, end = ranges[level]
            with torch.no_grad():
                fg, bg, rgb_overlap = region_masks(rb[ri], rb, centers[start:end], cfg)
                tfg, tbg, teacher_overlap = region_masks(tb[ti], tb, centers[start:end], cfg)
            sd, sv = pool_relative_logits(scores[bi, :, start:end], fg, bg, cfg.minimum_foreground, cfg.minimum_background)
            with torch.no_grad():
                td, tv = pool_relative_logits(ts[bi, :, start:end], tfg, tbg, cfg.minimum_foreground, cfg.minimum_background)
                rd, rv = pool_relative_logits(rs[bi, :, start:end], fg, bg, cfg.minimum_foreground, cfg.minimum_background)
                if not torch.equal(common[:, li], sv & tv & rv):
                    raise RuntimeError("C1 regions differ from the frozen C0 validity mask")
            sd_levels.append(sd)
            td_levels.append(td)
            rd_levels.append(rd)
            region_rows.append((bi, level, start, end, first, len(ri), fg, bg, tfg, tbg, rgb_overlap, teacher_overlap))
        sd_rows.append(torch.stack(sd_levels, 1))
        td_rows.append(torch.stack(td_levels, 1))
        rd_rows.append(torch.stack(rd_levels, 1))

    stats = dict(arm="paired", rgb_gt_count=len(rgb_boxes), teacher_gt_count=len(ir_boxes), common_count=0,
                 valid_region_count=0, reference_candidate_count=0, base_count=0, teacher_correct_base_count=0,
                 eligible_count=0, selected_count=0, normalizer=1, nominal_dose=0., loss_unweighted=0.,
                 selected_object_ids=[], config=asdict(cfg))
    if not es_rows:
        empty_delta = scores[:, :, :0].permute(0, 2, 1).reshape(0, 1, nc).expand(0, len(cfg.levels), nc)
        empty_long = torch.empty(0, dtype=torch.long, device=scores.device)
        return ClassificationSelection(empty_delta, empty_delta.detach(), empty_delta.detach(),
            torch.empty((0, len(cfg.levels)), dtype=torch.bool, device=scores.device), empty_long.bool(), empty_long,
            empty_long, empty_long, empty_long, (), (), empty_long.float(), empty_long.bool(), (), zero, stats,
            cfg, sources, tuple(_tensor_version(t) for t in sources), batch)

    es, et, er = torch.cat(es_rows), torch.cat(et_rows).detach(), torch.cat(er_rows).detach()
    common, ref_ok, teacher_ok = torch.cat(common_rows), torch.cat(ref_rows), torch.cat(correct_rows)
    base = common.any(1) & ref_ok
    q = (F.softplus(-er) - F.softplus(-et)).clamp_min(0).detach()
    eligible = base & teacher_ok & (q > 0)
    selected_matched = original._choose(q, eligible, base, "paired", cfg.rho, selection_seed)
    base_to_matched = torch.nonzero(base, as_tuple=False).flatten()
    matched_to_base = torch.full((len(object_ids),), -1, dtype=torch.long, device=scores.device)
    matched_to_base[base_to_matched] = torch.arange(len(base_to_matched), device=scores.device)
    selected = torch.zeros(len(base_to_matched), dtype=torch.bool, device=scores.device)
    selected[matched_to_base[selected_matched]] = True
    normalizer = max(1, int(base.sum().item()))
    target = et.clamp(-cfg.target_clip, cfg.target_clip)
    c0 = (F.smooth_l1_loss(es[selected_matched], target[selected_matched], reduction="sum", beta=cfg.smooth_l1_beta)
          / normalizer) if len(selected_matched) else zero
    if not bool(torch.isfinite(c0)):
        raise FloatingPointError("Nonfinite original C0 scalar")
    stats.update(common_count=len(es), valid_region_count=int(common.any(1).sum()),
                 reference_candidate_count=int(ref_ok.sum()), base_count=int(base.sum()),
                 teacher_correct_base_count=int((base & teacher_ok).sum()), eligible_count=int(eligible.sum()),
                 selected_count=len(selected_matched), normalizer=normalizer,
                 nominal_dose=len(selected_matched) / normalizer, loss_unweighted=float(c0.detach()),
                 selected_object_ids=[object_ids[i] for i in selected_matched.tolist()])
    for name, values in (("student_evidence", es.detach()), ("teacher_evidence", et), ("reference_evidence", er), ("quality", q)):
        subset = values[base]
        stats[name + "_base_mean"] = float(subset.mean()) if len(subset) else None
        stats[name + "_base_min"] = float(subset.min()) if len(subset) else None
        stats[name + "_base_max"] = float(subset.max()) if len(subset) else None
        stats[name + "_base_sd"] = float(subset.std(unbiased=False)) if len(subset) else None
        stats[name + "_selected_mean"] = float(values[selected_matched].mean()) if len(selected_matched) else None
        stats[name + "_selected_sd"] = float(values[selected_matched].std(unbiased=False)) if len(selected_matched) else None
    p = target[selected_matched].sigmoid()
    stats["target_binary_entropy_selected"] = float(-(p * p.clamp_min(1e-9).log() + (1-p) * (1-p).clamp_min(1e-9).log()).mean()) if len(p) else None
    stats["target_gap_to_positive_selected"] = float((1-p).mean()) if len(p) else None
    stats["target_clipped_count"] = int((et[selected_matched].abs() > cfg.target_clip).sum())
    regions = []
    for bi, level, start, end, first, count, fg, bg, tfg, tbg, overlap, toverlap in region_rows:
        keep = base[first:first+count]
        rows = matched_to_base[first:first+count][keep]
        if len(rows):
            regions.append(RegionLevel(bi, level, start, end, rows, fg[keep], bg[keep], tfg[keep], tbg[keep], overlap[keep], toverlap[keep]))
    return ClassificationSelection(torch.cat(sd_rows)[base], torch.cat(td_rows)[base].detach(),
        torch.cat(rd_rows)[base].detach(), common[base], selected, torch.cat(class_rows)[base],
        base_to_matched, matched_to_base, selected_matched, tuple(object_ids),
        tuple(object_ids[i] for i in base_to_matched.tolist()), q[base], eligible[base], tuple(regions),
        c0, stats, cfg, sources, tuple(_tensor_version(t) for t in sources), batch)
