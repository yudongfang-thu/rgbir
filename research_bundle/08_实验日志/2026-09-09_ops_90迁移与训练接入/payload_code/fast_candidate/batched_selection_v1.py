"""Independent C1 fast path: batch objects, not Python objects / CUDA scalars.

Bind pinned modules explicitly. No patch, CUDA execution, RNG or filesystem work
on import. Scientific gates, matched/base order, stable q ties and loss stay fixed.
Full diagnostics use original same-raw selector at the original cadence only.
"""
from dataclasses import asdict, dataclass
from math import ceil
import time
import types

import numpy as np
import torch
from torch.nn import functional as F

VERSION = 'C1_batched_selection_v1'
OBJECT_CHUNK = 64


def _clone(fn, **bindings):
    result = types.FunctionType(fn.__code__, dict(fn.__globals__, **bindings),
                                fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


def _timed(fn, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        value = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


@dataclass
class TrainingSelection:
    learning: object
    diagnostics: object = None
    thin_path_used: bool = True
    fallback_reason: object = None
    full_diagnostics_requested: bool = False
    diagnostics_seconds: float = 0.

    def __getattr__(self, key):
        return getattr(self.learning, key)


def _cpu_labels(original, batch, batch_size, classes, size):
    """One small transfer, exact original CPU validation and float32 conversion.

    A float64 transport table preserves supported label scalar values before the
    original float32 xywh -> xyxy arithmetic. No detector scores leave the GPU.
    """
    tables, lengths, pixels = [], [], []
    for labels in (batch, batch.get('teacher_batch', batch.get('ir_batch'))):
        if labels is None:
            raise ValueError('Independent teacher labels required')
        idx, cls, boxes = (labels[key].detach() for key in ('batch_idx', 'cls', 'bboxes'))
        if boxes.ndim != 2 or boxes.shape[1] != 4 or idx.numel() != len(boxes) or cls.numel() != len(boxes):
            raise ValueError('Invalid GT field shapes')
        tables.append(torch.cat((idx.reshape(-1, 1).double(), cls.reshape(-1, 1).double(), boxes.double()), 1))
        lengths.append(len(boxes))
        # Keep original device arithmetic for all geometry gates.
        b = boxes.float()
        scale = b.new_tensor((size[1], size[0]))
        pixels.append(torch.cat(((b[:, :2] - b[:, 2:] / 2) * scale,
                                 (b[:, :2] + b[:, 2:] / 2) * scale), -1))
    host = torch.cat(tables, 0).cpu()
    answer, offset = [], 0
    for length in lengths:
        part = host[offset:offset + length]
        labels = dict(batch_idx=part[:, 0], cls=part[:, 1], bboxes=part[:, 2:])
        answer.append(original._labels(labels, batch_size, classes, torch.device('cpu'), size))
        offset += length
    return answer, pixels


def _regions(original, matched_boxes, all_boxes, all_batch_ids, matched_batch_ids,
             centers, batch_size, cfg):
    # Counts are integer and do not depend on floating accumulation order.
    all_inside = original._inside(all_boxes, centers)
    counts = torch.zeros((batch_size, centers.shape[0]), dtype=torch.int32, device=centers.device)
    counts.index_add_(0, all_batch_ids, all_inside.to(torch.int32))
    owned = counts[matched_batch_ids]
    midpoint = (matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2
    half = (matched_boxes[:, 2:] - matched_boxes[:, :2]) * (cfg.background_scale / 2)
    outer = torch.cat((midpoint - half, midpoint + half), -1)
    fg = original._inside(matched_boxes, centers)
    bg = original._inside(outer, centers) & (owned == 0)
    overlap = (fg & (owned > 1)).sum(1).float() / fg.sum(1).clamp_min(1)
    nfg, nbg = fg.sum(1), bg.sum(1)
    valid = (nfg >= cfg.minimum_foreground) & (nbg >= cfg.minimum_background)
    return fg, bg, overlap, nfg, nbg, valid


def _pool_gt(z, region, temperature):
    fg, bg, _, nfg, nbg, valid = region
    a = z.masked_fill(~fg, -1e30).logsumexp(1) - nfg.clamp_min(1).float().log()
    b = z.masked_fill(~bg, -1e30).logsumexp(1) - nbg.clamp_min(1).float().log()
    return torch.where(valid, (a - b) / temperature, torch.zeros_like(a))


def _pool_classes(scores, image_ids, start, end, fg, bg, minimum_foreground,
                  minimum_background, chunk=OBJECT_CHUNK):
    """Vectorized FP32 masked LME; class channels share one raw FP32 cast."""
    nfg, nbg = fg.sum(1), bg.sum(1)
    valid = (nfg >= minimum_foreground) & (nbg >= minimum_background)
    result = []
    classes, anchors = scores.shape[1], end - start
    for lo in range(0, len(image_ids), chunk):
        hi = min(lo + chunk, len(image_ids))
        z = scores[image_ids[lo:hi], :, start:end]
        a = z.masked_fill(~fg[lo:hi, None], -1e30).reshape(-1, anchors).logsumexp(1).reshape(hi-lo, classes)
        b = z.masked_fill(~bg[lo:hi, None], -1e30).reshape(-1, anchors).logsumexp(1).reshape(hi-lo, classes)
        delta = a - nfg[lo:hi].clamp_min(1).float().log()[:, None] - (b - nbg[lo:hi].clamp_min(1).float().log()[:, None])
        result.append(torch.where(valid[lo:hi, None], delta, torch.zeros_like(delta)))
    return torch.cat(result, 0) if result else scores.new_empty((0, classes))


def _candidate(pred, probabilities, gt, images, conf, threshold, classes=None,
               chunk=OBJECT_CHUNK):
    """Same pre-NMS existence test, now across images and bounded object chunks."""
    max_prob, max_class = probabilities.max(1)
    result = []
    for lo in range(0, len(gt), chunk):
        hi = min(lo + chunk, len(gt))
        bi, g = images[lo:hi], gt[lo:hi]
        p = pred[bi]
        overlap = (torch.minimum(g[:, None, 2:], p[..., 2:]) - torch.maximum(g[:, None, :2], p[..., :2])).clamp_min(0).prod(-1)
        ag = (g[:, 2:] - g[:, :2]).clamp_min(0).prod(-1)
        ap = (p[..., 2:] - p[..., :2]).clamp_min(0).prod(-1)
        iou = overlap / (ag[:, None] + ap - overlap).clamp_min(1e-9)
        if classes is None:
            ok = max_prob[bi] >= conf
        else:
            cls = classes[lo:hi]
            ok = (probabilities[bi, cls] >= conf) & (max_class[bi] == cls[:, None])
        result.append((ok & (iou >= threshold)).any(1))
    return torch.cat(result) if result else torch.empty(0, dtype=torch.bool, device=gt.device)


class BatchedAPI:
    def __init__(self, adapter, classification):
        self.adapter, self.classification = adapter, classification
        self.original_selector = adapter.build_classification_selection
        self.original_loss = classification.classification_loss_from_selection

    def build(self, student, teacher, reference, batch, *, strides=(8, 16, 32),
              config=None, selection_seed=0, full_diagnostics=False):
        a, o = self.adapter, self.adapter.original
        cfg = a.evidence_config(config)
        centers, step, ranges, size = o._layout(student, cfg, strides)
        for name, raw in (('teacher', teacher), ('reference', reference)):
            _, _, rr, ss = o._layout(raw, cfg, strides)
            if raw['scores'].shape != student['scores'].shape or rr != ranges or ss != size or raw['scores'].device != student['scores'].device:
                raise ValueError(name + ' layout differs')
        sources = a._source_tensors(student, teacher, reference, batch)
        raw_scores = [raw['scores'] for raw in (student, teacher, reference)]
        fallback = None
        if any(z.dtype not in (torch.float16, torch.float32) for z in raw_scores):
            fallback = 'RAW_SCORE_DTYPE_OUTSIDE_FAST_DOMAIN'
        elif any(z.dtype == torch.float32 for z in raw_scores):
            if not bool(torch.stack([z.abs().max().float() <= 65504 for z in raw_scores]).all()):
                fallback = 'RAW_SCORE_OUTSIDE_FINITE_HALF_RANGE'
        if fallback:
            learning = self.original_selector(student, teacher, reference, batch, strides=strides,
                config=config, selection_seed=selection_seed)
            return TrainingSelection(learning, learning, False, fallback, full_diagnostics)
        scores = student['scores'].float()
        device, (batch_size, nc, _) = scores.device, scores.shape
        # Single student cast before all branches is essential for AMP gradients.
        zero = scores[:, :, :0].sum() * 0.
        cpu, pixels = _cpu_labels(o, batch, batch_size, nc, size)
        (ri_cpu, rc_cpu, rb_cpu), (ti_cpu, tc_cpu, tb_cpu) = cpu
        rb, tb = pixels
        rgb_idx, ir_idx = ri_cpu.to(device), ti_cpu.to(device)
        rgb_classes, ir_classes = rc_cpu.to(device), tc_cpu.to(device)
        rgb_groups = [torch.nonzero(ri_cpu == bi, as_tuple=False).flatten().tolist() for bi in range(batch_size)]
        ir_groups = [torch.nonzero(ti_cpu == bi, as_tuple=False).flatten().tolist() for bi in range(batch_size)]
        # Original float32 GPU IoU, threshold and cardinality bonus; a single
        # compact cost transfer replaces one CUDA synchronization per image.
        with torch.no_grad():
            overlaps = o._iou(rb, tb)
            eligible = ((rgb_idx[:, None] == ir_idx[None]) &
                        (rgb_classes[:, None] == ir_classes[None]) & (overlaps >= cfg.match_iou))
            bonus = overlaps.new_tensor([min(len(rgb_groups[int(bi)]), len(ir_groups[int(bi)])) + 1. for bi in ri_cpu.tolist()])
            costs = (eligible * (bonus[:, None] + overlaps)).cpu().numpy()
        object_ids, image_groups = [], []
        for bi in range(batch_size):
            rg, tg = rgb_groups[bi], ir_groups[bi]
            if not rg or not tg:
                continue
            local = costs[np.ix_(rg, tg)]
            ri, ti = o.linear_sum_assignment(-local)
            pairs = [(int(r), int(t)) for r, t in zip(ri, ti) if local[r, t] > 0]
            first = len(object_ids)
            object_ids.extend((bi, rg[r], tg[t]) for r, t in pairs)
            if pairs:
                image_groups.append((bi, first, len(pairs)))
        m = len(object_ids)
        base_stats = dict(arm='paired', rgb_gt_count=len(rb_cpu), teacher_gt_count=len(tb_cpu),
            common_count=m, valid_region_count=0, reference_candidate_count=0, base_count=0,
            teacher_correct_base_count=0, eligible_count=0, selected_count=0, normalizer=1,
            nominal_dose=0., loss_unweighted=0., selected_object_ids=[], config=asdict(cfg))
        if not m:
            delta = scores[:, :, :0].permute(0, 2, 1).reshape(0, 1, nc).expand(0, len(cfg.levels), nc)
            empty = torch.empty(0, dtype=torch.long, device=device)
            learning = a.ClassificationSelection(delta, delta.detach(), delta.detach(),
                torch.empty((0, len(cfg.levels)), dtype=torch.bool, device=device), empty.bool(), empty,
                empty, empty, empty, (), (), empty.float(), empty.bool(), (), zero, base_stats, cfg,
                sources, tuple(a._tensor_version(t) for t in sources), batch)
        else:
            ids = torch.tensor(object_ids, dtype=torch.long, device=device)
            images, rg, tg = ids.unbind(1)
            labels = rgb_classes[rg]
            mr, mt = rb[rg], tb[tg]
            with torch.no_grad():
                ts, rs = teacher['scores'].detach().float(), reference['scores'].detach().float()
                tbox, rbox = o._decode_boxes(teacher, centers, step), o._decode_boxes(reference, centers, step)
                tprob, rprob = ts.sigmoid(), rs.sigmoid()
                region_data, et_levels, er_levels, valid_levels = [], [], [], []
                for level in cfg.levels:
                    start, end = ranges[level]
                    rreg = _regions(o, mr, rb, rgb_idx, images, centers[start:end], batch_size, cfg)
                    treg = _regions(o, mt, tb, ir_idx, images, centers[start:end], batch_size, cfg)
                    valid_levels.append(rreg[-1] & treg[-1])
                    et_levels.append(_pool_gt(ts[images, labels, start:end], treg, cfg.temperature))
                    er_levels.append(_pool_gt(rs[images, labels, start:end], rreg, cfg.temperature))
                    region_data.append((level, start, end, rreg, treg))
                common = torch.stack(valid_levels, 1)
                denom = common.sum(1).clamp_min(1)
                et = (torch.stack(et_levels, 1) * common).sum(1) / denom
                er = (torch.stack(er_levels, 1) * common).sum(1) / denom
                ref_ok = _candidate(rbox, rprob, mr, images, cfg.reference_conf, cfg.reference_iou)
                teacher_ok = _candidate(tbox, tprob, mt, images, cfg.teacher_conf, cfg.teacher_iou, labels)
                q = (F.softplus(-er) - F.softplus(-et)).clamp_min(0)
                # Only this compact table crosses to CPU for stable selection.
                table = torch.cat((q[:, None], ref_ok[:, None].float(), teacher_ok[:, None].float(), common.float()), 1).cpu()
            valid_cpu = table[:, 3:].bool()
            base_cpu = valid_cpu.any(1) & table[:, 1].bool()
            eligible_cpu = base_cpu & table[:, 2].bool() & (table[:, 0] > 0)
            base_indices = torch.nonzero(base_cpu, as_tuple=False).flatten().tolist()
            population = torch.nonzero(eligible_cpu, as_tuple=False).flatten().tolist()
            selected_indices = sorted(population, key=lambda i: -float(table[i, 0]))[:ceil(cfg.rho * len(population))]
            selected_set = set(selected_indices)
            mapping_cpu = [-1] * m
            for i, j in enumerate(base_indices):
                mapping_cpu[j] = i
            selected_base = [i for i, j in enumerate(base_indices) if j in selected_set]
            pooled_matches = [base_indices[i] for i in selected_base]
            bt = torch.tensor(base_indices, dtype=torch.long, device=device)
            mb = torch.tensor(mapping_cpu, dtype=torch.long, device=device)
            sm = torch.tensor(selected_indices, dtype=torch.long, device=device)
            sb = torch.tensor(selected_base, dtype=torch.long, device=device)
            pm = torch.tensor(pooled_matches, dtype=torch.long, device=device)
            selected = torch.tensor([j in selected_set for j in base_indices], dtype=torch.bool, device=device)
            sd = scores.new_zeros((len(base_indices), len(cfg.levels), nc)) + zero
            td, rd = torch.zeros_like(sd).detach(), torch.zeros_like(sd).detach()
            if selected_base:
                sd_values, td_values = [], []
                for level, start, end, rreg, treg in region_data:
                    sd_values.append(_pool_classes(scores, images[pm], start, end, rreg[0][pm], rreg[1][pm], cfg.minimum_foreground, cfg.minimum_background))
                    with torch.no_grad():
                        td_values.append(_pool_classes(ts, images[pm], start, end, treg[0][pm], treg[1][pm], cfg.minimum_foreground, cfg.minimum_background))
                sd = sd.index_copy(0, sb, torch.stack(sd_values, 1))
                td = td.index_copy(0, sb, torch.stack(td_values, 1)).detach()
                # C0 uses only selected S evidence; reuse the already pooled data.
                ranked_base = mb[sm]
                es = sd[ranked_base].gather(-1, labels[sm, None, None].expand(-1, len(cfg.levels), 1)).squeeze(-1) / cfg.temperature
                es = (es * common[sm]).sum(1) / common[sm].sum(1).clamp_min(1)
                c0 = F.smooth_l1_loss(es, et[sm].clamp(-cfg.target_clip, cfg.target_clip), reduction='sum', beta=cfg.smooth_l1_beta) / max(1, len(base_indices))
            else:
                c0 = zero
            normalizer = max(1, len(base_indices))
            base_stats.update(valid_region_count=int(valid_cpu.any(1).sum()), reference_candidate_count=int(table[:, 1].sum()),
                base_count=len(base_indices), teacher_correct_base_count=int((base_cpu & table[:, 2].bool()).sum()),
                eligible_count=len(population), selected_count=len(selected_indices), normalizer=normalizer,
                nominal_dose=len(selected_indices)/normalizer, loss_unweighted=None,
                selected_object_ids=[object_ids[i] for i in selected_indices])
            # Metadata only: constructing per-image views does not synchronize CUDA.
            regions = []
            for bi, first, count in image_groups:
                matched = [j for j in range(first, first+count) if mapping_cpu[j] >= 0]
                if not matched:
                    continue
                mi = torch.tensor(matched, dtype=torch.long, device=device)
                rows = torch.tensor([mapping_cpu[j] for j in matched], dtype=torch.long, device=device)
                for level, start, end, rreg, treg in region_data:
                    regions.append(a.RegionLevel(bi, level, start, end, rows, rreg[0][mi], rreg[1][mi],
                        treg[0][mi], treg[1][mi], rreg[2][mi], treg[2][mi]))
            learning = a.ClassificationSelection(sd, td, rd, common[bt], selected, labels[bt], bt, mb, sm,
                tuple(object_ids), tuple(object_ids[i] for i in base_indices), q[bt],
                eligible_cpu.to(device)[bt], tuple(regions), c0, base_stats, cfg, sources,
                tuple(a._tensor_version(t) for t in sources), batch)
        payload = TrainingSelection(learning, full_diagnostics_requested=full_diagnostics)
        if full_diagnostics:
            payload.diagnostics, payload.diagnostics_seconds = _timed(lambda: self.original_selector(
                student, teacher, reference, batch, strides=strides, config=config, selection_seed=selection_seed), device)
        return payload

    def loss(self, payload, *, temperature=2., raw_teacher_clip=16., off_target_weight=.25, return_records=False):
        if not isinstance(payload, TrainingSelection):
            raise TypeError('Use BatchedAPI.build payload')
        if not payload.thin_path_used:
            value, stats = self.original_loss(payload.learning, temperature=temperature,
                raw_teacher_clip=raw_teacher_clip, off_target_weight=off_target_weight, return_records=return_records)
            stats.update(candidate_version=VERSION, thin_path_used=False,
                thin_path_fallback_reason=payload.fallback_reason, full_diagnostics_collected=True,
                full_diagnostics_seconds=0.)
            return value, stats
        if return_records and payload.diagnostics is None:
            raise ValueError('Full records require full_diagnostics=True')
        s = payload.learning
        kwargs = dict(temperature=temperature, raw_teacher_clip=raw_teacher_clip, off_target_weight=off_target_weight)
        terms = self.classification._classification_terms(s.student_delta, s.teacher_delta, s.valid_levels, s.selected, s.labels, **kwargs)
        values = torch.stack((terms['loss'].detach(), terms['target_loss'].detach(), terms['off_target_loss_unit'].detach(), s.c0_loss.detach())).cpu().tolist()
        if not all(__import__('math').isfinite(v) for v in values):
            raise FloatingPointError('Nonfinite C1 or C0 scalar')
        if payload.diagnostics is not None:
            (_, stats), seconds = _timed(lambda: self.original_loss(payload.diagnostics, return_records=return_records, **kwargs), s.student_delta.device)
            payload.diagnostics_seconds += seconds
        else:
            stats = dict(s.c0_stats)
            missing = ['target_binary_entropy_selected', 'target_gap_to_positive_selected',
                'target_clipped_count', 'target_clipping_population', 'target_clipped_fraction',
                'gt_channel_relative_entropy_selected', 'per_level', 'per_class',
                'rgb_other_gt_foreground_fraction_selected_level_mean',
                'teacher_other_gt_foreground_fraction_selected_level_mean', 'base_records']
            missing += [name + '_delta_class_mean_' + scope for name in ('student', 'teacher', 'reference') for scope in ('base_valid', 'selected_valid')]
            missing += [name + '_' + field for name in ('student_evidence', 'teacher_evidence', 'reference_evidence', 'quality') for field in ('base_mean','base_min','base_max','base_sd','selected_mean','selected_sd')]
            stats.update({key: None for key in missing})
            stats['missing_diagnostics'] = missing
        m, levels, classes = s.student_delta.shape
        stats.update(arm='C1_y' if off_target_weight == 0 else 'C1', candidate_version=VERSION,
            selection_version=self.adapter.SELECTION_VERSION, selection_source=str(self.adapter.LEGACY_SOURCE),
            temperature=temperature, raw_teacher_clip=raw_teacher_clip, off_target_weight=off_target_weight,
            class_count=classes, supervised_class_channels=classes if off_target_weight > 0 else 1,
            payload_shape=[m, levels, classes], normalizer=max(1,m), loss_unweighted=values[0],
            c0_loss_unweighted=values[3], target_loss_unweighted=values[1], off_target_loss_unit=values[2],
            off_target_loss_unweighted=off_target_weight*values[2], thin_path_used=True,
            thin_path_fallback_reason=None, full_diagnostics_collected=payload.diagnostics is not None,
            full_diagnostics_seconds=payload.diagnostics_seconds,
            learning_delta_coverage='selected_S_T_only; other_slots_private_placeholders',
            relative_evidence_not_detector_confidence=True, student_background_has_gradient=True,
            teacher_gradient_enabled=False, target_binary_entropy_semantics='relative_evidence')
        return terms['loss'], stats


def make_api(adapter, classification):
    return BatchedAPI(adapter, classification)


def full_diagnostics_due(calls, log_every_batches, sanity, observer_due):
    if calls < 1 or log_every_batches < 1:
        raise ValueError('Positive calls and logging interval required')
    return bool(sanity or calls <= 3 or calls % log_every_batches == 0 or observer_due)


def make_criterion_type(criterion_module, api, selection_observer=None):
    original_type = criterion_module.IndependentCriterion
    original_call = original_type.__call__

    def call(self, prediction, batch):
        if self.method_arm not in ('C1', 'C1_y'):
            raise ValueError('Fast path supports independent C1/C1_y only')

        def select(*args, **kwargs):
            due = full_diagnostics_due(self.calls, self.cfg['log_every_batches'], self.sanity,
                self.shared_gradient_observer.should_observe(self.trainer.epoch))
            payload = api.build(*args, full_diagnostics=due, **kwargs)
            if selection_observer is not None:
                selection_observer(self, payload)
            return payload

        def loss(payload, **kwargs):
            value, stats = api.loss(payload, **kwargs)
            self.selected_only_diagnostics_seconds_total = getattr(self, 'selected_only_diagnostics_seconds_total', 0.) + payload.diagnostics_seconds
            for name, increment in (('thin_learning_batches', int(payload.thin_path_used)), ('full_diagnostics_batches', int(stats['full_diagnostics_collected'])), ('fallback_batches', int(not payload.thin_path_used))):
                setattr(self, name, getattr(self, name, 0) + increment)
                stats[name] = getattr(self, name)
            return value, stats

        return _clone(original_call, build_classification_selection=select,
            classification_loss_from_selection=loss)(self, prediction, batch)

    return type('BatchedSelectionC1Criterion', (original_type,), {'__call__': call})
