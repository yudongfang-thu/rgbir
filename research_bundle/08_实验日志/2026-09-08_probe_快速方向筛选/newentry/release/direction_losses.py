"""C2 class-balanced relative KL and selected GT-ROI feature relations.

Pure losses only: no new selector, trainable projection, optimizer or schedule.
The caller binds the pinned classification_logit module before invoking C2.
"""
from contextlib import nullcontext
import torch
from torch.nn import functional as F

TEMPERATURE = 2.
TEACHER_CLIP = 16.
OFF_TARGET_WEIGHT = .25
LEVELS = (0, 1)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _learning(selection):
    return getattr(selection, 'learning', selection)


def class_balanced_loss(selection):
    """Mean across classes present in pre-teacher base, not selected classes."""
    from classification_logit import _classification_terms
    s = _learning(selection)
    terms = _classification_terms(s.student_delta, s.teacher_delta, s.valid_levels,
        s.selected, s.labels, temperature=TEMPERATURE, raw_teacher_clip=TEACHER_CLIP,
        off_target_weight=OFF_TARGET_WEIGHT)
    m, _, nc = s.student_delta.shape
    counts = torch.bincount(s.labels, minlength=nc)
    present = counts > 0
    nclasses = int(present.sum())
    if nclasses <= 1:
        # Algebraically C1; reuse its reduction verbatim to avoid changing
        # floating-point summation when the base contains only one class.
        loss = terms['loss']
    else:
        scale = s.valid_levels.float() / s.valid_levels.sum(-1).clamp_min(1)[:, None]
        obj = ((terms['target'] + OFF_TARGET_WEIGHT * terms['non_target']) * scale).sum(-1)
        obj = obj * (TEMPERATURE ** 2) * s.selected
        loss = (obj / counts[s.labels].to(obj.dtype)).sum() / nclasses
    _require(bool(torch.isfinite(loss)), 'Nonfinite C2 loss')
    return loss, dict(arm='C2', loss_unweighted=float(loss.detach()), base_count=m,
        selected_count=int(s.selected.sum()), normalizer='class_base_count_then_present_base_class_mean',
        base_class_counts=counts.detach().cpu().tolist(), present_base_class_count=nclasses,
        selected_valid_level_count=int((s.valid_levels & s.selected[:, None]).sum()),
        temperature=TEMPERATURE, raw_teacher_clip=TEACHER_CLIP, off_target_weight=OFF_TARGET_WEIGHT,
        selection_changed=False, teacher_gradient_enabled=False, trainable_projection=False)


def _sample_roi_tokens(feature, boxes, canvas_hw):
    """One image, many ROIs: one input map and a stacked M*3 by3 grid."""
    _require(feature.ndim == 4 and feature.shape[0] == 1, 'Sample one image without ROI feature expansion')
    _require(boxes.ndim == 2 and boxes.shape[1] == 4, 'ROI boxes must be pixel xyxy[M,4]')
    h, w = canvas_hw
    _require(h > 0 and w > 0 and bool(torch.isfinite(boxes).all()), 'Invalid ROI/canvas')
    _require(bool((boxes[:, 2:] > boxes[:, :2]).all()), 'Nonpositive GT ROI')
    n = len(boxes)
    fractions = (torch.arange(3, device=feature.device, dtype=torch.float32) + .5) / 3.
    xs = boxes[:, 0, None] + (boxes[:, 2]-boxes[:, 0])[:, None] * fractions[None]
    ys = boxes[:, 1, None] + (boxes[:, 3]-boxes[:, 1])[:, None] * fractions[None]
    grid = torch.stack((xs[:, None, :].expand(n, 3, 3) * (2./w)-1.,
                        ys[:, :, None].expand(n, 3, 3) * (2./h)-1.), -1).reshape(1, n*3, 3, 2)
    sampled = F.grid_sample(feature.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    tokens = sampled[0].reshape(feature.shape[1], n, 3, 3).permute(1, 2, 3, 0).reshape(n, 9, feature.shape[1])
    _require(bool(torch.isfinite(tokens).all()), 'Nonfinite sampled features')
    return tokens


def _relation(tokens):
    normalized = F.normalize(tokens, p=2, dim=-1, eps=1e-6)
    return torch.bmm(normalized, normalized.transpose(1, 2))


def _pixel_boxes(labels, canvas_hw, device):
    cls = labels['cls'].detach().reshape(-1).to(device)
    idx = labels['batch_idx'].detach().reshape(-1).to(device)
    boxes = labels['bboxes'].detach().to(device=device, dtype=torch.float32)
    _require(boxes.shape == (len(cls), 4) and len(idx) == len(cls), 'GT label shapes differ')
    _require(bool(torch.isfinite(boxes).all()) and bool((boxes[:, 2:] > 0).all()), 'Invalid GT xywh')
    _require(bool((cls == cls.long()).all()) and bool((idx == idx.long()).all()), 'Noninteger GT identity')
    h, w = canvas_hw
    xywh = boxes * boxes.new_tensor([w, h, w, h])
    return idx.long(), cls.long(), torch.cat((xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2), -1)


def feature_relation_loss(selection, student, teacher, batch):
    """Selected canonical ROI cosine-Gram MSE, scale mean then /M_base."""
    s = _learning(selection)
    levels = tuple(s.config.levels)
    _require(levels == LEVELS, 'F-rel is fixed to head-input P3/P4')
    m = len(s.labels)
    _require(s.selected.dtype == torch.bool and s.selected.shape == (m,), 'Invalid selected mask')
    _require(s.valid_levels.dtype == torch.bool and s.valid_levels.shape == (m, 2), 'Invalid common valid-level mask')
    _require(not bool((s.selected & ~s.valid_levels.any(-1)).any()), 'Selected object lacks a valid level')
    sf, tf = student['feats'], teacher['feats']
    _require(len(sf) >= 3 and len(tf) >= 3, 'Expected raw head-input P3/P4/P5 feature list')
    device = sf[0].device
    _require(s.selected.device == device and s.valid_levels.device == device, 'Selection/feature device mismatch')
    zero = sum((sf[level].reshape(-1)[:0].float().sum()*0. for level in levels))
    stats = dict(arm='F-rel', base_count=m, selected_count=int(s.selected.sum()),
        normalizer=max(1, m), levels=list(levels), roi_grid=[3, 3], relations_per_roi_level=72,
        channel_matching=False, trainable_projection=False, teacher_gradient_enabled=False,
        selection_changed=False, align_corners=False, padding_mode='zeros')
    if not m or not bool(s.selected.any()):
        stats.update(loss_unweighted=0., sampled_roi_levels=0, sample_calls=0)
        return zero, stats
    if hasattr(s, 'source_batch'):
        _require(s.source_batch is batch, 'Selection belongs to another batch')
    other = batch.get('teacher_batch', batch.get('ir_batch'))
    _require(other is not None and 'strong_img' in batch, 'Separate IR labels/image required')
    rgb_image, ir_image = batch['img'], batch['strong_img']
    _require(rgb_image.ndim == 4 and ir_image.ndim == 4 and rgb_image.shape[0] == ir_image.shape[0], 'Image batch shapes differ')
    rgb_hw, ir_hw = tuple(rgb_image.shape[-2:]), tuple(ir_image.shape[-2:])
    ri, rc, rb = _pixel_boxes(batch, rgb_hw, device)
    ti, tc, tb = _pixel_boxes(other, ir_hw, device)
    _require(len(s.base_object_ids) == m, 'Base object identity count differs')
    ids = torch.tensor(s.base_object_ids, device=device, dtype=torch.long)
    _require(ids.shape == (m, 3), 'Expected (image, global RGB row, global IR row)')
    _require(bool(((ids[:, 0] >= 0) & (ids[:, 0] < rgb_image.shape[0])).all()), 'Image identity out of range')
    _require(bool(((ids[:, 1] >= 0) & (ids[:, 1] < len(rb)) & (ids[:, 2] >= 0) & (ids[:, 2] < len(tb))).all()), 'Global GT row out of range')
    _require(torch.equal(ri[ids[:, 1]], ids[:, 0]) and torch.equal(ti[ids[:, 2]], ids[:, 0]), 'Global GT row belongs to a different image')
    _require(torch.equal(rc[ids[:, 1]], s.labels) and torch.equal(tc[ids[:, 2]], s.labels), 'GT/selection class mapping differs')
    per_object = zero + torch.zeros(m, device=device, dtype=torch.float32)
    diagonal_mask = ~torch.eye(9, device=device, dtype=torch.bool)
    sampled_count = calls = 0
    # Explicitly keep grid sampling, normalization, Gram and MSE in FP32 under AMP.
    context = torch.cuda.amp.autocast(enabled=False) if device.type == 'cuda' else nullcontext()
    with context:
        for li, level in enumerate(levels):
            _require(sf[level].ndim == 4 and tf[level].ndim == 4, 'Features must be BCHW')
            _require(sf[level].shape[0] == rgb_image.shape[0] and tf[level].shape[0] == ir_image.shape[0], 'Feature image count differs')
            _require(sf[level].device == device and tf[level].device == device, 'Teacher/student feature devices differ')
            active = s.selected & s.valid_levels[:, li]
            for bi in range(rgb_image.shape[0]):
                rows = torch.nonzero(active & (ids[:, 0] == bi), as_tuple=False).flatten()
                if not len(rows):
                    continue
                rgb_tokens = _sample_roi_tokens(sf[level][bi:bi+1], rb[ids[rows, 1]], rgb_hw)
                with torch.no_grad():
                    ir_tokens = _sample_roi_tokens(tf[level][bi:bi+1].detach(), tb[ids[rows, 2]], ir_hw)
                    teacher_gram = _relation(ir_tokens).detach()
                student_gram = _relation(rgb_tokens)
                roi_loss = ((student_gram-teacher_gram)[:, diagonal_mask] ** 2).mean(-1)
                per_object = per_object.index_add(0, rows, roi_loss)
                sampled_count += len(rows); calls += 2
        loss = (per_object / s.valid_levels.sum(-1).clamp_min(1)).sum() / max(1, m)
    _require(sampled_count == int((s.selected[:, None] & s.valid_levels).sum()), 'Sampled ROI/valid-level count differs')
    _require(bool(torch.isfinite(loss)), 'Nonfinite feature relation loss')
    stats.update(loss_unweighted=float(loss.detach()), sampled_roi_levels=sampled_count, sample_calls=calls,
                 feature_shapes_student=[list(sf[x].shape) for x in levels],
                 feature_shapes_teacher=[list(tf[x].shape) for x in levels])
    return loss, stats
