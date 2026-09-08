"""L3 full object-coordinate DFL distribution; never an L1 geometry admission.

Caller puts the pinned rgbir_independent_kd_v2 and task_conditional_reference
on sys.path (as its existing runtime does). Raw boxes are DFL logits, not xyxy.
Loss is unweighted; caller retains complete native loss and applies lambda once.
"""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from probability_transport import transport_logits

VERSION = 'L3_object_coordinate_full_distribution_v1'
VARIANTS = ('L3-DFL', 'L3-GT')
TEMPERATURE = 2.
FIXED = dict(match_iou=.5, pair_iou=.8, reference_conf=.05, reference_iou=.1,
             reliable_conf=.25, reference_iou_max=.70, teacher_ir_iou_min=.5,
             mapped_rgb_iou_min=.60, localization_margin=.05,
             support_epsilon=.01, native_candidate_epsilon=1e-9,
             levels=(0, 1))


def float32_target_or_none(probabilities,device):
    """Reject positive target mass lost by the learning dtype; legitimate zero stays zero."""
    q64=torch.tensor(probabilities,dtype=torch.float64)
    if q64.shape!=(4,16) or not bool(torch.isfinite(q64).all()) or bool((q64<0).any()):
        raise ValueError('Invalid transported probability target')
    q32=q64.float()
    if bool(((q64>0)&(q32==0)).any()):return None
    return q32.to(device=device)


def _helpers():
    import selection_adapter
    import localization_loss
    expected = Path(selection_adapter.REFERENCE) / 'localization_loss.py'
    if Path(localization_loss.__file__).resolve() != expected.resolve():
        raise RuntimeError('L2 native helper must come from pinned task_conditional_reference')
    return selection_adapter.original, localization_loss


def _valid_boxes(boxes):
    return torch.isfinite(boxes).all(-1) & ((boxes[..., 2:] - boxes[..., :2]) > 0).all(-1)


def object_relative(boxes, gt):
    """Input-pixel xyxy to GT-relative xyxy, with no clipping."""
    if boxes.shape != gt.shape or boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError('boxes and GT must be matching [N,4] xyxy')
    if not bool(_valid_boxes(boxes).all()) or not bool(_valid_boxes(gt).all()):
        raise ValueError('Nonfinite or nonpositive-width/height boxes')
    wh = gt[:, 2:] - gt[:, :2]
    return (boxes - gt[:, :2].repeat(1, 2)) / wh.repeat(1, 2)


def map_teacher_box(boxes, ir_gt, rgb_gt):
    """Canonical instance affine map, NOT estimated physical registration."""
    relative = object_relative(boxes, ir_gt)
    if rgb_gt.shape != ir_gt.shape or not bool(_valid_boxes(rgb_gt).all()):
        raise ValueError('Invalid RGB affine target GT')
    return rgb_gt[:, :2].repeat(1, 2) + relative * (rgb_gt[:, 2:] - rgb_gt[:, :2]).repeat(1, 2)


def decode_selected(raw, batch_ids, anchor_ids, centers, stride_values):
    """Pinned native DFL expectation arithmetic, retaining student gradient.

    The pinned _decode_boxes intentionally detaches; this selected slice uses
    its same float32 softmax/bin expectation and xyxy construction without detach.
    """
    logits = raw['boxes'].float()
    reg_max = logits.shape[1] // 4
    selected = logits.permute(0, 2, 1)[batch_ids, anchor_ids].reshape(-1, 4, reg_max)
    bins = torch.arange(reg_max, dtype=torch.float32, device=logits.device)
    distances = (selected.softmax(-1) * bins).sum(-1) * stride_values[anchor_ids]
    xy = centers[anchor_ids]
    return torch.cat((xy - distances[:, :2], xy + distances[:, 2:]), -1)


def _best(mask, primary, secondary):
    ids = torch.nonzero(mask, as_tuple=False).flatten()
    if len(ids) == 0:
        return None
    ids = ids[primary[ids] == primary[ids].max()]
    ids = ids[secondary[ids] == secondary[ids].max()]
    return int(ids[0])  # ascending anchor ID is the final deterministic tie


def _layout(raw, original, strides):
    if tuple(strides) != (8, 16, 32):
        raise ValueError('L2 diagnostic is frozen to P3/P4/P5 strides 8/16/32')
    cfg = SimpleNamespace(input_size=tuple(int(v) * strides[0] for v in raw['feats'][0].shape[-2:]))
    layout = original._layout(raw, cfg, strides)
    if (raw['boxes'].shape[1] < 8 or not raw['boxes'].is_floating_point()
            or not raw['scores'].is_floating_point()
            or raw['boxes'].device != raw['scores'].device):
        raise ValueError('Floating raw logits with reg_max >= 2 on a common device required')
    return layout


@torch.no_grad()
def _select(teacher, reference, batch, strides, original, native):
    centers, stride_values, ranges, size = _layout(reference, original, strides)
    tcens, tstrides, tranges, tsize = _layout(teacher, original, strides)
    if (tsize != size or teacher['scores'].shape != reference['scores'].shape
            or teacher['boxes'].shape != reference['boxes'].shape or tranges != ranges
            or not torch.equal(centers,tcens) or not torch.equal(stride_values,tstrides)
            or teacher['scores'].device != reference['scores'].device):
        raise ValueError('Pairing IoU requires same input canvas, batch/classes and device')
    batch_size, nc, count = reference['scores'].shape
    device = centers.device
    rgb_idx, rgb_cls, rgb_boxes = original._labels(batch, batch_size, nc, device, size)
    teacher_batch = batch.get('teacher_batch', batch.get('ir_batch'))
    if teacher_batch is None:
        raise ValueError('Separate teacher_batch/ir_batch labels are required')
    ir_idx, ir_cls, ir_boxes = original._labels(teacher_batch, batch_size, nc, device, tsize)
    rpred = original._decode_boxes(reference, centers, stride_values)
    tpred = original._decode_boxes(teacher, tcens, tstrides)
    rconf, rclass = reference['scores'].detach().float().sigmoid().max(1)
    tconf, tclass = teacher['scores'].detach().float().sigmoid().max(1)
    active = torch.zeros(count, dtype=torch.bool, device=device)
    tactive = torch.zeros(teacher['scores'].shape[2], dtype=torch.bool, device=device)
    for li in FIXED['levels']:
        active[ranges[li][0]:ranges[li][1]] = True
        tactive[tranges[li][0]:tranges[li][1]] = True
    stats = dict(version=VERSION, method_family='object_coordinate_full_distribution',
        scope='L3_DISTRIBUTION_DIAGNOSTIC_ONLY', L1_geometry_admitted=False,
        physical_registration_verified=False, same_anchor_required=True,
        same_DFL_bin_claim=False, target_coordinate_basis='annotation_coordinate_transport',
        shared_annotation_caveat='Shared RGB/IR labels (including LLVIP) do not establish independently annotated physical correspondence.',
        object_index_space='augmented_batch_global_gt_rows',
        normalization='coarse_R_base_before_R_reliability_and_teacher_quality',
        config=dict(FIXED), rgb_gt_count=len(rgb_boxes), teacher_gt_count=len(ir_boxes),
        common_count=0, pair_iou_count=0, reference_support_count=0,
        reference_unique_owner_count=0, base_count=0, reference_reliable_count=0,
        reference_gap_count=0, teacher_own_quality_count=0,
        mapped_rgb_quality_count=0, localization_margin_count=0, fullsupport_checked=0, fullsupport_rejected=0,
        teacher_same_anchor_unique_count=0, target_float32_rejected=0, selected_count=0,
        normalizer=1, selected_anchors=[], selected_object_ids=[], base_records=[])
    chosen, gt_rows, targets = [], [], []
    for bi in range(batch_size):
        rg = torch.nonzero(rgb_idx == bi, as_tuple=False).flatten()
        tg = torch.nonzero(ir_idx == bi, as_tuple=False).flatten()
        rb, tb = rgb_boxes[rg], ir_boxes[tg]
        ri, ti = original._match_objects(rb, rgb_cls[rg], tb, ir_cls[tg], FIXED['match_iou'])
        stats['common_count'] += len(ri)
        # ALL GT, including other classes and unmatched objects, own anchors.
        rins = native.native_candidate_mask(rb, centers, strides, FIXED['native_candidate_epsilon'])
        tins = native.native_candidate_mask(tb, tcens, strides, FIXED['native_candidate_epsilon'])
        runique, tunique = rins.sum(0) == 1, tins.sum(0) == 1
        for ril, til in zip(ri.tolist(), ti.tolist()):
            rgi, tgi = int(rg[ril]), int(tg[til])
            cls = int(rgb_cls[rgi])
            pair_iou = float(original._iou(rb[ril:ril+1], tb[til:til+1])[0, 0])
            if pair_iou < FIXED['pair_iou']:
                continue
            stats['pair_iou_count'] += 1
            distances = native.unclamped_distances(rb[ril].expand(count, -1), centers, stride_values)
            support = ((distances >= 0) & (distances <= reference['boxes'].shape[1] // 4 - 1 - FIXED['support_epsilon'])).all(1)
            candidate = active & support
            if not bool(candidate.any()):
                continue
            stats['reference_support_count'] += 1
            candidate = candidate & rins[ril] & runique
            if not bool(candidate.any()):
                continue
            stats['reference_unique_owner_count'] += 1
            overlaps = original._iou(rb[ril:ril+1], rpred[bi])[0]
            candidate &= _valid_boxes(rpred[bi]) & (rconf[bi] >= FIXED['reference_conf']) & (overlaps >= FIXED['reference_iou'])
            ai = _best(candidate, rconf[bi], overlaps)
            if ai is None:
                continue
            stats['base_count'] += 1  # fixed denominator, BEFORE all later gates
            riou = float(overlaps[ai])
            reliable = int(rclass[bi, ai]) == cls and float(rconf[bi, ai]) >= FIXED['reliable_conf']
            gap = reliable and riou < FIXED['reference_iou_max']
            stats['reference_reliable_count'] += int(reliable)
            stats['reference_gap_count'] += int(gap)
            record = dict(batch_index=bi, rgb_gt_index=rgi, ir_gt_index=tgi, class_id=cls,
                reference_anchor=ai, teacher_anchor=None, pair_iou=pair_iou,
                rgb_gt=rb[ril].tolist(), ir_gt=tb[til].tolist(),
                reference_box=rpred[bi, ai].tolist(), reference_iou=riou,
                reference_conf=float(rconf[bi, ai]), reference_reliable=reliable,
                reference_gap=gap, teacher_conf=None, teacher_own_iou=None,
                teacher_box=None, mapped_teacher_box=None, mapped_teacher_rgb_iou=None,
                teacher_same_anchor_unique=None, full_probability_support=None, fullsupport_reason=None,
                target_float32_supported=None, selected=False)
            if gap:
                tiou = original._iou(tb[til:til+1], tpred[bi])[0]
                tai=ai  # Fixed learning position; no independent teacher anchor search.
                unique=bool(tactive[tai] & tins[til,tai] & tunique[tai])
                stats['teacher_same_anchor_unique_count']+=int(unique)
                record.update(teacher_anchor=tai,teacher_same_anchor_unique=unique,
                    teacher_conf=float(tconf[bi,tai]),teacher_own_iou=float(tiou[tai]),teacher_box=tpred[bi,tai].tolist())
                teacher_quality=(unique and bool(_valid_boxes(tpred[bi,tai])) and int(tclass[bi,tai])==cls
                    and float(tconf[bi,tai])>=FIXED['reliable_conf'] and float(tiou[tai])>=FIXED['teacher_ir_iou_min'])
                if teacher_quality:
                    stats['teacher_own_quality_count'] += 1
                    mapped = map_teacher_box(tpred[bi, tai:tai+1], tb[til:til+1], rb[ril:ril+1])
                    mapped_iou = float(original._iou(rb[ril:ril+1], mapped)[0, 0])
                    quality = mapped_iou >= FIXED['mapped_rgb_iou_min']
                    stats['mapped_rgb_quality_count'] += int(quality)
                    margin=quality and mapped_iou>riou+FIXED['localization_margin']
                    stats['localization_margin_count']+=int(margin)
                    selected=False
                    if margin:
                        stats['fullsupport_checked']+=1
                        arguments=dict(teacher_gt=tb[til].tolist(),student_gt=rb[ril].tolist(),
                            teacher_center=tcens[tai].tolist(),student_center=centers[ai].tolist(),
                            teacher_stride=float(tstrides[tai]),student_stride=float(stride_values[ai]))
                        tlogits=teacher['boxes'][bi,:,tai].detach().float().reshape(4,16).cpu().tolist()
                        support=transport_logits(tlogits,**arguments)
                        if support['status']=='INVALID_INPUT':raise FloatingPointError('Invalid teacher distribution: '+support['reason'])
                        selected=support['status']=='SUPPORTED'
                        record.update(full_probability_support=selected,fullsupport_reason=support['status'])
                        stats['fullsupport_rejected']+=int(not selected)
                        if selected:
                            # Soften BEFORE transport. This is not sqrt of transported T=1 q.
                            softened=transport_logits([[v/TEMPERATURE for v in edge] for edge in tlogits],**arguments)
                            if softened['status']!='SUPPORTED':raise AssertionError('T=2 positive support differs from T=1')
                            q=float32_target_or_none(softened['target_probabilities'],device)
                            selected=q is not None
                            record['target_float32_supported']=selected
                            stats['target_float32_rejected']+=int(not selected)
                    record.update(teacher_anchor=tai, teacher_conf=float(tconf[bi, tai]),
                        teacher_own_iou=float(tiou[tai]), teacher_box=tpred[bi, tai].tolist(),
                        mapped_teacher_box=mapped[0].tolist(), mapped_teacher_rgb_iou=mapped_iou,
                        selected=selected)
                    if selected:
                        chosen.append((bi, ai, tai, rgi, tgi))
                        gt_rows.append(rb[ril])
                        targets.append(q)
            stats['base_records'].append(record)
    stats.update(selected_count=len(chosen), normalizer=max(1, stats['base_count']),
        selected_anchors=[(bi, ai, tai, rgi, tgi) for bi, ai, tai, rgi, tgi in chosen],
        selected_object_ids=[(bi, rgi, tgi) for bi, ai, tai, rgi, tgi in chosen])
    stats['nominal_dose'] = len(chosen) / stats['normalizer']
    stats['selection_identity'] = stats['selected_anchors']
    indices = torch.tensor(chosen, dtype=torch.long, device=device).reshape(-1, 5)
    gts = torch.stack(gt_rows) if gt_rows else centers.new_empty((0, 4))
    target = torch.stack(targets) if targets else centers.new_empty((0, 4, 16))
    return indices, gts, target, stats, centers, stride_values


def gt_temperature2_target(distances, bins=16):
    if distances.ndim!=2 or distances.shape[1]!=4 or not bool(torch.isfinite(distances).all()) or not bool(((distances>=0)&(distances<bins-1)).all()):
        raise ValueError('Unclamped RGB GT distances outside adjacent-bin support')
    lo=distances.floor().long();hi=lo+1;upper=distances-lo
    q=distances.new_zeros((*distances.shape,bins))
    q.scatter_add_(-1,lo.unsqueeze(-1),(1-upper).unsqueeze(-1))
    q.scatter_add_(-1,hi.unsqueeze(-1),upper.unsqueeze(-1))
    softened=q.sqrt();return (softened/softened.sum(-1,keepdim=True)).detach()

def distribution_loss(student, ids, targets, normalizer):
    if normalizer<1:raise ValueError('Invalid base normalizer')
    if len(ids)==0:return student['boxes'].float().reshape(-1)[:0].sum()
    logits=student['boxes'].float().permute(0,2,1)[ids[:,0],ids[:,1]].reshape(-1,4,16)
    q=targets.detach().to(device=logits.device,dtype=torch.float32)
    if q.shape!=logits.shape or not bool(torch.isfinite(q).all()) or bool((q<0).any()):raise ValueError('Invalid full distribution target')
    if not bool(torch.allclose(q.sum(-1),torch.ones_like(q.sum(-1)),atol=1e-6,rtol=0)):raise ValueError('Transported target mass differs')
    # F.kl_div handles q=0 exactly without evaluating log(0) in a manual branch.
    terms=F.kl_div(F.log_softmax(logits/TEMPERATURE,dim=-1),q,reduction='none').sum(-1).mean(-1)
    return terms.sum()*(TEMPERATURE**2)/normalizer

def compute(student, teacher, reference, batch, strides=(8, 16, 32), variant='L3-DFL'):
    """Return unweighted scalar and JSON-friendly fixed-selection statistics."""
    if variant not in VARIANTS:
        raise ValueError('Only independent L3-DFL and same-mask L3-GT are supported')
    original, native = _helpers()
    if any(raw['boxes'].shape[1]!=64 for raw in (student,teacher,reference)):raise ValueError('L3 is frozen to four-by-sixteen DFL')
    sc, ss, sr, sz = _layout(student, original, strides)
    rc, rs, rr, rz = _layout(reference, original, strides)
    if (sr != rr or sz != rz or student['scores'].shape != reference['scores'].shape
            or student['boxes'].shape != reference['boxes'].shape
            or student['boxes'].device != reference['boxes'].device):
        raise ValueError('Student must share the frozen RGB reference anchor grid and DFL support')
    ids, gt, targets, stats, centers, stride_values = _select(teacher, reference, batch, strides, original, native)
    if len(ids):
        if variant=='L3-GT':
            distances=native.unclamped_distances(gt,centers[ids[:,1]],stride_values[ids[:,1]])
            targets=gt_temperature2_target(distances)
        loss=distribution_loss(student,ids,targets,stats['normalizer'])
    else:
        # No unused logits are summed: avoids overflow in a mathematically zero path.
        loss = student['boxes'].float().reshape(-1)[:0].sum()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite L3 loss')
    stats.update(variant=variant, loss_unweighted=float(loss.detach()),
        target_kind='T2_then_full_object_coordinate_probability_transport' if variant == 'L3-DFL' else 'same_mask_RGB_GT_adjacent_bin_sqrt_normalized',
        temperature=TEMPERATURE,temperature_squared_applied=True,four_edges_mean=True,
        target_semantics='annotation_defined_object_coordinates_not_physical_registration',
        native_loss_replaced=False, classification_kd_evaluated=False,
        lambda_applied=False, batch_size_multiplier_applied=False)
    return loss, stats
