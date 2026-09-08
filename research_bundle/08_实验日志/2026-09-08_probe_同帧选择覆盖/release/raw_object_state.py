"""Fixed baseline object states, using the supplied pinned matching helpers.

Assignment is GT-assisted one-to-one spatial matching, not AP/post-NMS.
Own-GT and cross-modal-GT coordinate states must be named separately.
"""
import torch

COARSE_CONFIDENCE = .05
COARSE_IOU = .1
CORRECT_CONFIDENCE = .25
CORRECT_IOU = .5


def spatial_assign(gt_boxes, prediction_boxes, confidence, classes, *, original):
    """Exact baseline spatial_assign operations and thresholds; helper injected."""
    eligible = torch.nonzero(confidence >= COARSE_CONFIDENCE, as_tuple=False).flatten()
    candidate_boxes = prediction_boxes[eligible]
    zeros_gt = torch.zeros(len(gt_boxes), dtype=torch.long, device=gt_boxes.device)
    zeros_candidate = torch.zeros(len(eligible), dtype=torch.long, device=gt_boxes.device)
    gi, pi = original._match_objects(gt_boxes, zeros_gt, candidate_boxes, zeros_candidate, COARSE_IOU)
    result = {}
    for g, p in zip(gi.tolist(), pi.tolist()):
        anchor = int(eligible[p])
        result[g] = dict(anchor_index=anchor, **{'class':int(classes[anchor])},
            confidence=float(confidence[anchor]),
            own_gt_iou=float(original._iou(gt_boxes[g:g+1],prediction_boxes[anchor:anchor+1])[0,0]),
            box=prediction_boxes[anchor].tolist())
    return result, len(eligible)


def state(assigned, gt_class, *, evaluation_iou=None, coordinate_scope='own_gt'):
    """Baseline priority: low confidence precedes class/box error decomposition."""
    if assigned is None:
        return dict(state='no_candidate',candidate=False,confidence_ok=False,class_ok=False,
            localization_ok=False,correct=False,iou=None,assigned=None,coordinate_scope=coordinate_scope)
    iou=assigned['own_gt_iou'] if evaluation_iou is None else evaluation_iou
    c=assigned['confidence'] >= CORRECT_CONFIDENCE
    y=assigned['class'] == gt_class
    b=iou >= CORRECT_IOU
    if not c: name='low_confidence'
    elif not y and not b: name='class_and_localization'
    elif not y: name='class_only'
    elif not b: name='localization_only'
    else: name='correct'
    return dict(state=name,candidate=True,confidence_ok=bool(c),class_ok=bool(y),
        localization_ok=bool(b),correct=bool(c and y and b),iou=float(iou),
        assigned=assigned,coordinate_scope=coordinate_scope)
