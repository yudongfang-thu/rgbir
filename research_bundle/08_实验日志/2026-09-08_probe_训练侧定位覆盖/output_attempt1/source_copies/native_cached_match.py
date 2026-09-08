"""Execute accepted native source functions; capture actual GT/detection identities."""
import ast
from pathlib import Path
import sys
import numpy as np
import torch


def load_native(source_dir):
    source_dir=Path(source_dir);env=dict(np=np,torch=torch)
    for filename,name in [('7_metrics.py','box_iou'),('4_validator.py','match_predictions'),('3_val.py','_process_batch')]:
        path=source_dir/filename;tree=ast.parse(path.read_text(encoding='utf-8-sig'));found=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name==name:found.append(node)
        if len(found)!=1:raise ValueError('Exactly one native function required: '+name)
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+found,type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)
    Base=type('AcceptedNativeBase',(),{'match_predictions':env['match_predictions']})
    Validator=type('AcceptedNativeDetection',(Base,),{'_process_batch':env['_process_batch']})
    return Validator,Base,env['box_iou']


def call_with_pairs(validator,base,pred,gt):
    captured=[];prior=sys.getprofile()
    def capture(frame,event,arg):
        if frame.f_code is base.match_predictions.__code__ and event=='return':
            value=frame.f_locals.get('matches')
            captured.append(np.empty((0,2),dtype=int) if value is None else value.copy().astype(int))
    try:
        sys.setprofile(capture)
        tp=validator._process_batch(pred,gt)['tp']
    finally:sys.setprofile(prior)
    required=int(len(pred['cls'])>0 and len(gt['cls'])>0)
    if len(captured)!=required:raise AssertionError('Native matching return-frame capture missing/duplicated')
    pairs=captured[0] if captured else np.empty((0,2),dtype=int)
    check=np.zeros((len(pred['cls']),1),dtype=bool)
    if len(pairs):check[pairs[:,1],0]=True
    if not np.array_equal(tp,check):raise AssertionError('Native per-prediction TP differs from captured pairs')
    if len(set(pairs[:,0]))!=len(pairs) or len(set(pairs[:,1]))!=len(pairs):raise AssertionError('Non one-to-one native identity')
    return pairs,tp[:,0]


def match_row(row,confidence_cut,iou_threshold,native):
    """Preserve original post-NMS row order and original prediction IDs."""
    Validator,Base,box_iou=native
    conf=np.asarray(row['pred_confidence'],dtype=np.float32)
    keep=np.arange(len(conf),dtype=int) if confidence_cut is None else np.flatnonzero(conf>confidence_cut)
    pred=dict(bboxes=torch.tensor(np.asarray(row['pred_boxes'],dtype=np.float32).reshape(-1,4)[keep]),
        cls=torch.tensor(np.asarray(row['pred_classes'],dtype=np.float32)[keep]))
    gt=dict(bboxes=torch.tensor(np.asarray(row['gt_boxes'],dtype=np.float32).reshape(-1,4)),
        cls=torch.tensor(np.asarray(row['gt_classes'],dtype=np.float32)))
    validator=object.__new__(Validator);validator.iouv=torch.tensor([iou_threshold]);validator.niou=1
    pairs,tp=call_with_pairs(validator,Base,pred,gt)
    overlaps=box_iou(gt['bboxes'],pred['bboxes']);matched={}
    pred_to_gt={}
    for gi,pi in pairs:
        original_index=int(keep[pi]);matched[int(gi)]=dict(prediction_id=original_index,
            filtered_prediction_id=int(pi),iou=float(overlaps[gi,pi]),confidence=float(conf[original_index]))
        pred_to_gt[original_index]=int(gi)
    return dict(gt_matches=matched,prediction_matches=pred_to_gt,kept_prediction_ids=keep.tolist(),
        prediction_tp=tp.tolist(),tp=len(pairs),fp=len(keep)-len(pairs),fn=len(gt['cls'])-len(pairs))
