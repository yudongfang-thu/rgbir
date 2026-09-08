"""Existing-raw witnesses; FP32 OEv1 decode and native Detect decode are distinct."""
import copy
import inspect
import sys
import numpy as np
import torch

SCOPE='SAME_FORWARD_DETECTOR_WITNESS'


def native_components():
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import xywh2xyxy
    return DetectionValidator,BaseValidator,box_iou,non_max_suppression,xywh2xyxy


def native_matches(pred,gt,components):
    """Capture the actual native GT/detection pairs, and check every returned TP."""
    Validator,Base,_,_,_=components
    validator=object.__new__(Validator);validator.iouv=torch.tensor([.5]);validator.niou=1
    observed=[];previous=sys.getprofile()
    def trace(frame,event,arg):
        if frame.f_code is Base.match_predictions.__code__ and event=='return':
            matches=frame.f_locals.get('matches')
            observed.append(np.empty((0,2),dtype=int) if matches is None else matches.copy())
    try:
        sys.setprofile(trace)
        tp=validator._process_batch(pred,gt)['tp']
    finally:sys.setprofile(previous)
    pairs=observed[0].astype(int) if observed else np.empty((0,2),dtype=int)
    expected=np.zeros((len(pred['cls']),1),dtype=bool)
    if len(pairs):expected[pairs[:,1],0]=True
    required_calls=int(len(pred['cls'])>0 and len(gt['cls'])>0)
    assert len(observed)==required_calls and np.array_equal(tp,expected),'Actual native TP/pair identities disagree'
    assert len(set(pairs[:,0]))==len(pairs) and len(set(pairs[:,1]))==len(pairs),'Native matching not one-to-one'
    return pairs,tp


def dense_witness(boxes,probs,gt_boxes,gt_classes,gt_rows,gt_ids,box_iou,strict=False):
    overlaps=box_iou(gt_boxes.float(),boxes.float());confidence,classes=probs.float().max(0)
    ok=(overlaps>=.5)&(classes[None]==gt_classes[:,None])
    ok=ok&((confidence>.25) if strict else (confidence>=.25))[None]
    out=[]
    for j in range(len(gt_boxes)):
        ids=torch.nonzero(ok[j],as_tuple=False).flatten();w=None
        if len(ids):
            anchor=int(ids[confidence[ids].argmax()])
            other=[dict(global_gt_row=int(gt_rows[k]),stable_gt_id=gt_ids[k],iou=float(overlaps[k,anchor]))
                   for k in range(len(gt_boxes)) if k!=j and overlaps[k,anchor]>=.5]
            w=dict(anchor_index=anchor,confidence=float(confidence[anchor]),**{'class':int(classes[anchor])},
                   box=boxes[anchor].tolist(),iou=float(overlaps[j,anchor]),other_gt_overlaps=other)
        out.append(dict(dense_any_correct=bool(len(ids)),dense_correct_count=len(ids),dense_witness=w))
    return out


@torch.no_grad()
def analyze_witness(batch,raw_by_name,model_by_name,evidence_config,strides,existing_export):
    import selection_adapter as adapter
    components=native_components();Validator,Base,box_iou,nms,xywh2xyxy=components
    cfg=adapter.evidence_config(evidence_config);original=adapter.original
    assert set(raw_by_name)==set(model_by_name)=={'S','T','R'}
    assert existing_export['scope']=='SAME_FORWARD_SELECTION_COVERAGE'
    records=copy.deepcopy(existing_export['records']);ir_records=copy.deepcopy(existing_export['ir_records'])
    frames=copy.deepcopy(existing_export['frames']);per_model={};decode={}
    for r in records:r['detector']={}
    for f in frames:f['native_detections']={}
    for name in ('S','T','R'):
        raw=raw_by_name[name];model=model_by_name[name];head=model.model[-1]
        assert not head.end2end and not head.xyxy and not head.export and head.nc==1,'Frozen LLVIP standard Detect required'
        versions=[(x,x._version) for x in [raw['scores'],raw['boxes']]+list(raw['feats'])]
        centers,sv,_,size=original._layout(raw,cfg,strides)
        labels=adapter._teacher_batch(batch) if name=='T' else batch
        indices,classes,gtboxes=original._labels(labels,raw['scores'].shape[0],head.nc,raw['scores'].device,size)
        fpboxes=original._decode_boxes(raw,centers,sv);probs=raw['scores'].detach().float().sigmoid()
        native=head._inference(raw)  # Original head decode only; no backbone/head prediction forward.
        assert bool(torch.isfinite(native).all())
        nativeboxes=xywh2xyxy(native[:,:4].transpose(1,2));nativeprobs=native[:,4:]
        decode[name]=dict(raw_dtype=str(raw['boxes'].dtype),native_dtype=str(native.dtype),
            nms_backend='torchvision.ops.nms' if native.device.type not in ('npu','xpu') and 'torchvision' in sys.modules else 'ultralytics.TorchNMS.nms',
            fp32_native_max_box_abs_difference=float((fpboxes-nativeboxes.float()).abs().max()))
        identity=ir_records if name=='T' else records;id_key='stable_ir_gt_id' if name=='T' else 'stable_rgb_gt_id'
        per_model[name]={}
        for bi in range(raw['scores'].shape[0]):
            rows=torch.nonzero(indices==bi,as_tuple=False).flatten();boxes=gtboxes[rows];cl=classes[rows]
            stable=[identity[int(row)][id_key] for row in rows]
            raw_states=dense_witness(fpboxes[bi],probs[bi],boxes,cl,rows,stable,original._iou)
            expected=original._has_candidate(fpboxes[bi],probs[bi],boxes,.25,.5,cl)
            assert [r['dense_any_correct'] for r in raw_states]==expected.tolist(),'Original dense gate differs'
            pre=dense_witness(nativeboxes[bi],nativeprobs[bi],boxes,cl,rows,stable,box_iou,strict=True)
            # Per-image calls avoid a batch-wide wall timeout silently skipping later images.
            dets,anchors=nms(native[bi:bi+1].clone(),conf_thres=.25,iou_thres=.7,nc=0,
                multi_label=True,agnostic=False,max_det=300,end2end=False,rotated=False,return_idxs=True)
            det=dets[0];anchor=anchors[0].flatten()
            pred=dict(bboxes=det[:,:4],conf=det[:,4],cls=det[:,5]);gt=dict(bboxes=boxes,cls=cl)
            pairs,tp=native_matches(pred,gt,components);matched={int(g):int(p) for g,p in pairs}
            overlap=box_iou(boxes,pred['bboxes'])
            frames[bi]['native_detections'][name]=[dict(prediction_index=k,anchor_index=int(anchor[k]),
                box=det[k,:4].tolist(),confidence=float(det[k,4]),**{'class':int(det[k,5])},native_tp_iou50=bool(tp[k,0])) for k in range(len(det))]
            for j,row in enumerate(rows.tolist()):
                witness=None
                if j in matched:
                    k=matched[j];witness=dict(frames[bi]['native_detections'][name][k],iou=float(overlap[j,k]))
                value=dict(raw_states[j],native_pre_nms_any_correct=pre[j]['dense_any_correct'],
                    native_pre_nms_correct_count=pre[j]['dense_correct_count'],native_pre_nms_witness=pre[j]['dense_witness'],
                    native_postnms_matched=j in matched,native_witness=witness,
                    own_gt_scope='ir' if name=='T' else 'rgb',global_gt_row=row,stable_gt_id=stable[j])
                per_model[name][row]=value
        assert all(x._version==v for x,v in versions),'Input raw mutated'
    for r in records:
        row=r['rgb_global_row'];ti=r['ir_global_row']
        r['detector']={n:per_model[n][row] for n in ('S','R')}
        r['detector']['T']=None if ti is None else per_model['T'][ti]
        if ti is not None:assert r['gates']['teacher_correct_own']==r['detector']['T']['dense_any_correct']
    for r in ir_records:r['detector_T']=per_model['T'][r['ir_global_row']]
    return dict(scope=SCOPE,records=records,ir_records=ir_records,frames=frames,
        objects_n=len(records),raw_decode='original OEv1 _decode_boxes detach/float32',
        native_decode='pinned Detect._inference(existing_raw), original raw dtype and caller AMP',
        dense_conf_operator='>=',native_conf_operator='>',confidence=.25,correct_iou=.5,
        native_profile=dict(conf=.25,iou=.7,max_det=300,multi_label=True,agnostic_nms=False,matching_iou=.5),
        nms=dict(iou=.7,max_det=300,multi_label=True,agnostic=False,per_image=True),
        native_match_provenance=dict(native_process_batch=inspect.getsourcefile(Validator._process_batch),
            native_match_predictions=inspect.getsourcefile(Base.match_predictions),captured_matches_exact=True),
        decode_diagnostics=decode,native_tp_and_gt_pairs_exact=True,
        source_files={name:inspect.getsourcefile(fn) for name,fn in [('postprocess',nms),('native_matching',Base.match_predictions),('native_process_batch',Validator._process_batch)]},
        new_model_forward=False,backward=False,optimizer_updates=0,new_hash_computed=False,AP_estimated=False)
