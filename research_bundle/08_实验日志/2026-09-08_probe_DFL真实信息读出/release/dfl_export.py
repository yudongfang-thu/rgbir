"""Lossless requested-anchor DFL readout; no NMS, matching, or loss."""
import copy
import torch

SCOPE='RAW_DFL_SINGLE_BATCH'
ROLES={'S':('historical_R_candidate','own_native_iou50'),'R':('historical_R_candidate','own_native_iou50'),
       'T':('same_historical_R_index','same_S_native_iou50_index','same_R_native_iou50_index','own_native_iou50','historical_selected_T')}

def require(ok,msg):
    if not ok:raise ValueError(msg)
def anchor_geometry(anchor,feats,strides):
    require(type(anchor) is int and anchor>=0,'Invalid anchor')
    offset=0
    for level,(shape,stride) in enumerate(zip(feats,strides)):
        h,w=shape[-2:]
        if anchor<offset+h*w:
            y,x=divmod(anchor-offset,w)
            return dict(anchor_index=anchor,level=level,stride=stride,center_xy=[(x+.5)*stride,(y+.5)*stride],row=y,col=x)
        offset+=h*w
    raise ValueError('Anchor outside raw layout')
def distances(gt,center,stride,bins=16):
    x,y=center;d=[(x-gt[0])/stride,(y-gt[1])/stride,(gt[2]-x)/stride,(gt[3]-y)/stride]
    return dict(unclamped_gt_distance_bins=d,gt_distance_in_range_per_side=[0<=v<bins-1 for v in d],gt_distance_all_in_range=all(0<=v<bins-1 for v in d))
def build_requests(anchor_objects):
    objects=[];requests=set()
    for old in anchor_objects:
        bi=old['image_index'];record=old['historical_L2_record'];ra=None if record is None else record['reference_anchor']
        roles={}
        for name in ('S','R','T'):
            native=old['native_iou50_matches'][name];na=None if native is None else native['anchor_index']
            choices=({'historical_R_candidate':ra,'own_native_iou50':na} if name!='T' else
                {'same_historical_R_index':ra,
                 'same_S_native_iou50_index':None if old['native_iou50_matches']['S'] is None else old['native_iou50_matches']['S']['anchor_index'],
                 'same_R_native_iou50_index':None if old['native_iou50_matches']['R'] is None else old['native_iou50_matches']['R']['anchor_index'],
                 'own_native_iou50':na,'historical_selected_T':record['teacher_anchor'] if record is not None and old['historical_L2_gates']['selected'] else None})
            roles[name]={}
            for role,anchor in choices.items():
                key=None if anchor is None else (name,bi,anchor)
                roles[name][role]=key
                if key is not None:requests.add(key)
        objects.append(dict(stable_rgb_gt_id=old['stable_rgb_gt_id'],stable_ir_gt_id=old['stable_ir_gt_id'],image_index=bi,frame_id=old['frame_id'],
            bucket=old['bucket'],rgb_gt_xyxy=old['rgb_gt_xyxy'],ir_gt_xyxy=old['ir_gt_xyxy'],C_selected=old['C_selected'],historical_L2_gates=copy.deepcopy(old['historical_L2_gates']),roles=roles))
    return objects,sorted(requests)
def distribution_id(key):return '%s::image:%s::anchor:%s'%key

@torch.no_grad()
def export_dfl(batch,raw_by_name,model_by_name,anchor_objects,original,evidence_config,strides):
    objects,requested=build_requests(anchor_objects);out={};decode={}
    require(set(raw_by_name)==set(model_by_name)=={'S','R','T'},'Model roster differs')
    for name in ('S','R','T'):
        raw=raw_by_name[name];head=model_by_name[name].model[-1];shape=[list(x.shape) for x in raw['feats']]
        require(raw['boxes'].ndim==3 and raw['boxes'].shape[1]==64 and head.reg_max==16,'Frozen four-by-sixteen DFL layout differs')
        require(not head.end2end and not head.xyxy and not head.export,'Frozen native head mode differs')
        keys=[k for k in requested if k[0]==name];versions=[(t,t._version) for t in [raw['boxes'],raw['scores']]+list(raw['feats'])]
        centers,sv,_,size=original._layout(raw,evidence_config,strides)
        require(tuple(size)==tuple(batch['img'].shape[-2:]),'Actual input shape differs')
        logits=raw['boxes'].detach().float().reshape(raw['boxes'].shape[0],4,16,-1)
        probs=logits.softmax(2);means=(probs*torch.arange(16,device=probs.device,dtype=torch.float32)[None,None,:,None]).sum(2)
        ds=means.transpose(1,2)*sv[None]
        reconstructed=torch.cat((centers[None]-ds[...,:2],centers[None]+ds[...,2:]),-1)
        fp=original._decode_boxes(raw,centers,sv)
        require(torch.equal(reconstructed,fp),'FP32 softmax-expectation decode differs')
        captured={};calls={'probability':0,'distance':0}
        def pre_conv(module,args):
            calls['probability']+=1;v=args[0]
            require(tuple(v.shape[1:3])==(16,4),'Native DFL probability layout differs')
            captured['probability_dtype']=str(v.dtype)
            captured['probabilities']={k:v[k[1],:,:,k[2]].transpose(0,1).detach().float().cpu().tolist() for k in keys}
        def post_dfl(module,args,value):
            calls['distance']+=1;captured['distances']=value.detach()
        hooks=[head.dfl.conv.register_forward_pre_hook(pre_conv),head.dfl.register_forward_hook(post_dfl)]
        try:native=head._inference(raw)
        finally:
            for hook in hooks:hook.remove()
        require(calls=={'probability':1,'distance':1},'Expected exactly one native DFL readout')
        decoded=head.decode_bboxes(captured['distances'],head.anchors.unsqueeze(0))*head.strides
        require(torch.equal(decoded,native[:,:4]),'Actual native DFL-output decode does not close')
        from ultralytics.utils.ops import xywh2xyxy
        native_xyxy=xywh2xyxy(native[:,:4].transpose(1,2))
        for key in keys:
            _,bi,ai=key;geometry=anchor_geometry(ai,shape,strides)
            require(torch.equal(centers[ai],torch.tensor(geometry['center_xy'],device=centers.device,dtype=centers.dtype)),'Anchor center identity differs')
            values=dict(distribution_id=distribution_id(key),model=name,image_index=bi,**geometry,bins=16,side_order=['left','top','right','bottom'],
                raw_dtype=str(raw['boxes'].dtype),raw_logits=logits[bi,:,:,ai].cpu().tolist(),probabilities_fp32=probs[bi,:,:,ai].cpu().tolist(),
                probabilities_native=captured['probabilities'][key],native_probability_dtype=captured['probability_dtype'],
                expectation_fp32_bins=means[bi,:,ai].cpu().tolist(),native_dfl_distances_bins=captured['distances'][bi,:,ai].float().cpu().tolist(),native_distance_dtype=str(captured['distances'].dtype),
                box_fp32_xyxy=fp[bi,ai].cpu().tolist(),box_native_xyxy=native_xyxy[bi,ai].float().cpu().tolist(),native_decode_dtype=str(native.dtype),input_shape_HW=list(size),raw_feature_shapes=shape,
                fp32_decode_exact=True,native_decode_exact=True,new_forward=True)
            require(all(torch.isfinite(t).all() for t in (logits[bi,:,:,ai],probs[bi,:,:,ai],fp[bi,ai],native_xyxy[bi,ai])),'Nonfinite DFL readout')
            out[values['distribution_id']]=values
        require(all(t._version==v for t,v in versions),'Original raw tensor changed')
        decode[name]=dict(fp32_expectation_decode_exact=True,native_DFL_output_decode_exact=True,native_DFL_calls=calls,raw_dtype=str(raw['boxes'].dtype),native_dtype=str(native.dtype),requested_unique_anchors=len(keys))
        del logits,probs,means,ds,fp,reconstructed,native,native_xyxy,decoded,captured
    for obj in objects:
        for name,roles in obj['roles'].items():
            own_gt=obj['ir_gt_xyxy'] if name=='T' else obj['rgb_gt_xyxy'];own_id=obj['stable_ir_gt_id'] if name=='T' else obj['stable_rgb_gt_id']
            for role,key in list(roles.items()):
                if key is None:roles[role]=None;continue
                did=distribution_id(key);d=out[did]
                roles[role]=dict(distribution_id=did,anchor_index=key[2],own_gt_id=own_id,own_gt_xyxy=own_gt,own_gt_modality='IR' if name=='T' else 'RGB',**distances(own_gt,d['center_xy'],d['stride']))
    return dict(scope=SCOPE,objects=objects,distributions=list(out.values()),decode=decode,roles=ROLES,
        full_probability_vectors=True,GT_distances_clamped=False,cross_anchor_KL=False,new_NMS=False,new_matching=False,new_loss=False)
