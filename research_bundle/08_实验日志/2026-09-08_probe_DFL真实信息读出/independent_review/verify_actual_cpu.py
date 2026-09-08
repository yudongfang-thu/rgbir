"""Read-only independent verification of an executed DFL JSON export; no model imports."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct


def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def require(value,msg):
    if not value:raise AssertionError(msg)
def geometry(ai):
    base=0
    for level,(w,s) in enumerate([(80,8),(40,16),(20,32)]):
        if base<=ai<base+w*w:
            y,x=divmod(ai-base,w)
            return dict(level=level,stride=s,row=y,col=x,center_xy=[(x+.5)*s,(y+.5)*s])
        base+=w*w
    raise AssertionError('out of range anchor')
def flat(x):
    for v in x:
        if isinstance(v,list):yield from flat(v)
        else:yield v
def expected_roles(o):
    hist=o['historical_L2_record'];r=None if hist is None else hist['reference_anchor']
    n={k:None if v is None else v['anchor_index'] for k,v in o['native_iou50_matches'].items()}
    return dict(S=dict(historical_R_candidate=r,own_native_iou50=n['S']),
                R=dict(historical_R_candidate=r,own_native_iou50=n['R']),
                T=dict(same_historical_R_index=r,same_S_native_iou50_index=n['S'],same_R_native_iou50_index=n['R'],
                       own_native_iou50=n['T'],historical_selected_T=hist['teacher_anchor'] if hist is not None and o['historical_L2_gates']['selected'] else None))


def run(a):
    source=a.workspace/'08_实验日志/2026-09-08_probe_定位学习位置与目标/anchor_join/output_attempt3/objects.jsonl'
    old=rows(source);current=rows(a.probe/'objects.jsonl');raw=rows(a.probe/'anchor_distributions.jsonl')
    receipt=read(a.probe/'completion_receipt.json');contract=read(a.probe/'dfl_contract.json')
    identity=read(a.probe/'identity_contract.json');old_by={x['stable_rgb_gt_id']:x for x in old}
    require(receipt['status']=='RAW_DFL_SINGLE_BATCH_COMPLETED','producer not complete')
    require((len(old),len(current),len(identity['frames']))==(80,80,32),'fixed population')
    require(len(set(o['stable_rgb_gt_id'] for o in current))==80,'duplicated GT IDs')
    require({o['stable_rgb_gt_id'] for o in current}==set(old_by),'GT roster')
    fid=receipt['current_forward_id'];require(fid==contract['current_forward_id'] and fid!='llvip_first32_seed42','new forward identity')
    expected_requests=set();role_counts=Counter();missing_counts=Counter();valid_counts=Counter();refs=Counter()
    raw_by={d['distribution_id']:d for d in raw};require(len(raw_by)==len(raw),'unique distribution IDs')
    for obj in current:
        o=old_by[obj['stable_rgb_gt_id']]
        require(obj['current_forward_id']==fid,'object forward identity')
        for k in ['stable_ir_gt_id','image_index','frame_id','rgb_gt_xyxy','ir_gt_xyxy','bucket','C_selected','historical_L2_gates']:
            require(obj[k]==o[k],'historical identity '+k)
        expected=expected_roles(o);require(set(obj['roles'])==set(expected),'model roles')
        for model,roles in expected.items():
            require(set(obj['roles'][model])==set(roles),'role schema '+model)
            for role,anchor in roles.items():
                value=obj['roles'][model][role];label=model+'/'+role;role_counts[label]+=1
                if anchor is None:
                    require(value is None,'invented missing role');missing_counts[label]+=1;continue
                require(value is not None,'lost role')
                did='%s::image:%s::anchor:%s'%(model,o['image_index'],anchor)
                expected_requests.add(did);refs[did]+=1
                require(value['distribution_id']==did and value['anchor_index']==anchor,'anchor role identity')
                d=raw_by[did];gt=o['ir_gt_xyxy'] if model=='T' else o['rgb_gt_xyxy']
                gid=o['stable_ir_gt_id'] if model=='T' else o['stable_rgb_gt_id']
                require(value['own_gt_xyxy']==gt and value['own_gt_id']==gid,'own GT identity')
                require(value['own_gt_modality']==('IR' if model=='T' else 'RGB'),'own GT modality')
                x,y=d['center_xy'];s=d['stride'];target=[(x-gt[0])/s,(y-gt[1])/s,(gt[2]-x)/s,(gt[3]-y)/s]
                require(value['unclamped_gt_distance_bins']==target,'unclamped GT distances')
                valid=[math.isfinite(v) and 0<=v<15 for v in target]
                require(value['gt_distance_in_range_per_side']==valid and value['gt_distance_all_in_range']==all(valid),'support flags')
                valid_counts[label]+=int(all(valid))
    require(set(raw_by)==expected_requests,'missing or invented anchor distributions')
    maxima=dict(FP64_vs_saved_FP32_probability=0.,FP64_vs_saved_FP32_mean_bin=0.,FP64_vs_saved_FP32_box_px=0.,
                native_probability_sum_error=0.,native_vs_FP32_box_px=0.,native_conv_vs_unrounded_native_probability_mean_bin=0.)
    dtype_counts=Counter()
    for d in raw:
        require(d['current_forward_id']==fid,'distribution forward identity')
        require(d['bins']==16 and d['side_order']==['left','top','right','bottom'],'bin layout')
        require(d['raw_feature_shapes']==[[32,64,80,80],[32,128,40,40],[32,256,20,20]],'feature geometry')
        require(d['input_shape_HW']==[640,640],'input canvas')
        g=geometry(d['anchor_index'])
        for k,v in g.items():require(d[k]==v,'geometry '+k)
        logits=d['raw_logits'];require(len(logits)==4 and all(len(v)==16 for v in logits),'raw 4x16')
        require(all(isinstance(v,(int,float)) and math.isfinite(v) for v in flat(logits)),'finite raw')
        dtype_counts[d['raw_dtype']]+=1
        if d['raw_dtype']=='torch.float16':
            require(all(struct.unpack('e',struct.pack('e',v))[0]==v for v in flat(logits)),'lossless FP16 JSON values')
        mean=[]
        for side,v in enumerate(logits):
            top=max(v);e=[math.exp(x-top) for x in v];z=sum(e);p=[x/z for x in e];m=sum(i*x for i,x in enumerate(p));mean.append(m)
            maxima['FP64_vs_saved_FP32_probability']=max(maxima['FP64_vs_saved_FP32_probability'],max(abs(x-y) for x,y in zip(p,d['probabilities_fp32'][side])))
            maxima['FP64_vs_saved_FP32_mean_bin']=max(maxima['FP64_vs_saved_FP32_mean_bin'],abs(m-d['expectation_fp32_bins'][side]))
            pn=d['probabilities_native'][side];require(len(pn)==16 and all(math.isfinite(x) and 0<=x<=1 for x in pn),'native probabilities')
            maxima['native_probability_sum_error']=max(maxima['native_probability_sum_error'],abs(sum(pn)-1))
            native_mean=sum(i*x for i,x in enumerate(pn))
            maxima['native_conv_vs_unrounded_native_probability_mean_bin']=max(maxima['native_conv_vs_unrounded_native_probability_mean_bin'],abs(native_mean-d['native_dfl_distances_bins'][side]))
        x,y=g['center_xy'];s=g['stride'];box=[x-s*mean[0],y-s*mean[1],x+s*mean[2],y+s*mean[3]]
        maxima['FP64_vs_saved_FP32_box_px']=max(maxima['FP64_vs_saved_FP32_box_px'],max(abs(x-y) for x,y in zip(box,d['box_fp32_xyxy'])))
        maxima['native_vs_FP32_box_px']=max(maxima['native_vs_FP32_box_px'],max(abs(x-y) for x,y in zip(d['box_native_xyxy'],d['box_fp32_xyxy'])))
        require(d['fp32_decode_exact'] is True and d['native_decode_exact'] is True,'producer closure flags')
    require(maxima['FP64_vs_saved_FP32_probability']<4e-6,'FP32 probability closure')
    require(maxima['FP64_vs_saved_FP32_mean_bin']<1e-5,'FP32 expected distance closure')
    require(maxima['FP64_vs_saved_FP32_box_px']<5e-4,'FP32 box closure')
    require(receipt['raw_forward_counts']==dict(student=1,reference=1,teacher=1),'measured batch forward counts')
    for k in ['student_full_state_unchanged','auxiliary_full_state_unchanged','all_gradients_absent','first_batch_stream_exact','identity_exact']:
        require(receipt[k] is True,'producer executed guard '+k)
    for k in ['optimizer_updates','ema_updates','backward','training']:require(receipt[k]==0,'zero-update guard '+k)
    result=dict(status='PASS_INDEPENDENT_ACTUAL_JSON_CPU',auditor='/root/dfl_readout_review',model_identity='unavailable',
                objects=80,frames=32,unique_distributions=len(raw),role_counts=dict(role_counts),missing_roles=dict(missing_counts),
                valid_all4_roles=dict(valid_counts),bucket_counts=dict(Counter(o['bucket'] for o in current)),raw_dtype_counts=dict(dtype_counts),
                closure_maxima=maxima,current_forward_id=fid,inputs=[str(source),str(a.probe)],
                audited_input_hashes='not computed; task boundary',new_GPU=False,new_model_forward=False,
                limits=['FP64 independent closure uses numerical tolerance, not FP32 bitwise emulation.',
                        'Native decode closure is an executed producer assertion on actual pinned operations; FP64/native differences are reported, not silently equated.',
                        'State/forward/update facts are from the reviewed executed producer guards; independent reviewer did not run a second forward.',
                        'This verifies the fixed single-batch readout and cannot establish physical registration or KD utility.'])
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(result['status'],len(raw),maxima)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
