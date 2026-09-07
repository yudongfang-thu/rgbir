"""Frozen-model diagnostic only. No optimizer, no training or test data."""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
SRC=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/release_v8')
sys.path.insert(0,str(SRC));sys.path.insert(0,str(REPO))
from diagnose_opportunities import (load_batch, stratified_rows, spatial_assign,
    dataset_config, reject_test_path, write_json)
from localization_loss import LocalizationConfig, _layout, _labels, _decode_boxes, _iou, _match_objects
from legacy_bridge import legacy

def roi_vector(feature, scoremap, box, all_boxes, stride, valid_box):
    c,h,w=feature.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=feature.device),torch.arange(w,device=feature.device),indexing='ij')
    centers=torch.stack((xx+.5,yy+.5),-1)*stride
    valid=((centers>=valid_box[:2])&(centers<valid_box[2:])).all(-1)
    fg=((centers>=box[:2])&(centers<box[2:])).all(-1)&valid
    mid=(box[:2]+box[2:])/2;wh=box[2:]-box[:2]
    bg=((centers>=mid-wh)&(centers<mid+wh)).all(-1)&~fg&valid
    if len(all_boxes):
        inside=((centers[...,None,:]>=all_boxes[None,None,:,:2])&(centers[...,None,:]<all_boxes[None,None,:,2:])).all(-1).any(-1)
        bg &= ~inside
    x1,y1=torch.floor(box[:2]/stride).long().tolist();x2,y2=torch.ceil(box[2:]/stride).long().tolist()
    x1=max(0,min(x1,w-1));x2=max(x1+1,min(x2,w));y1=max(0,min(y1,h-1));y2=max(y1+1,min(y2,h))
    bgmean=feature[:,bg].mean(-1) if bg.any() else torch.zeros(c,device=feature.device)
    pooled=F.adaptive_avg_pool2d(feature[:,y1:y2,x1:x2],(2,2))-bgmean[:,None,None]
    def lme(values):return torch.logsumexp(values,-1)-np.log(values.shape[-1])
    relative=lme(scoremap[:,fg])-lme(scoremap[:,bg]) if fg.any() and bg.any() else torch.zeros(scoremap.shape[0],device=feature.device)
    return pooled.flatten().cpu().numpy(),relative.cpu().numpy(),{'fg_tokens':int(fg.sum()),'bg_tokens':int(bg.sum()),'valid':bool(fg.any() and bg.any())}

def background_boxes(all_boxes, rng, valid_box):
    grid=[(x,y) for y in (80,240,400,560) for x in (80,240,400,560)]
    result=[]
    for i in rng.permutation(len(grid)):
        x,y=grid[i];b=all_boxes.new_tensor([x-32,y-32,x+32,y+32])
        if not bool((b[:2]>=valid_box[:2]).all() and (b[2:]<=valid_box[2:]).all()):continue
        if not len(all_boxes) or not ((_iou(b[None],all_boxes)>0).any()):result.append(b)
        if len(result)==4:break
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--canary',action='store_true');a=p.parse_args()
    spec=json.loads(a.spec.read_text());s=spec[a.dataset];cfg=yaml.safe_load(Path(s['config']).read_text())
    lease=legacy.require_bound_lease_from_environment();assert len(lease['gpus'])==1
    assert str(torch.__version__)==cfg['torch_version'];assert str(legacy.ultralytics.__version__)==cfg['ultralytics_version']
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();started=time.time()
    a.output.mkdir(parents=True,exist_ok=False)
    rosters=[]
    for split in ('train','val'):
        rows=json.loads(Path(s[split+'_roster']).read_text())['roster']
        rows=stratified_rows(rows,1024 if split=='train' else 200)
        if a.canary:rows=rows[:1]
        for row in rows:
            for key in ('rgb_path','ir_path','rgb_label','ir_label'):reject_test_path(row[key])
            assert row['split']==split
        rosters.extend(rows)
    write_json(a.output/'frozen_roster.json',{'roster':rosters,'spec':s,'test_accessed':False})
    names=dataset_config(cfg['paths']['student_data_yaml'])['names']
    if isinstance(names,list):names=dict(enumerate(names))
    models={};identity={};captured={};hooks=[]
    for key,path in s['models'].items():
        data=cfg['paths']['privileged_data_yaml'] if key=='T42' else cfg['paths']['student_data_yaml']
        models[key]=legacy.load_frozen(path,data,names).cuda()
        st=Path(path).stat();identity[key]={'path':path,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'names':models[key].names,'args':yaml.safe_load((Path(path).parents[1]/'args.yaml').read_text())}
        assert identity[key]['args']['seed']==(0 if key=='N0' else 42) and identity[key]['args']['epochs']==200
        def capture(module,inputs,key=key):captured[key]=tuple(inputs[0])
        hooks.append(models[key].model[-1].register_forward_pre_hook(capture))
    write_json(a.output/'model_identity.json',identity)
    strides=tuple(int(v) for v in models['N42'].stride)
    assert all(tuple(int(v) for v in m.stride)==strides for m in models.values())
    lc=LocalizationConfig(input_size=640);features={};logits={};object_count=0;rng=np.random.default_rng(20260907)
    with (a.output/'objects.jsonl').open('x') as out,torch.inference_mode():
        for ni,row in enumerate(rosters):
            batch,br=load_batch([row],640)
            batch['img']=batch['img'].cuda();batch['strong_img']=batch['strong_img'].cuda()
            gpu_raw={key:legacy.raw_prediction(model(batch['strong_img'] if key=='T42' else batch['img'])) for key,model in models.items()}
            if ni==0:assert all(torch.equal(f,h) for key,value in gpu_raw.items() for f,h in zip(value['feats'],captured[key]))
            # One image-level transfer avoids hundreds of per-object GPU syncs.
            raw={key:{'scores':v['scores'].cpu(),'boxes':v['boxes'].cpu(),'feats':[f.cpu() for f in captured[key]]} for key,v in gpu_raw.items()}
            del gpu_raw
            centers,sv,_,size=_layout(raw['N42'],lc,strides)
            for key,value in raw.items():
                cc,ss,_,_=_layout(value,lc,strides)
                assert torch.equal(cc,centers) and torch.equal(ss,sv)
                assert value['scores'].shape==raw['N42']['scores'].shape and value['boxes'].shape==raw['N42']['boxes'].shape and value['boxes'].shape[1]==64
            _,rc,rb=_labels(batch,1,len(names),centers.device,size)
            _,tc,tb=_labels(batch['teacher_batch'],1,len(names),centers.device,size)
            pr,pt=_match_objects(rb,rc,tb,tc,.5);pairs=dict(zip(pr.tolist(),pt.tolist()))
            boxes={key:_decode_boxes(value,centers,sv)[0] for key,value in raw.items()}
            assignments={};best={};predcls={};maps={}
            for key,value in raw.items():
                conf,cl=value['scores'][0].sigmoid().max(0);best[key]=conf;predcls[key]=cl
                assignments[key]=spatial_assign(tb if key=='T42' else rb,boxes[key],conf,cl)[0]
                offset=0;maps[key]=[]
                for f in value['feats']:
                    h,w=f.shape[-2:];maps[key].append(value['scores'][0,:,offset:offset+h*w].reshape(len(names),h,w));offset+=h*w
            # The common probe anchor comes only from N P3/P4, class-independent.
            p34=sum(int(f.shape[-2]*f.shape[-1]) for f in raw['N42']['feats'][:2])
            common=spatial_assign(rb,boxes['N42'][:p34],best['N42'][:p34],predcls['N42'][:p34])[0]
            meta=batch['pair_info'][0];bounds=[]
            for shape,matrix in [(meta['original_shape'],meta['rgb_matrix']),(meta['ir_original_shape'],meta['ir_matrix'])]:
                h,w=shape;m=np.asarray(matrix);bounds.append([m[0,2],m[1,2],m[0,2]+w*m[0,0],m[1,2]+h*m[1,1]])
            valid_box=rb.new_tensor([max(b[0] for b in bounds),max(b[1] for b in bounds),min(b[2] for b in bounds),min(b[3] for b in bounds)])
            all_boxes=torch.cat((rb,tb));bg=background_boxes(all_boxes,rng,valid_box)
            units=[(i,b,int(rc[i]),False) for i,b in enumerate(rb)]+[(i,b,len(names),True) for i,b in enumerate(bg)]
            for gi,box,cl,isbg in units:
                anchor=common.get(gi,{}).get('anchor_index') if not isbg else None
                has=anchor is not None
                if anchor is None:
                    mid=(box[:2]+box[2:])/2;xy=torch.floor(mid/8).long().clamp(0,79);anchor=int(xy[1]*80+xy[0])
                iridx=pairs.get(gi) if not isbg else None
                pairiou=float(_iou(box[None],tb[iridx:iridx+1])[0,0]) if iridx is not None else None
                obj={'object_id':row['rgb_path']+('::background:' if isbg else '::gt:')+str(gi),'image':row['rgb_path'],'ir_image':row['ir_path'],
                    'split':row['split'],'class':cl,'is_background':isbg,'source_group':row['source_group'],**br[0],
                    'gt_box_input':box.tolist(),'ir_gt_box_input':tb[iridx].tolist() if iridx is not None else None,'paired_gt_iou':pairiou,
                    'anchor_index':anchor,'anchor_stride':float(sv[anchor]),'anchor_center':centers[anchor].tolist(),
                    'anchor_has_reference_candidate':has,'rgb_gt_local_index':None if isbg else gi,'pair_info':batch['pair_info'][0],
                    'region_policy':'common_RGB_GT_window' if not isbg else 'fixed_annotation_background_window','valid_content_box':valid_box.tolist()}
                area=float((box[2:]-box[:2]).prod());obj['scale_bin']='small' if area<1024 else ('medium' if area<9216 else 'large')
                for key,value in raw.items():
                    ai=iridx if key=='T42' else gi
                    assigned=assignments[key].get(ai) if ai is not None and not isbg else None
                    to_rgb=float(_iou(box[None],box.new_tensor([assigned['box']]))[0,0]) if assigned is not None else None
                    obj[key]={'assigned':assigned,'iou_to_rgb_gt':to_rgb,'same_anchor':{'confidence':float(best[key][anchor]),'pred_class':int(predcls[key][anchor]),
                        'iou_to_rgb_gt':None if isbg else float(_iou(box[None],boxes[key][anchor:anchor+1])[0,0]),'box':boxes[key][anchor].tolist()},'regions':{}}
                    logits.setdefault(key+'_cls',[]).append(value['scores'][0,:,anchor].cpu().numpy().copy())
                    logits.setdefault(key+'_dfl',[]).append(value['boxes'][0,:,anchor].reshape(4,-1).cpu().numpy().copy())
                    rel=[]
                    for level in (0,1):
                        vector,evidence,meta=roi_vector(value['feats'][level][0],maps[key][level],box,all_boxes,strides[level],valid_box)
                        features.setdefault(key+'_P'+str(level+3),[]).append(vector);rel.append(evidence);obj[key]['regions']['P'+str(level+3)]=meta
                        feature=value['feats'][level][0];h,w=feature.shape[-2:];xy=torch.floor(centers[anchor]/strides[level]).long();x,y=int(xy[0]),int(xy[1])
                        patch=feature[:,max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)]
                        features.setdefault(key+'_anchor_P'+str(level+3),[]).append(F.adaptive_avg_pool2d(patch,(2,2)).flatten().cpu().numpy())
                    logits.setdefault(key+'_region_cls',[]).append(np.stack(rel))
                out.write(json.dumps(obj,allow_nan=False)+'\n');object_count+=1
            if (ni+1)%25==0 or ni+1==len(rosters):
                out.flush();write_json(a.output/'progress.json',{'images':ni+1,'total':len(rosters),'objects_including_background':object_count,'seconds':time.time()-started})
                print(json.dumps({'images':ni+1,'total':len(rosters),'objects':object_count}),flush=True)
    np.savez_compressed(a.output/'features.npz',**{k:np.stack(v).astype(np.float32) for k,v in features.items()})
    np.savez_compressed(a.output/'logits.npz',**{k:np.stack(v).astype(np.float32) for k,v in logits.items()})
    summary={'status':'completed','dataset':a.dataset,'images':len(rosters),'objects_including_background':object_count,'models':s['models'],
        'input_box_format':'xyxy','anchor_center_units':'input_pixels','reg_max':16,'reference_model':'N42','feature_shapes':{k:[len(v),len(v[0])] for k,v in features.items()},
        'feature_region_gt_privileged':True,'anchor_patch_region_gt_privileged':False,'anchor_association_gt_privileged':True,'head_input_verified_against_raw_feats':True,'padding_excluded_from_background':True,'background_class':len(names),'background_is_annotation_proxy':True,'same_input_protocol':True,
        'training_recipes_equal':False,'postprocessing_device':'CPU_float32','gpu_allocated_peak_mib':torch.cuda.max_memory_allocated()/2**20,'gpu_reserved_peak_mib':torch.cuda.max_memory_reserved()/2**20,
        'resources':legacy.bound_lease_resource_record_from_environment(),'seconds':time.time()-started,'official_test_accessed':False,'formal_training_authorized':False}
    write_json(a.output/'summary.json',summary)
    legacy.emit_bound_run_receipt(run_dir=a.output/'run_evidence',method_identity='PROTOCOL-ADAPTED',dataset=a.dataset,data_role='development_train_val',seed=20260907,run_kind='feature',
        trainers=[Path(__file__),SRC/'diagnose_opportunities.py'],losses=[SRC/'localization_loss.py'],configs=[a.spec,Path(s['config'])],split_rosters=[a.output/'frozen_roster.json'],
        metric_files=[a.output/'summary.json'],environment={'torch':str(torch.__version__),'ultralytics':legacy.ultralytics.__version__},inputs={'models':s['models'],'diagnostic_only':True})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
