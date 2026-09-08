"""Frozen LLVIP full-dev cached native post-NMS opportunity readout; CPU only."""
import argparse
from collections import Counter
import gzip
import itertools
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from native_cached_match import load_native,match_row

SCOPE='LLVIP_FULL_DEV_CACHED_NATIVE_OPPORTUNITIES'
GROUPS={'all_iou50':(None,.5),'all_iou75':(None,.75),'gt025_iou50':(.25,.5),'gt025_iou75':(.25,.75)}
BUCKETS=('both_correct','T_only','N_only','both_unmatched')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def stat(path):
    p=Path(path);return dict(path=str(p.absolute()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
def bucket(n,t):return 'both_correct' if n and t else 'T_only' if t else 'N_only' if n else 'both_unmatched'


def validate_row(row):
    require(isinstance(row.get('image'),str) and row['image'],'Missing image identity')
    for k in ('canvas_shape','original_shape'):
        require(len(row[k])==2 and all(type(x) is int and x>0 for x in row[k]),'Invalid shape')
    require(len(row['gt_boxes'])==len(row['gt_classes']),'GT arrays differ')
    require(len(row['pred_boxes'])==len(row['pred_classes'])==len(row['pred_confidence']),'Prediction arrays differ')
    for key in ('gt_boxes','pred_boxes'):
        a=np.asarray(row[key],dtype=np.float32).reshape(-1,4)
        require(np.isfinite(a).all() and (a[:,2:]>=a[:,:2]).all(),'Nonfinite or inverted box')
    for key in ('gt_classes','pred_classes'):require(all(x==0 for x in row[key]),'LLVIP single class changed')
    require(all(math.isfinite(x) and .001<x<=1 for x in row['pred_confidence']),'Native confidence range differs')
    a=np.asarray(row['gt_boxes'],dtype=np.float32).reshape(-1,4);h,w=row['canvas_shape']
    require((a>=0).all() and (a[:,[0,2]]<=w).all() and (a[:,[1,3]]<=h).all(),'GT outside canvas')
    # Native predictions may cross the canvas edge; never clamp or reject those.
    a=np.asarray(row['pred_boxes'],dtype=np.float32).reshape(-1,4)
    return int(((a<0).any(1)|(a[:,[0,2]]>w).any(1)|(a[:,[1,3]]>h).any(1)).sum())


def load_inputs(campaign):
    sources=[];models={};roster_roots={};population_stats={}
    accepted=campaign/'completed_verification_receipt.json';sources.append(accepted)
    require(read(accepted)['status']=='PASS_LIMITED_EVALUATION_SCOPE','Accepted original campaign missing')
    for name,modality in (('N','visible'),('T','infrared')):
        root=campaign/'remote_completed_attempt2'/(name+'42_full_attempt1')
        paths={k:root/v for k,v in dict(cache='capture/objects.jsonl.gz',contract='capture_contract.json',
            identity='model_identity.json',population='population.json',summary='summary.json').items()}
        sources.extend(paths.values());c=read(paths['contract']);identity=read(paths['identity'])['capture'];p=read(paths['population']);s=read(paths['summary'])
        require(s['status']=='completed' and s['model']==name+'42' and s['dataset']=='llvip' and s['images']==2406 and s['gt_count']==7879,'Full endpoint differs')
        require(c['expected_val_images']==c['observed_images']==2406 and c['expected_gt']==c['loader_gt']==c['captured_gt']==7879,'Population counts differ')
        require(c['loader_per_image_labels_exact'] is True and c['official_test_accessed'] is False,'Label/test contract differs')
        require(c['evaluator_identity']['torch']=='2.10.0+cu128' and c['evaluator_identity']['ultralytics']=='8.4.115','Pinned evaluator differs')
        expected=dict(imgsz=640,batch=32,workers=4,quantize=None,conf=.001,iou=.7,max_det=300,
            agnostic_nms=False,single_cls=False,rect=True,augment=False,half=False)
        require(c['effective_kwargs']==expected,'Actual evaluation profile differs')
        model_path='/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/llvip/'+modality+'_seed42_native_b32a2/weights/last.pt'
        require(identity['path']==s['checkpoint']==model_path and identity['scope']=='historical_baseline_not_new_protocol_N','Model identity differs')
        require(identity['names']=={'0':'person'} and identity['training_args']['epochs']==200 and identity['training_args']['seed']==42,'Historical model/seed differs')
        aliases=c['alias_to_canonical'];require(len(aliases)==len(set(aliases.values()))==2406,'Alias not bijection')
        with gzip.open(paths['cache'],'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f if line.strip()]
        require(len(rows)==2406 and len({r['image'] for r in rows})==2406,'Duplicate/missing images')
        require({r['image'] for r in rows}==set(aliases),'Captured alias set differs')
        canonical=[aliases[r['image']] for r in rows]
        require(canonical==c['actual_loader_roster'] and set(canonical)==set(c['roster'])==set(p['full_roster']),'Native actual roster/order differs')
        outside=sum(validate_row(r) for r in rows)
        require(sum(len(r['gt_boxes']) for r in rows)==7879,'Full GT denominator differs')
        models[name]={r['image']:r for r in rows};roster_roots[name]='/mnt/dataset/yudongfang/projects/RGBT_campaign/data/processed/llvip/yolo/grouped_v1/'+modality+'/images/dev/'
        population_stats[name]=dict(images=len(rows),GT=7879,predictions=sum(len(r['pred_boxes']) for r in rows),
            empty_predictions=sum(not r['pred_boxes'] for r in rows),confidence_exact_025=sum(x==.25 for r in rows for x in r['pred_confidence']),
            native_predictions_cross_canvas=outside,checkpoint_identity=identity)
    pairpath=campaign/'verified_pair_manifest.jsonl';sources.append(pairpath)
    pairs=[json.loads(s) for s in pairpath.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    require(len(pairs)==len({p['pair_key'] for p in pairs})==2406,'Pair manifest duplicate/missing')
    for p in pairs:
        for name,key in (('N','rgb_image'),('T','ir_image')):
            require(p[key].startswith(roster_roots[name]) and p[key][len(roster_roots[name]):]==p['pair_key'],'Pair relative path differs')
        n,t=models['N'][p['rgb_image']],models['T'][p['ir_image']]
        for key in ('gt_boxes','gt_classes','canvas_shape','original_shape'):require(n[key]==t[key],'Paired GT/order/shape differ: '+key)
        require(p['gt_count']==len(n['gt_boxes']) and p['canvas_shape']==n['canvas_shape'] and p['original_shape']==n['original_shape'],'Pair population differs')
    for name,key in (('N','rgb_image'),('T','ir_image')):require({p[key] for p in pairs}==set(models[name]),'Incomplete paired image set')
    return models,pairs,population_stats,sources


def build_summary(objects,metrics,image_counts,input_stats):
    groups={}
    for group in GROUPS:
        count=Counter(r['groups'][group]['bucket'] for r in objects)
        groups[group]=dict(confidence=GROUPS[group][0],iou=GROUPS[group][1],GT=len(objects),models=metrics[group],
            buckets={b:dict(objects=count[b],fraction=count[b]/len(objects),images=len({r['pair_key'] for r in objects if r['groups'][group]['bucket']==b})) for b in BUCKETS})
        require(sum(count.values())==len(objects),'Bucket sum differs')
        require(count['both_correct']+count['N_only']==metrics[group]['N']['tp'],'N TP/GT bucket differs')
        require(count['both_correct']+count['T_only']==metrics[group]['T']['tp'],'T TP/GT bucket differs')
    transitions={}
    for suffix in ('50','75'):
        left,right='all_iou'+suffix,'gt025_iou'+suffix
        counts=Counter((r['groups'][left]['bucket'],r['groups'][right]['bucket']) for r in objects)
        transitions['iou'+suffix]=[dict(low_threshold_bucket=a,gt025_bucket=b,objects=counts[(a,b)],
            images=len({r['pair_key'] for r in objects if r['groups'][left]['bucket']==a and r['groups'][right]['bucket']==b})) for a in BUCKETS for b in BUCKETS]
    keys=('N_iou50','N_iou75','T_iou50','T_iou75')
    def joint(r):return tuple(r['groups']['gt025_iou'+iou][model]['correct'] for model,iou in (('N','50'),('N','75'),('T','50'),('T','75')))
    counts=Counter(joint(r) for r in objects)
    joint_table=[dict(zip(keys,bits),objects=counts[bits],fraction=counts[bits]/len(objects),
        images=len({r['pair_key'] for r in objects if joint(r)==bits})) for bits in itertools.product((False,True),repeat=4)]
    require(sum(r['objects'] for r in joint_table)==len(objects),'Joint state denominator differs')
    return dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',scope=SCOPE,dataset='llvip',seed=42,images=image_counts,
        paired_GT=len(objects),unpaired_GT=dict(N=0,T=0),input_statistics=input_stats,groups=groups,
        low_to_gt025_transitions=transitions,gt025_joint_iou50_iou75=joint_table,
        denominators='All paired cached GT rows. Each confidence/IoU combination independently matched.',
        model_scope='historical native visible42/infrared42 E200; no FT endpoint',
        matching='Actual accepted native _process_batch/match_predictions; one threshold per invocation, original prediction order retained',
        AP_estimated=False,KD_gain_claim=False,loss_carrier_ranking=False,selector_coverage_inferred=False,
        new_GPU=False,new_model_forward=False,checkpoint_loaded=False,new_hash_computed=False,official_test_accessed=False,
        drone_gap='Full N/C0 six endpoints available locally; full IR42 post-NMS cache not located. No substituted dev200 or new inference.')


def markdown(value):
    lines=['# LLVIP完整dev缓存原生匹配读出','',
        '2406图/7879个配对GT；旧native RGB42/IR42。只在原缓存上保序筛分和独立原生匹配，未推理/训练/NMS重跑/AP计算。结果待独立验收。','',
        '|缓存子集|IoU|双方正确|仅T正确|仅N正确|双方未匹配|N TP/FP/FN|T TP/FP/FN|',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for name,g in value['groups'].items():
        counts=[str(g['buckets'][b]['objects']) for b in BUCKETS]
        models=['/'.join(str(g['models'][m][k]) for k in ('tp','fp','fn')) for m in ('N','T')]
        lines.append('|'+('原低阈值' if g['confidence'] is None else 'score>.25')+'|'+str(g['iou'])+'|'+'|'.join(counts+models)+'|')
    lines+=['','## score>.25的逐GT联合定位状态','',
        '|N@.5|N@.75|T@.5|T@.75|GT|图|','|---|---|---|---|---:|---:|']
    for r in value['gt025_joint_iou50_iou75']:
        if r['objects']:lines.append('|'+ '|'.join(str(r[k]) for k in ('N_iou50','N_iou75','T_iou50','T_iou75','objects','images'))+'|')
    lines+=['','四组分别调用实际native匹配；高IoU匹配不假设是低IoU子集。低阈值→>.25的16格转移、比例、图像覆盖见summary.json；对象和原预测ID在objects.jsonl.gz / prediction_matches.jsonl.gz。',
        '', '仅T匹配是检测互补线索，不是可学习的KD收益或教师oracle。两模态GT数组相同来自LLVIP标签登记，并非独立物理配准证明。缓存不能恢复dense候选、NMS被抑制框、原C门或训练剂量；旧200dev的assigned低置信不是这套真实post-NMS未匹配定义。Drone IR完整缓存缺口仍单列。']
    return '\n'.join(lines)+'\n'


def run(args):
    require(not args.output.exists(),'New output directory required');args.output.mkdir(parents=True)
    started=time.perf_counter()
    try:
        models,pairs,input_stats,sources=load_inputs(args.campaign)
        native=load_native(args.native_source_dir)
        sourcefiles=[args.native_source_dir/name for name in ('3_val.py','4_validator.py','7_metrics.py')]
        sources+=sourcefiles+[Path(__file__),Path(__file__).with_name('native_cached_match.py'),args.protocol]
        write(args.output/'input_manifest.json',dict(scope=SCOPE,sources=[stat(p) for p in sources],groups=GROUPS,new_hash_computed=False))
        snapshots=args.output/'source_copies';snapshots.mkdir()
        for i,p in enumerate(sourcefiles+[Path(__file__),Path(__file__).with_name('native_cached_match.py'),args.protocol]):shutil.copyfile(p,snapshots/(str(i)+'_'+p.name))
        metrics={g:{n:dict(tp=0,fp=0,fn=0,predictions=0) for n in ('N','T')} for g in GROUPS};objects=[]
        with gzip.open(args.output/'prediction_matches.jsonl.gz','xt',encoding='utf-8') as pf:
            for pair in pairs:
                per_model={}
                for name,imagekey in (('N','rgb_image'),('T','ir_image')):
                    row=models[name][pair[imagekey]];conditions={}
                    for group,(cut,iou) in GROUPS.items():
                        m=match_row(row,cut,iou,native);conditions[group]=m
                        for k in ('tp','fp','fn'):metrics[group][name][k]+=m[k]
                        metrics[group][name]['predictions']+=len(m['kept_prediction_ids'])
                    per_model[name]=conditions
                    pred_rows=[]
                    for pi,(box,cl,conf) in enumerate(zip(row['pred_boxes'],row['pred_classes'],row['pred_confidence'])):
                        pred_rows.append(dict(prediction_id=pi,box=box,**{'class':cl},confidence=conf,
                            groups={g:dict(included=pi in set(m['kept_prediction_ids']),gt_row=m['prediction_matches'].get(pi),
                                gt_id=None if pi not in m['prediction_matches'] else pair['pair_key']+'::cached_gt:'+str(m['prediction_matches'][pi])) for g,m in conditions.items()}))
                    pf.write(json.dumps(dict(pair_key=pair['pair_key'],model=name,image=row['image'],predictions=pred_rows),allow_nan=False)+'\n')
                nrow=models['N'][pair['rgb_image']]
                for gi,(box,cl) in enumerate(zip(nrow['gt_boxes'],nrow['gt_classes'])):
                    groups={}
                    for g in GROUPS:
                        values={n:dict(correct=gi in per_model[n][g]['gt_matches'],witness=per_model[n][g]['gt_matches'].get(gi)) for n in ('N','T')}
                        groups[g]=dict(values,bucket=bucket(values['N']['correct'],values['T']['correct']))
                    objects.append(dict(gt_id=pair['pair_key']+'::cached_gt:'+str(gi),pair_key=pair['pair_key'],gt_row=gi,
                        gt_box=box,gt_class=cl,rgb_image=pair['rgb_image'],ir_image=pair['ir_image'],canvas_shape=pair['canvas_shape'],groups=groups))
        with gzip.open(args.output/'objects.jsonl.gz','xt',encoding='utf-8') as f:
            for row in objects:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        value=build_summary(objects,metrics,len(pairs),input_stats);value['seconds']=time.perf_counter()-started
        value['torch_CPU_runtime']=str(torch.__version__);value['source_native_runtime']='accepted torch2.10/ultralytics8.4.115 source functions with local CPU float32'
        write(args.output/'summary.json',value)
        (args.output/'README.md').write_text(markdown(value),encoding='utf-8')
        write(args.output/'completion.json',dict(status=value['status'],scope=SCOPE,seconds=value['seconds'],objects=len(objects),images=len(pairs),new_GPU=False,new_hash_computed=False))
        print(json.dumps(dict(status=value['status'],seconds=value['seconds'],groups=value['groups']),ensure_ascii=False))
    except BaseException as e:
        write(args.output/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-started,new_GPU=False,new_hash_computed=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--native-source-dir',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);run(a)
