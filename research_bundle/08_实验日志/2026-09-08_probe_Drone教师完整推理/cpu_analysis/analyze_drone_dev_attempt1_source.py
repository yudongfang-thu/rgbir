"""CPU-only four-group Drone dev detector readout, independent IR GT association."""
import argparse
from collections import Counter
import gzip,itertools,json,shutil,time,traceback
from pathlib import Path
import torch
from native_cached_match import load_native,match_row
from pairing_core import CLASSES,GROUPS,BUCKETS,require,load_original,gt_pair,bind_images,original_image_fallback,validate_row,bucket,gt_id

HERE=Path(__file__).resolve().parent
SCOPE='DRONE_FULL_DEV_CACHED_NATIVE_PAIRED_OPPORTUNITIES'
B='/mnt/dataset/yudongfang/projects/RGBT_campaign'
N_MODEL=B+'/runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/weights/last.pt'
T_MODEL=B+'/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt'
PROFILE=dict(imgsz=640,batch=32,workers=4,quantize=None,conf=.001,iou=.7,max_det=300,agnostic_nms=False,single_cls=False,rect=True,augment=False,half=False)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def stat(p):
    p=Path(p);s=p.stat();return dict(path=str(p.absolute()),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def lines(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def dump_rows(p,rows):
    with gzip.open(p,'xt',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def fraction(a,b):return a/b if b else None

def load_inputs(a):
    models={};aliases={};sources=[];stats={}
    for name,root,count in [('N',a.n_root,22462),('T',a.t_root,24490)]:
        if name=='N':
            paths=[root/'predictions/objects.jsonl.gz',root/'evaluation_contract.json',root/'evaluation_val.json',root/'reevaluation_receipt.json',root/'origin_manifest.json']
            cache,cpath,spath,rpath,opath=paths;c,s,r,o=map(read,[cpath,spath,rpath,opath])
            require(s['status']==r['status']=='completed' and s['checkpoint']==r['checkpoint']==o['original_checkpoint']==N_MODEL,'Wrong N endpoint')
            require(s['dataset']=='dronevehicle' and s['seed']==42 and s['arm']=='weight0' and s['normalized_method_arm']=='N','Wrong N method identity')
            require(r['full_dev_images']==1469 and r['historical_five_metrics_exact'] is True and r['official_test_accessed'] is False,'N reevaluation not accepted')
            require([x['name'] for x in s['per_class']]==CLASSES and [x['class_id'] for x in s['per_class']]==list(range(5)),'N class order differs')
            identity=dict(path=N_MODEL,stat=o['checkpoint_stat_before'],scope='weight0_native_N42_E200')
        else:
            paths=[root/'capture/objects.jsonl.gz',root/'capture_contract.json',root/'summary.json',root/'model_identity.json',root/'population.json']
            cache,cpath,spath,ipath,ppath=paths;c,s,ident,p=map(read,[cpath,spath,ipath,ppath]);identity=ident['capture']
            require(s['status']=='completed' and s['scope']=='DRONE_IR42_FULL_DEV_CAPTURE' and s['dataset']=='dronevehicle' and s['model']=='T42' and s['canary'] is False,'Wrong T full capture')
            require(s['images']==1469 and s['gt_count']==s['full_gt']==24490 and s['new_hash_computed'] is False,'T full population/scope differs')
            require(s['checkpoint']==identity['path']==T_MODEL and identity['scope']=='historical_native_infrared_teacher_seed42_E200','Wrong T model')
            require(identity['names']==dict(zip(map(str,range(5)),CLASSES)) and identity['training_args']['seed']==42 and identity['training_args']['epochs']==200,'T training/class identity differs')
            require(c['expected_gt']==c['loader_gt']==c['captured_gt']==24490 and c['loader_per_image_labels_exact'] is True,'T independent labels not verified')
            require(p['full_gt']==24490 and p['full_class_counts']==[20588,918,1470,789,725],'T full label inventory differs')
        sources+=paths
        require(c['expected_val_images']==c['observed_images']==1469 and c['official_test_accessed'] is False,'Native full dev contract differs')
        require(c['effective_kwargs']==PROFILE and c['evaluator_identity']['torch']=='2.10.0+cu128' and c['evaluator_identity']['ultralytics']=='8.4.115','Pinned native profile differs')
        rows=lines(cache);require(len(rows)==1469 and len({r['image'] for r in rows})==1469,'Missing or duplicate captured images')
        if name=='N':
            require([r['image'] for r in rows]==c['actual_loader_roster'],'N captured aliases not canonical roster; explicit alias provenance needed')
            alias={r['image']:r['image'] for r in rows}
        else:alias=c['alias_to_canonical']
        require(set(alias)=={r['image'] for r in rows},'Alias/capture population differs')
        require([alias[r['image']] for r in rows]==c['actual_loader_roster'] and set(alias.values())==set(c['roster']),'Actual native order/roster differs')
        outside=sum(validate_row(r) for r in rows);require(sum(len(r['gt_boxes']) for r in rows)==count,'Full own-GT denominator differs')
        models[name]={r['image']:r for r in rows};aliases[name]=alias
        stats[name]=dict(images=len(rows),GT=count,predictions=sum(len(r['pred_boxes']) for r in rows),empty_GT_images=sum(not r['gt_boxes'] for r in rows),empty_prediction_images=sum(not r['pred_boxes'] for r in rows),native_predictions_cross_canvas=outside,identity=identity)
    if a.image_mapping:
        sources.append(a.image_mapping);mapping=read(a.image_mapping);origin='external_explicit_dev_mapping'
    else:
        mapping=original_image_fallback(models,aliases,HERE/'frozen_sources/paired_rgbir_data.py')
        origin='Executed original DualLabelRGBIRDataset.__init__ strong_by_weak=None; remote canonical identity comes from capture contracts. Expected old val JSON did not exist.'
    pairs=bind_images(mapping,models,aliases)
    return models,pairs,stats,sources,mapping,origin

def count_buckets(objects,g):
    count=Counter(r['groups'][g]['bucket'] for r in objects)
    return {b:dict(objects=count[b],fraction=fraction(count[b],len(objects)),images=len({r['pair_key'] for r in objects if r['groups'][g]['bucket']==b})) for b in BUCKETS}

def summarize(objects,all_gt,frames,metrics,inputs):
    unpaired={n:[r for r in all_gt if r['model']==n and r['partner_gt_id'] is None] for n in ('N','T')}
    for n in ('N','T'):require(len(objects)+len(unpaired[n])==inputs[n]['GT'],'Paired + unpaired own GT does not close')
    groups={}
    for g,(cut,iou) in GROUPS.items():
        counts=count_buckets(objects,g);require(sum(x['objects'] for x in counts.values())==len(objects),'Four bucket denominator differs')
        up={n:dict(objects=len(unpaired[n]),images=len({r['image'] for r in unpaired[n]}),correct=sum(r['groups'][g]['correct'] for r in unpaired[n]),unmatched=sum(not r['groups'][g]['correct'] for r in unpaired[n])) for n in ('N','T')}
        for n,side in [('N','N_only'),('T','T_only')]:require(counts['both_correct']['objects']+counts[side]['objects']+up[n]['correct']==metrics[g][n]['tp'],'Pair/unpaired/native TP closure differs')
        perclass=[]
        for cl,name in enumerate(CLASSES):
            selected=[r for r in objects if r['gt_class']==cl]
            perclass.append(dict(class_id=cl,name=name,paired_GT=len(selected),buckets=count_buckets(selected,g),
                unpaired={n:dict(GT=sum(r['gt_class']==cl for r in unpaired[n]),correct=sum(r['gt_class']==cl and r['groups'][g]['correct'] for r in unpaired[n])) for n in ('N','T')}))
        groups[g]=dict(confidence=cut,iou=iou,paired_GT=len(objects),buckets=counts,whole_model_metrics=metrics[g],unpaired=up,per_class=perclass)
    transitions={}
    for suffix in ('50','75'):
        l,r='all_iou'+suffix,'gt025_iou'+suffix;counter=Counter((o['groups'][l]['bucket'],o['groups'][r]['bucket']) for o in objects)
        transitions['iou'+suffix]=[dict(low_threshold_bucket=x,gt025_bucket=y,objects=counter[x,y]) for x in BUCKETS for y in BUCKETS]
    def bits(o):return tuple(o['groups']['gt025_iou'+iou][n]['correct'] for n,iou in [('N','50'),('N','75'),('T','50'),('T','75')])
    keys=('N_iou50','N_iou75','T_iou50','T_iou75');counter=Counter(bits(o) for o in objects)
    joint=[dict(zip(keys,b),objects=counter[b],images=len({o['pair_key'] for o in objects if bits(o)==b})) for b in itertools.product((False,True),repeat=4)]
    return dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',scope=SCOPE,dataset='dronevehicle',seed=42,images=len(frames),paired_GT=len(objects),unpaired_GT={n:len(unpaired[n]) for n in unpaired},input_statistics=inputs,groups=groups,low_to_gt025_transitions=transitions,gt025_joint_iou50_iou75=joint,
        denominator='Four buckets and joint tables use paired GT only; all own-GT native metrics and unpaired GT are separate.',
        GT_pairing='Original same-class maximum-cardinality then maximum-IoU assignment at IoU>=.5 on native canvas; no physical registration claim.',
        AP_estimated=False,KD_gain_claim=False,selector_coverage_inferred=False,new_GPU=False,new_model_forward=False,checkpoint_loaded=False,new_hash_computed=False,official_test_accessed=False)

def run(a):
    require(not a.output.exists(),'New output directory required');a.output.mkdir(parents=True);started=time.perf_counter()
    try:
        models,pairs,inputs,sources,mapping,origin=load_inputs(a);native=load_native(HERE/'frozen_sources');original=load_original(HERE/'frozen_sources/object_evidence_loss.py')
        write(a.output/'image_mapping.json',mapping)
        write(a.output/'image_mapping_provenance.json',dict(origin=origin,images=len(pairs),canonicalization='Accepted remote alias-to-canonical contracts; no local Path.resolve on remote image paths',source=stat(HERE/'frozen_sources/paired_rgbir_data.py'),new_hash_computed=False))
        code=[p for p in HERE.iterdir() if p.is_file()]+list((HERE/'frozen_sources').iterdir())+[HERE.parent/'PLAN_CPU_PAIRING.md']
        write(a.output/'input_manifest.json',dict(scope=SCOPE,input_files=[stat(p) for p in sources],source_files=[stat(p) for p in code],new_hash_computed=False))
        copy=a.output/'source_copies';copy.mkdir()
        for i,p in enumerate(code):shutil.copyfile(p,copy/(str(i)+'_'+p.name))
        metrics={g:{n:dict(tp=0,fp=0,fn=0,predictions=0) for n in ('N','T')} for g in GROUPS};objects=[];all_gt=[];frames=[]
        with gzip.open(a.output/'prediction_matches.jsonl.gz','xt',encoding='utf-8') as pf:
            for pair in pairs:
                rows={'N':models['N'][pair['rgb_image']],'T':models['T'][pair['ir_image']]};matched=gt_pair(rows['N'],rows['T'],original)
                mapping={'N':{x['N_gt_row']:x['T_gt_row'] for x in matched},'T':{x['T_gt_row']:x['N_gt_row'] for x in matched}}
                conditions={n:{g:match_row(rows[n],cut,iou,native) for g,(cut,iou) in GROUPS.items()} for n in ('N','T')}
                frames.append(dict(pair,canvas_shape=rows['N']['canvas_shape'],original_shape=rows['N']['original_shape'],N_GT=len(rows['N']['gt_boxes']),T_GT=len(rows['T']['gt_boxes']),paired_GT=len(matched),GT_pairs=matched))
                for n in ('N','T'):
                    row=rows[n];other='T' if n=='N' else 'N'
                    for g,m in conditions[n].items():
                        for k in ('tp','fp','fn'):metrics[g][n][k]+=m[k]
                        metrics[g][n]['predictions']+=len(m['kept_prediction_ids'])
                    pred=[]
                    for pi,(box,cl,conf) in enumerate(zip(row['pred_boxes'],row['pred_classes'],row['pred_confidence'])):
                        pred.append(dict(prediction_id=pi,box=box,pred_class=int(cl),confidence=conf,groups={g:dict(included=pi in m['kept_prediction_ids'],gt_row=m['prediction_matches'].get(pi),gt_id=gt_id(row['image'],m['prediction_matches'][pi]) if pi in m['prediction_matches'] else None) for g,m in conditions[n].items()}))
                    pf.write(json.dumps(dict(pair_key=pair['pair_key'],model=n,image=row['image'],predictions=pred),allow_nan=False)+'\n')
                    for gi,(box,cl) in enumerate(zip(row['gt_boxes'],row['gt_classes'])):
                        partner=mapping[n].get(gi)
                        all_gt.append(dict(model=n,image=row['image'],gt_id=gt_id(row['image'],gi),gt_row=gi,gt_box=box,gt_class=int(cl),partner_gt_id=None if partner is None else gt_id(rows[other]['image'],partner),groups={g:dict(correct=gi in m['gt_matches'],witness=m['gt_matches'].get(gi),partner_correct=None if partner is None else partner in conditions[other][g]['gt_matches']) for g,m in conditions[n].items()}))
                for match in matched:
                    ni,ti=match['N_gt_row'],match['T_gt_row'];groups={}
                    for g in GROUPS:
                        states={n:dict(correct=gi in conditions[n][g]['gt_matches'],witness=conditions[n][g]['gt_matches'].get(gi)) for n,gi in [('N',ni),('T',ti)]}
                        groups[g]=dict(states,bucket=bucket(states['N']['correct'],states['T']['correct']))
                    objects.append(dict(pair_key=pair['pair_key'],N_gt_id=gt_id(rows['N']['image'],ni),T_gt_id=gt_id(rows['T']['image'],ti),N_gt_row=ni,T_gt_row=ti,N_gt_box=rows['N']['gt_boxes'][ni],T_gt_box=rows['T']['gt_boxes'][ti],gt_class=int(rows['N']['gt_classes'][ni]),pair_iou=match['iou'],groups=groups))
        dump_rows(a.output/'paired_objects.jsonl.gz',objects);dump_rows(a.output/'all_gt_objects.jsonl.gz',all_gt);dump_rows(a.output/'frames.jsonl.gz',frames)
        value=summarize(objects,all_gt,frames,metrics,inputs);value['seconds']=time.perf_counter()-started;value['torch_CPU_runtime']=str(torch.__version__)
        write(a.output/'summary.json',value)
        md=['# Drone完整dev缓存配对检出读出','',f"配对GT {len(objects)}；未配对RGB {value['unpaired_GT']['N']} / IR {value['unpaired_GT']['T']}。仅缓存CPU读出，待独立验收。",'', '|组|双方正确|仅T正确|仅N正确|双方未匹配|','|---|---:|---:|---:|---:|']
        for g,v in value['groups'].items():md.append('|'+g+'|'+'|'.join(str(v['buckets'][b]['objects']) for b in BUCKETS)+'|')
        md+=['','上述分母仅配对GT。所有GT、双方未配对、逐类、原预测ID与四组独立native匹配见JSON小产物。T始终在IR ownGT评价，再映到配对RGB对象；标签配对不证明物理配准。不是AP、可蒸馏收益或训练选择覆盖。']
        (a.output/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
        write(a.output/'completion.json',dict(status=value['status'],scope=SCOPE,seconds=value['seconds'],images=len(frames),paired_GT=len(objects),new_GPU=False,new_hash_computed=False));print(json.dumps(dict(status=value['status'],seconds=value['seconds'],paired_GT=len(objects))))
    except BaseException as e:
        write(a.output/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n-root',type=Path,required=True);p.add_argument('--t-root',type=Path,required=True);p.add_argument('--image-mapping',type=Path);p.add_argument('--output',type=Path,required=True)
    torch.set_num_threads(2);run(p.parse_args())
