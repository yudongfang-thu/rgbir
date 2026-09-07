"""Independent CPU closure review of persisted natural-flow results; no model inference."""
import argparse, collections, importlib.util, json, math, sys, time, traceback
from pathlib import Path
import torch, yaml
HERE=Path(__file__).resolve().parent
TASK=HERE.parent
WORKSPACE=HERE.parents[3]
RELEASE=WORKSPACE/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
OLD=WORKSPACE/'08_实验日志/2026-09-07_train_IndependentKD实施/remote_cpu1'
sys.path[:0]=[str(RELEASE),str(RELEASE/'task_conditional_reference')]
from localization_loss import build_localization_selection, LocalizationConfig
COUNT_KEYS=('rgb_gt_count','teacher_gt_count','common_count','pair_iou_count','geometry_count','inside_count',
 'support_count','unique_owner_count','reference_candidate_count','base_count','reference_reliable_count',
 'both_reliable_count','reference_localization_gap_count','teacher_rgb_quality_count','teacher_own_quality_count',
 'eligible_count','selected_count','selected_quality_count')
GT_KEYS=COUNT_KEYS[:8]
POST_KEYS=COUNT_KEYS[10:]

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]
def require(ok,message):
    if not ok:raise AssertionError(message)
def eq(a,b,message):require(a==b,message+' actual='+str(a)[:300]+' expected='+str(b)[:300])
def plain(counter):return dict(counter)
def close(a,b):return math.isfinite(a) and abs(a-b)<=1e-12

def context(records,images,selected=False):
    items=[r for r in records if not selected or r['selected']]
    pos=sorted({r['batch_index'] for r in items})
    return dict(objects=len(items),class_counts=dict(collections.Counter(str(r['class']) for r in items)),
        image_positions=pos,unique_images=len(pos),source_groups=sorted({images[p]['governed_group'] for p in pos if images[p]['governed_group'] is not None}),
        unknown_group_images=sum(images[p]['governed_group'] is None for p in pos),
        stride_counts=dict(collections.Counter(str(r['stride']) for r in items)))

def gt_recompute(trace,cfg,nc):
    def labels(prefix):
        classes=[];boxes=[];indices=[]
        for i,row in enumerate(trace):
            classes+=row[prefix+'_classes'];boxes+=row[prefix+'_boxes_normalized_xywh'];indices += [i]*len(row[prefix+'_classes'])
        return dict(batch_idx=torch.tensor(indices,dtype=torch.long),cls=torch.tensor(classes,dtype=torch.float32).reshape(-1,1),
                    bboxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4))
    batch=labels('rgb');batch['teacher_batch']=labels('ir');batch['im_file']=[r['image'] for r in trace]
    raw=dict(scores=torch.full((32,nc,8400),-100.),boxes=torch.zeros(32,64,8400),
        feats=[torch.zeros(32,1,s,s) for s in (80,40,20)])
    with torch.no_grad():out=build_localization_selection(raw,raw,batch,config=LocalizationConfig(**cfg),geometry_eligible=None,geometry_verified=False)
    eq(out.stats['base_count'],0,'Synthetic score sentinel must not create candidates')
    return {k:out.stats[k] for k in GT_KEYS}

def post_gate(record,cfg):
    gate=record['reference_correct'];out={'reference_reliable_count':int(gate)}
    gate=gate and record['teacher_correct'];out['both_reliable_count']=int(gate)
    gate=gate and record['reference_iou']<cfg['reference_iou_max'];out['reference_localization_gap_count']=int(gate)
    gate=gate and record['teacher_rgb_iou']>=cfg['teacher_rgb_iou_min'];out['teacher_rgb_quality_count']=int(gate)
    gate=gate and record['teacher_ir_iou']>=cfg['teacher_ir_iou_min'];out['teacher_own_quality_count']=int(gate)
    gate=gate and record['teacher_rgb_iou']-record['reference_iou']>cfg['localization_margin']
    out.update(eligible_count=int(gate),selected_count=int(gate),selected_quality_count=int(gate))
    return out

def inspect_run(root,key,batches):
    run=root/(key+('_canary_attempt1' if batches==2 else '_full_attempt1'))
    summary=read(run/'summary.json');cfg=yaml.safe_load((run/'input_config.yaml').read_text(encoding='utf-8'))
    trace=rows(run/'natural_batches.jsonl');selection=rows(run/'selection_batches.jsonl')
    old=rows(OLD/('coverage_'+key+'_attempt1')/'natural_batches.jsonl')
    eq(trace,old[:batches],key+' original natural trace')
    eq(rows(run/'original_natural_batches.jsonl'),old,key+' original full trace copy')
    eq(len(selection),batches,'selection rows');eq(summary['batches'],batches,'summary batches')
    eq(summary['status'],'COMPLETED','summary status');eq(summary['diagnostic_status'],'UNVERIFIED_GEOMETRY_DIAGNOSTIC','status scope')
    eq(summary['dataset'],{'llvip':'llvip','drone':'dronevehicle'}[key],'dataset');eq(summary['seed'],20260907,'seed')
    eq(summary['canary_only'],batches==2,'canary');eq(summary['pixel_bytes_comparable'],False,'pixel scope')
    for name in ('backward_executed','training_executed','calibration_executed','validation_or_test_accessed','new_hash_computed'):
        eq(summary[name],False,name)
    require(summary['trace_exact_all_recorded_fields'],'trace scope')
    eq(summary['localization']['geometry_verified'],False,'geometry');eq(summary['localization']['formal_L1_admitted'],False,'L admission')
    eq(summary['classification']['C0_C1_selection_identical'],True,'C equality')
    manifest=read(run/'input_manifest.json');eq(manifest['hash_computed'],False,'input no hash')
    source_checks=[]
    sealed=read(HERE/'attempt2_source_stat_manifest.json')['sources']
    snapshot_by_name={Path(x['source']).name:Path(x['snapshot']) for x in sealed}
    for item in manifest['sources']:
        snap=run/'sources'/Path(item['snapshot']).name
        original=Path(item['source'])
        if original.name=='diagnose_natural_flow.py':local=snapshot_by_name[original.name]
        else:
            relative=str(original).replace('\\','/').split('/release_gpu5/',1)[1];local=RELEASE/relative
        require(snap.read_bytes()==local.read_bytes(),'source bytes: '+str(snap))
        eq(snap.stat().st_size,item['bytes'],'source bytes declared')
        source_checks.append(dict(snapshot=str(snap),reference=str(local),byte_identity=True,bytes=snap.stat().st_size))
    identities=read(run/'model_identity.json')
    for role in ('teacher','reference'):
        m=identities[role];eq(m['path'],cfg[role],'model path');eq(m['stat'],manifest['inputs'][cfg[role]],'model stat')
        eq(m['args']['seed'],42,'aux seed');eq(m['args']['epochs'],200,'aux epochs');eq(m['model_eval'],True,'frozen eval');eq(m['requires_grad'],False,'frozen gradient')
        eq(len(m['names']),cfg['expected_nc'],'class size')
    C=collections.Counter();L=collections.Counter();classes=collections.defaultdict(collections.Counter)
    c_images=set();c_groups=set();gimages=collections.defaultdict(set);ggroups=collections.defaultdict(set)
    c_nonzero=0;l_nonzero=0;class_l=collections.defaultdict(collections.Counter);gt_total=collections.Counter()
    for bi,(tr,s) in enumerate(zip(trace,selection)):
        eq(s['batch'],bi,'batch');eq(s['actual_B'],32,'B32');require(s['trace_exact'],'batch exact')
        eq(len(tr['images']),32,'trace B32');images=s['images'];eq(len(images),32,'image population')
        for i,(im,t) in enumerate(zip(images,tr['images'])):
            eq(im['image'],t['image'],'selection image');eq(im['rgb_source'],t['rgb_source'],'selection source')
            eq(t['position'],i,'position')
        rgb_ids=[(i,int(y)) for i,t in enumerate(tr['images']) for y in t['rgb_classes']]
        ir_ids=[(i,int(y)) for i,t in enumerate(tr['images']) for y in t['ir_classes']]
        c=s['classification'];cs=c['counts'];cr=c['base_records'];cchoose=[i for i,r in enumerate(cr) if r['selected']]
        eligible=[i for i,r in enumerate(cr) if r['eligible']]
        expected=sorted(eligible,key=lambda i:-cr[i]['quality'])[:math.ceil(cfg['evidence']['rho']*len(eligible))]
        eq(set(cchoose),set(expected),'C global stable rank/quota');eq(cs['base_count'],len(cr),'C base')
        eq(cs['eligible_count'],len(eligible),'C eligible');eq(cs['selected_count'],len(expected),'C selected')
        eq(cs['normalizer'],max(1,len(cr)),'C normalizer');require(close(cs['nominal_dose'],len(expected)/max(1,len(cr))),'C dose')
        cc={str(y):dict(base=sum(r['class']==y for r in cr),eligible=sum(r['class']==y and r['eligible'] for r in cr),selected=sum(r['class']==y and r['selected'] for r in cr)) for y in sorted({r['class'] for r in cr})}
        eq(c['class_counts'],cc,'C perclass')
        positions=sorted({r['batch_index'] for r in cr if r['selected']});eq(c['selected_image_positions'],positions,'C images')
        eq(c['selected_unique_images'],len(positions),'C image count')
        eq(c['selected_source_groups'],sorted({images[p]['governed_group'] for p in positions if images[p]['governed_group'] is not None}),'C groups')
        for r in cr:
            eq(r['image'],images[r['batch_index']]['image'],'C record identity');eq(r['governed_group'],images[r['batch_index']]['governed_group'],'C group identity')
            eq(rgb_ids[r['rgb_gt_index']],(r['batch_index'],r['class']),'C RGB GT class identity')
            eq(ir_ids[r['ir_gt_index']],(r['batch_index'],r['class']),'C IR GT class identity')
            require(not r['eligible'] or r['quality']>0,'C eligible quality');require(any(r['valid_levels']),'C base validlevel')
        C.update({k:v for k,v in cs.items() if k.endswith('_count')});c_nonzero+=bool(expected)
        for y,x in cc.items():classes[y].update(x)
        for p in positions:
            c_images.add(images[p]['rgb_source'])
            if images[p]['governed_group'] is not None:c_groups.add(images[p]['governed_group'])
        l=s['localization'];lr=l['base_records'];per=s['localization_per_image'];eq(len(per),32,'L perimage population')
        for field in ('geometry_verified','geometry_mask_supplied','calibration','training_admitted'):eq(l[field],False,'L '+field)
        eq(l['per_image_replay_counts_and_ids_exact'],True,'real L replay');eq(l['mode'],'teacher','L mode');eq(l['input_dtype'],'FP32','FP32')
        eq(l['normalizer'],max(1,len(lr)),'L normalizer');eq(l['base_count'],len(lr),'L base');eq(l['reference_candidate_count'],len(lr),'L reference candidate base')
        for context_name,chosen in [('base_context',False),('selected_context',True)]:eq(l[context_name],context(lr,images,chosen),'L '+context_name)
        pcounts=collections.Counter();byposition=collections.defaultdict(collections.Counter)
        chosen=[]
        for r in lr:
            require(not any(k.startswith('reference_logit_') or k.endswith('_dfl_ce') or k.endswith('_dfl_entropy') for k in r),'omitted learning proxies')
            eq(r['image_id'],images[r['batch_index']]['image'],'L record identity')
            eq(rgb_ids[r['rgb_gt_index']],(r['batch_index'],r['class']),'L RGB GT class identity')
            eq(ir_ids[r['ir_gt_index']],(r['batch_index'],r['class']),'L IR GT class identity')
            require(0<=r['anchor_index']<8000 and r['stride'] in (8.,16.),'L P3/P4 anchor')
            require(all(0<=x<=15-cfg['localization']['support_epsilon'] for k in ('rgb_distances','ir_distances') for x in r[k]),'L DFL support')
            require(r['reference_conf']>=cfg['localization']['reference_conf'] and r['reference_iou']>=cfg['localization']['reference_iou'],'R candidate thresholds')
            require(r['pair_iou']>=cfg['localization']['pair_iou'],'pair threshold')
            if r['reference_correct']:require(r['reference_conf']>=cfg['localization']['reliable_conf'],'R reliable conf')
            if r['teacher_correct']:require(r['teacher_conf']>=cfg['localization']['reliable_conf'],'T reliable conf')
            gates=post_gate(r,cfg['localization']);pcounts.update(gates);byposition[r['batch_index']].update(gates)
            eq(r['quality_gate'],bool(gates['eligible_count']),'quality from IoU/conf record');eq(r['selected'],r['quality_gate'],'teacher selected')
            if r['selected']:chosen.append(r)
            class_l[str(r['class'])].update(base=1,selected=int(r['selected']))
        for k in POST_KEYS:eq(l[k],pcounts[k],'L postgate '+k)
        eq(l['selected_anchors'],[[r['batch_index'],r['anchor_index'],r['rgb_gt_index']] for r in chosen],'L selected anchors')
        eq(l['selected_object_ids'],[[r['batch_index'],r['rgb_gt_index'],r['ir_gt_index']] for r in chosen],'L selected IDs')
        require(close(l['nominal_dose'],len(chosen)/max(1,len(lr))),'L dose')
        totals=collections.Counter()
        for p,item in enumerate(per):
            eq(item['position'],p,'perimage position');eq(item['image'],images[p]['image'],'perimage image')
            eq(item['rgb_source'],images[p]['rgb_source'],'perimage source');eq(item['governed_group'],images[p]['governed_group'],'perimage group')
            for k,v in item['gate_counts'].items():
                require(isinstance(v,int) and v>=0,'nonnegative gate count');totals[k]+=v
                if k in POST_KEYS:eq(v,byposition[p][k],'perimage postgate')
                if v:
                    gimages[k].add(images[p]['rgb_source'])
                    if images[p]['governed_group'] is not None:ggroups[k].add(images[p]['governed_group'])
        for k in COUNT_KEYS:eq(totals[k],l[k],'all gate perimage->batch '+k)
        gt=gt_recompute(tr['images'],cfg['localization'],cfg['expected_nc'])
        for k,v in gt.items():eq(l[k],v,'CPU GT-only recompute '+k)
        gt_total.update(gt);L.update({k:l[k] for k in COUNT_KEYS});l_nonzero+=bool(chosen)
        if (bi+1)%8==0:print(key,batches,'verified_batches',bi+1,flush=True)
    eq(dict(C),summary['classification']['counts'],'C summary totals');eq(dict(L),summary['localization']['counts'],'L summary totals')
    eq(c_nonzero,summary['classification']['selected_batches'],'C selected batches');eq(l_nonzero,summary['localization']['selected_batches'],'L selected batches')
    eq(sorted(c_images),summary['classification']['selected_images'],'C all images');eq(len(c_images),summary['classification']['selected_unique_images'],'C all image count')
    eq(sorted(c_groups),summary['classification']['selected_source_groups'],'C all groups');eq({k:dict(v) for k,v in classes.items()},summary['classification']['class_counts'],'C summary classes')
    for k in COUNT_KEYS:eq(summary['localization']['each_gate'][k],dict(objects=L[k],unique_images=len(gimages[k]),images=sorted(gimages[k]),source_groups=sorted(ggroups[k])),'L summary gate '+k)
    peak_values=list(summary['resources']['per_gpu_peak_vram_mib'].values())+[summary['resources']['peak_rss_mib'],summary['gpu_reserved_peak_mib'],summary['gpu_allocated_peak_mib']]
    require(peak_values and all(math.isfinite(v) and v>0 for v in peak_values),'finite positive resource measurements')
    return dict(status='PASS',dataset=key,batches=batches,trace_images=batches*32,source_checks=source_checks,
        classification_counts=dict(C),localization_counts=dict(L),L_class_counts={k:dict(v) for k,v in class_l.items()},
        C_class_counts={k:dict(v) for k,v in classes.items()},cpu_gt_only_gate_counts=dict(gt_total),
        resources=summary['resources'],gpu_reserved_peak_mib=summary['gpu_reserved_peak_mib'],gpu_allocated_peak_mib=summary['gpu_allocated_peak_mib'])

def review_resources(root):
    reports=[]
    for key in ('llvip','drone'):
        for kind in ('canary','full'):
            prefix='natural_'+key+'_'+kind;queue=root/'queue_attempt1'
            admission=read(queue/(prefix+'_admission.json'));profile=read(queue/(prefix+'_resource_profile.json'))
            eq(profile['status'],'COMPLETED','resource completion');eq(profile['exit_code'],0,'resource exit');eq(profile['monitor_errors'],[],'monitor')
            require(len(admission['active_gpus_after'])<=4,'project GPU cap')
            if len(admission['active_gpus_after'])==4:
                require(admission['four_gpu_exception'] and len(admission['empty_gpus_after'])>=2,'fourGPU exception')
            samples=profile['samples'];require(bool(samples),'sample evidence')
            require(min(s['memory_free_mib'] for s in samples)>=2048,'2GiB margin')
            require(max(s['project_rss_mib'] for s in samples)<=300*1024,'300GiB RSS')
            require(profile['launch']['cuda_processes_per_gpu']<=3,'perGPU process cap')
            eq(profile['minimum_free_mib'],min(s['memory_free_mib'] for s in samples),'minimum sampled memory')
            reports.append(dict(job=prefix,admission=admission,min_sampled_free_mib=profile['minimum_free_mib'],
                max_sampled_project_rss_mib=max(s['project_rss_mib'] for s in samples),samples=len(samples),launch=profile['launch']))
    return reports

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False);torch.set_num_threads(1);started=time.time()
    try:
        results=[]
        for key in ('llvip','drone'):
            for n in (2,64):results.append(inspect_run(args.input,key,n))
        resources=review_resources(args.input);eq(read(args.input/'queue_attempt1/completion.json')['status'],'COMPLETED','campaign complete')
        result=dict(status='PASS',date='2026-09-07',auditor='/root/ap_error',runs=results,resource_review=resources,
            real_forward_independently_recomputed=False,gt_only_prebase_recomputed=True,postbase_gates_recomputed=True,
            missing_all_anchor_logits_boundary='Reference-candidate existence/rank and C q values cannot be recomputed from unpersisted logits.',
            torch=torch.__version__,cuda_initialized=torch.cuda.is_initialized(),gpu_or_ssh_used=False,new_hash_computed=False,seconds=time.time()-started)
    except BaseException as error:
        result=dict(status='FAIL',error=repr(error),traceback=traceback.format_exc(),seconds=time.time()-started)
    (args.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','resource_review')},ensure_ascii=False))
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
