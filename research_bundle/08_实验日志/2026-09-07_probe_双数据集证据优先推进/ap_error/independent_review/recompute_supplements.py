"""CPU review of class-oracle arithmetic and all raw baseline object states.
No TIDE engine execution, no hashes, no writes outside independent_review.
"""
import collections,gzip,json,math,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def stat(p):
    p=Path(p);s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def dump(name,x):(HERE/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def same(a,b):assert abs(float(a)-float(b))<1e-9,(a,b)
started=time.time()

# Resolve the prior single wording warning without changing the old audit.
wording_paths=[ROOT/'README.md',ROOT/'README_DRONE.md',ROOT/'summarize_drone.py']
for p in wording_paths:
    content=p.read_text(encoding='utf-8')
    assert '六类main error中' in content and 'special FP' in content
old_review=read(HERE/'EXPERIMENT_AUDIT.json')
assert old_review['reason_code']=='NUMERIC_CHECKS_PASS_MAIN_VS_SPECIAL_HEADLINE_NEEDS_QUALIFIER'
dump('FINAL_SCOPE_ACCEPTANCE.json',{
    'date':'2026-09-07','auditor':'/root/loc_stress','overall_verdict':'pass','integrity_status':'pass',
    'scope':'Existing six-endpoint Drone TIDE descriptive AP/oracle/sorting analysis only',
    'supersedes_warning_only':'EXPERIMENT_AUDIT.json:largest_error_oracle_headline',
    'old_audit_preserved':True,'warning_resolution':'Both reports and generation source now limit the maximum claim to six main errors and separately disclose special FP.',
    'wording_sources':[stat(p) for p in wording_paths],'numeric_evidence':'recompute_receipt.json','additional_truth':'tie_truth_receipt.json',
    'unchanged_limits':['No LLVIP inputs accepted here','Not a claim of default COCOeval bitwise equality','No teacher/student learnability or causal KD gains','No L1 or training admission'],
    'new_hashes_computed':False})

classdir=ROOT/'class_oracle_v1';bridgedir=ROOT/'baseline_class_bridge_v2'
cs=read(classdir/'summary.json');main=read(ROOT/'drone_results_v1/summary.json')['results'];bs=read(bridgedir/'summary.json')
rawpath=Path(bs['input']['path']);oldsource=Path(bs['source']['path'])
observed=[ROOT/'class_oracle_supplement.py',classdir/'runner_source.py',classdir/'summary.json',
          ROOT/'bridge_baseline_classes.py',bridgedir/'runner_source.py',bridgedir/'summary.json',
          bridgedir/'object_state_bridge.jsonl',bridgedir/'analyze_baseline_probe_source.py',oldsource,rawpath,
          ROOT/'BRIDGE_PROTOCOL.md',classdir/'README.md',bridgedir/'README.md']
before={str(p):stat(p) for p in observed}
assert (classdir/'runner_source.py').read_bytes()==(ROOT/'class_oracle_supplement.py').read_bytes()
assert (bridgedir/'runner_source.py').read_bytes()==(ROOT/'bridge_baseline_classes.py').read_bytes()
assert (bridgedir/'analyze_baseline_probe_source.py').read_bytes()==oldsource.read_bytes()
classchecks={};minor={}
for name,end in cs.items():
    assert end==read(classdir/(name+'.json'))
    classchecks[name]={}
    for t,r in end.items():
        ref=main[name]['thresholds'][t];per=r['per_class']
        assert set(per)==set(map(str,range(5)))
        assert sum(v['gt'] for v in per.values())==22462
        for c,v in per.items():
            assert v['gt']==ref['independent_ap']['per_class'][c]['gt']
            same(v['AP'],ref['independent_ap']['per_class'][c]['AP'])
            same(v['Cls_oracle_AP']-v['AP'],v['Cls_oracle_dAP'])
        original=float(np.mean([v['AP'] for v in per.values()]))
        fixed=float(np.mean([v['Cls_oracle_AP'] for v in per.values()]))
        delta=float(np.mean([v['Cls_oracle_dAP'] for v in per.values()]))
        same(original,ref['AP']);same(max(delta,0),ref['main_dAP']['Cls']);same(r['macro_Cls_dAP'],ref['main_dAP']['Cls']);same(fixed-original,delta)
        matrices=r['confusion_counts_predicted_row_true_column']
        for label,bins in [('all',slice(None)),('ge25',slice(3,None)),('ge50',slice(4,None))]:
            m=np.array(matrices[label]);assert m.shape==(5,5) and np.all(m>=0) and np.all(np.diag(m)==0)
            for c in range(5):
                expected=sum(b['error_counts'].get('Cls',0) for b in ref['score_diagnostic'][str(c)]['bins'][bins])
                assert int(m[c].sum())==expected
            if label=='all':assert int(m.sum())==ref['error_counts']['Cls']
        classchecks[name][t]={'macro_AP':original,'macro_Cls_oracle_AP':fixed,'macro_Cls_dAP':delta,'per_class_and_main_closure':True,'matrix_row_and_all_Cls_count_closure':True}
for t in ('0.5','0.75'):
    for arm in ('N','C0'):
        sums=np.array([[cs[f'{arm}_s{s}'][t]['per_class'][str(c)]['Cls_oracle_dAP'] for c in range(5)] for s in (0,42,123)])
        ratio=sums[:,[1,2,4]].sum(1)/sums.sum(1)
        pooled=sums[:,[1,2,4]].mean(0).sum()/sums.mean(0).sum()
        minor[t+'_'+arm]={'per_seed_fraction_freight_truck_van':ratio.tolist(),'ratio_of_three_seed_means':float(pooled)}

# Verify all report table means, sample SD and paired class deltas from JSON.
table=[]
for line in (classdir/'README.md').read_text(encoding='utf-8').splitlines():
    if line.startswith('|0.'):
        f=line.strip('|').split('|');t,arm,classname=f[:3]
        c=next(k for k,v in cs[f'{arm}_s0'][t]['per_class'].items() if v['name']==classname)
        ap=[cs[f'{arm}_s{s}'][t]['per_class'][c]['AP'] for s in (0,42,123)]
        dp=[cs[f'{arm}_s{s}'][t]['per_class'][c]['Cls_oracle_dAP'] for s in (0,42,123)]
        dif=[cs[f'C0_s{s}'][t]['per_class'][c]['Cls_oracle_dAP']-cs[f'N_s{s}'][t]['per_class'][c]['Cls_oracle_dAP'] for s in (0,42,123)]
        assert f[4]==f'{np.mean(ap):.4f} ± {np.std(ap,ddof=1):.4f}'
        assert f[5]==f'{np.mean(dp):.4f} ± {np.std(dp,ddof=1):.4f}'
        assert f[6]==', '.join(f'{x:+.4f}' for x in dif)
        table.append((t,arm,c))
assert len(table)==20

def one_iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]));ih=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=iw*ih;return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
geom_errors=collections.defaultdict(list)
def view(r,model):
    m=r.get(model) or {};a=m.get('assigned')
    if not a:return {'candidate':False,'confidence_ok':False,'class_ok':False,'localization_ok':False,'correct':False,'state':'no_candidate'}
    val=m.get('iou_to_rgb_gt',a.get('iou_to_rgb_gt'))
    loc=val is not None and float(val)>=.5
    geometry=one_iou(a['box'],r['gt_box_input'])
    if val is not None:
        geom_errors[model].append(abs(geometry-val));assert loc==(geometry>=.5)
    conf=a['confidence']>=.25;cls=a.get('pred_class',a.get('class'))==r['class']
    state='low_confidence' if not conf else ('class_and_localization' if not cls and not loc else ('class_only' if not cls else ('localization_only' if not loc else 'correct')))
    return {'candidate':True,'confidence_ok':conf,'class_ok':cls,'localization_ok':loc,'correct':conf and cls and loc,'state':state}

allraw=lines(rawpath);raw=[r for r in allraw if r['split']=='val' and not r.get('is_background',False)]
devimages=set(r['image'] for r in allraw if r['split']=='val');gtimages=set(r['image'] for r in raw)
assert (len(raw),len(devimages),len(gtimages))==(3084,200,194)
assert sorted(devimages-gtimages)==bs['images_without_gt']
saved=lines(bridgedir/'object_state_bridge.jsonl');assert len(saved)==len(raw)
rebuilt=[]
views=[]
for r,previous in zip(raw,saved):
    vv={m:view(r,m) for m in ('N42','T42','N0')};views.append(vv)
    nc,tc,zc=[vv[m]['correct'] for m in ('N42','T42','N0')];paired=r.get('paired_gt_iou') is not None and r['paired_gt_iou']>=.5
    new={'object_id':r['object_id'],'image':r['image'],'gt_class':r['class'],'paired':paired,
         'N_state':vv['N42']['state'],'T_state':vv['T42']['state'],'N0_state':vv['N0']['state'],
         'T_repair':paired and not nc and tc,'T_harm':paired and nc and not tc,'N0_repair':not nc and zc,'N0_harm':nc and not zc}
    assert new==previous,(r['object_id'],new,previous)
    rebuilt.append(new)
state_names=['no_candidate','low_confidence','class_only','localization_only','class_and_localization','correct']
bridgechecks={}
for key,result in bs['results'].items():
    indices=[i for i,r in enumerate(rebuilt) if key=='all' or r['gt_class']==int(key)]
    sel=[rebuilt[i] for i in indices];n=len(sel)
    correct=sum(r['N_state']=='correct' for r in sel);errors=n-correct
    assert (result['gt'],result['N_correct'],result['N_errors'])==(n,correct,errors)
    for state in state_names:
        assert result['N_states'][state]==sum(r['N_state']==state for r in sel)
        for model in ('T','N0'):assert result['repairs_by_state'][model][state]==sum(r['N_state']==state and r[model+'_repair'] for r in sel)
    paired=[r for r in sel if r['paired']]
    common={'gt':len(paired),'N_errors':sum(r['N_state']!='correct' for r in paired),'N_correct':sum(r['N_state']=='correct' for r in paired),
            'both_repair':sum(r['T_repair'] and r['N0_repair'] for r in paired),
            'T_only_repair':sum(r['T_repair'] and not r['N0_repair'] for r in paired),
            'N0_only_repair':sum(r['N0_repair'] and not r['T_repair'] for r in paired)}
    assert common==result['common_eligible']
    for model,select,require_pair in [('T',sel,True),('N0',sel,False),('N0_on_T_eligible',paired,False)]:
        prefix='T' if model=='T' else 'N0';data=result[model]
        repair=sum(r[prefix+'_repair'] for r in select);harm=sum(r[prefix+'_harm'] for r in select)
        den=len(select);err=sum(r['N_state']!='correct' for r in select)
        assert data['all_rgb_gt']==den and data['repair']['n']==repair and data['harm_if_teacher_replaces_native']['n']==harm
        same(data['repair']['fraction_all_gt'],repair/den);same(data['repair']['fraction_native_errors'],repair/err)
        assert data['comparison_eligible_gt']['n']==(len(paired) if require_pair else den)
    assert common['both_repair']+common['T_only_repair']==result['T']['repair']['n']
    assert common['both_repair']+common['N0_only_repair']==result['N0_on_T_eligible']['repair']['n']
    bridgechecks[key]={'gt':n,'N_correct':correct,'N_errors':errors,'common_eligible':common,
                      'T_repair':result['T']['repair']['n'],'T_harm':result['T']['harm_if_teacher_replaces_native']['n'],
                      'N0_repair':result['N0']['repair']['n'],'N0_harm':result['N0']['harm_if_teacher_replaces_native']['n']}
for field in ('gt','N_correct','N_errors','T_repair','T_harm','N0_repair','N0_harm'):
    assert sum(bridgechecks[str(c)][field] for c in range(5))==bridgechecks['all'][field]
for field in bridgechecks['all']['common_eligible']:
    assert sum(bridgechecks[str(c)]['common_eligible'][field] for c in range(5))==bridgechecks['all']['common_eligible'][field]

# Rebuild N_s0 Cls matrix directly from raw boxes and score-greedy assignment, without invoking TIDE.
entry=next(e for e in read(ROOT/'drone_results_v1/input_manifest.json') if e['name']=='N_s0')
with gzip.open(entry['path'],'rt',encoding='utf-8') as h:detections=[json.loads(s) for s in h if s.strip()]
matrixchecks={}
for threshold in (.5,.75):
    mats={k:np.zeros((5,5),dtype=np.int64) for k in ('all','ge25','ge50')}
    for r in detections:
        if not r['gt_boxes'] or not r['pred_boxes']:continue
        gt=np.array(r['gt_boxes']);pred=np.array(r['pred_boxes']);gc=np.array(r['gt_classes'],int);pc=np.array(r['pred_classes'],int);scores=np.array(r['pred_confidence'])
        intersection=np.maximum(0,np.minimum(pred[:,None,2:],gt[None,:,2:])-np.maximum(pred[:,None,:2],gt[None,:,:2])).prod(-1)
        overlap=intersection/((pred[:,2:]-pred[:,:2]).prod(-1)[:,None]+(gt[:,2:]-gt[:,:2]).prod(-1)[None,:]-intersection)
        used=np.zeros(len(gt),bool);tp=np.zeros(len(pred),bool)
        for j in np.argsort(-scores,kind='stable')[:300]:
            candidates=np.where((gc==pc[j]) & ~used,overlap[j],-1)
            k=int(candidates.argmax())
            if candidates[k]>=threshold:tp[j]=True;used[k]=True
        for j in np.flatnonzero(~tp):
            same_iou=np.where(gc==pc[j],overlap[j],0).max()
            if .1<=same_iou<=threshold:continue
            opposite=np.where(gc!=pc[j],overlap[j],0);k=int(opposite.argmax())
            if opposite[k]>=threshold:
                for label,lo in [('all',.001),('ge25',.25),('ge50',.5)]:
                    if scores[j]>=lo:mats[label][pc[j],gc[k]]+=1
    expected=cs['N_s0'][str(threshold)]['confusion_counts_predicted_row_true_column']
    for k,m in mats.items():assert np.array_equal(m,np.array(expected[k])),(threshold,k,m,np.array(expected[k]))
    matrixchecks[str(threshold)]={k:m.tolist() for k,m in mats.items()}

for p,prev in before.items():assert stat(p)==prev,('changed during review',p)
dump('supplement_recompute_receipt.json',{'status':'PASS_SCOPED_SUPPLEMENT_REVIEW','auditor':'/root/loc_stress','new_hashes_computed':False,
     'sources_before_and_after_stat_equal':list(before.values()),'source_copies_byte_equal':True,'class_oracle_checks':classchecks,
     'class_report_20_rows_means_SD_deltas_exact':True,'minority_class_contribution_fractions':minor,
     'bridge_raw_all_3084_rows_exact':True,'bridge_200_images_194_with_GT_6_background_only_verified':True,
     'bridge_all_five_class_counts_and_common_eligible_closure':bridgechecks,
     'assigned_box_iou_vs_saved_rgb_iou_max_abs':{k:max(v) for k,v in geom_errors.items()},'assigned_box_vs_saved_iou_threshold_mismatches':0,
     'independent_numpy_N_s0_complete_Cls_matrices':matrixchecks,'TIDE_engine_rerun':False,'seconds':time.time()-started})
print('Supplement checks PASS; all 3084 raw object rows and all class totals; no TIDE engine rerun.',flush=True)
