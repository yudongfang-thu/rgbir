"""Independent CPU review: six inventories/arithmetic, N_s0 TIDE rerun.
No hashes; no writes outside this independent_review directory.
"""
import ast
import collections
import copy
import csv
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT/'official_tide')]
import numpy as np
from tidecv import TIDE
from tidecv.data import Data

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def stat(p):
    p=Path(p);s=p.stat();return {'path':str(p.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def rows_read(p):
    with gzip.open(p,'rt',encoding='utf-8') as h:return [json.loads(s) for s in h if s.strip()]
def ast_funcs(p):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def assert_recursive(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:assert_recursive(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):assert_recursive(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert abs(a-b)<1e-9,(path,a,b)
    else:assert a==b,(path,a,b)

start=time.time()
manifest=read(ROOT/'drone_results_v1/input_manifest.json')
summary=read(ROOT/'drone_results_v1/summary.json')
initial_paths=[ROOT/p for p in ['PROTOCOL.md','PROTOCOL_ADDENDUM.md','run_tide_audit.py','drone_results_v1/runner_source.py',
    'official_tide/tidecv/data.py','official_tide/tidecv/ap.py','official_tide/tidecv/quantify.py',
    'official_tide/tidecv/errors/main_errors.py','official_tide/tidecv/errors/error.py','drone_results_v1/summary.json',
    'drone_aggregate.json','README_DRONE.md','drone_results_v1/synthetic_truth.json','drone_results_v1/synthetic_validation.json']]
for e in manifest:
    initial_paths += [Path(e[k]) for k in ('path','metric_path','contract_path','receipt_path')]
    initial_paths.append(ROOT/'drone_results_v1'/(e['name']+'.json'))
initial={str(p):stat(p) for p in initial_paths}
source_bytes={str(p):p.read_bytes() for p in initial_paths if p.suffix in ('.py','.md')}
old,new=ast_funcs(ROOT/'drone_results_v1/runner_source.py'),ast_funcs(ROOT/'run_tide_audit.py')
algorithm_names=['xywh','make_data','iou_matrix','independent_ap','score_diagnostic','evaluate','summarize_run','synthetic_rows','validate_synthetic']
assert all(old[k]==new[k] for k in algorithm_names)
assert (ROOT/'PROTOCOL.md').read_bytes()==(ROOT/'drone_results_v1/PROTOCOL.md').read_bytes()
inventories={};first=None;target_rows=None
for e in manifest:
    rows=rows_read(e['path']);contract=read(e['contract_path']);receipt=read(e['receipt_path']);saved=read(ROOT/'drone_results_v1'/(e['name']+'.json'))
    assert saved==summary['results'][e['name']]
    assert len(rows)==1469 and len(set(r['image'] for r in rows))==1469
    assert set(r['image'] for r in rows)==set(contract['roster']) and len(contract['roster'])==1469
    gt=sorted([{k:r[k] for k in ('image','canvas_shape','original_shape','gt_boxes','gt_classes')} for r in rows],key=lambda r:r['image'])
    if first is None:first=gt
    else:assert first==gt
    classes=collections.Counter(int(c) for r in rows for c in r['gt_classes'])
    assert classes=={0:18965,1:710,2:1336,3:751,4:700}
    assert all(len(r['pred_boxes'])==len(r['pred_classes'])==len(r['pred_confidence'])<=300 for r in rows)
    assert all(len(r['gt_boxes'])==len(r['gt_classes']) for r in rows)
    assert all(0<=int(c)<5 and c==int(c) for r in rows for k in ('gt_classes','pred_classes') for c in r[k])
    assert set(tuple(r['canvas_shape']) for r in rows)=={(544,672)}
    assert min(s for r in rows for s in r['pred_confidence'])>=.001
    assert contract['effective_kwargs']['max_det']==300 and contract['effective_kwargs']['conf']==.001
    assert contract['effective_kwargs']['iou']==.7 and contract['effective_kwargs']['half'] is False
    assert receipt['status']=='completed' and receipt['official_test_accessed'] is False and receipt['historical_five_metrics_exact'] is True
    assert receipt['seed']==e['seed'] and receipt['normalized_method_arm']==e['arm']
    assert receipt['checkpoint'].endswith('/weights/last.pt')
    metrics=read(e['metric_path'])
    for key in ('AP50','AP75','mAP50_95','per_class'):assert saved['pinned_metrics'][key]==metrics[key]
    inv={'images':len(rows),'gt':sum(classes.values()),'gt_classes':dict(classes),'predictions':sum(len(r['pred_boxes']) for r in rows),
         'empty_pred_images':sum(not r['pred_boxes'] for r in rows),'empty_gt_images':sum(not r['gt_boxes'] for r in rows),
         'images_at_max_det':sum(len(r['pred_boxes'])==300 for r in rows),'checkpoint':receipt['checkpoint'],
         'contract_roster_exact':True,'same_gt_records_exact':True}
    assert inv['predictions']==saved['input_audit']['predictions']
    assert inv['empty_pred_images']==len(saved['input_audit']['empty_prediction_images'])
    assert inv['empty_gt_images']==len(saved['input_audit']['empty_gt_images'])
    for t,rec in saved['thresholds'].items():
        assert abs(rec['AP']-rec['independent_ap']['AP'])<1e-9
        for c,counts in rec['independent_ap']['per_class'].items():assert counts['gt']==classes[int(c)] and counts['tp']+counts['fn']==counts['gt']
        diag=rec['score_diagnostic']
        for g,d in diag.items():
            b=d['bins'];q=d['score_quantiles']
            assert sum(x['tp'] for x in b)==q['TP']['count'] and sum(x['fp'] for x in b)==q['FP']['count']
            assert all(sum(x['error_counts'].values())==x['tp']+x['fp'] for x in b)
            assert all(x['error_counts'].get('TP',0)==x['tp'] for x in b)
        assert diag['pooled']['score_quantiles']['TP']['count']+diag['pooled']['score_quantiles']['FP']['count']==inv['predictions']
    inventories[e['name']]=inv
    if e['name']=='N_s0':target_rows=rows

def official_direct(rows,names,threshold):
    gt,pred=Data('review_gt',max_dets=300),Data('review_pred',max_dets=300)
    for c,n in enumerate(names):gt.add_class(c,n);pred.add_class(c,n)
    for i,r in enumerate(rows):
        gt.add_image(i,r['image']);pred.add_image(i,r['image'])
        for b,c in zip(r['gt_boxes'],r['gt_classes']):gt.add_ground_truth(i,int(c),box=[b[0],b[1],b[2]-b[0],b[3]-b[1]])
        for b,c,s in zip(r['pred_boxes'],r['pred_classes'],r['pred_confidence']):pred.add_detection(i,int(c),float(s),box=[b[0],b[1],b[2]-b[0],b[3]-b[1]])
    return TIDE().evaluate(gt,pred,pos_threshold=threshold,background_threshold=.1,mode=TIDE.BOX,use_for_errors=True)

rerun={}
for t in (.5,.75):
    run=official_direct(target_rows,['car','freight car','truck','bus','van'],t)
    expected=summary['results']['N_s0']['thresholds'][str(t)]
    assert abs(run.ap-expected['AP'])<1e-9
    before=[p.get('used','MISSING') for p in run.preds.annotations]
    main={c.short_name:v for c,v in run.fix_main_errors().items()}
    special={c.short_name:v for c,v in run.fix_special_errors().items()}
    assert before==[p.get('used','MISSING') for p in run.preds.annotations]
    assert all(x is True or x is False for x in before)
    assert_recursive(main,expected['main_dAP']);assert_recursive(special,expected['special_dAP'])
    points=[]
    for c,obj in run.ap_data.objs.items():
        for pid,(score,tp,info) in obj.data_points.items():
            assert run.preds.annotations[pid]['used'] is tp
            points.append((c,pid,score,tp))
    errors={e.pred['_id']:type(e).short_name for e in run.errors if hasattr(e,'pred')}
    ranges=[.001,.01,.05,.25,.5,1.0000001]
    for group in ('pooled','0','1','2','3','4'):
        pp=points if group=='pooled' else [p for p in points if p[0]==int(group)]
        diag=expected['score_diagnostic'][group]
        for j,(lower,upper) in enumerate(zip(ranges[:-1],ranges[1:])):
            selected=[p for p in pp if lower<=p[2]<upper]
            tp=sum(p[3] for p in selected)
            assert tp==diag['bins'][j]['tp'] and len(selected)-tp==diag['bins'][j]['fp']
            assert dict(collections.Counter('TP' if p[3] else errors[p[1]] for p in selected))==diag['bins'][j]['error_counts']
        for label,state in [('TP',True),('FP',False)]:
            values=[p[2] for p in pp if p[3] is state]
            quant=np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist() if values else None
            assert quant==diag['score_quantiles'][label]['quantiles']
    rerun[str(t)]={'AP':run.ap,'AP_difference_pp':run.ap-expected['AP'],'main_dAP':main,'special_dAP':special,
                   'all_bins_quantiles_error_counts_exact':True,'all_prediction_used_matches_ap_data':True,
                   'used_unchanged_by_independent_oracles':True,'evaluated_predictions':len(points)}
    print('N_s0 independent direct official rerun',t,'PASS',flush=True)

# Execute existing synthetic validator only in this review's new subdirectory.
spec=importlib.util.spec_from_file_location('adapter_for_synthetic_review',ROOT/'run_tide_audit.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
synthetic_out=HERE/'synthetic_rerun';synthetic_out.mkdir(exist_ok=False)
adapter.validate_synthetic(synthetic_out)
assert read(synthetic_out/'synthetic_truth.json')==read(ROOT/'drone_results_v1/synthetic_truth.json')
assert read(synthetic_out/'synthetic_validation.json')==read(ROOT/'drone_results_v1/synthetic_validation.json')

# Recompute every aggregate and paired delta from the original endpoint numbers.
agg=read(ROOT/'drone_aggregate.json');delta={}
def stats(v):return {'values':v,'mean':float(np.mean(v)),'sample_SD':float(np.std(v,ddof=1)),'positive_seeds':sum(x>0 for x in v)}
for t in ('0.5','0.75'):
    for arm in ('N','C0'):
        for category,keys in [('main_dAP',['Cls','Loc','Both','Dupe','Bkg','Miss']),('special_dAP',['FalsePos','FalseNeg'])]:
            for key in keys:
                values=[summary['results'][f'{arm}_s{s}']['thresholds'][t][category][key] for s in (0,42,123)]
                assert_recursive(stats(values),agg[t][arm][key])
for label,t,category,key in [('TIDE AP50','0.5',None,'AP'),('TIDE AP75','0.75',None,'AP'),('AP50 Cls dAP','0.5','main_dAP','Cls'),('AP50 Bkg dAP','0.5','main_dAP','Bkg'),('AP75 Loc dAP','0.75','main_dAP','Loc'),('AP75 Bkg dAP','0.75','main_dAP','Bkg')]:
    def get(arm,s):
        r=summary['results'][f'{arm}_s{s}']['thresholds'][t]
        return (r[category] if category else r)[key]
    delta[label]=stats([get('C0',s)-get('N',s) for s in (0,42,123)])
    assert_recursive(delta[label],agg[label])
delta['pinned_mAP50_95_C0_minus_N_pp']=stats([(summary['results'][f'C0_s{s}']['pinned_metrics']['mAP50_95']-summary['results'][f'N_s{s}']['pinned_metrics']['mAP50_95'])*100 for s in (0,42,123)])

for p,previous in initial.items():assert stat(p)==previous,('changed input during review',p)
for p,content in source_bytes.items():assert Path(p).read_bytes()==content
receipt={'status':'PASSED_SCOPED_CPU_RECOMPUTATION','auditor':'/root/loc_stress','started_source_stat':list(initial.values()),
         'source_stat_and_source_bytes_unchanged':True,'new_hashes_computed':False,'cpu_only':True,'test_accessed':False,
         'algorithm_ast_same_between_executed_and_current_runner':algorithm_names,'protocol_bytes_match_executed_copy':True,
         'six_endpoint_inventories':inventories,'direct_official_N_s0_rerun':rerun,'synthetic_rerun_exact':True,
         'all_three_seed_aggregates_verified':True,'paired_deltas':delta,'seconds':time.time()-start}
(HERE/'recompute_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('Scoped independent checks PASS, seconds',receipt['seconds'],flush=True)
