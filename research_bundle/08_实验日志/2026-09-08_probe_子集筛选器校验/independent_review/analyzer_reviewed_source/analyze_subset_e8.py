"""Prospective receipt-only known-positive-control subset screener validation."""
import argparse,json,math,shutil
from pathlib import Path

SCOPE='DRONE_SUBSET2048_PRETRAIN_E8_CHECK'
ENDPOINT='SUBSET2048_PRETRAIN_E8_LAST_EMA'
ARMS=('N','C0');COEFFICIENTS={'N':0.,'C0':.1}
AP=('mAP50_95','AP50','AP75');METRICS=AP+('precision','recall')
SUBSET_KEYS=('student_data_yaml','privileged_data_yaml','paired_train_mapping')

def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def number(x,upper=1.):return type(x) in (int,float) and math.isfinite(x) and 0<=x<=upper
def file_identity(value):
    return isinstance(value,dict) and isinstance(value.get('path'),str) and bool(value['path']) and type(value.get('bytes')) is int and value['bytes']>0 and type(value.get('mtime_ns')) is int and value['mtime_ns']>0

def validate(receipt,arm):
    require(arm in ARMS,'Unknown arm')
    fixed=dict(status='SUBSET_SCREEN_EVALUATION_COMPLETED',scope=SCOPE,endpoint=ENDPOINT,
        dataset='dronevehicle',arm=arm,seed=42,single_seed=True,epochs=8,independent_lr_horizon=8,
        full_dev_images=1469,full_dev_gt_objects=22462,expected_train_images=2048,
        metric_units='fraction_0_to_1',bn_running_statistics='normal_training',
        formal_e200_complete=False,formal_paper_gain_claim=False,official_test_accessed=False,new_hash_computed=False)
    for key,value in fixed.items():
        require(receipt.get(key)==value and (type(value) is not bool or receipt.get(key) is value),'Receipt identity differs: '+key)
    require(number(receipt.get('classification_coefficient')) and receipt['classification_coefficient']==COEFFICIENTS[arm],'Wrong fixed arm coefficient')
    require(all(number(receipt.get(k)) for k in METRICS),'Invalid overall fraction')
    classes=receipt.get('per_class',[])
    require(len(classes)==5 and all(type(r.get('class_id')) is int for r in classes),'Wrong class count/ID')
    require(sorted(r['class_id'] for r in classes)==list(range(5)),'Missing or duplicate class ID')
    require(all(isinstance(r.get('name'),str) and r['name'] for r in classes),'Missing class name')
    require(all(number(r.get(k)) for r in classes for k in AP),'Invalid per-class fraction')
    for key in AP:require(math.isclose(math.fsum(r[key] for r in classes)/5,receipt[key],rel_tol=0,abs_tol=1e-12),'Class macro differs: '+key)
    init=receipt.get('initialization',{});require(all(file_identity(init.get(k)) for k in ('model','teacher','reference')),'Missing initialization stat identity')
    subset=receipt.get('training_subset_identity',{})
    require(all(isinstance(subset.get(k),str) and subset[k] for k in SUBSET_KEYS),'Missing subset/mapping identity')
    require(all(file_identity(receipt.get(k)) for k in ('training_configuration','training_completion')),'Missing executed config/completion stat')
    require(number(receipt.get('seconds'),float('inf')),'Invalid evaluation duration')
    return receipt

def compatible(a,b):
    require(a['initialization']==b['initialization'],'Common initial model/teacher/reference differs')
    require(a['training_subset_identity']==b['training_subset_identity'],'Common subset paths/mapping differs')
    require({r['class_id']:r['name'] for r in a['per_class']}=={r['class_id']:r['name'] for r in b['per_class']},'Class mapping differs')

def direction(value):return 'positive' if value>0 else 'negative' if value<0 else 'zero'

def summarize(receipts,unavailable=None):
    unavailable=unavailable or {};require(set(receipts).issubset(ARMS) and set(unavailable).issubset(ARMS),'Unknown arm input')
    require(not set(receipts)&set(unavailable),'Completed and unavailable arm conflict')
    valid={arm:validate(r,arm) for arm,r in receipts.items()}
    if len(valid)==2:compatible(valid['N'],valid['C0'])
    rows=[]
    for arm in ARMS:
        if arm not in valid:
            state=unavailable.get(arm,dict(status='MISSING',reason='No terminal evaluation receipt'))
            require(state.get('status') in ('MISSING','FAILED','CONFLICT'),'Invalid unavailable status')
            rows.append(dict(arm=arm,**state,raw_fraction=None,display_percent=None,per_class=None));continue
        r=valid[arm]
        rows.append(dict(arm=arm,status='COMPLETED',classification_coefficient=r['classification_coefficient'],
            raw_fraction={k:r[k] for k in METRICS},display_percent={k:r[k]*100 for k in METRICS},
            per_class=[dict(class_id=c['class_id'],name=c['name'],raw_fraction={k:c[k] for k in AP},display_percent={k:c[k]*100 for k in AP}) for c in sorted(r['per_class'],key=lambda x:x['class_id'])],
            evaluation_seconds=r['seconds'],initialization=r['initialization'],training_subset_identity=r['training_subset_identity'],
            training_configuration=r['training_configuration'],training_completion=r['training_completion']))
    contrast=dict(contrast='C0 - N',status='UNAVAILABLE',delta_pp=None,direction=None,per_class=None)
    if len(valid)==2:
        n,c=valid['N'],valid['C0'];pp={k:(c[k]-n[k])*100 for k in METRICS}
        nn={r['class_id']:r for r in n['per_class']};cc={r['class_id']:r for r in c['per_class']}
        contrast.update(status='COMPUTED',delta_pp=pp,direction={k:direction(v) for k,v in pp.items()},
            per_class=[dict(class_id=i,name=cc[i]['name'],delta_pp={k:(cc[i][k]-nn[i][k])*100 for k in AP}) for i in range(5)])
    return dict(status='COMPLETE_SUBSET_E8_READOUT' if len(valid)==2 else 'PARTIAL_SUBSET_E8_READOUT',scope=SCOPE,endpoint=ENDPOINT,
        purpose='known_positive_control_screener_validation',dataset='dronevehicle',seed=42,n_seeds=1,epochs=8,independent_lr_horizon=8,
        train_images=2048,dev_images=1469,dev_gt_objects=22462,standard_deviation=None,
        raw_metric_units='fraction_0_to_1',display_metric_units='percent',difference_units='percentage_points',arms=rows,comparison=contrast,
        primary_direction=None if contrast['direction'] is None else contrast['direction']['mAP50_95'],
        new_method_causal_claim=False,causal_modality_claim=False,general_method_ranking_validated=False,
        automatically_admit_e200=False,automatically_extend_matrix=False,formal_e200_complete=False,formal_paper_gain_claim=False,new_hash_computed=False,
        scope_limit='Only the known N/C0 direction in one fixed subset/seed/independent E8 schedule; no general efficacy inference. Actual training identity remains the responsibility of its collected execution receipts.')

def collect(campaign):
    receipts={};unavailable={};sources=[]
    for arm in ARMS:
        folder=campaign/'evaluations'/arm;p=folder/'short_evaluation_receipt.json';f=folder/'short_evaluation_failure.json'
        for path in (p,f):
            if path.is_file():s=path.stat();sources.append(dict(path=str(path.absolute()),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
        if p.is_file() and f.is_file():unavailable[arm]=dict(status='CONFLICT',reason='Completed and failed receipts coexist; no outcome selected')
        elif f.is_file():
            failure=read(f);require(failure.get('status')=='SUBSET_SCREEN_EVALUATION_FAILED','Unexpected failure status')
            require(failure.get('scope',SCOPE)==SCOPE,'Foreign failure scope')
            unavailable[arm]=dict(status='FAILED',reason=str(failure.get('error','Evaluation failed')))
        elif p.is_file():receipts[arm]=read(p)
    result=summarize(receipts,unavailable);result['input_sources']=sources;return result

def markdown(result):
    out=['# Fixed subset N/C0 direction check','',
        'Known-positive-control screener validation: one seed42, generic pretrained initialization, normal BN, fixed2048 train subset and independent E8 schedule. AP displays percent; differences are percentage points. No SD, new-method causal claim or automatic E200 admission.','',
        '|Arm|Status|mAP50-95 (%)|AP50 (%)|AP75 (%)|','|---|---|---:|---:|---:|']
    for r in result['arms']:
        values=[f"{r['display_percent'][k]:.6f}" for k in AP] if r['display_percent'] is not None else ['NA']*3
        out.append('| '+' | '.join([r['arm'],r['status']]+values)+' |')
    c=result['comparison'];out+=['','|Contrast|Status|delta mAP (pp)|delta AP50 (pp)|delta AP75 (pp)|','|---|---|---:|---:|---:|']
    values=[f"{c['delta_pp'][k]:+.6f}" for k in AP] if c['delta_pp'] is not None else ['NA']*3
    out.append('| '+' | '.join(['C0 - N',c['status']]+values)+' |')
    out+=['','Primary observed direction: '+str(result['primary_direction'])+'. Missing/failed/conflicting inputs are null, never zero. Per-class raw values, P/R and fixed C0-N differences are in summary.json.',
        'Positive only indicates retention of this known control direction here. Nonpositive means this fixed screener check did not retain that direction; it does not establish C0 ineffectiveness. Neither outcome validates general method ranking.']
    return '\n'.join(out)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(a.campaign.is_dir() and not a.output.exists(),'Existing campaign and new output required')
    result=collect(a.campaign);a.output.mkdir(parents=True)
    with (a.output/'summary.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    (a.output/'README.md').write_text(markdown(result),encoding='utf-8')
    source=a.output/'analyze_subset_e8_source.py';shutil.copyfile(__file__,source);require(Path(__file__).read_bytes()==source.read_bytes(),'Source copy differs')
    print(result['status'])
if __name__=='__main__':main()
