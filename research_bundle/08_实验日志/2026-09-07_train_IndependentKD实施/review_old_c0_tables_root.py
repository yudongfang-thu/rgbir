"""Independent arithmetic/identity cross-check of the actual supplemental tables."""
import datetime
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'old_c0_object_diagnostics_v1'
table=json.loads((BASE/'tables_attempt2/summary_tables.json').read_text(encoding='utf-8'))
assert statistics.mean([1,2,3])==2 and statistics.stdev([1,2,3])==1
seeds=(0,42,123);metrics=('mAP50_95','AP50','AP75');checks=0
def equal(a,b):
    global checks
    assert math.isclose(a,b,rel_tol=0,abs_tol=1e-10),(a,b)
    checks+=1
raw={};background=[]
for seed in seeds:
    folder=BASE/'attempt2_outputs'/('seed'+str(seed))
    summary=json.loads((folder/'pair_summary.json').read_text(encoding='utf-8'))
    receipt=json.loads((folder/'analysis_receipt.json').read_text(encoding='utf-8'))
    assert receipt['status']=='COMPLETED' and receipt['analyzer_acceptance']=='ACCEPTED'
    for name in ('object_error_analysis.py','test_object_error_analysis.py','OBJECT_ERROR_ANALYSIS_RULES.md'):
        assert (folder/'analysis_source'/name).read_bytes()==(ROOT/'object_error_analyzer_rect_v2'/name).read_bytes()
    raw[seed]={role:json.loads((folder/'input_evidence'/role/'000_evaluation_val.json').read_text(encoding='utf-8')) for role in ('baseline','candidate')}
    for role,arm in (('baseline','weight0'),('candidate','paired')):
        value=raw[seed][role]
        assert value['seed']==seed and value['arm']==arm and value['historical_five_metrics_exact']
        for metric in metrics:equal(statistics.mean(r[metric] for r in value['per_class']),value[metric])
    s=summary['summary']
    assert s['gt_objects']==22462
    assert s['baseline_correct']+s['repaired']-s['damaged']==s['candidate_correct']
    equal(table['overall']['net_correct']['values_by_seed'][str(seed)],s['repaired']-s['damaged'])
    equal(table['overall']['repair_rate_percent']['values_by_seed'][str(seed)],100*s['repaired']/s['baseline_incorrect'])
    equal(table['overall']['damage_rate_percent']['values_by_seed'][str(seed)],100*s['damaged']/s['baseline_correct'])
    delta=summary['candidate']['background_fp']-summary['baseline']['background_fp'];background.append(delta)
    equal(table['overall']['background_delta_per_image']['values_by_seed'][str(seed)],delta/1469)
for ci in range(5):
    for metric in metrics:
        values={role:[100*next(r[metric] for r in raw[s][role]['per_class'] if r['class_id']==ci) for s in seeds] for role in ('baseline','candidate')}
        for key,seq in (('N_percent',values['baseline']),('C0_percent',values['candidate']),('delta_pp',[b-a for a,b in zip(values['baseline'],values['candidate'])])):
            observed=table['per_class'][str(ci)]['metrics'][metric][key]
            assert observed['ddof']==1 and observed['n']==3
            for seed,value in zip(seeds,seq):equal(observed['values_by_seed'][str(seed)],value)
            equal(observed['mean'],statistics.mean(seq));equal(observed['sample_sd'],statistics.stdev(seq))
assert background==[3,5,46] and table['review_signals']['background_increase_all_three']
result=dict(status='ACCEPTED_DESCRIPTIVE_TABLE_REVIEW',reviewer='/root (independent of table author)',
    reviewed_at=datetime.datetime.now().astimezone().isoformat(),arithmetic_checks=checks,
    checks=['Raw same-seed actual metrics and independently accepted rect-v2 source bytes',
            'Known arithmetic1/2/3 gives mean2 and sampleSD1',
            'All5classes x3APmetrics x3seeds direct standard-library mean/SD cross-check',
            'MacroAP equals mean of5raw classAP; repair/damage denominators and netcounts verified',
            'Background delta3/5/46 over1469images verified'],
    background_increase_triggers_C0_review=True,C0_automatic_extension_paused=True,
    existing_training_stopped=False,new_C1_result=False,formal_posthoc_class_adapter_accepted=False)
(BASE/'tables_attempt2/root_independent_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],checks=checks,background_delta=background)))
