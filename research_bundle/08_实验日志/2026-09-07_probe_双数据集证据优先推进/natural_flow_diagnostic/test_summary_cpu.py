"""Known-value readout tests, no real-result evaluation or GPU."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import summarize_completed as s

HERE=Path(__file__).resolve().parent

def main():
    valid={'status':'COMPLETED','dataset':'llvip','batches':64,'canary_only':False,'seed':20260907,
        'diagnostic_status':'UNVERIFIED_GEOMETRY_DIAGNOSTIC','trace_exact_all_recorded_fields':True,
        'classification':{'C0_C1_selection_identical':True},
        'localization':{'geometry_verified':False,'formal_L1_admitted':False},
        'backward_executed':False,'training_executed':False,'calibration_executed':False,'validation_or_test_accessed':False,'new_hash_computed':False}
    s.require_completed(valid,'llvip')
    rejected=[]
    for key,value in [('status','RUNNING'),('batches',2),('canary_only',True),('seed',42),('dataset','dronevehicle'),('training_executed',True),('new_hash_computed',True)]:
        altered=deepcopy(valid);altered[key]=value
        try:s.require_completed(altered,'llvip')
        except ValueError:rejected.append(key)
        else:raise AssertionError('Invalid completed input accepted: '+key)
    try:s.load_dataset(HERE/'nonexistent_full_input_fixture','llvip','llvip')
    except ValueError:pass
    else:raise AssertionError('Missing full input accepted')
    c=s.concentration(Counter({'A':4,'B':3,'C':2,'D':1,'E':1,'F':1}))
    assert c['objects']==12 and c['top1_objects']==4 and c['top5_objects']==11
    assert c['top1_fraction']==4/12 and c['top5_fraction']==11/12
    empty=s.concentration({});assert empty['objects']==0 and empty['top1_fraction'] is None
    assert s.fraction(0,0) is None and s.fraction(0,5)==0
    values=[100,80,80,80,60,50,40,30,30,20,18,10,9,8,6,6]
    counts=dict(zip(s.L_CHAIN,values));images=defaultdict(set);groups=defaultdict(set);batches=Counter()
    rows=s.gate_rows('fixture',counts,images,groups,batches)
    assert rows[1]['lost_objects']==20 and rows[1]['loss_fraction_previous']==.2
    assert rows[8]['lost_objects']==0 and rows[-1]['retention_fraction_rgb_gt']==.06
    altered=dict(counts);altered['support_count']=99
    try:s.gate_rows('fixture',altered,images,groups,batches)
    except ValueError:pass
    else:raise AssertionError('Noncumulative gate accepted')
    result={'status':'PASSED_CPU_READOUT_CONTRACT_ONLY','checks':{
        'completed_64_allowed':True,'partial_canary_seed_dataset_training_rejected':rejected,
        'missing_full_files_rejected':True,'top1_top5_known_object_denominators':True,
        'empty_fraction_NA':True,'gate_loss_previous_denominator_and_increase_rejected':True},
        'real_completed_results_read':False,'gpu':False,'hash_computed':False}
    with (HERE/'summary_cpu_tests_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
