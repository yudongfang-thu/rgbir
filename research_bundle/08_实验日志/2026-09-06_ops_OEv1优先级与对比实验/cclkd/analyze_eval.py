"""Audit the collected independent last evaluations and compute sample SD."""
import json,statistics,shutil
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
LOGS=HERE.parents[1]
NATIVE=LOGS/'2026-09-06_audit_RGBIR夜间结果与GitHub更新/osssl/raw/runs/cgkd_w1'
RUNS=HERE/'eval_raw/runs/oev1_comparators_20260906'
REMOTE=PurePosixPath('/mnt/dataset/yudongfang/projects/RGBT_campaign')
metrics=('mAP50_95','AP50','AP75')
rows=[];audits=[];rosters=[];natives=[]
for seed in (0,42,123):
    run=RUNS/f'cclkd_partial_s{seed}_full_attempt1'
    result=json.loads((run/'evaluation_val.json').read_text())
    receipt_path=run/'eval_evidence/run_receipt.json'
    receipt=json.loads(receipt_path.read_text())
    roster=(run/'evaluation_roster.txt').read_text().splitlines()
    expected=str(REMOTE/f'runs/rgbt_cclkd_adapted_v1/cclkd_drone_seed{seed}_b32_e200/weights/last.pt')
    assert result['status']=='evaluation_completed' and result['seed']==seed
    assert result['checkpoint']==expected and result['arm']=='cclkd_partial'
    assert result['endpoint']=='fixed_budget_last_ema' and result['split']=='val'
    assert result['evaluated_images']==1469 and len(roster)==len(set(roster))==1469
    assert not result['canary'] and not result['official_test_accessed']
    assert result['metric_units']=='fraction_0_to_1'
    assert receipt['terminal_status']=='COMPLETED' and receipt['run_kind']=='eval'
    assert receipt['seed']==seed and receipt['method_identity']=='PROTOCOL-ADAPTED'
    assert receipt['inputs']['historical_training_source_not_fully_bound']
    assert not receipt['inputs']['teacher_labels_used_by_kd']
    assert receipt['inputs']['checkpoint']==expected
    assert receipt['resources']['gpu_ids']==[2]
    assert receipt['resources']['cuda_pid_counts']=={'2':1}
    assert receipt['resources']['lease_id'].startswith(f'cclkd_partial_s{seed}_full_attempt1-')
    assert receipt['test_exposure']['confirmatory'] is False
    snaps=receipt['source_snapshots']
    for p in [v for values in snaps.values() for v in values]+receipt['metric_snapshots']:
        assert (receipt_path.parent/p).is_file(),p
    assert json.loads((receipt_path.parent/receipt['metric_snapshots'][0]).read_text())==result
    assert (receipt_path.parent/snaps['split_roster'][0]).read_text().splitlines()==roster
    rosters.append(roster)
    npath=NATIVE/f'native_rgb_s{seed}_e200'
    ndest=HERE/'historical_native'/npath.name;ndest.mkdir(parents=True,exist_ok=True)
    for name in ('metrics_record.json','args.yaml','completion_receipt.json'):
        shutil.copyfile(npath/name,ndest/name)
    native=json.loads((ndest/'metrics_record.json').read_text())
    assert native['seed']==seed and native['split']=='val' and native['arm']=='native_weight0'
    assert native['checkpoint'].endswith('/weights/last.pt')
    natives.append(native)
    rows.append({'seed':seed,'cclkd_partial_pct':{m:result[m]*100 for m in metrics},
                 'historical_native_pct':{m:native['metrics'][m]*100 for m in metrics},
                 'delta_pp':{m:(result[m]-native['metrics'][m])*100 for m in metrics},
                 'evaluation_file':str((run/'evaluation_val.json').relative_to(HERE)),
                 'native_file':str((ndest/'metrics_record.json').relative_to(HERE))})
    audits.append({'seed':seed,'status':'PASS','eval_receipt':str(receipt_path.relative_to(HERE)),
                   'evaluated_images':1469,'split':'val','checkpoint':expected,
                   'sources_describe_eval_not_historical_training':True,
                   'resource_lease_id':receipt['resources']['lease_id'],
                   'resources':receipt['resources']})
assert rosters[0]==rosters[1]==rosters[2]
def agg(vals):return {'mean':statistics.mean(vals),'sample_sd_ddof1':statistics.stdev(vals),'n':len(vals)}
summary={'method_id':'CCLKD-ADAPTED-PARTIAL-LLD-CCL-HISTORICAL','seeds':[0,42,123],
         'endpoint':'fixed_budget_last_ema','split':'val','evaluated_images_each':1469,
         'all_three_ordered_val_rosters_equal':True,'paired_rows':rows,'audits':audits,
         'cclkd_partial_pct':{m:agg([r['cclkd_partial_pct'][m] for r in rows]) for m in metrics},
         'historical_native_pct':{m:agg([r['historical_native_pct'][m] for r in rows]) for m in metrics},
         'descriptive_delta_pp':{m:{**agg([r['delta_pp'][m] for r in rows]),
                                    'positive_seeds':sum(r['delta_pp'][m]>0 for r in rows)} for m in metrics},
         'limitations':['Partial LLD+CCL implementation, not complete CCLKD',
             'Historical training source snapshot is incomplete; new sources bind evaluation only',
             'Historical native comparator; not a newly matched CCLKD weight0 run',
             'No three-seed shuffled/same-modal attribution controls for this CCLKD variant',
             'OEv1 directly uses IR labels during KD whereas this CCLKD variant does not',
             'All are development-val results, not confirmatory test results']}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:summary[k] for k in ('cclkd_partial_pct','historical_native_pct','descriptive_delta_pp','paired_rows')},indent=2))
