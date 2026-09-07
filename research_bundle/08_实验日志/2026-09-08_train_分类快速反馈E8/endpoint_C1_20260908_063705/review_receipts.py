"""Reuse C0 receipt checks for C1; add actual weighted-KD scalar closure; decimal 300 GB limit."""
from pathlib import Path
import csv
import json
import math
import struct
import yaml

B = Path(__file__).parent
STAGE = B.parent
NATIVE = STAGE.parent / '2026-09-07_train_IndependentKD实施/remote_admission_1532/evaluator_profile_attempt2'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def load_yaml(p):
    return yaml.safe_load(p.read_text(encoding='utf-8-sig'))
t = read(B/'runs/C1/short_training_receipt.json')
e = read(B/'evaluations/C1/short_evaluation_receipt.json')
c = read(B/'evaluations/C1/short_evaluation_contract.json')
r = read(B/'runs/C1/runtime_ready.json')
cfg = load_yaml(B/'runs/C1/short_screen_config.yaml')
args = load_yaml(B/'runs/C1/args.yaml')
frozen = load_yaml(STAGE/'launch_evidence_attempt1/source/configs/drone_C1_s42_E8.yaml')
accepted = read(NATIVE/'evidence_contract.json')
binding = read(NATIVE/'evaluation_profile_binding/binding.json')
rows = list(csv.reader((B/'runs/C1/results.csv').read_text().splitlines()))
kd = [json.loads(x) for x in (B/'runs/C1/kd_batches.jsonl').read_text().splitlines() if x]
checks = {}
checks['complete_e8'] = t['status']=='SHORT_SCREEN_TRAINING_COMPLETED' and t['epochs_configured']==t['last_epoch']==t['independent_lr_horizon']==8 and [int(x[0]) for x in rows[1:]]==list(range(1,9))
checks['full_batch_count'] = r['train_images']==17990 and r['train_batches']==563 and t['batches']==8*563==4504
checks['counter_closure'] = t['optimizer_updates']==2667 and t['amp_skips']==7 and t['attempts']==t['optimizer_updates']+t['amp_skips']==t['ema_updates']==2674
checks['frozen_config_and_original_init'] = cfg==frozen and args['model']==cfg['model']==t['model'] and args['resume'] is False and args['pretrained'] is True and args['epochs']==8
COEFFICIENT = .09227393550836771
f32 = lambda v: struct.unpack('f',struct.pack('f',v))[0]
weighted_errors = [abs(x['weighted_kd_total']-x['loss_unweighted']*x['actual_B']*COEFFICIENT) for x in kd]
total_errors = [abs(x['total_loss']-f32(x['native_total']+f32(x['loss_unweighted']*f32(x['actual_B']*COEFFICIENT)))) for x in kd]
checks['actual_c1_learning_coefficient'] = cfg['classification_coefficient']==t['classification_coefficient']==COEFFICIENT and cfg['localization_coefficient']==t['localization_coefficient']==0 and all(x['arm']=='C1' and x['source']=='paired' and x['coefficient']==COEFFICIENT and x['actual_B']==len(x['student_files']) and x['weighted_kd_total']>0 for x in kd)
checks['weighted_kd_and_total_closure'] = max(weighted_errors)==max(total_errors)==0
checks['logged_batch_cadence'] = [x['batch'] for x in kd]==[1,2,3]+list(range(100,4501,100))
checks['same_checkpoint_stat'] = t['checkpoint']==e['checkpoint'] and t['checkpoint']['bytes']==10753555
checks['short_identity'] = e['status']=='SHORT_SCREEN_EVALUATION_COMPLETED' and all(x['scope']=='SHORT_SCREEN' and x['arm']=='C1' and x['seed']==42 and x['single_seed'] is True and x['formal_e200_complete'] is False and x['formal_paper_gain_claim'] is False and x['official_test_accessed'] is False for x in (t,e))
roster = (B/'evaluations/C1/development_roster.txt').read_text().splitlines()
checks['canonical_full_roster'] = len(roster)==len(set(roster))==1469 and roster==c['roster']==accepted['roster']==binding['roster'] and sorted(c['actual_loader_roster'])==sorted(roster)
checks['actual_native_kwargs'] = c['effective_kwargs']==accepted['effective_kwargs']==binding['actual_effective_kwargs'] and c['evaluator_identity']==accepted['evaluator_identity']
checks['population_counts'] = c['observed_images']==e['full_dev_images']==1469 and c['gt_objects_before_inference']==c['gt_objects_captured']==e['full_dev_gt_objects']==22462
names = ['car','freight car','truck','bus','van']
checks['classes_units_finite'] = e['metric_units']=='fraction_0_to_1' and [(x['class_id'],x['name']) for x in e['per_class']]==list(enumerate(names)) and all(math.isfinite(x[k]) and 0<=x[k]<=1 for x in e['per_class'] for k in ('AP50','AP75','mAP50_95'))
means = {k:dict(recomputed=sum(x[k] for x in e['per_class'])/5,reported=e[k]) for k in ('AP50','AP75','mAP50_95')}
for v in means.values(): v['absolute_difference']=abs(v['recomputed']-v['reported'])
checks['class_means_close'] = all(v['absolute_difference']<=1e-15 for v in means.values())
resource = {}
for phase, receipt in [('train',t),('eval',e)]:
    p = read(B/f'queue/short_C1_s42_E8_{phase}_resource_profile.json')
    status = read(B/f'queue/short_C1_s42_E8_{phase}_status.json')
    done = read(B/f'queue/short_C1_s42_E8_{phase}_completed.json')
    samples = p['samples']; peak = receipt['resources']
    resource[phase] = dict(samples=len(samples),minimum_free_mib=min(x['memory_free_mib'] for x in samples),max_project_rss_mib=max(x['project_rss_mib'] for x in samples),peak_job_vram_mib=max(peak['per_gpu_peak_vram_mib'].values()),peak_job_rss_mib=peak['peak_rss_mib'],monitor_errors=p['monitor_errors'])
    v=resource[phase]
    checks[phase+'_queue_resources'] = p['status']==status['status']=='COMPLETED' and p['exit_code']==status['exit_code']==0 and not p['monitor_errors'] and not status['monitor_errors'] and done['status']=='SHORT_SCREEN_STAGE_COMPLETED' and done['step']['arm']=='C1' and done['step']['stage']==phase and v['minimum_free_mib']==p['minimum_free_mib']>=2048 and v['max_project_rss_mib']*2**20<=300*10**9 and v['peak_job_vram_mib']<=p['launch']['expected_vram_mib'] and v['peak_job_rss_mib']<=p['launch']['expected_rss_mib']
    manifest=read(B/('runs/C1/source_manifest.json' if phase=='train' else 'evaluations/C1/source_manifest.json'))
    checks[phase+'_source_receipt_byte_identity'] = all(x['byte_identity'] is True for x in manifest['files'])

# Additional C1 provenance and three-arm comparison-eligibility checks only.
admission=read(B/'runs/C1/short_screen_admission.json')
launch_admission=read(STAGE/'launch_evidence_attempt1/admissions/C1.json')
train_manifest=read(B/'runs/C1/source_manifest.json')['files']
candidate_records=[x for x in train_manifest if x['path']==t['candidate']['source']]
local_candidate=STAGE/'launch_evidence_attempt1/selected_only_v1.py'
reviewed_candidate=STAGE.parent/'2026-09-08_ops_训练吞吐诊断/performance_candidate/selected_only_v1.py'
checks['candidate_factory_approved_source'] = t['candidate']==admission['candidate']==launch_admission['candidate'] and t['candidate']['kind']=='criterion_factory' and cfg['short_screen']['candidate']==dict(kind='criterion_factory',implementation='selected_only_v1') and len(candidate_records)==1 and candidate_records[0]['byte_identity'] is True and candidate_records[0]['bytes']==len(local_candidate.read_bytes())==14574 and local_candidate.read_bytes()==reviewed_candidate.read_bytes()
checks['logged_cumulative_thin_no_fallback_through_4500'] = all(x['thin_path_used'] is True and x['thin_path_fallback_reason'] is None and x['full_diagnostics_collected'] is True and x['thin_learning_batches']==x['batch'] and x['full_diagnostics_batches']==i+1 and x['fallback_batches']==0 for i,x in enumerate(kd))
three={a:STAGE/f'endpoint_{a}_{stamp}' for a,stamp in [('N','20260908_043812'),('C0','20260908_053913'),('C1','20260908_063705')]}
configs={a:load_yaml(p/f'runs/{a}/short_screen_config.yaml') for a,p in three.items()}
def differences(x,y,path=''):
    if isinstance(x,dict) and isinstance(y,dict):
        return sum((differences(x.get(k),y.get(k),path+'.'+k) for k in sorted(x.keys()|y.keys())),[])
    return [dict(field=path,left=x,right=y)] if x!=y else []
config_diffs={a:differences(configs['N'],configs[a]) for a in ('C0','C1')}
allowed={'.arm','.classification_coefficient','.method_id','.short_screen.source_E20_draft','.short_screen.accepted_N_C0_compatibility_may_be_reused','.short_screen.candidate.implementation','.short_screen.candidate.kind','.short_screen.needs_new_24update_review'}
checks['three_arm_config_differences_expected'] = all(x['field'] in allowed for d in config_diffs.values() for x in d)
actual_args={a:load_yaml(p/f'runs/{a}/args.yaml') for a,p in three.items()}
actual_arg_diffs={a:differences(actual_args['N'],actual_args[a]) for a in ('C0','C1')}
checks['three_arm_actual_args_match'] = all(x['field'] in {'.name','.save_dir'} for d in actual_arg_diffs.values() for x in d)
contracts={a:read(p/f'evaluations/{a}/short_evaluation_contract.json') for a,p in three.items()}
checks['three_arm_actual_eval_contract_exact'] = contracts['N']==contracts['C0']==contracts['C1']
endpoints={a:read(p/f'evaluations/{a}/short_evaluation_receipt.json') for a,p in three.items()}
keys=['scope','single_seed','seed','dataset','endpoint','epochs','independent_lr_horizon','formal_e200_complete','formal_paper_gain_claim','metric_units','native_metric_definition','native_profile_binding','full_dev_images','full_dev_gt_objects','official_test_accessed']
checks['three_arm_endpoint_identity_common'] = all(len({str(x[k]) for x in endpoints.values()})==1 for k in keys)
source_comparison={}
for phase in ('runs','evaluations'):
    manifests={a:{x['path']:(x['bytes'],x['mtime_ns'],x['byte_identity']) for x in read(p/f'{phase}/{a}/source_manifest.json')['files']} for a,p in three.items()}
    common=set.intersection(*(set(x) for x in manifests.values()))
    mismatch=[p for p in common if not manifests['N'][p]==manifests['C0'][p]==manifests['C1'][p]]
    unique={a:sorted(set(x)-common) for a,x in manifests.items()}
    expected_unique={a:([f'admissions/{a}.json',f'configs/drone_{a}_s42_E8.yaml']+(['performance_candidate/selected_only_v1.py'] if a=='C1' else [])) if phase=='runs' else [f'runs/{a}/short_screen_config.yaml'] for a in three}
    checks[phase+'_three_arm_source_stat_scope'] = not mismatch and all(len(unique[a])==len(expected_unique[a]) and all(any(path.endswith('/'+suffix) for path in unique[a]) for suffix in expected_unique[a]) for a in three)
    source_comparison[phase]=dict(common_count=len(common),common_stat_mismatches=mismatch,noncommon_paths=unique,scope='Path/bytes/mtime and source-copy byte_identity receipts; not freshly retrieved remote source bytes.')

result = dict(status='PASS_RECEIPT_CROSSCHECK_SINGLE_SEED_E8_C1' if all(checks.values()) else 'FAIL',checks=checks,metric_means=means,metric_percent={k:e[k]*100 for k in means},resources=resource,training_counters={k:t[k] for k in ('batches','optimizer_updates','attempts','amp_skips','ema_updates')},checkpoint_stat=t['checkpoint'],csv_header_columns=len(rows[0]),csv_row_columns=[len(x) for x in rows[1:]],logged_kd_rows=len(kd),weighted_kd_double_maxabs=max(weighted_errors),total_loss_float32_maxabs=max(total_errors),project_rss_limit_decimal_bytes=300*10**9,three_arm_config_differences=config_diffs,three_arm_actual_args_differences=actual_arg_diffs,three_arm_source_comparison=source_comparison,thin_last_logged={k:kd[-1][k] for k in ('batch','thin_learning_batches','full_diagnostics_batches','fallback_batches')},thin_final_counters_available=all(k in t for k in ('thin_learning_batches','full_diagnostics_batches','fallback_batches')),limitations=['Training CSV final row is short; do not read AP by its header. Independent evaluation receipt is the AP source.','Population and checkpoint stat crosscheck, not per-image GT byte audit or checkpoint content equality.','Counter closure uses executed receipts and sparse logs; no independent replay of all optimizer steps.','Final training receipt omits thin/full/fallback counters; cumulative logs verify through batch 4500 only, not the final four batches.','No inference, GPU, weights/state loading, new hashes, or gain judgment.'])
with (B/'endpoint_review_receipt.json').open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
print(json.dumps(result,ensure_ascii=True,indent=2))
