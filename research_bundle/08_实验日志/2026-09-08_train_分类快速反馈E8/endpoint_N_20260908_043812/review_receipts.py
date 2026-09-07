"""Small CPU receipt crosscheck; no inference, weights, or hashes."""
from pathlib import Path
import csv
import json
import math
import yaml

B = Path(__file__).parent
STAGE = B.parent
NATIVE = STAGE.parent / '2026-09-07_train_IndependentKD实施/remote_admission_1532/evaluator_profile_attempt2'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def load_yaml(p):
    return yaml.safe_load(p.read_text(encoding='utf-8-sig'))
t = read(B/'runs/N/short_training_receipt.json')
e = read(B/'evaluations/N/short_evaluation_receipt.json')
c = read(B/'evaluations/N/short_evaluation_contract.json')
r = read(B/'runs/N/runtime_ready.json')
cfg = load_yaml(B/'runs/N/short_screen_config.yaml')
args = load_yaml(B/'runs/N/args.yaml')
frozen = load_yaml(STAGE/'launch_evidence_attempt1/source/configs/drone_N_s42_E8.yaml')
accepted = read(NATIVE/'evidence_contract.json')
binding = read(NATIVE/'evaluation_profile_binding/binding.json')
rows = list(csv.reader((B/'runs/N/results.csv').read_text().splitlines()))
kd = [json.loads(x) for x in (B/'runs/N/kd_batches.jsonl').read_text().splitlines() if x]
checks = {}
checks['complete_e8'] = t['status']=='SHORT_SCREEN_TRAINING_COMPLETED' and t['epochs_configured']==t['last_epoch']==t['independent_lr_horizon']==8 and [int(x[0]) for x in rows[1:]]==list(range(1,9))
checks['full_batch_count'] = r['train_images']==17990 and r['train_batches']==563 and t['batches']==8*563==4504
checks['counter_closure'] = t['optimizer_updates']==2667 and t['amp_skips']==7 and t['attempts']==t['optimizer_updates']+t['amp_skips']==t['ema_updates']==2674
checks['frozen_config_and_original_init'] = cfg==frozen and args['model']==cfg['model']==t['model'] and args['resume'] is False and args['pretrained'] is True and args['epochs']==8
checks['zero_learning_coefficient'] = cfg['classification_coefficient']==t['classification_coefficient']==cfg['localization_coefficient']==t['localization_coefficient']==0 and all(x['kd_weight']==x['weighted_kd_total']==0 and x['total_loss']==x['native_total'] for x in kd)
checks['logged_batch_cadence'] = [x['batch'] for x in kd]==[1,2,3]+list(range(100,4501,100))
checks['same_checkpoint_stat'] = t['checkpoint']==e['checkpoint'] and t['checkpoint']['bytes']==10753555
checks['short_identity'] = e['status']=='SHORT_SCREEN_EVALUATION_COMPLETED' and all(x['scope']=='SHORT_SCREEN' and x['arm']=='N' and x['seed']==42 and x['single_seed'] is True and x['formal_e200_complete'] is False and x['formal_paper_gain_claim'] is False and x['official_test_accessed'] is False for x in (t,e))
roster = (B/'evaluations/N/development_roster.txt').read_text().splitlines()
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
    p = read(B/f'queue/short_N_s42_E8_{phase}_resource_profile.json')
    status = read(B/f'queue/short_N_s42_E8_{phase}_status.json')
    done = read(B/f'queue/short_N_s42_E8_{phase}_completed.json')
    samples = p['samples']; peak = receipt['resources']
    resource[phase] = dict(samples=len(samples),minimum_free_mib=min(x['memory_free_mib'] for x in samples),max_project_rss_mib=max(x['project_rss_mib'] for x in samples),peak_job_vram_mib=max(peak['per_gpu_peak_vram_mib'].values()),peak_job_rss_mib=peak['peak_rss_mib'],monitor_errors=p['monitor_errors'])
    v=resource[phase]
    checks[phase+'_queue_resources'] = p['status']==status['status']=='COMPLETED' and p['exit_code']==status['exit_code']==0 and not p['monitor_errors'] and not status['monitor_errors'] and done['status']=='SHORT_SCREEN_STAGE_COMPLETED' and done['step']['arm']=='N' and done['step']['stage']==phase and v['minimum_free_mib']==p['minimum_free_mib']>=2048 and v['max_project_rss_mib']<=300*1024 and v['peak_job_vram_mib']<=p['launch']['expected_vram_mib'] and v['peak_job_rss_mib']<=p['launch']['expected_rss_mib']
    manifest=read(B/('runs/N/source_manifest.json' if phase=='train' else 'evaluations/N/source_manifest.json'))
    checks[phase+'_source_receipt_byte_identity'] = all(x['byte_identity'] is True for x in manifest['files'])
result = dict(status='PASS_RECEIPT_CROSSCHECK_SINGLE_SEED_E8_N' if all(checks.values()) else 'FAIL',checks=checks,metric_means=means,metric_percent={k:e[k]*100 for k in means},resources=resource,training_counters={k:t[k] for k in ('batches','optimizer_updates','attempts','amp_skips','ema_updates')},checkpoint_stat=t['checkpoint'],csv_header_columns=len(rows[0]),csv_row_columns=[len(x) for x in rows[1:]],logged_kd_rows=len(kd),limitations=['Training CSV final row is short; do not read AP by its header. Independent evaluation receipt is the AP source.','Population and checkpoint stat crosscheck, not per-image GT byte audit or checkpoint content equality.','Counter closure uses executed receipts and sparse logs; no independent replay of all optimizer steps.','No inference, GPU, weights/state loading, new hashes, or gain judgment.'])
with (B/'endpoint_review_receipt.json').open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
print(json.dumps(result,ensure_ascii=True,indent=2))
