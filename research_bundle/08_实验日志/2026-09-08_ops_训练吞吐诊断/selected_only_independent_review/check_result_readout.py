"""Independent JSON/JSONL readout audit only; no state tensors, model, or hashes."""
import json
from pathlib import Path
import statistics

OPS=Path(__file__).resolve().parents[1]
RUN=OPS/'remote_selected_update24_probe_attempt1/comparison_attempt1'
def read(path):return json.loads(path.read_text(encoding='utf8'))
def jsonl(path):return [json.loads(row) for row in path.read_text(encoding='utf8').splitlines() if row.strip()]
summary=read(OPS/'selected_update24_readout_attempt1/summary.json')
source=read(OPS/'selected_update24_readout_attempt1/source_input_review.json')
comparison=read(RUN/'comparison/comparison.json')
categories={}
for row in comparison['records']:
    value=categories.setdefault(row['category'],dict(records=0,bitwise_exact=True,numeric_agreement=True,
        all_finite=True,max_abs=0.,finite_outside_tolerance=0,nonfinite_elements=0))
    value['records']+=1
    for key in ('bitwise_exact','numeric_agreement','all_finite'):value[key]&=row[key]
    value['max_abs']=max(value['max_abs'],row['max_abs'])
    for key in ('finite_outside_tolerance','nonfinite_elements'):value[key]+=row[key]
checks=dict(category_records_close=categories==comparison['categories']==summary['categories'])
workers=[read(RUN/name/'worker_receipt.json') for name in ('old','selected_only')]
timings=[]
for index,worker in enumerate(workers):
    claimed=summary['workers'][index]
    checks['worker_'+str(index)+'_result']=worker['result']==claimed['result']
    checks['worker_'+str(index)+'_coverage']=worker['candidate_coverage']==claimed['candidate_coverage']
    rows=worker['timing']['batches'];retained=[row for row in rows if not row['warmup']]
    checks['worker_'+str(index)+'_rows']=rows==claimed['timing']['all_batch_rows']
    medians={key:statistics.median(row[key] for row in retained) for key in (
        'wall_seconds','wall_minus_audit_seconds','wall_minus_audit_and_extra_diagnostics_seconds')}
    for key,value in medians.items():checks['worker_'+str(index)+'_'+key]=value==claimed['timing']['post_warmup'][key]['median']
    checks['worker_'+str(index)+'_subtractions']=all(
        row['wall_seconds']-row['audit_seconds']==row['wall_minus_audit_seconds'] and
        row['wall_seconds']-row['audit_seconds']-row['extra_full_diagnostics_seconds']==row['wall_minus_audit_and_extra_diagnostics_seconds']
        for row in rows)
    timings.append(medians)
fields=('native_total','loss_unweighted','weighted_kd_total','total_loss','target_loss_unweighted','off_target_loss_unweighted')
batch_rows=[jsonl(RUN/name/'kd_batches.jsonl') for name in ('old','selected_only')]
checks['kd_batch_counts_30']=len(batch_rows[0])==len(batch_rows[1])==30
for field in fields:checks['all_batch_'+field]=[row[field] for row in batch_rows[0]]==[row[field] for row in batch_rows[1]]
for file in ('shared_gradient_observations.jsonl','gradient_checks.jsonl'):
    checks[file+'_exact']=jsonl(RUN/'old'/file)==jsonl(RUN/'selected_only'/file)
manifests=[read(RUN/name/'source_manifest.json') for name in ('old','selected_only')]
checks['source_65_pairs']=len(manifests[0]['files'])==len(manifests[1]['files'])==65
for i,(a,b) in enumerate(zip(manifests[0]['files'],manifests[1]['files'])):
    checks['source_'+str(i)]=a['path']==b['path'] and (RUN/'old/sources'/Path(a['copy']).name).read_bytes()==(RUN/'selected_only/sources'/Path(b['copy']).name).read_bytes()
checks['models_stat_exact']=manifests[0]['models']==manifests[1]['models']==source['models']
checks['config_copy_exact']=(RUN/'old/sources/input_config.yaml').read_bytes()==(RUN/'selected_only/sources/input_config.yaml').read_bytes()
for row in source['current_candidate_and_harness_bindings']:
    current=Path(row['current_local_source'])
    matches=list((RUN/'selected_only/sources').glob('*_'+row['name']))
    checks['current_'+row['name']]=len(matches)==1 and matches[0].read_bytes()==current.read_bytes()
ratio=timings[0]['wall_minus_audit_and_extra_diagnostics_seconds']/timings[1]['wall_minus_audit_and_extra_diagnostics_seconds']
checks['median_ratio_exact']=ratio==summary['observed_post_warmup_old_over_new_median']['wall_minus_audit_and_extra_diagnostics_seconds']
receipt=dict(status='PASS_JSON_READOUT_SCOPE' if all(checks.values()) else 'FAIL',checks=checks,
    state_pt_loaded=False,model_loaded=False,new_hash_computed=False,
    observed_median_ratio=ratio,complete_training_span_seconds=[w['timing']['training_span_seconds'] for w in workers],
    thin_coverage=workers[1]['candidate_coverage'],
    scope='Independent JSON/JSONL/byte-copy audit; executed state comparator supplies tensor equality')
with (Path(__file__).resolve().parent/'result_readout_receipt.json').open('x',encoding='utf8') as f:
    json.dump(receipt,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({key:value for key,value in receipt.items() if key!='checks'}))
if not all(checks.values()):print('FAILURES',[key for key,value in checks.items() if not value]);raise SystemExit(2)
