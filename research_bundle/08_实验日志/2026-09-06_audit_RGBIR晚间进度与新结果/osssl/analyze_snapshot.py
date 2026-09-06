"""Compute same-seed CSV contrasts without claiming accepted final evaluation."""
import csv
import json
import pathlib
import yaml

O=pathlib.Path(__file__).resolve().parent
R=O/'raw/runs'
snapshot=json.loads((O/'snapshot.json').read_text(encoding='utf-8'))
report={'timestamp':snapshot['timestamp'],'metric_identity':'training CSV last row; per-epoch validation, not independent eval_rgbt_detector last.pt endpoint','test_used':False,'contrasts':[],'progress':[],'native_independent_metrics':[]}
for arm in ('paired','sar_only','shuffled'):
    for seed in (0,42,123):
        d=R/'osssl_ir_20260906'/f'{arm}_rgb_s{seed}_e200'
        n=R/'cgkd_w1'/f'native_rgb_s{seed}_e200'
        progress={'arm':arm,'seed':seed,'status':'queued','epoch_completed':0,'has_independent_metrics_record':(d/'metrics_record.json').exists()}
        if (d/'results.csv').exists():
            rows=list(csv.DictReader((d/'results.csv').read_text().splitlines()))
            row=rows[-1]
            progress.update(status='completed' if (d/'completion_receipt.json').exists() else 'running',epoch_completed=int(row['epoch']))
            if progress['status']=='completed':
                native=list(csv.DictReader((n/'results.csv').read_text().splitlines()))[-1]
                metrics={}
                for k in ('metrics/mAP50(B)','metrics/mAP50-95(B)'):
                    metrics[k]={'arm_pp':float(row[k])*100,'native_pp':float(native[k])*100,'delta_pp':(float(row[k])-float(native[k]))*100}
                config=yaml.safe_load((d/'args.yaml').read_text())
                nconfig=yaml.safe_load((n/'args.yaml').read_text())
                diffs={k:{'arm':config.get(k),'native':nconfig.get(k)} for k in sorted(config.keys()|nconfig.keys()) if config.get(k)!=nconfig.get(k)}
                report['contrasts'].append({'arm':arm,'seed':seed,'epoch':int(row['epoch']),'metrics':metrics,'args_differences':diffs,'caution':'native and SSL detector initialization differ; cannot attribute this contrast exclusively to SSL'})
        report['progress'].append(progress)
for p in sorted((R/'cgkd_w1').glob('*/metrics_record.json')):
    report['native_independent_metrics'].append(json.loads(p.read_text()))
report['completed_finetunes']=sum(p['status']=='completed' for p in report['progress'])
report['remaining_finetunes']=9-report['completed_finetunes']
report['conclusions']={
 'positive_ssl_claim_supported':False,
 'paired_attribution_supported':False,
 'ssl_vs_native_initialization_confounded':True,
 'ssl_internal_nonbackbone_template_shared':True,
 'target_rgb_only_ssl_control_missing':True,
 'primary_gate_ap50_pp':1.0,
 'attribution_gates_ap50_pp':0.5,
 'pretraining_seed_repetitions':1,
 'next_actions':['Finish existing frozen arms','Run fixed last/EMA val evaluator for all endpoints','Add same nc5/non-backbone-template native with zero SSL steps','Add RGB-only SSL target-modal control','Keep route-specific recipes and evaluation identities separate; do not rank OEv1 independent last endpoint against OS-SSL CSV']}
(O/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('timestamp','completed_finetunes','contrasts','progress','conclusions')},ensure_ascii=False,indent=2))
