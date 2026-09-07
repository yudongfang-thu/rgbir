"""Build derived audit products from an immutable read-only SSH snapshot."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
ANALYZER = WORKSPACE / '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/analyze_results.py'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    args=parser.parse_args()
    snapshot_dir=args.snapshot.resolve()
    snapshot=json.loads((snapshot_dir/'snapshot.json').read_text(encoding='utf-8'))
    module_spec=importlib.util.spec_from_file_location('result_analyzer',ANALYZER)
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    manifest={'captured_at':snapshot['captured_at'],'implementation_checks_passed':False,
              'runs':[{'arm':label[0],'seed':int(label[1:]),'source_arm':{'C':'paired','N':'weight0','R':'paired_random'}[label[0]],
                       'path':f'raw/{label}','protocol_id':'Drone_OEv1_frozen_E200_b32n64_workers4_lastEMA'} for label in snapshot['runs']]}
    result=module.analyze_manifest(manifest,snapshot_dir)
    derived=snapshot_dir/'derived'
    derived.mkdir(exist_ok=False)
    (snapshot_dir/'analysis_input.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (derived/'results_analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    resources=json.loads((snapshot_dir/'resource_leases_raw.json').read_text())
    leases=resources['leases']
    live={'capture':snapshot['captured_at'],'active_leases':len(leases),'physical_gpus':sorted({g for lease in leases.values() for g in lease['gpus']}),
          'reserved_rss_mib':sum(lease['expected_rss_mib'] for lease in leases.values()),
          'measured_project_rss_gib':snapshot['project_rss_kib_readonly']/2**20,
          'leases':[{'job_id':lease['job_id'],'gpus':lease['gpus'],'expected_rss_mib':lease['expected_rss_mib'],
                     'expected_vram_mib':lease['expected_vram_mib'],'peak_rss_mib':lease.get('peak_rss_mib'),
                     'per_gpu_peak_vram_mib':lease.get('per_gpu_peak_vram_mib')} for lease in leases.values()]}
    (derived/'resource_summary.json').write_text(json.dumps(live,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for label,run in snapshot['runs'].items():
        eta=run['eta'] or {}
        complete=run['completion']
        # progress.epoch is current in-progress epoch; CSV epoch is completed.
        # Final CSV row lacks normal validation columns when val=False; completion wins.
        completed=complete['last_epoch'] if complete else eta.get('completed_epoch',0)
        remaining=0 if complete else max(0,200-completed)*eta.get('recent_median_epoch_seconds',0)/3600
        rows.append({'label':label,'completed_epochs':completed,'current_progress_epoch':(run['progress'] or {}).get('epoch'),
                     'training_complete':bool(complete),'independent_endpoint':run['eval_receipt_exists'] and bool(run['evaluation']),
                     'estimated_remaining_train_hours':remaining,'recent_median_epoch_seconds':eta.get('recent_median_epoch_seconds')})
    (derived/'progress_eta.json').write_text(json.dumps({'captured_at':snapshot['captured_at'],'rows':rows,
        'assumptions':'Recent throughput unchanged. Excludes resource queue and evaluation. progress.epoch is in-progress; CSV epoch is completed. Completion receipt overrides irregular last CSV row.'},indent=2)+'\n',encoding='utf-8')
    with (derived/'endpoint_table.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=['arm','seed','status',*module.METRICS])
        writer.writeheader()
        for record in result['records']:
            writer.writerow({'arm':record['arm'],'seed':record['seed'],'status':record['status'],**(record['metrics_percent'] or {})})
    print(json.dumps({'analysis':str(derived/'results_analysis.json'),
          'status':{f"{r['arm']}{r['seed']}":{'status':r['status'],'issues':r['issues']} for r in result['records']},
          'pair':result['comparisons']['C_minus_N'],'resource':live,'progress':rows},indent=2))


if __name__=='__main__':main()
