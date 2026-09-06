"""Run on 94 with CPU Python. Copy small evidence; leave all original runs untouched."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import io
import json
import math
import runpy
import shutil
import statistics
import subprocess

BASE = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
OUT = BASE / 'artifacts/audit_rgbir_evening_20260906_1732'
OUT.mkdir(parents=True, exist_ok=False)
started = datetime.now(timezone.utc).isoformat()
collector = BASE / 'artifacts/rgbir_object_evidence_expand_20260906/analyze_three_seed_endpoints.py'
shutil.copyfile(collector, OUT / 'analyze_three_seed_endpoints.py')
module = runpy.run_path(str(collector))
endpoint = module['write_snapshot'](BASE, OUT / 'oev1_endpoints')
sources = []

def copy_small(src, dst):
    if not src.is_file():
        return
    before = src.stat()
    if before.st_size > 2 * 1024**2:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    after = src.stat()
    sources.append({'source': str(src), 'copy': str(dst.relative_to(OUT)), 'bytes': dst.stat().st_size,
                    'mtime_utc': datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
                    'changed_while_copying': (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)})

def window(rows, n=5):
    result = {}
    for k in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']:
        values = [float(r[k]) for r in rows if k in r]
        result[k] = statistics.mean(values) if values else None
    return result

health = []
for cell in endpoint['cells']:
    run = Path(cell['run'])
    item = {'seed': cell['seed'], 'arm': cell['arm'], 'endpoint_status': cell['status'], 'run': str(run)}
    if not run.is_dir():
        health.append(item)
        continue
    dest = OUT / 'raw_runs' / run.name
    for name in ['args.yaml', 'progress.json', 'results.csv', 'launch_manifest.json', 'completion_receipt.json',
                 'evaluation_val.json', 'failure_receipt.json', 'protocol_config.yaml', 'runtime_ready.json',
                 'run_evidence/run_receipt.json', 'eval_evidence/run_receipt.json']:
        copy_small(run/name, dest/name)
    for sub in ['run_evidence/metrics', 'eval_evidence/metrics']:
        if (run/sub).is_dir():
            for p in (run/sub).rglob('*.json'):
                copy_small(p, dest/p.relative_to(run))
    progress = run/'progress.json'
    item['progress'] = json.loads(progress.read_text()) if progress.is_file() else None
    csv_path = dest/'results.csv'
    parsed = list(csv.DictReader(csv_path.read_text().splitlines())) if csv_path.exists() else []
    rows = [{k.strip(): v for k,v in r.items()} for r in parsed if all(k is not None and isinstance(v,str) and v.strip() for k,v in r.items())]
    item['ignored_incomplete_csv_rows'] = len(parsed)-len(rows)
    item['completed_epochs'] = max((int(float(r['epoch'])) for r in rows), default=0)
    item['first5_train_means'] = window(rows[:5])
    item['last5_train_means'] = window(rows[-5:])
    item['csv_all_finite'] = all(math.isfinite(float(v)) for r in rows for v in r.values())
    if len(rows) >= 6 and 'time' in rows[-1]:
        item['recent_epoch_seconds'] = (float(rows[-1]['time']) - float(rows[-6]['time']))/5
        item['estimated_remaining_hours_at_recent_rate'] = (200-item['completed_epochs'])*item['recent_epoch_seconds']/3600
    kd_path = run/'kd_batches.jsonl'
    if kd_path.is_file():
        kd = []
        ignored = 0
        for line in kd_path.read_text().splitlines():
            try:
                kd.append(json.loads(line))
            except json.JSONDecodeError:
                ignored += 1
        last = [r for r in kd if max(0,item['completed_epochs']-5) <= r['epoch'] < item['completed_epochs']]
        def summarize(rows):
            keys = ['native_total', 'loss_unweighted', 'weighted_kd_total', 'total_loss', 'selected_count', 'quality_selected_mean', 'target_clipped_count', 'teacher_evidence_selected_mean', 'student_evidence_selected_mean']
            report = {k: statistics.mean(float(r[k]) for r in rows if k in r) for k in keys if any(k in r for r in rows)}
            report['logged_batches'] = len(rows)
            report['selected_fraction_of_base'] = sum(r['selected_count'] for r in rows)/max(1,sum(r['base_count'] for r in rows))
            report['zero_selection_batches'] = sum(r['selected_count']==0 for r in rows)
            report['zero_kd_batches'] = sum(r['loss_unweighted']==0 for r in rows)
            report['numeric_nonfinite'] = sum(not math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
            return report
        item['kd_all'] = summarize(kd)
        item['kd_last5_epochs'] = summarize(last)
        item['ignored_partial_jsonl_lines'] = ignored
        # Full stream stays on 94; retain explicitly labelled samples for inspection.
        (dest/'kd_first30_last30.partial.json').write_text(json.dumps({'source':str(kd_path),'full_rows':len(kd),'first30':kd[:30],'last30':kd[-30:]},indent=2)+'\n')
    health.append(item)

for campaign in ['rgbir_object_evidence_v1_20260906', 'rgbir_object_evidence_expand_20260906']:
    source = BASE/'artifacts'/campaign
    for pattern in ['*status*.json', 'workers/*/*status*.json', 'workers/*/*receipt*.json']:
        for p in source.glob(pattern):
            copy_small(p, OUT/'queue_artifacts'/campaign/p.relative_to(source))
    for pattern in ['*.log', 'workers/*/*.log']:
        for p in source.glob(pattern):
            with p.open('rb') as stream:
                stream.seek(max(0,p.stat().st_size-32768))
                raw=stream.read()
            dst=OUT/'queue_logs'/campaign/(p.name+'.partial_tail.txt')
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes(raw)

def command(argv):
    r = subprocess.run(argv, capture_output=True, text=True)
    return {'argv':argv,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}

system = {
    'gpu': command(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
    'cuda_processes': command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory,process_name','--format=csv,noheader']),
    'screens': command(['screen','-ls']),
    'free': command(['free','-m']),
}
ps = command(['ps','-eo','pid=,ppid=,user=,rss=,args='])
system['project_processes'] = [line for line in ps['stdout'].splitlines() if ('RGBT_campaign' in line or 'osssl' in line or 'object_evidence' in line) and 'yudongfang' in line]
guard=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/runs/.project_resource_leases.json')
copy_small(guard, OUT/'resource_leases.json')
(OUT/'system_snapshot.json').write_text(json.dumps(system,indent=2)+'\n')
summary={'started_at_utc':started,'ended_at_utc':datetime.now(timezone.utc).isoformat(),
         'complete_endpoints':endpoint['complete_endpoints'],'complete_seed_pairs':endpoint['complete_seed_pairs'],
         'health':health,'limits':'Epoch rates are recent-history estimates; fixed E200 endpoints are judged by the unchanged collector, not CSV AP. Full raw KD streams remain at source; local samples are explicitly partial.'}
(OUT/'health_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
(OUT/'source_inventory.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps(summary,indent=2))
