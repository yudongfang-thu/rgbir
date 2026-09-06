"""94 CPU-only overview refresh. No changes to runs, queues or artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import runpy
import subprocess

base=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
m=runpy.run_path(str(base/'artifacts/rgbir_object_evidence_expand_20260906/analyze_three_seed_endpoints.py'))
cells=[m['collect_cell'](base,seed,arm) for seed in (0,42,123) for arm in ('paired','weight0')]
for cell in cells:
    run=Path(cell['run'])
    progress=run/'progress.json'
    cell['progress']=json.loads(progress.read_text()) if progress.exists() else None
    p=run/'results.csv'
    rows=list(csv.reader(p.read_text().splitlines())) if p.exists() else []
    cell['completed_csv_epochs']=max((int(float(row[0])) for row in rows[1:] if row and row[0]),default=0)
ssl=[]
for arm in ['paired','sar_only','shuffled']:
    for seed in (0,42,123):
        run=base/'runs/osssl_ir_20260906'/f'{arm}_rgb_s{seed}_e200'
        p=run/'results.csv'
        rows=[{k.strip():v for k,v in row.items()} for row in csv.DictReader(p.read_text().splitlines())] if p.exists() else []
        epoch=max((int(float(row['epoch'])) for row in rows),default=0)
        receipt=run/'completion_receipt.json'
        metrics=list(run.glob('*metrics*.json'))+list(run.glob('evaluation*.json'))
        ssl.append({'arm':arm,'seed':seed,'run':str(run),'completed_epochs':epoch,'has_completion_receipt':receipt.exists(),
                    'independent_metric_files':[str(p) for p in metrics],
                    'last_csv_metrics':{k:rows[-1].get(k) for k in ['metrics/mAP50(B)','metrics/mAP50-95(B)']} if rows else None})
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True)
    return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
result={'captured_at_utc':datetime.now(timezone.utc).isoformat(),'oev1':{'cells':cells,**m['aggregate'](cells)},'osssl':ssl,
        'gpu':cmd(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader']),
        'cuda_pids':cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory,process_name','--format=csv,noheader']),
        'scope':'Read-only status refresh. Old completed endpoint proof is stored in 21:46 audit; no new GPU evaluation.'}
print(json.dumps(result,ensure_ascii=False,indent=2))
