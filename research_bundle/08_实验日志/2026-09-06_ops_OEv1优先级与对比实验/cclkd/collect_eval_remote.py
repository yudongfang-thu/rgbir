"""Read-only collection of completed evaluation evidence; no weights/cache/images."""
import base64,datetime,json
from pathlib import Path
ROOT=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
ART=ROOT/'artifacts/oev1_priority_comparators_20260906'
RUNS=ROOT/'runs/oev1_comparators_20260906'
files={}
def take(path):
    if path.is_file() and path.suffix in {'.py','.json','.yaml','.yml','.txt','.md','.log'} and path.stat().st_size<3000000:
        data=path.read_bytes()
        files[str(path.relative_to(ROOT))]={'source':str(path),'size_bytes':len(data),
            'mtime':datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            'base64':base64.b64encode(data).decode('ascii')}
for name in ('cclkd_eval_summary.json','cclkd_eval_allocation.json','evaluate_cclkd_partial.py','cclkd_eval_worker.py'):
    take(ART/name)
for name in ['cclkd_partial_s42_canary_attempt1']+[f'cclkd_partial_s{s}_full_attempt1' for s in (0,42,123)]:
    run=RUNS/name
    for path in run.rglob('*'): take(path)
    for log in (ART/f'{name}.log',RUNS/'logs'/f'{name}.log'):take(log)
print(json.dumps({'captured_at_server':datetime.datetime.now().isoformat(),'files':files}))
