import json,subprocess
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
base=root/'artifacts/oev1_priority_comparators_20260906'
out={}
for name in ['cclkd_eval_worker_failure.json','cclkd_eval_summary.json','cclkd_eval_allocation.json']:
    p=base/name
    if p.exists():out[name]=json.loads(p.read_text())
for p in sorted(base.glob('cclkd_partial*.log')):out[p.name]=p.read_text(errors='replace')[-2800:]
out['worker_log']=(base/'cclkd_eval_worker.log').read_text(errors='replace')[-5000:]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader'],text=True)
print(json.dumps(out,indent=2))
