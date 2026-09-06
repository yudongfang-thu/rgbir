import json,subprocess,runpy
from pathlib import Path
from datetime import datetime,timezone
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
base=root/'artifacts/rgbir_oev1_random_20260906'
run=root/'runs/rgbir_oev1_random_20260906/full_paired_random_s42_attempt1'
out={'utc':datetime.now(timezone.utc).isoformat(),'random':{},'osssl_remaining':[]}
for name in ['queue_status.json','canary_comparison_s42_attempt2.json']:
    p=base/name
    if p.exists():out['random'][name]=json.loads(p.read_text())
for name in ['launch_manifest.json','runtime_ready.json','progress.json','failure_receipt.json']:
    p=run/name
    if p.exists():out['random'][name]=json.loads(p.read_text())
out['random']['log_tail']=(base/'oev1_random_full_s42.log').read_text(errors='replace')[-1800:]
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader'],text=True)
allps=subprocess.check_output(['ps','-u','yudongfang','-o','pid,ppid,args'],text=True)
out['osssl_remaining']=[line for line in allps.splitlines() if '/artifacts/osssl_ir_20260906/' in line and ('train_native_rgbt.py' in line or 'bash /mnt/' in line and '/worker3.sh' in line)]
out['oev1_processes']=[line for line in allps.splitlines() if 'train_object_evidence.py' in line and ('--output' in line)]
print(json.dumps(out,indent=2))
