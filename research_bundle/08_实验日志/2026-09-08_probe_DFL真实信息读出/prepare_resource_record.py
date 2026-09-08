"""Read GPU preflight and prior measured reservation provenance; no GPU work."""
from pathlib import Path
import datetime,json,subprocess
root=Path(__file__).parent
old=root.parent/'2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe/completion_receipt.json'
r=json.loads(old.read_text(encoding='utf-8'))
p=subprocess.run(['ssh','94','nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
gpus=[]
for line in p.stdout.decode().splitlines():
    a=[x.strip() for x in line.split(',')];gpus.append(dict(index=int(a[0]),name=a[1],total_mib=int(a[2]),used_mib=int(a[3]),free_mib=int(a[4]),utilization_percent=int(a[5])))
out=dict(status='READ_ONLY_PREFLIGHT',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpus=gpus,
    requested_reservation=dict(vram_mib=8192,rss_mib=32768),old_witness_resources=r['resources'],
    old_framework_allocated_mib=r['gpu_allocated_peak_mib'],old_framework_reserved_mib=r['gpu_reserved_peak_mib'],
    old_witness_receipt=str(old),not_GPU_admission=True,dynamic_original_lease_rechecks_at_start=True,new_hash_computed=False)
with (root/'RESOURCE_PREFLIGHT.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('PREFLIGHT_SAVED',len(gpus),'GPUs; dynamic lease still required')
