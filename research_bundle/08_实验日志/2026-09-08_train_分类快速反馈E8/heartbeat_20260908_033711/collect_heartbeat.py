"""Read-only E8 queue/status/resource collection, no CUDA or file hashes."""
import datetime
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REMOTE=r'''
import datetime,json,subprocess
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_short_screen_E8_20260908_attempt1')
guard='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/tools/project_resource_guard.py'
import sys
resources=json.loads(subprocess.check_output([sys.executable,guard,'inspect']))
rows={}
for arm in ('N','C0','C1'):
    row={}
    for key,path in {
        'queue_status':root/'queue'/('short_'+arm+'_s42_E8_train_status.json'),
        'progress':root/'runs'/arm/'progress.json',
        'training':root/'runs'/arm/'short_training_receipt.json',
        'training_failure':root/'runs'/arm/'short_training_failure.json',
        'evaluation':root/'evaluations'/arm/'short_evaluation_receipt.json',
        'evaluation_failure':root/'evaluations'/arm/'short_evaluation_failure.json',
    }.items():
        row[key]=json.loads(path.read_text()) if path.exists() else None
    rows[arm]=row
failure=root/'queue/failure.json'
print(json.dumps(dict(captured_at=datetime.datetime.now().astimezone().isoformat(),arms=rows,
    queue_failure=json.loads(failure.read_text()) if failure.exists() else None,
    resources=resources,new_cuda_workloads=0,new_hash_computed=False,remote_mutations=False)))
'''
r=subprocess.run(['ssh','94',PY,'-'],input=REMOTE.encode(),capture_output=True,check=True)
data=json.loads(r.stdout)
out=ROOT/('heartbeat_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir(exist_ok=False)
(out/'snapshot.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(out/'collect_heartbeat.py').write_bytes(Path(__file__).read_bytes())
usage=data['resources']['usage']
checks=dict(no_queue_failure=data['queue_failure'] is None,
    no_arm_failure=all(not a['training_failure'] and not a['evaluation_failure'] for a in data['arms'].values()),
    physical_gpus_at_most_three=len(usage['active_gpus'])<=3,
    two_cuda_per_gpu=all(v<=2 for v in usage['actual_cuda_pids'].values()),
    full_gpu_free_at_least_2048=all(data['resources']['gpus'][str(g)]['memory_free_mib']>=2048 for g in usage['active_gpus']),
    project_vram_below70=all(v<.7*data['resources']['gpus'][str(g)]['memory_total_mib'] for g,v in usage['actual_vram_mib'].items()),
    total_rss_below300GB=usage['project_rss_mib']*2**20<=300_000_000_000)
(out/'checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(dict(output=str(out),checks=checks,arms={a:{k:v for k,v in row.items() if k!='queue_status'} for a,row in data['arms'].items()},
    resources=dict(usage=usage,gpus=data['resources']['gpus'])),ensure_ascii=True))
