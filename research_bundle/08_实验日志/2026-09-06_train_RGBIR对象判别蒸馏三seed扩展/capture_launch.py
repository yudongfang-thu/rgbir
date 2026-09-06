"""Persist a read-only launch snapshot of the three queues and bound resources."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
R=B/'artifacts/rgbir_object_evidence_expand_20260906'
S=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
out=R/'launch_verification_snapshot1'
out.mkdir(exist_ok=False)
read=lambda p:json.loads(p.read_text())
record={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_unchanged':str(B/'artifacts/rgbir_object_evidence_v1_20260906/release_v2'),
        'runs':{},'workers':{},'canaries':{}}
for seed,arm in [(42,'paired'),(0,'weight0'),(123,'paired')]:
    campaign='rgbir_object_evidence_v1_20260906' if seed==42 else 'rgbir_object_evidence_expand_20260906'
    run=B/'runs'/campaign/f'full_{arm}_s{seed}_attempt1'
    record['runs'][str(seed)]={'path':str(run),'progress':read(run/'progress.json'),
        'launch':read(run/'launch_manifest.json'),'ready':read(run/'runtime_ready.json')}
for seed in [0,123]:
    record['workers'][str(seed)]=read(R/f'workers/full_s{seed}_attempt1/status.json')
    record['canaries'][str(seed)]=read(R/f'canary_comparison_s{seed}_attempt1.json')
guard=subprocess.run([sys.executable,str(S/'tools/project_resource_guard.py'),'inspect'],
                     check=True,capture_output=True,text=True)
record['resource_guard']=json.loads(guard.stdout)
(out/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'timestamp_utc':record['timestamp_utc'],
    'runs':{k:v['progress'] for k,v in record['runs'].items()},
    'resources':record['resource_guard']['usage']},indent=2))
