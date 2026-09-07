"""Wait for this exact admission queue, then run both calibrated real canaries."""
import json
from pathlib import Path
import subprocess
import time

B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907')
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
queue=B/'admission_queue_attempt5'
while not (queue/'completion.json').exists():
    failures=sorted(queue.glob('*_failure.json'))
    if failures:
        raise RuntimeError('Admission failed; preserve attempts and do not launch canary: '+str(failures[0]))
    time.sleep(30)
if json.loads((queue/'completion.json').read_text()).get('status')!='COMPLETED':
    raise RuntimeError('Admission queue is not complete')
subprocess.run([PY,str(B/'prepare_canary_queue_gpu5.py')],check=True)
subprocess.run([PY,str(B/'release_gpu5/resource_dispatch.py'),
    '--manifest',str(B/'canary_manifest_gpu5.json'),
    '--output',str(B/'canary_queue_gpu5')],check=True)
