"""Snapshot finalized candidate into a new remote attempt with byte comparison."""
import base64
import datetime
import json
from pathlib import Path
import shlex
import subprocess

LOCAL = Path(__file__).resolve().parent
DEST = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_throughput_20260908/real_probe_attempt1'
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
names = ['run_real_probe.py', 'performance_candidate/pool_block16.py',
         'performance_candidate/benchmark_candidate.py',
         'performance_candidate/capture_real_batch.py',
         'performance_candidate/README.md', 'performance_candidate/candidate_review.md']
files = {name: base64.b64encode((LOCAL/name).read_bytes()).decode() for name in names}
code = '''import base64,json,sys
from pathlib import Path
x=json.loads(sys.stdin.read())
r=Path(x['destination'])
r.mkdir(parents=True,exist_ok=False)
rows=[]
for name,payload in x['files'].items():
 p=r/name
 p.parent.mkdir(parents=True,exist_ok=True)
 raw=base64.b64decode(payload)
 p.write_bytes(raw)
 assert p.read_bytes()==raw
 rows.append({'path':str(p),'size':len(raw),'byte_comparison':True})
print(json.dumps(rows))
'''
command = ' '.join(shlex.quote(s) for s in [PY, '-c', code])
result = subprocess.run(['ssh', '94', command],
                        input=json.dumps(dict(destination=DEST, files=files)).encode(),
                        capture_output=True, check=True)
receipt = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
               destination=DEST, files=json.loads(result.stdout),
               new_hashes=False, training_source_changed=False)
with (LOCAL/'real_probe_deployment.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
