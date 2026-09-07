"""Copy an explicit project-local source list into a new remote probe directory."""
import argparse
import base64
import datetime
import json
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--attempt', required=True)
parser.add_argument('--file', action='append', required=True)
args = parser.parse_args()
if not args.attempt.replace('_', '').isalnum():
    raise ValueError('Simple attempt name required')
local = Path(__file__).resolve().parent
destination = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_throughput_20260908/'+args.attempt
python = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
files = {}
for name in args.file:
    path = (local/name).resolve()
    path.relative_to(local)
    files[name] = base64.b64encode(path.read_bytes()).decode()
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
result = subprocess.run(['ssh', '94', ' '.join(shlex.quote(s) for s in [python, '-c', code])],
    input=json.dumps(dict(destination=destination, files=files)).encode(), capture_output=True, check=True)
receipt = dict(created_at=datetime.datetime.now().astimezone().isoformat(), destination=destination,
    files=json.loads(result.stdout), new_hashes=False, production_source_changed=False)
with (local/(args.attempt+'_deployment.json')).open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
