"""Collect small artifacts from one named performance attempt without weights."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--attempt', required=True)
args = parser.parse_args()
if not args.attempt.replace('_', '').isalnum():
    raise ValueError('Simple attempt name required')
local = Path(__file__).resolve().parent
python = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
root = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_throughput_20260908/'+args.attempt
code = '''import base64,json
from pathlib import Path
r=Path(ROOT_LITERAL)
files={}
stats=[]
for p in r.rglob('*'):
 if not p.is_file():continue
 s=p.stat()
 stats.append({'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns})
 if p.suffix in ('.json','.jsonl','.py','.md','.log','.yaml','.csv') and s.st_size<2_000_000:
  files[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
print(json.dumps({'files':files,'stats':stats}))
'''.replace('ROOT_LITERAL', repr(root))
result = subprocess.run(['ssh', '94', ' '.join(shlex.quote(s) for s in [python, '-c', code])],
    capture_output=True, check=True)
payload = json.loads(result.stdout)
destination = local/('remote_'+args.attempt)
destination.mkdir(exist_ok=False)
for name, content in payload['files'].items():
    target = destination/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(content))
(destination/'collection_manifest.json').write_text(json.dumps(dict(remote=root,
    files=payload['stats'], new_hashes=False, large_tensors_downloaded=False), indent=2))
print(json.dumps(dict(destination=str(destination), files=len(payload['files']))))
