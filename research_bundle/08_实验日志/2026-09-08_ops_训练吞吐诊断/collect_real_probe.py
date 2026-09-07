"""Read small receipts/source only; retain large raw tensor bundles on 94."""
import base64
import json
from pathlib import Path
import shlex
import subprocess

LOCAL = Path(__file__).resolve().parent
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
ROOT = '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_throughput_20260908/real_probe_attempt1'
code = '''import base64,json
from pathlib import Path
r=Path(ROOT_LITERAL)
files={}
stats=[]
for p in r.rglob('*'):
 if not p.is_file():continue
 s=p.stat()
 row={'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns}
 stats.append(row)
 if p.suffix in ('.json','.jsonl','.py','.md','.log','.yaml','.csv') and s.st_size<2_000_000:
  files[str(p.relative_to(r))]=base64.b64encode(p.read_bytes()).decode()
print(json.dumps({'files':files,'stats':stats}))
'''.replace('ROOT_LITERAL', repr(ROOT))
command = ' '.join(shlex.quote(s) for s in [PY, '-c', code])
result = subprocess.run(['ssh', '94', command], capture_output=True, check=True)
payload = json.loads(result.stdout)
destination = LOCAL/'remote_real_probe_attempt1'
destination.mkdir(exist_ok=False)
for name, content in payload['files'].items():
    target = destination/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(content))
(destination/'collection_manifest.json').write_text(json.dumps(dict(remote=ROOT,
    files=payload['stats'], new_hashes=False, raw_tensor_bundles_downloaded=False), indent=2))
print(json.dumps(dict(destination=str(destination), files=len(payload['files']))))
