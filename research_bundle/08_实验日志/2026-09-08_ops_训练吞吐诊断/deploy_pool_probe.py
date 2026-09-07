"""Copy only finalized new candidate scripts; preserve existing remote files."""
import base64
import datetime
import json
from pathlib import Path
import shlex
import subprocess

LOCAL=Path(__file__).resolve().parent
DEST='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_throughput_20260908/pool_probe_attempt1'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
names=['run_pool_probe.py','performance_candidate/pool_block16.py',
       'performance_candidate/benchmark_candidate.py']
files={name:base64.b64encode((LOCAL/name).read_bytes()).decode() for name in names}
code='''import base64,json,sys\nfrom pathlib import Path\nx=json.loads(sys.stdin.read())\nr=Path(x['destination'])\nr.mkdir(parents=True,exist_ok=False)\nrows=[]\nfor name,payload in x['files'].items():\n p=r/name\n p.parent.mkdir(parents=True,exist_ok=True)\n raw=base64.b64decode(payload)\n p.write_bytes(raw)\n assert p.read_bytes()==raw\n rows.append({'path':str(p),'size':len(raw),'byte_comparison':True})\nprint(json.dumps(rows))\n'''
command=' '.join(shlex.quote(s) for s in [PY,'-c',code])
r=subprocess.run(['ssh','94',command],input=json.dumps(dict(destination=DEST,files=files)).encode(),
                 capture_output=True,check=True)
receipt=dict(created_at=datetime.datetime.now().astimezone().isoformat(),destination=DEST,
             files=json.loads(r.stdout),new_hashes=False,training_source_changed=False)
with (LOCAL/'pool_probe_deployment.json').open('x',encoding='utf-8') as stream:
    json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
