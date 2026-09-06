import ast,json,subprocess
from pathlib import Path
base=Path(__file__).resolve().parent
names=['evaluate_cclkd_partial.py','stop_osssl_priority.py']
files={name:(base/name).read_text(encoding='utf-8') for name in names}
for name,content in files.items():ast.parse(content,filename=name)
payload=json.dumps(files)
code="""import json,subprocess
from pathlib import Path
base=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/oev1_priority_comparators_20260906')
base.mkdir(parents=True,exist_ok=True)
files=json.loads(PAYLOAD)
for name,content in files.items():
    dest=base/name
    if dest.exists():assert dest.read_text()==content
    else:
        with dest.open('x') as f:f.write(content)
print(subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout)
repo=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
print(subprocess.run([str(repo/'environments/sn6-int8-kd/bin/python'),str(repo/'tools/project_resource_guard.py'),'inspect'],capture_output=True,text=True).stdout)
""".replace('PAYLOAD',repr(payload))
r=subprocess.run(['ssh','-o','BatchMode=yes','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=code.encode(),capture_output=True)
(base/'eval_deploy_preflight.txt').write_bytes(r.stdout)
print(r.stdout.decode(errors='replace'))
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
