"""Deploy source only to a new immutable release; no GPU job launch."""
from pathlib import Path
import base64,json,subprocess,shutil,sys
root=Path(__file__).parent;version=sys.argv[1]
release=root/'release'
queue=release/'run_capture_queue.py'
if queue.exists() and queue.read_bytes()!=(root/'run_capture_queue.py').read_bytes():raise ValueError('Existing release queue differs')
if not queue.exists():shutil.copyfile(root/'run_capture_queue.py',queue)
rows=[]
for p in sorted(release.iterdir()):
    if not p.is_file() or p.suffix not in {'.py','.json','.md'}:continue
    b=p.read_bytes();rows.append(dict(path=p.name,data=base64.b64encode(b).decode()))
dest='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_drone_teacher_capture_20260908/release_v'+version
script="""from pathlib import Path
import ast,base64,json
root=Path(%r);root.mkdir(parents=True,exist_ok=False);rows=[]
for item in json.loads(%r):
 p=root/item['path'];b=base64.b64decode(item['data'])
 if p.parent!=root:raise ValueError('Invalid filename')
 if p.suffix=='.py':ast.parse(b,filename=str(p))
 with p.open('xb') as f:f.write(b)
 if p.read_bytes()!=b:raise ValueError('Deployment differs')
 rows.append(dict(path=str(p),bytes=len(b),byte_exact=True))
r=dict(status='SOURCE_DEPLOYED',directory=str(root),files=rows,new_hash_computed=False,GPU_started=False)
(root/'deployment.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
"""%(dest,json.dumps(rows))
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
r=subprocess.run(['ssh','94',py,'-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
result=json.loads(r.stdout)
with (root/('deployment_v'+version+'.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(result['status'],len(result['files']))
