"""Read pinned remote sources and compare bytes, without file hashes or GPU work."""
from pathlib import Path
import base64,json,subprocess
root=Path(__file__).parent
local=root.parent/'2026-09-08_probe_快速方向筛选/newentry/release'
names=['localization_box_v2.py','direction_criterion.py','calibrate_direction.py']
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/release_v1'
source="""from pathlib import Path
import base64,json
root=Path(%r);out=[]
for name in %r:
 p=root/name;s=p.stat();out.append(dict(name=name,path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,data=base64.b64encode(p.read_bytes()).decode()))
print(json.dumps(out))
"""%(remote,names)
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=subprocess.run(['ssh','94',py,'-'],input=source.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
out=[];dest=root/'source_identity';dest.mkdir(exist_ok=False)
for r in json.loads(p.stdout):
 b=base64.b64decode(r.pop('data'));p=local/r['name']
 if b!=p.read_bytes():raise ValueError('Local and executed source differ: '+r['name'])
 with (dest/r['name']).open('xb') as f:f.write(b)
 r.update(local_path=str(p),byte_exact=True);out.append(r)
receipt=dict(status='PINNED_EXECUTED_SOURCE_BYTE_EXACT',files=out,new_hash_computed=False,new_GPU=False)
with (root/'SOURCE_IDENTITY.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(receipt['status'],len(out))
