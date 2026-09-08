"""Collect attributable small capture outputs; do not download weights/images or overwrite snapshots."""
from pathlib import Path
import base64,json,subprocess,sys
root=Path(__file__).parent;tag=sys.argv[1];attempt=sys.argv[2]
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_drone_teacher_capture_20260908'
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
script="""from pathlib import Path
import json,base64
root=Path(%r);attempt=%r;rows=[]
p=root/('attempt'+attempt)
if not p.is_dir():raise FileNotFoundError(p)
paths=list(p.rglob('*'))
for file in paths:
 if not file.is_file() or '__pycache__' in file.parts or file.suffix.lower() not in {'.json','.jsonl','.yaml','.gz','.txt'}:continue
 if 'sources' in file.relative_to(p).parts:continue
 if file.stat().st_size>8000000:raise ValueError('Unexpected large capture artifact: '+str(file))
 b=file.read_bytes();rows.append(dict(path=file.relative_to(p).as_posix(),bytes=len(b),data=base64.b64encode(b).decode()))
for name in ['pinned_cpu.json','deployment.json']:
 file=root/'release_v1'/name;b=file.read_bytes()
 rows.append(dict(path='pinned_source/'+name,bytes=len(b),data=base64.b64encode(b).decode()))
file=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/mappings/rgb_to_infrared_val.json')
unavailable=[]
if file.is_file():
 b=file.read_bytes()
 rows.append(dict(path='input_mapping/rgb_to_infrared_val.json',bytes=len(b),data=base64.b64encode(b).decode()))
else:unavailable.append(dict(path=str(file),status='NOT_FOUND',scope='unverified expected name from preparation code; not an observed mapping'))
print(json.dumps(dict(status='SMALL_CAPTURE_COLLECTED',remote=str(p),files=rows,unavailable_inputs=unavailable,new_hash_computed=False)))
"""%(remote,attempt)
r=subprocess.run(['ssh','94',py,'-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
payload=json.loads(r.stdout);dest=root/('evidence_'+tag);dest.mkdir(exist_ok=False)
for row in payload['files']:
    p=dest/row['path']
    if dest.resolve() not in p.resolve().parents:raise ValueError('Invalid capture path')
    p.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(row.pop('data'))
    with p.open('xb') as f:f.write(b)
payload['local']=str(dest)
(dest/'collection_receipt.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(payload['status'],len(payload['files']),sum(r['bytes'] for r in payload['files']))
