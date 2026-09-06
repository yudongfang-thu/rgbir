"""Collect small original stage artifacts; weights/tensors remain on 94."""
from pathlib import Path
import argparse
import base64
import json
import subprocess

HERE=Path(__file__).resolve().parent
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);a=p.parse_args()
    output=HERE/'snapshots'/a.snapshot;output.mkdir(parents=True,exist_ok=False)
    script='''from pathlib import Path
import json,base64,datetime
b=Path(BASE)
files={};omitted=[]
for p in b.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(b)
 if rel.parts[0].startswith(('release_','geometry','audits','d2_drone_','d2_llvip_')):continue
 if '__pycache__' in rel.parts or p.suffix not in ('.py','.json','.jsonl','.yaml','.txt','.log'):continue
 if p.stat().st_size>10*2**20:
  omitted.append({'path':str(p),'bytes':p.stat().st_size});continue
 files[rel.as_posix()]=base64.b64encode(p.read_bytes()).decode('ascii')
extra=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/tools/project_resource_guard.py')
files['external_runtime/project_resource_guard.py']=base64.b64encode(extra.read_bytes()).decode('ascii')
donors=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/task_conditional_shuffled_loader_cpu_20260907')
for p in donors.glob('*.json'):
 if p.stat().st_size<10*2**20:files['shuffled_loader_cpu/'+p.name]=base64.b64encode(p.read_bytes()).decode('ascii')
formal=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_task_conditional_c_attribution_20260907')
for p in formal.glob('*/*.json'):
 if p.stat().st_size<3*2**20:files['formal_runs/'+p.relative_to(formal).as_posix()]=base64.b64encode(p.read_bytes()).decode('ascii')
print(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'files':files,'omitted':omitted}))
'''.replace('BASE',repr(BASE))
    result=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=script.encode(),capture_output=True,check=True)
    data=json.loads(result.stdout)
    rows=[]
    for name,content in data.pop('files').items():
        dst=output/name;dst.parent.mkdir(parents=True,exist_ok=True)
        raw=base64.b64decode(content);dst.write_bytes(raw)
        rows.append({'remote':BASE+'/'+name,'local':str(dst),'bytes':len(raw)})
    data['files']=rows
    (output/'collection_receipt.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'snapshot':str(output),'files':len(rows),'bytes':sum(r['bytes'] for r in rows)},ensure_ascii=False))

if __name__=='__main__':main()
