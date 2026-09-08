"""Read small immutable evidence from current direction run; never weights/images."""
from pathlib import Path
import subprocess,sys,json,base64
root=Path(__file__).parent
tag=sys.argv[1]
target=root/('results_'+tag);target.mkdir(exist_ok=False)
remote='''from pathlib import Path
import json,base64
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/screen_attempt1')
names={'calibration_receipt.json','completion_receipt.json','canary.json','initialization_check.json','direction_evaluation_receipt.json','evaluation_contract.json','actual_native_evaluation_profile.json','development_roster.txt','failure.json','direction_failure.json','direction_evaluation_failure.json','args.yaml','direction_config.yaml','calibration_batches.jsonl'}
files=[]
for p in root.rglob('*'):
 if not p.is_file() or 'source_copies' in p.parts or 'weights' in p.parts:continue
 rel=p.relative_to(root)
 if p.name in names or (rel.parts[0]=='queue' and p.suffix=='.json') or (rel.parts[0]=='effective_configs' and p.suffix=='.yaml'):
  if p.stat().st_size>5000000:continue
  files.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'content':base64.b64encode(p.read_bytes()).decode()})
print(json.dumps({'files':files}))
'''
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
r=subprocess.run(['ssh','94',PY,'-'],input=remote.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
data=json.loads(r.stdout);manifest=[]
for f in data['files']:
    p=target/f['path']
    if target.resolve() not in p.resolve().parents:raise ValueError('Invalid evidence relative path')
    p.parent.mkdir(parents=True,exist_ok=True);contents=base64.b64decode(f['content'])
    with p.open('xb') as out:out.write(contents)
    if len(contents)!=f['bytes']:raise ValueError('File changed while reading')
    manifest.append({k:f[k] for k in ('path','bytes')})
(target/'collection_receipt.json').write_text(json.dumps(dict(files=manifest,new_hash_computed=False,weights_downloaded=False),indent=2))
for p in target.rglob('direction_evaluation_receipt.json'):
    d=json.loads(p.read_text());print(d['dataset'],d['arm'],{k:d.get(k) for k in ('mAP50_95','AP50','AP75','seconds')})
print('COLLECTED',len(manifest),'files',sum(f['bytes'] for f in manifest),'bytes')
