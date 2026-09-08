"""Collect a new snapshot of small probe artifacts only."""
from pathlib import Path
import json,base64,subprocess,sys
root=Path(__file__).parent;tag=sys.argv[1];attempt=sys.argv[2]
version=sys.argv[3] if len(sys.argv)>3 else attempt
if not attempt.isdigit() or not version.isdigit():raise ValueError('Numeric attempt/release required')
dest=root/('evidence_'+tag);dest.mkdir(exist_ok=False)
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_selection_coverage_20260908/attempt'+attempt
source="""from pathlib import Path
import json,base64
root=Path(%r);rows=[]
for p in root.rglob('*'):
 if not p.is_file() or 'source_copies' in p.parts or 'weights' in p.parts:continue
 if p.suffix not in ('.json','.jsonl','.yaml'):continue
 if p.stat().st_size>5000000:raise ValueError('Unexpected evidence size')
 rows.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,content=base64.b64encode(p.read_bytes()).decode()))
release=root.parent/('release_v'+%r)
for name in ('mapping_cpu.json','selection_cpu.json'):
 p=release/name
 if p.is_file():rows.append(dict(path='pinned_cpu/'+name,bytes=p.stat().st_size,content=base64.b64encode(p.read_bytes()).decode()))
print(json.dumps(dict(files=rows)))
"""%(remote,version)
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
r=subprocess.run(['ssh','94',PY,'-'],input=source.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
rows=json.loads(r.stdout)['files'];manifest=[]
for row in rows:
    p=dest/row['path']
    if dest.resolve() not in p.resolve().parents:raise ValueError('Invalid relative path')
    p.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(row['content'])
    if len(data)!=row['bytes']:raise AssertionError('Evidence size changed while read')
    with p.open('xb') as f:f.write(data)
    manifest.append({k:row[k] for k in ('path','bytes')})
with (dest/'collection_receipt.json').open('x',encoding='utf-8') as f:json.dump(dict(remote=remote,files=manifest,
    new_hash_computed=False,weights_downloaded=False,images_downloaded=False),f,indent=2)
print('COLLECTED',len(rows),'files',sum(x['bytes'] for x in rows),'bytes',dest)
