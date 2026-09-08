"""Collect immutable small JSON evidence snapshots; no images, weights, or tensors."""
from pathlib import Path
import base64,json,subprocess,sys
root=Path(__file__).parent;tag,attempt,version=sys.argv[1:4]
if not attempt.isdigit() or not version.isdigit():raise ValueError('Numeric attempt/version')
dest=root/('evidence_'+tag)
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_dfl_information_20260908/attempt'+attempt
code="""from pathlib import Path
import base64,json
root=Path(%r);rows=[]
for p in root.rglob('*'):
 if not p.is_file() or 'source_copies' in p.parts or 'weights' in p.parts:continue
 if p.suffix not in ('.json','.jsonl','.yaml'):continue
 if p.stat().st_size>8000000:raise ValueError('Unexpected evidence size '+str(p))
 rows.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,data=base64.b64encode(p.read_bytes()).decode()))
release=root.parent/('release_v'+%r)
for name in ('pinned_cpu.json','deployment.json'):
 p=release/name
 if p.is_file():rows.append(dict(path='pinned_source/'+name,bytes=p.stat().st_size,data=base64.b64encode(p.read_bytes()).decode()))
print(json.dumps(dict(files=rows)))
"""%(remote,version)
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
rows=json.loads(p.stdout)['files'];dest.mkdir(exist_ok=False);manifest=[]
for row in rows:
    p=dest/row['path'];b=base64.b64decode(row['data'])
    if dest.resolve() not in p.resolve().parents or len(b)!=row['bytes']:raise ValueError('Invalid collected file')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(b)
    manifest.append({k:row[k] for k in ('path','bytes')})
with (dest/'collection_receipt.json').open('x',encoding='utf-8') as f:json.dump(dict(remote=remote,files=manifest,weights_downloaded=False,images_downloaded=False,new_hash_computed=False),f,ensure_ascii=False,indent=2)
print('COLLECTED',len(manifest),sum(x['bytes'] for x in manifest),'bytes')
