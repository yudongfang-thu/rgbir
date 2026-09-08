"""Deploy fresh immutable source only; no model or GPU invocation."""
from pathlib import Path
import base64,io,json,subprocess,sys,tarfile
root=Path(__file__).parent;release=root/'release';version=sys.argv[1]
if not version.isdigit():raise ValueError('Numeric release version required')
dest='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_dfl_information_20260908/release_v'+version
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as ar:
    for p in sorted(release.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.yaml','.json','.md') and '__pycache__' not in p.parts:ar.add(p,arcname=p.relative_to(release).as_posix())
code="""from pathlib import Path
import io,tarfile,base64,json,ast
root=Path(%r);root.mkdir(parents=True,exist_ok=False);rows=[]
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(%r)),mode='r:gz') as ar:
 for m in ar:
  p=root/m.name
  if not m.isfile() or root not in p.resolve().parents:raise ValueError('Invalid source path')
  b=ar.extractfile(m).read();p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
  if p.read_bytes()!=b:raise ValueError('Source differs')
  if p.suffix=='.py':ast.parse(b.decode('utf-8-sig'),filename=str(p))
  rows.append(dict(path=str(p),bytes=len(b),byte_exact=True))
r=dict(status='SOURCE_DEPLOYED_PARSED_NOT_LAUNCHED',release=str(root),files=rows,new_hash_computed=False)
(root/'deployment.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
"""%(dest,base64.b64encode(buf.getvalue()).decode())
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
r=json.loads(p.stdout)
with (root/('deployment_v'+version+'.json')).open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
print(r['status'],len(r['files']))
