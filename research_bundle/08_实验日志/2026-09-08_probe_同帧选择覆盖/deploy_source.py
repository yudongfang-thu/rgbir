"""Deploy a new immutable probe source directory; do not launch GPU work."""
from pathlib import Path
import base64,io,json,subprocess,tarfile,sys
root=Path(__file__).parent;release=root/'release'
version=sys.argv[1]
if not version.isdigit():raise ValueError('Numeric new release version required')
dest='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_selection_coverage_20260908/release_v'+version
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for p in sorted(release.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.yaml','.json','.md') and '__pycache__' not in p.parts:
            archive.add(p,arcname=p.relative_to(release).as_posix())
script="""from pathlib import Path
import io,tarfile,base64,json,py_compile
root=Path(%r);root.mkdir(parents=True,exist_ok=False);rows=[]
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(%r)),mode='r:gz') as ar:
 for member in ar.getmembers():
  p=root/member.name
  if not member.isfile() or root not in p.resolve().parents:raise ValueError('Invalid source member')
  p.parent.mkdir(parents=True,exist_ok=True);b=ar.extractfile(member).read()
  with p.open('xb') as f:f.write(b)
  if p.read_bytes()!=b:raise AssertionError('Source copy differs')
  rows.append(dict(path=str(p),bytes=len(b)))
for p in root.glob('*.py'):py_compile.compile(str(p),doraise=True)
print(json.dumps(dict(status='SOURCE_DEPLOYED_COMPILED_NOT_LAUNCHED',release=str(root),files=rows,new_hash_computed=False)))
"""%(dest,base64.b64encode(buffer.getvalue()).decode())
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
r=subprocess.run(['ssh','94',PY,'-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
data=json.loads(r.stdout)
with (root/('deployment_v'+version+'.json')).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
print(data['status'],len(data['files']),data['release'])
