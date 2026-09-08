"""Publish a fresh small source release then launch the existing lease driver."""
import base64,io,json,subprocess,tarfile,sys
from pathlib import Path
root=Path(__file__).parent
release=root/'newentry/release'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/release_v1'
out='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/screen_attempt1'
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for p in sorted(release.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.yaml','.md','.json') and '__pycache__' not in p.parts:
            archive.add(p,arcname=p.relative_to(release).as_posix())
encoded=base64.b64encode(buffer.getvalue()).decode()
source='''import base64,io,tarfile,json,subprocess,sys,py_compile
from pathlib import Path
release=Path(%r);out=Path(%r)
if release.exists() or out.exists():raise FileExistsError('Use a fresh release and attempt')
release.mkdir(parents=True)
rows=[]
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(%r)),mode='r:gz') as ar:
    for m in ar.getmembers():
        p=release/m.name
        if not m.isfile() or release not in p.resolve().parents:raise ValueError('Invalid archive entry')
        p.parent.mkdir(parents=True,exist_ok=True)
        data=ar.extractfile(m).read()
        with p.open('xb') as f:f.write(data)
        rows.append(dict(path=str(p),bytes=len(data),byte_identity=p.read_bytes()==data))
for p in release.glob('*.py'):py_compile.compile(str(p),doraise=True)
(release/'deployment_receipt.json').write_text(json.dumps(dict(status='SOURCE_DEPLOYED_COMPILED',files=rows,new_hash_computed=False)))
subprocess.run(['screen','-dmS','direction_screen_s42','bash','-lc',%r],check=True)
print(json.dumps(dict(status='DISPATCHED',release=str(release),output=str(out),source_files=len(rows),existing_global_lease=True)))
'''%(remote,out,encoded,'exec '+PY+' '+remote+'/run_direction_queue.py --release-dir '+remote+' --output '+out+' > '+remote+'/launch.log 2>&1')
r=subprocess.run(['ssh','94',PY,'-'],input=source.encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
sys.stdout.buffer.write(r.stdout)
if r.returncode==0:
    (root/'deployment_attempt1.json').write_bytes(r.stdout)
sys.exit(r.returncode)
