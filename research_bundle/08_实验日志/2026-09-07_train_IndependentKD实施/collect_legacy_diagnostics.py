"""Copy completed/current small supplemental evaluation evidence read-only."""
import datetime
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/legacy_diagnostics_v1'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
script='''from pathlib import Path
import io,json,sys,tarfile
root=Path(REMOTE)
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() in ('.json','.jsonl','.csv','.yaml','.yml','.txt','.py','.log','.gz') and p.stat().st_size<=20*2**20)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as stream:
 for path in files:stream.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
'''
result=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=('REMOTE='+repr(REMOTE)+'\n'+script).encode('utf-8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
out=HERE/('legacy_diagnostics_snapshot_'+stamp);out.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as stream:
    for entry in stream.getmembers():
        if not entry.isfile() or Path(entry.name).is_absolute() or '..' in Path(entry.name).parts:
            raise ValueError('Unexpected archive member')
    stream.extractall(out)
files=sorted(p for p in out.rglob('*') if p.is_file())
manifest=dict(collected_at=datetime.datetime.now().astimezone().isoformat(),remote_root=REMOTE,
    files=[dict(relative=p.relative_to(out).as_posix(),remote=REMOTE+'/'+p.relative_to(out).as_posix(),bytes=p.stat().st_size) for p in files])
(out/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
completed=[]
for path in out.glob('*/reevaluation_receipt.json'):
    row=json.loads(path.read_text());completed.append(dict(arm=row['normalized_method_arm'],seed=row['seed'],status=row['status'],historical_exact=row['historical_five_metrics_exact']))
failures=[p.relative_to(out).as_posix() for p in out.rglob('*failure*.json')]
print(json.dumps(dict(output=str(out),files=len(files),bytes=sum(p.stat().st_size for p in files),completed=completed,failures=failures)))
