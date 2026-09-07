"""Copy small source/receipt evidence; exclude all tensors, weights and archives."""
import argparse
import base64
import datetime
import json
from pathlib import Path
import subprocess

REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    code='''import base64,json
from pathlib import Path
root=Path(REMOTE_VALUE)
files=[]
allowed={'.py','.json','.jsonl','.yaml','.yml','.md','.txt','.csv','.log','.sh','.gz'}
for path in root.rglob('*'):
 if not path.is_file() or path.suffix not in allowed or '__pycache__' in path.parts or path.stat().st_size>8*2**20:continue
 if any(part.startswith('release_') for part in path.relative_to(root).parts) or path.name.endswith(('.tar.gz','.tgz')):continue
 files.append(dict(relative=path.relative_to(root).as_posix(),remote=str(path),bytes=path.stat().st_size,data=base64.b64encode(path.read_bytes()).decode()))
print(json.dumps(files))
'''.replace('REMOTE_VALUE',repr(REMOTE))
    result=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=code.encode(),capture_output=True,check=True)
    records=json.loads(result.stdout);a.output.mkdir(parents=True,exist_ok=False)
    for record in records:
        target=a.output/record['relative'];target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(base64.b64decode(record.pop('data')))
    (a.output/'source_manifest.json').write_text(json.dumps(dict(captured_at=datetime.datetime.now().astimezone().isoformat(),
        files=records,excludes='all pt/pth/weights/archives/full image corpus'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),bytes=sum(x['bytes'] for x in records),output=str(a.output))))


if __name__=='__main__':main()
