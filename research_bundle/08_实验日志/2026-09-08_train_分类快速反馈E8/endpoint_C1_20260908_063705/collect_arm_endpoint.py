"""Collect completed E8 arm small receipts; large raw predictions remain remote."""
import argparse
import base64
import datetime
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['N','C0','C1'],required=True);a=p.parse_args()
REMOTE=r'''
import base64,json
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_short_screen_E8_20260908_attempt1')
arm=ARM
names=['args.yaml','short_screen_config.yaml','short_screen_admission.json','runtime_ready.json',
       'short_training_receipt.json','results.csv','source_manifest.json','kd_batches.jsonl']
files=[root/'runs'/arm/n for n in names]
ev=root/'evaluations'/arm
files += [ev/n for n in ('short_evaluation_receipt.json','short_evaluation_contract.json',
                         'source_manifest.json','development_roster.txt')]
files += list((root/'queue').glob('short_'+arm+'_s42_E8*json'))
files += list(ev.glob('native_capture/*json'))
assert json.loads((root/'runs'/arm/'short_training_receipt.json').read_text())['status']=='SHORT_SCREEN_TRAINING_COMPLETED'
assert json.loads((ev/'short_evaluation_receipt.json').read_text())['status']=='SHORT_SCREEN_EVALUATION_COMPLETED'
rows=[]
for path in sorted(set(files)|set(ev.glob('native_capture/*.gz'))):
    if not path.is_file():continue
    s=path.stat();row=dict(path=str(path),relative=str(path.relative_to(root)),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    if path.suffix in ('.json','.jsonl','.yaml','.txt','.csv') and s.st_size<=3*1024**2:
        row['data']=base64.b64encode(path.read_bytes()).decode()
    else:row['remote_only']=True
    rows.append(row)
print(json.dumps(dict(arm=arm,files=rows,new_hash_computed=False,remote_mutations=False,new_gpu_workloads=0)))
'''.replace('ARM',repr(a.arm))
r=subprocess.run(['ssh','94',PY,'-'],input=REMOTE.encode(),capture_output=True,check=True)
data=json.loads(r.stdout)
out=ROOT/('endpoint_'+a.arm+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir(exist_ok=False)
for row in data['files']:
    if 'data' in row:
        dest=out/row['relative'];dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(base64.b64decode(row.pop('data')))
data['collected_at']=datetime.datetime.now().astimezone().isoformat()
(out/'collection_receipt.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(out/'collect_arm_endpoint.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(output=str(out),files=len(data['files'])),ensure_ascii=True))
