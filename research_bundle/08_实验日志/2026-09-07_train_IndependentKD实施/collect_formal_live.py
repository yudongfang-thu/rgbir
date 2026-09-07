"""Record actual launch events, optimizer progress and resource samples."""
import datetime
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
CODE=r'''
import datetime,json
from pathlib import Path
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907')
C=B/'formal_C1_gpu5_attempt2'
result=dict(read_at=datetime.datetime.now().astimezone().isoformat(),campaign=str(C),state=json.loads((C/'state.json').read_text()),seeds=[])
for seed in (42,0,123):
 path=C/'dispatch'/('ikdv2_C1_'+str(seed)+'_train_a1_events.jsonl')
 events=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
 launches=[x for x in events if x.get('status')=='LAUNCHED']
 row=dict(seed=seed,launched=launches,latest_dispatch_event=events[-1] if events else None)
 run=C/'runs'/('C1_seed'+str(seed))
 for filename in ('progress.json','runtime_ready.json','completion_receipt.json','failure_receipt.json','evaluation_val.json'):
  target=run/filename
  if target.exists():
   try:row[filename]=json.loads(target.read_text())
   except json.JSONDecodeError:row[filename]=dict(status='writer_in_progress')
 result['seeds'].append(row)
print(json.dumps(result))
'''
result=subprocess.run(['ssh','94',PY,'-'],input=CODE.encode(),stdout=subprocess.PIPE,check=True)
data=json.loads(result.stdout)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
path=ROOT/('formal_live_'+stamp+'.json')
path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in data['seeds']:
    print(json.dumps(dict(seed=row['seed'],launched=row['launched'],progress=row.get('progress.json'),
                         resources=(row.get('latest_dispatch_event') or {}).get('resources')),ensure_ascii=True))
print('Saved '+str(path))
