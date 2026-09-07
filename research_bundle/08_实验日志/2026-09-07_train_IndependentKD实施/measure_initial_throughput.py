"""Read epoch wall times only; do not inspect intermediate AP for decisions."""
import datetime
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/formal_C1_gpu5_attempt2/runs'
code='''import csv,json,statistics
from pathlib import Path
rows=[]
for seed in (42,0,123):
 path=Path(BASE)/('C1_seed'+str(seed))/'results.csv'
 with path.open() as stream: raw=[{k.strip():v for k,v in r.items()} for r in csv.DictReader(stream)]
 times=[dict(epoch=int(float(r['epoch'])),cumulative_seconds=float(r['time'])) for r in raw]
 differences=[times[i]['cumulative_seconds']-times[i-1]['cumulative_seconds'] for i in range(1,len(times))]
 median=statistics.median(differences) if differences else None
 rows.append(dict(seed=seed,completed_epoch_times=times,subsequent_epoch_seconds=median,
   e200_hours_at_same_speed=None if median is None else 200*median/3600))
print(json.dumps(dict(rows=rows,scope='initial throughput extrapolation, not promised ETA; no AP inspected')))
'''
result=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=('BASE='+repr(BASE)+'\n'+code).encode(),stdout=subprocess.PIPE,check=True)
value=json.loads(result.stdout);value['collected_at']=datetime.datetime.now().astimezone().isoformat()
out=ROOT/('initial_throughput_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(output=str(out),**value)))
