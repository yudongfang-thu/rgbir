"""Compact read-only status for this deployment; no checkpoint or image reads."""
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
keys=('status','arm','seed','family','successful_updates','loader_batches','trajectory_exact',
 'optimizer_updates','optimizer_update_attempts','amp_skipped_updates','total_batches','nonzero_batches',
 'lambda_C1','resources','seconds','native_evidence_metrics_exact','metric_differences','selected_objects','error')
rows=[]
for pattern in ('compat_*/compatibility_receipt.json','c1_calibration*/calibration_receipt.json',
 'C1*canary*/completion_receipt.json','evaluator_profile_attempt2/evaluator_profile_receipt.json',
 'admission_queue_attempt5/*failure.json','canary_queue_gpu5/*failure.json','formal_C1_gpu5*/state.json'):
 for path in sorted(B.glob(pattern)):
  value=json.loads(path.read_text())
  rows.append(dict(path=str(path),**({k:value[k] for k in keys if k in value} or value)))
for queue in ('admission_queue_attempt5','canary_queue_gpu5'):
 for path in sorted((B/queue).glob('*events.jsonl')):
  lines=path.read_text().splitlines()
  if lines:
   last=json.loads(lines[-1]);rows.append(dict(queue=queue,job=path.stem,last_event=last))
print(json.dumps(dict(read_at=datetime.datetime.now().astimezone().isoformat(),records=rows)))
'''
result=subprocess.run(['ssh','94',PY,'-'],input=CODE.encode(),stdout=subprocess.PIPE,check=True)
data=json.loads(result.stdout)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
path=ROOT/('stage_status_'+stamp+'.json')
path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in data['records']:
    if 'last_event' not in row:print(json.dumps(row,ensure_ascii=True))
print('Raw status: '+str(path))
