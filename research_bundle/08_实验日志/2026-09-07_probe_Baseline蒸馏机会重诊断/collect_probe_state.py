from pathlib import Path
import json,subprocess,datetime
ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
CODE=r'''
from pathlib import Path
import json,datetime
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_baseline_information_20260907')
state={'read_at':datetime.datetime.now().astimezone().isoformat(),'root':str(root),'records':{}}
for pattern in ('attempt3/*/progress.json','attempt3/*/summary.json','attempt3/queue_attempt1/*_status.json','attempt3/queue_attempt1/*acceptance.json','attempt3/queue_attempt1/completion.json','attempt2/throughput_stop_receipt.json'):
 for p in root.glob(pattern):state['records'][str(p.relative_to(root))]=json.loads(p.read_text())
print(json.dumps(state))
'''
r=subprocess.run(['ssh','94',PY,'-'],input=CODE.encode(),capture_output=True,check=True)
v=json.loads(r.stdout);stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
out=ROOT/('state_'+stamp+'.json');out.write_text(json.dumps(v,indent=2)+'\n')
print(str(out))
for k,q in v['records'].items():
 if k.endswith('progress.json') or k.endswith('completion.json'):print(k,json.dumps(q))
