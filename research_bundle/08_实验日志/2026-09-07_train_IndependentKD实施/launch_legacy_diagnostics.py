"""Launch six accepted read-only evaluations through the existing shared lease."""
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907'
REPO='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction'
PY=REPO+'/environments/sn6-int8-kd/bin/python'
CODE=BASE+'/legacy_endpoint_eval_prepare_v1'
OUT=BASE+'/legacy_diagnostics_v1/dispatch_attempt1'
SESSION='ikdv2_legacy_diag_20260907'
payload='''from pathlib import Path
import datetime,json,subprocess,sys
base=Path(BASE);code=Path(CODE);out=Path(OUT)
if out.exists():raise FileExistsError('Preserve prior dispatcher attempt')
sys.path.insert(0,str(code))
from legacy_checkpoint_evaluate import require_review
require_review(code/'review_receipt.json')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,memory.free','--format=csv,noheader'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
argv=['screen','-dmS',SESSION,PY,str(base/'release_gpu5/resource_dispatch.py'),'--manifest',str(code/'candidate_bundle_v1/queue_candidate.json'),'--output',str(out),'--repo',REPO]
subprocess.run(argv,cwd=REPO,check=True)
value=dict(status='SCREEN_DISPATCH_STARTED',time=datetime.datetime.now().astimezone().isoformat(),session=SESSION,argv=argv,output=str(out),nvidia_smi_before=gpu,compute_apps_before=processes,
    shared_lease_required=True,physical_gpu_hardcoded=False,fourth_gpu_exception_invoked=False,new_training_jobs=0,evaluation_jobs=6)
with (base/'legacy_diagnostics_launch_attempt1.json').open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps(value))
'''
for name,value in (('BASE',BASE),('CODE',CODE),('OUT',OUT),('REPO',REPO),('PY',PY),('SESSION',SESSION)):
    payload=name+'='+repr(value)+'\n'+payload
result=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=payload.encode('utf-8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
value=json.loads(result.stdout)
with (HERE/'legacy_diagnostics_launch_attempt1.json').open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
print(json.dumps({k:value[k] for k in ('status','time','session','output','evaluation_jobs')}))
