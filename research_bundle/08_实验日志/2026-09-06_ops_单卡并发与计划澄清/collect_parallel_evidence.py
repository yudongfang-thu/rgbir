import json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
code='''import json,subprocess,datetime
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
base=root/'artifacts/oev1_concurrency_20260906'
repo=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
py=str(repo/'environments/sn6-int8-kd/bin/python')
names=['concurrency_summary.json','profiled_reservation_revision_20260906_before.json','profiled_reservation_revision_20260906_intent.json','profiled_reservation_revision_20260906_after.json','profiled_reservation_revision_20260906_receipt.json','parallel_dispatch.json']
for seed in [0,123]:names += [f'seed{seed}_status.json',f'seed{seed}_ownership.json',f'concurrent_canary_s{seed}.json',f'parallel_canary_comparison_s{seed}.json']
files={n:(base/n).read_text() for n in names if (base/n).exists()}
runs={'R42':'rgbir_oev1_random_20260906/full_paired_random_s42_attempt1','R0':'rgbir_oev1_random_20260906/full_paired_random_s0_attempt1','R123':'rgbir_oev1_random_20260906/full_paired_random_s123_attempt1','N42':'rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1','P0':'rgbir_object_evidence_expand_20260906/full_paired_s0_attempt1','N123':'rgbir_object_evidence_expand_20260906/full_weight0_s123_attempt1'}
progress={}
for label,relative in runs.items():
    p=root/'runs'/relative
    progress[label]={'exists':p.exists(),'run':str(p),'progress':json.loads((p/'progress.json').read_text()) if (p/'progress.json').exists() else None,'full_complete':(p/'completion_receipt.json').exists(),'eval_receipt':(p/'eval_evidence/run_receipt.json').exists()}
state=json.loads(subprocess.check_output([py,str(repo/'tools/project_resource_guard.py'),'inspect'],cwd=repo,text=True))
snapshot={'captured_at':datetime.datetime.now().astimezone().isoformat(),'runs':progress,'guard':state,'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)}
files['parallel_live_snapshot.json']=json.dumps(snapshot,indent=2)
print(json.dumps({'files':files}))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=code.encode(),capture_output=True)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
data=json.loads(r.stdout)
out=HERE/'parallel_evidence';out.mkdir(exist_ok=True)
for n,contents in data['files'].items():(out/n).write_text(contents,encoding='utf-8')
snapshot=json.loads(data['files']['parallel_live_snapshot.json'])
print(json.dumps({'captured_at':snapshot['captured_at'],'runs':snapshot['runs'],'gpus':snapshot['gpus'],'usage':snapshot['guard']['usage'],'files':list(data['files'])},indent=2))
