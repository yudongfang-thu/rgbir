"""Deploy bounded scheduling helpers, then explicitly dry-run/apply/start."""
import argparse,ast,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/oev1_concurrency_20260906'
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['deploy','dry-run','apply','start','status']);a=parser.parse_args()
if a.action=='deploy':
    files={n:(HERE/p).read_text(encoding='utf-8') for n,p in [('revise_profiled_reservations.py','concurrency/revise_profiled_reservations.py'),('parallel_seed_worker.py','parallel_seed_worker.py'),('PARALLEL_HANDOFF_PLAN.md','PARALLEL_HANDOFF_PLAN.md')]}
    for name,source in files.items():
        if name.endswith('.py'):ast.parse(source)
    code='from pathlib import Path\nimport json\nbase=Path('+repr(BASE)+')\nfiles='+repr(files)+'''\nfor name,source in files.items():
    p=base/name
    if p.exists():assert p.read_text()==source, str(p)+' differs; do not overwrite'
    else:
        with p.open('x') as f:f.write(source)
print(json.dumps({'status':'deployed','files':list(files)}))
'''
elif a.action in ['dry-run','apply']:
    code='import subprocess\nsubprocess.run('+repr([PY,BASE+'/revise_profiled_reservations.py']+(['--apply'] if a.action=='apply' else []))+',check=True)\n'
elif a.action=='start':
    code='from pathlib import Path\nimport json,subprocess,time\nbase=Path('+repr(BASE)+')\npy='+repr(PY)+'''\nassert json.loads((base/'profiled_reservation_revision_20260906_receipt.json').read_text())['status']=='COMMITTED'
assert all(not (base/f'seed{s}_ownership.json').exists() for s in [0,123])
receipt=base/'parallel_dispatch.json'
with receipt.open('x') as f:json.dump({'time':time.time(),'seeds':[0,123],'gpus':[2,4],'seed123_full_waits_N42_eval':True},f,indent=2)
for seed,gpu in [(0,2),(123,4)]:
    name=f'oev1_random_parallel_s{seed}'
    command=f'exec {py} {base}/parallel_seed_worker.py --seed {seed} --gpu {gpu} > {base}/seed{seed}_worker.log 2>&1'
    subprocess.run(['screen','-dmS',name,'bash','-c',command],check=True)
print(receipt.read_text())
'''
else:
    code='from pathlib import Path\nimport json,subprocess,datetime\nbase=Path('+repr(BASE)+')\nresult={"captured_at":datetime.datetime.now().astimezone().isoformat()}\n'+'''for seed in [0,123]:
    p=base/f'seed{seed}_status.json'
    result[str(seed)]=json.loads(p.read_text()) if p.exists() else None
    result[f'log{seed}']=(base/f'seed{seed}_worker.log').read_text()[-2000:] if (base/f'seed{seed}_worker.log').exists() else None
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(result,indent=2))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=code.encode(),capture_output=True)
(HERE/(a.action+'_result.json')).write_bytes(r.stdout)
(HERE/(a.action+'_stderr.txt')).write_bytes(r.stderr)
print(r.stdout.decode(errors='replace'))
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
