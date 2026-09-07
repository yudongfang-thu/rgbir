"""Collect current dependency source and checkpoint stat without loading weights."""
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent/'runtime_sources'
OUT.mkdir(exist_ok=False)
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
CODE=r'''
import importlib.util,json
from pathlib import Path
pkg=Path(importlib.util.find_spec('ultralytics').origin).parent
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
paths={'engine_trainer.py':pkg/'engine/trainer.py','data_build.py':pkg/'data/build.py',
       'resource_dispatch.py':root/'artifacts/rgbir_task_conditional_v1_20260907/release_v8/resource_dispatch.py'}
sources={}
for name,path in paths.items():
    stat=path.stat()
    sources[name]={'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'text':path.read_text()}
runs={str(s):root/'artifacts/rgbir_independent_kd_v2_20260907/formal_C1_gpu5_attempt2/runs'/('C1_seed'+str(s)) for s in (42,0,123)}
runs.update({name:root/'runs/rgbir_task_conditional_c_attribution_20260907'/('full_'+name+'_s42_attempt1') for name in ('c_shuffled','c_same_modal')})
checkpoints={}
for key,run in runs.items():
    checkpoints[key]={}
    for name in ('last.pt','best.pt'):
        p=run/'weights'/name
        stat=p.stat() if p.exists() else None
        checkpoints[key][name]={'path':str(p),'exists':stat is not None,
            'size':None if stat is None else stat.st_size,'mtime_ns':None if stat is None else stat.st_mtime_ns}
print(json.dumps({'sources':sources,'checkpoints':checkpoints}))
'''
r=subprocess.run(['ssh','94',PY,'-'],input=CODE.encode(),capture_output=True,check=True)
data=json.loads(r.stdout)
for name,row in data['sources'].items():
    (OUT/name).write_text(row.pop('text'),encoding='utf-8')
(OUT/'source_and_checkpoint_stat.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps(data))
