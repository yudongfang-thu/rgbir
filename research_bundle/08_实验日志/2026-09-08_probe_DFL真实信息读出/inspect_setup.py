"""Read actual setup and AMP-check sources; never invoke their routines."""
from pathlib import Path
import base64,json,subprocess
root=Path(__file__).parent
code="""import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import inspect,json,base64
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.checks import check_amp
from pathlib import Path
out=[]
for name,fn in [('BaseTrainer_setup_train',BaseTrainer._setup_train),('check_amp',check_amp)]:
 data=inspect.getsource(fn).encode();out.append(dict(name=name,path=inspect.getsourcefile(fn),data=base64.b64encode(data).decode()))
for name,path in [('runtime','/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5/runtime.py'),('train_direction','/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/release_v1/train_direction.py')]:
 p=Path(path);out.append(dict(name=name,path=path,data=base64.b64encode(p.read_bytes()).decode()))
print(json.dumps(dict(sources=out,no_functions_invoked=True,no_GPU=True)))
"""
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
raw=json.loads(p.stdout);dest=root/'setup_sources';dest.mkdir(exist_ok=False);rows=[]
for r in raw['sources']:
 b=base64.b64decode(r.pop('data'))
 with (dest/(r['name']+'.py')).open('xb') as f:f.write(b)
 rows.append(dict(r,bytes=len(b)))
with (dest/'receipt.json').open('x',encoding='utf-8') as f:json.dump(dict(sources=rows,no_functions_invoked=True,no_GPU=True,new_hash_computed=False),f,ensure_ascii=False,indent=2)
print('SETUP_SOURCE_INSPECTED',len(rows))
