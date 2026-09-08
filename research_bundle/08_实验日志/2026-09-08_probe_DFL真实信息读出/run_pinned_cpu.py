"""Actual pinned CPU tests; CUDA hidden, no checkpoint and no GPU lease needed."""
from pathlib import Path
import json,subprocess,sys
root=Path(__file__).parent;version=sys.argv[1]
if not version.isdigit():raise ValueError('Numeric release version required')
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
base='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts'
release=base+'/rgbir_dfl_information_20260908/release_v'+version
code="""from pathlib import Path
import os,json,subprocess,sys,time
root=Path(%r);os.environ['CUDA_VISIBLE_DEVICES']='';start=time.time()
if (root/'pinned_cpu.json').exists():raise FileExistsError('Preserve pinned receipt')
commands=[[sys.executable,'test_dfl_cpu.py','--output',str(root/'helper_cpu.json')],
 [sys.executable,'test_installed_native_cpu.py','--reference-dir',%r,'--output',str(root/'installed_native_cpu.json')]]
rows=[]
for i,cmd in enumerate(commands):
 p=subprocess.run(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=os.environ.copy())
 with (root/('pinned_cpu_'+str(i)+'.log')).open('xb') as f:f.write(p.stdout)
 rows.append(dict(command=cmd,returncode=p.returncode,log='pinned_cpu_'+str(i)+'.log'))
 if p.returncode:break
ok=len(rows)==2 and all(x['returncode']==0 for x in rows)
native=json.loads((root/'installed_native_cpu.json').read_text()) if (root/'installed_native_cpu.json').exists() else None
helper=json.loads((root/'helper_cpu.json').read_text()) if (root/'helper_cpu.json').exists() else None
ok=ok and native is not None and native['status']=='PASS' and native['GPU_used'] is False and helper is not None and helper['status']=='PASS'
r=dict(status='PASS' if ok else 'FAIL',tests=(helper['tests'] if helper else 0)+(native['checks'] if native else 0),failures=0 if ok else 1,errors=0,
 commands=rows,installed_native=native,seconds=time.time()-start,CUDA_VISIBLE_DEVICES='',new_hash_computed=False)
with (root/'pinned_cpu.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r));sys.exit(0 if ok else 1)
"""%(release,base+'/rgbir_independent_kd_v2_20260907/release_gpu5')
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.stdout:
 r=json.loads(p.stdout)
 with (root/('pinned_cpu_v'+version+'.json')).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
 print(json.dumps(r,indent=2))
if p.returncode:raise RuntimeError('Pinned tests failed; '+p.stderr.decode(errors='replace'))
