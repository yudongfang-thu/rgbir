"""Run the fixed new source's CPU tests under pinned Python with CUDA hidden."""
from pathlib import Path
import json,subprocess,sys
root=Path(__file__).parent;version=sys.argv[1]
if not version.isdigit():raise ValueError('Numeric version')
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_dfl_screen_20260908/release_v'+version
code="""from pathlib import Path
import os,json,sys,unittest,time,io,contextlib
root=Path(%r);os.chdir(root);os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(root))
if (root/'pinned_cpu.json').exists():raise FileExistsError('Preserve receipt')
start=time.time();buf=io.StringIO()
with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):
 suite=unittest.defaultTestLoader.discover(str(root),pattern='test_*cpu.py')
 result=unittest.TextTestRunner(stream=buf,verbosity=2).run(suite)
text=buf.getvalue()
with (root/'pinned_cpu.log').open('x') as f:f.write(text)
r=dict(status='PASS' if result.wasSuccessful() and result.testsRun>0 else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
 seconds=time.time()-start,scope='NEW_L3_CPU_TESTS_ONLY',CUDA_VISIBLE_DEVICES='',new_hash_computed=False)
with (root/'pinned_cpu.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(dict(receipt=r,log=text)));sys.exit(0 if r['status']=='PASS' else 1)
"""%remote
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.stdout:
 r=json.loads(p.stdout)
 with (root/('PINNED_TRAINING_CPU_v'+version+'.json')).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
 print(json.dumps(r['receipt']))
 if p.returncode:print(r['log'])
if p.returncode:raise RuntimeError('Pinned CPU failed '+p.stderr.decode(errors='replace'))
