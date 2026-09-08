"""Run declared test entry points against a fresh immutable release; CUDA hidden."""
from pathlib import Path
import json,subprocess,sys
root=Path(__file__).parent
version=sys.argv[1]
if not version.isdigit():raise ValueError('Numeric version')
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_dfl_screen_20260908/release_v'+version
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
code="""from pathlib import Path
import os,json,sys,unittest,time,io,contextlib,subprocess
root=Path(%r);os.chdir(root);os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(root))
if (root/'pinned_cpu_accepted.json').exists():raise FileExistsError('Preserve receipt')
start=time.time();buf=io.StringIO()
with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):
 suite=unittest.defaultTestLoader.loadTestsFromNames(['test_object_dfl_wrappers_cpu','test_object_dfl_queue_cpu'])
 result=unittest.TextTestRunner(stream=buf,verbosity=2).run(suite)
p=subprocess.run([sys.executable,'test_l3_cpu.py','--reference-dir','/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5','--output',str(root/'pinned_l3_cpu.json')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
l3=json.loads((root/'pinned_l3_cpu.json').read_text())
cli=subprocess.run([sys.executable,'calibrate_object_dfl.py','--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
ok=result.wasSuccessful() and result.testsRun==22 and p.returncode==0 and l3['status']=='PASS' and l3['tests']==14 and cli.returncode==0
log=buf.getvalue()+p.stdout+'\\nCALIBRATOR_CLI\\n'+cli.stdout
r=dict(status='PASS' if ok else 'FAIL',tests=result.testsRun+l3['tests'],wrapper_and_queue_tests=result.testsRun,L3=l3,calibrator_cli_exit=cli.returncode,source_changed=False,GPU_used=False,CUDA_VISIBLE_DEVICES='',new_hash_computed=False,seconds=time.time()-start,scope='ACTUAL_PINNED_CPU_V2_DECLARED_ENTRY_POINTS')
with (root/'pinned_cpu_accepted.json').open('x') as f:json.dump(r,f,indent=2)
with (root/'pinned_cpu_accepted.log').open('x') as f:f.write(log)
print(json.dumps(dict(receipt=r,log=log)));sys.exit(0 if ok else 1)
"""%remote
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.stdout:
 r=json.loads(p.stdout)
 with (root/('PINNED_TRAINING_CPU_ACCEPTED_v'+version+'.json')).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
 print(json.dumps(r['receipt']))
if p.returncode:raise RuntimeError('Pinned CPU failed '+p.stderr.decode(errors='replace'))
