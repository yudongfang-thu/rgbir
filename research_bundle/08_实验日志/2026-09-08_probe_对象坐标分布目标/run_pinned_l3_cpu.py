"""Correct the L3 CLI invocation; preserve failed discover receipt and tested source."""
from pathlib import Path
import json,subprocess,sys
root=Path(__file__).parent;version=sys.argv[1]
if not version.isdigit():raise ValueError('Numeric version')
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_dfl_screen_20260908/release_v'+version
code="""from pathlib import Path
import os,json,subprocess,sys,time
root=Path(%r);os.environ['CUDA_VISIBLE_DEVICES']='';start=time.time()
previous=json.loads((root/'pinned_cpu.json').read_text());log=(root/'pinned_cpu.log').read_text()
if previous['tests']!=23 or previous['failures']!=0 or previous['errors']!=1 or 'SystemExit: 2' not in log or 'test_l3_cpu (unittest.loader._FailedTest)' not in log:raise ValueError('Unexpected original preflight failure')
p=subprocess.run([sys.executable,str(root/'test_l3_cpu.py'),'--reference-dir','/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5','--output',str(root/'pinned_l3_cpu.json')],cwd=root,env=os.environ.copy(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
with (root/'pinned_l3_cpu.log').open('xb') as f:f.write(p.stdout)
l3=json.loads((root/'pinned_l3_cpu.json').read_text()) if (root/'pinned_l3_cpu.json').exists() else None
passed=p.returncode==0 and l3 is not None and l3['status']=='PASS' and l3['tests']==14
r=dict(status='PASS' if passed else 'FAIL',tests=22+(l3['tests'] if l3 else 0),failures=0 if passed else 1,errors=0,
 scope='ACTUAL_PINNED_CPU_LOSS_WRAPPER_QUEUE',prior_discovery_receipt='pinned_cpu.json',prior_failure_preserved=True,
 resolved_issue='L3 CLI requires explicit reference-dir/output; unchanged source invoked via its declared entry point.',
 wrapper_tests_passed=15,queue_tests_passed=7,L3=l3,source_changed=False,GPU_used=False,new_hash_computed=False,seconds=time.time()-start)
with (root/'pinned_cpu_accepted.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(dict(receipt=r,log=p.stdout.decode(errors='replace'))));sys.exit(0 if passed else 1)
"""%remote
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
p=subprocess.run(['ssh','94',py,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if p.stdout:
 r=json.loads(p.stdout)
 with (root/('PINNED_TRAINING_CPU_ACCEPTED_v'+version+'.json')).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
 print(json.dumps(r['receipt']))
 if p.returncode:print(r['log'])
if p.returncode:raise RuntimeError('Pinned L3 CPU failed '+p.stderr.decode(errors='replace'))
