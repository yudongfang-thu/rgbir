"""Start the accepted canary/full queue in screen, after pinned CPU checks."""
from pathlib import Path
import json,subprocess,sys
root=Path(__file__).parent;version=sys.argv[1];attempt=sys.argv[2]
review=(root/'SOURCE_REVIEW.md').read_text(encoding='utf-8')
if 'READY_FOR_CANARY' not in review.splitlines()[0]:raise ValueError('Independent canary source acceptance required')
base='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_drone_teacher_capture_20260908'
release=base+'/release_v'+version;output=base+'/attempt'+attempt
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
script="""from pathlib import Path
import json,subprocess
release=Path(%r);output=Path(%r);py=%r
receipt=json.loads((release/'pinned_cpu.json').read_text())
if receipt.get('status')!='PASS' or receipt.get('tests')!=6 or receipt.get('failures')!=0 or receipt.get('errors')!=0:raise ValueError('Pinned CPU contract not accepted')
if output.exists():raise FileExistsError('New attempt required')
session='drone_teacher_T42_s42_a'+%r
existing=subprocess.run(['screen','-ls'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True).stdout
if session in existing:raise ValueError('Session already exists')
log=output.parent/(output.name+'_queue.log')
cmd=[py,str(release/'run_capture_queue.py'),'--release-dir',str(release),'--output',str(output)]
subprocess.run(['screen','-L','-Logfile',str(log),'-dmS',session]+cmd,check=True)
print(json.dumps(dict(status='QUEUE_DISPATCHED',session=session,output=str(output),release=str(release),log=str(log),command=cmd,new_resource_pool=False,new_hash_computed=False)))
"""%(release,output,py,attempt)
r=subprocess.run(['ssh','94',py,'-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
value=json.loads(r.stdout)
with (root/('queue_launch_attempt'+attempt+'.json')).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
print(value['status'],value['session'])
