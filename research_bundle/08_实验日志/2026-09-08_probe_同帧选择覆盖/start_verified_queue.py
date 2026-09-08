"""Start the only GPU stage after CPU acceptance has been checked by the caller."""
from pathlib import Path
import json,subprocess,sys
root=Path(__file__).parent
version,attempt=sys.argv[1:3]
if not version.isdigit() or not attempt.isdigit():raise ValueError('Numeric new version and attempt required')
B='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_selection_coverage_20260908'
release=B+'/release_v'+version;out=B+'/attempt'+attempt
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
script="""from pathlib import Path
import json,subprocess
release=Path(%r);out=Path(%r)
if out.exists():raise FileExistsError('Preserve previous attempts')
for name in ('mapping_cpu.json','selection_cpu.json'):
 p=release/name
 if not p.is_file():raise ValueError('Required executed pinned CPU receipt missing: '+name)
 acceptance=json.loads(p.read_text())
 if acceptance.get('status')!='PASS' or acceptance.get('tests',0)<1 or acceptance.get('failures',0)!=0 or acceptance.get('errors',0)!=0:raise ValueError('Pinned CPU acceptance did not pass: '+name)
cmd=%r
subprocess.run(['screen','-dmS','selection_coverage_probe_s42','bash','-lc',cmd],check=True)
print(json.dumps(dict(status='QUEUED_EXISTING_GLOBAL_LEASE',release=str(release),output=str(out),screen='selection_coverage_probe_s42',new_hash_computed=False)))
"""%(release,out,'exec '+PY+' '+release+'/run_queue.py --release-dir '+release+' --output '+out+' > '+release+'/queue_launch_attempt'+attempt+'.log 2>&1')
r=subprocess.run(['ssh','94',PY,'-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
data=json.loads(r.stdout)
with (root/('queue_launch_attempt'+attempt+'.json')).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
print(data['status'],data['output'])
