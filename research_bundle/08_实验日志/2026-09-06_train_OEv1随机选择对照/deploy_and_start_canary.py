import json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
files={p.name:p.read_text(encoding='utf-8') for p in (here/'code').iterdir() if p.suffix in ('.py','.yaml')}
files['EXPERIMENT_PLAN.md']=(here/'EXPERIMENT_PLAN.md').read_text(encoding='utf-8').replace('源代码diff和哈希核验','源代码diff和直接字节比较')
remote="""import json,subprocess,os
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
base=root/'artifacts/rgbir_oev1_random_20260906'
source=base/'release_v1'
source.mkdir(parents=True,exist_ok=False)
for name,content in json.loads(PAYLOAD).items():
    with (source/name).open('x') as f:f.write(content)
snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[list(map(int,x.split(','))) for x in snapshot.strip().splitlines()]
empty=[x[0] for x in rows if x[1]<100]
assert 2 in empty and len(empty)>=3,'Need empty GPU2 and >=2 empty cards after fourth-card admission'
repo=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
py=str(repo/'environments/sn6-int8-kd/bin/python')
run=root/'runs/rgbir_oev1_random_20260906/canary_paired_random_s42_attempt1'
assert not run.exists()
cmd=[py,str(repo/'tools/project_resource_guard.py'),'run','--job-id','oev1_random_canary_s42',
 '--kind','train','--candidate-gpu','2','--expected-vram-mib','10000','--expected-rss-mib','49152',
 '--free-safety-mib','2048','--non-formal-train','--',py,str(source/'train_object_evidence.py'),
 '--config',str(source/'config_drone.yaml'),'--output',str(run),'--arm','paired_random','--seed','42','--max-steps','24']
import shlex
shell='export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; exec '+shlex.join(cmd)+' > '+shlex.quote(str(base/'canary_s42.log'))+' 2>&1'
record={'source':str(source),'run':str(run),'screen':'oev1_random_canary_s42','command':cmd,'physical_gpu':2,
 'preflight':snapshot,'empty_before':empty,'policy':'User AGENTS 2.1 fourth-card exception; >=2 fully empty GPUs remain; global guard used'}
with (base/'allocation.json').open('x') as f:json.dump(record,f,indent=2)
subprocess.run(['screen','-dmS',record['screen'],'bash','-c',shell],check=True)
print(json.dumps(record,indent=2))
""".replace('PAYLOAD',repr(json.dumps(files)))
r=subprocess.run(['ssh','-o','BatchMode=yes','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=remote.encode(),capture_output=True)
(here/'canary_launch.json').write_bytes(r.stdout)
(here/'canary_launch.stderr.txt').write_bytes(r.stderr)
print(r.stdout.decode(errors='replace'))
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
