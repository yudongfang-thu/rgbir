import ast,json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
content=(here/'random_worker.py').read_text(encoding='utf-8');ast.parse(content)
script="""import json,subprocess
from pathlib import Path
base=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_oev1_random_20260906')
check=json.loads((base/'canary_comparison_s42_attempt2.json').read_text())
assert check['status']=='passed'
target=base/'random_worker.py'
with target.open('x') as f:f.write(CONTENT)
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
cmd=f'exec {py} {target} > {base}/random_worker.log 2>&1'
subprocess.run(['screen','-dmS','oev1_random_queue_3seed','bash','-c',cmd],check=True)
print(json.dumps({'screen':'oev1_random_queue_3seed','gpu':2,'seeds':[42,0,123],'status':'dispatched','canary_check':check}))
""".replace('CONTENT',repr(content))
r=subprocess.run(['ssh','-o','BatchMode=yes','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=script.encode(),capture_output=True)
(here/'full_queue_dispatch.json').write_bytes(r.stdout)
(here/'full_queue_dispatch.stderr.txt').write_bytes(r.stderr)
print(r.stdout.decode(errors='replace'))
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
