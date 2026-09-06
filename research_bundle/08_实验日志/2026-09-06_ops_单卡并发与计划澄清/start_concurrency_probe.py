import ast,json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
source=(here/'concurrency_probe.py').read_text(encoding='utf-8');ast.parse(source)
code="""import json,subprocess
from pathlib import Path
base=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/oev1_concurrency_20260906')
base.mkdir(parents=True,exist_ok=False)
script=base/'concurrency_probe.py'
with script.open('x') as f:f.write(SOURCE)
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
cmd=f'exec {py} {script} > {base}/worker.log 2>&1'
subprocess.run(['screen','-dmS','oev1_concurrency_probe_s0','bash','-c',cmd],check=True)
print(json.dumps({'status':'dispatched','screen':'oev1_concurrency_probe_s0','source':str(script),'gpu':2}))
""".replace('SOURCE',repr(source))
r=subprocess.run(['ssh','-o','BatchMode=yes','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=code.encode(),capture_output=True)
(here/'launch.json').write_bytes(r.stdout)
(here/'launch.stderr.txt').write_bytes(r.stderr)
print(r.stdout.decode(errors='replace'))
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
