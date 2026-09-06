import json,subprocess
from pathlib import Path
base=Path(__file__).resolve().parent
files={name:(base/name).read_text(encoding='utf-8') for name in ['cclkd_eval_worker.py']}
code="""import json,subprocess
from pathlib import Path
base=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/oev1_priority_comparators_20260906')
for name,content in json.loads(PAYLOAD).items():
    target=base/name
    assert not target.exists(),target
    with target.open('x') as f:f.write(content)
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
cmd=f'exec {py} {base}/cclkd_eval_worker.py > {base}/cclkd_eval_worker.log 2>&1'
r=subprocess.run(['screen','-dmS','oev1comp_cclkd_eval_3seed','bash','-c',cmd],capture_output=True,text=True)
print(json.dumps({'screen':'oev1comp_cclkd_eval_3seed','returncode':r.returncode,'stderr':r.stderr}))
assert r.returncode==0
""".replace('PAYLOAD',repr(json.dumps(files)))
r=subprocess.run(['ssh','-o','BatchMode=yes','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=code.encode(),capture_output=True)
(base/'cclkd_eval_launch.json').write_bytes(r.stdout)
print(r.stdout.decode(errors='replace'))
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
