import json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent
content=(here/'code/validate_random_canary.py').read_text(encoding='utf-8')
remote="""import json,subprocess,os
from pathlib import Path
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
base=root/'artifacts/rgbir_oev1_random_20260906'
source=base/'release_v1'
dest=source/'validate_random_canary_reviewed.py'
if not dest.exists():
    with dest.open('x') as f:f.write(CONTENT)
else:assert dest.read_text()==CONTENT
run=root/'runs/rgbir_oev1_random_20260906/canary_paired_random_s42_attempt1'
if not (run/'completion_receipt.json').exists():
    print(json.dumps({'status':'not_completed','log':(base/'canary_s42.log').read_text(errors='replace')[-4500:]}))
else:
    py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
    out=base/'canary_comparison_s42_attempt2.json'
    if not out.exists():
        cmd=[py,str(dest),'--paired-run',str(root/'runs/rgbir_object_evidence_v1_20260906/canary_paired_s42_attempt1'),
             '--random-run',str(run),'--gpu','2','--output',str(out)]
        r=subprocess.run(cmd,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        print(r.stdout)
        if r.returncode:print(r.stderr)
    else:print(out.read_text())
""".replace('CONTENT',repr(content))
r=subprocess.run(['ssh','-o','BatchMode=yes','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=remote.encode(),capture_output=True)
target=here/'canary_comparison_s42_attempt2.json'
assert not target.exists()
target.write_bytes(r.stdout)
print(r.stdout.decode(errors='replace'))
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
