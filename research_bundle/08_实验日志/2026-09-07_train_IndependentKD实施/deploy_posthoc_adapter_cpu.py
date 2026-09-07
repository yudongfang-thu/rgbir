"""Mirror two small CPU sources and execute tests in the pinned interpreter."""
import base64
import json
from pathlib import Path
import subprocess

LOG=Path(__file__).resolve().parent
folder=LOG/'posthoc_class_adapter_v1'
payload={name:base64.b64encode((folder/name).read_bytes()).decode() for name in
         ('posthoc_class_adapter.py','test_posthoc_class_adapter.py')}
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/posthoc_class_adapter_cpu_v1'
py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
code='REMOTE='+repr(remote)+'\nPAYLOAD='+repr(payload)+'\n'+r'''
import base64,json,subprocess,sys
from pathlib import Path
p=Path(REMOTE)
p.mkdir(exist_ok=False)
for name,data in PAYLOAD.items():
    (p/name).write_bytes(base64.b64decode(data))
r=subprocess.run([sys.executable,'-m','unittest','test_posthoc_class_adapter','-v'],cwd=str(p),capture_output=True,text=True)
(p/'tests_pinned.txt').write_text(r.stdout+r.stderr)
(p/'README.md').write_text('CPU adapter tests only; no GPU or training. Local source: E:/SHARE/光sar/08_实验日志/2026-09-07_train_IndependentKD实施/posthoc_class_adapter_v1.\n')
print(json.dumps(dict(remote=str(p),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,python=sys.version,gpu_used=False)))
'''
r=subprocess.run(['ssh','94',py,'-'],input=code.encode('utf-8'),capture_output=True)
if r.returncode:
    raise RuntimeError(r.stderr.decode(errors='replace'))
result=json.loads(r.stdout)
with (folder/'pinned_cpu_receipt.json').open('x',encoding='utf-8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps(dict(remote=remote,exit_code=result['exit_code'],gpu_used=False)))
raise SystemExit(result['exit_code'])
