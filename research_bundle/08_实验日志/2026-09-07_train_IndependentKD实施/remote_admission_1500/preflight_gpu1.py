import json,os,subprocess,sys,time
from pathlib import Path
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907')
REL=B/'release_gpu1'
PY=sys.executable
reports=[]
for name,command,cwd in [('operator_tests',[PY,'-m','unittest','discover','-s',str(REL),'-p','test_*.py'],REL),('reference_package',[PY,'-m','pytest','-q',str(REL/'reference_package')],REL)]:
 with (B/(name+'_gpu1.log')).open('x') as log:
  result=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
 reports.append(dict(name=name,exit_code=result.returncode))
(B/'cpu_gpu1_receipt.json').write_text(json.dumps(dict(stages=reports,time=time.time())))
if any(r['exit_code'] for r in reports):raise SystemExit(1)
