import json,subprocess,sys,time
from pathlib import Path
import torch,ultralytics
B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907')
REL=B/'release_gpu5'
reports=[]
for name,command in [('operator_tests',[sys.executable,'-m','unittest','discover','-s',str(REL),'-p','test_*.py']),('reference_package',[sys.executable,'-m','pytest','-q',str(REL/'reference_package')])]:
 logpath=B/(name+'_gpu5.log')
 with logpath.open('x') as log:
  result=subprocess.run(command,cwd=REL,stdout=log,stderr=subprocess.STDOUT)
 reports.append(dict(name=name,exit_code=result.returncode,command=command,log=str(logpath)))
with (B/'cpu_gpu5_receipt.json').open('x') as stream:
 json.dump(dict(release=str(REL),environment=dict(torch=str(torch.__version__),ultralytics=str(ultralytics.__version__)),stages=reports,time=time.time()),stream,indent=2)
if any(r['exit_code'] for r in reports):raise SystemExit(1)
