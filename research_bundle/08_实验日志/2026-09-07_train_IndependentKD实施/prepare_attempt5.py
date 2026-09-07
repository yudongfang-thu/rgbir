"""Prepare the next immutable admission attempt after the validator API fix."""
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
remote='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907'
repo='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction'
py=repo+'/environments/sn6-int8-kd/bin/python'
text=(root/'admission_manifest_attempt4.json').read_text(encoding='utf-8')
text=text.replace('release_gpu4','release_gpu5').replace('admission_queue_attempt4','admission_queue_attempt5')
text=text.replace('evaluator_profile_a1','evaluator_profile_a2').replace('evaluator_profile_attempt1','evaluator_profile_attempt2')
manifest=root/'admission_manifest_attempt5.json'
with manifest.open('x',encoding='utf-8') as stream:stream.write(text)
preflight=root/'preflight_gpu5.py'
with preflight.open('x',encoding='utf-8') as stream:
    stream.write("""import json,subprocess,sys,time
from pathlib import Path
import torch,ultralytics
B=Path(%r)
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
""" % remote)
launch=root/'launch_admission_attempt5.sh'
body=(root/'launch_admission_attempt4.sh').read_text(encoding='utf-8').replace('gpu4','gpu5').replace('attempt4','attempt5')
with launch.open('x',encoding='utf-8',newline='\n') as stream:stream.write(body)
for file in (manifest,preflight,launch):
    subprocess.run(['scp',str(file),'94:'+remote+'/'+file.name],check=True)
print('Prepared only; release deployment and screen launch remain separate.')
