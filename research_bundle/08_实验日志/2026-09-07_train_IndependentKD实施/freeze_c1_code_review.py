"""Archive the completed cross-author C1 code reviews and exact reviewed release."""
import datetime
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REPORTS=('ROOT_INTEGRATION_RUNNABILITY_REVIEW.md','C1_GRADIENT_INDEPENDENT_CODE_REVIEW.md',
    'COMPATIBILITY_FRESH_PROCESS_CODE_REVIEW.md','EVIDENCE_BINDINGS_INDEPENDENT_REVIEW.md',
    'C1_FORMAL_ADMISSION_CODE_REVIEW.md','EVALUATOR_INDEPENDENT_REVIEW.md',
    'EVALUATOR_PROFILE_DISPATCH_CODE_REVIEW.md','EVALUATOR_CALLBACK_8_4_115_FIX.md','ANALYZER_REVIEW.md')
payload={name:(ROOT/name).read_text(encoding='utf-8-sig') for name in REPORTS}
script="""import datetime,json,sys
from pathlib import Path
data=json.load(sys.stdin)
B=Path(%r)
REL=B/'release_gpu5'
OUT=B/'c1_aggregate_code_review_gpu5'
OUT.mkdir(exist_ok=False)
reports=OUT/'review_reports';reports.mkdir()
for name,content in data.items():(reports/name).write_text(content,encoding='utf-8')
source_files=[]
for path in sorted(REL.rglob('*.py')):
 relative=path.relative_to(REL)
 target=OUT/'accepted_source'/relative
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(path.read_bytes())
 source_files.append(dict(relative=relative.as_posix(),accepted_copy=str(target)))
receipt=dict(status='ACCEPTED',blocking_issues_remaining=0,
 reviewer='/root integrating cross-author reviews in attached reports',
 scope='Code scope: paired C1; technical readiness separately requires actual six compatibility trajectories, 64-batch calibration, canary and same-release pinned tests.',
 excluded_claims=['No GPU acceptance fabricated by this code review','No L geometry acceptance','No C1 efficacy or four-arm attribution claim'],
 release=str(REL),review_reports=[str(reports/name) for name in data],source_files=source_files,
 created_at=datetime.datetime.now().astimezone().isoformat())
with (OUT/'review_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(str(OUT/'review_receipt.json'))
""" % BASE
# Ship the script and its JSON separately through the SSH Python command string.
import base64
encoded=base64.b64encode(script.encode('utf-8')).decode('ascii')
command='import base64;exec(base64.b64decode("'+encoded+'"))'
subprocess.run(['ssh','94',PY,'-c',"'"+command+"'"],input=json.dumps(payload).encode('utf-8'),check=True)
