"""Collect small E8 launch evidence and immutable source copies; no hashes."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
OPS=ROOT.parent/'2026-09-08_ops_训练吞吐诊断'
DEST=ROOT/'launch_evidence_attempt1'
DEST.mkdir(exist_ok=False)
REMOTE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_short_screen_E8_20260908_attempt1'
rows=[]
for name in ('manifest.json','short_N_s42_E8_train_status.json','short_N_s42_E8_train_admission.json'):
    source=REMOTE+'/queue/'+name
    result=subprocess.run(['ssh','94','cat',source],capture_output=True,check=True)
    parsed=json.loads(result.stdout)
    (DEST/name).write_bytes(result.stdout)
    rows.append(dict(source=source,copy=name))
    if name.endswith('_status.json'):
        assert parsed['status']=='RUNNING'
shutil.copytree(OPS/'short_screen_E8_release',DEST/'source')
shutil.copyfile(OPS/'performance_candidate/selected_only_v1.py',DEST/'selected_only_v1.py')
shutil.copyfile(OPS/'selected_only_independent_review/E8_RELEASE_REVIEW.md',DEST/'E8_RELEASE_REVIEW.md')
shutil.copytree(OPS/'remote_short_screen_E8_release_attempt1/admissions',DEST/'admissions')
shutil.copyfile(OPS/'remote_short_screen_E8_release_attempt1/freeze_receipt.json',DEST/'freeze_receipt.json')
for src in (OPS/'short_screen_E8_release').rglob('*'):
    if src.is_file():
        assert src.read_bytes()==(DEST/'source'/src.relative_to(OPS/'short_screen_E8_release')).read_bytes()
(DEST/'collection_receipt.json').write_text(json.dumps(dict(
    collected_at=datetime.datetime.now().astimezone().isoformat(),files=rows,
    source_byte_comparison=True,new_hash_computed=False,
    scope='Queue launched and N running; no completed E8 result'),indent=2),encoding='utf-8')
print(json.dumps(dict(status='COLLECTED',destination=str(DEST)),ensure_ascii=True))
