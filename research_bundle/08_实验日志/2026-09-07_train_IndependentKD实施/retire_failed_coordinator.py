"""Retire only the old, paused coordinator that failed before any training batch."""
import subprocess

code=r'''
import datetime,json,subprocess
from pathlib import Path
p=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/formal_C1_gpu5')
s=json.loads((p/'state.json').read_text())
log=(p/'dispatch/ikdv2_C1_42_train_a1.log').read_text()
assert s['paused'] and all(s['seeds'][str(x)]['status']=='NOT_STARTED' for x in (0,123))
assert "ModuleNotFoundError: No module named 'ultralytics'" in log
assert not (p/'runs/C1_seed42').exists()
r=subprocess.run(['screen','-S','ikdv2_C1_coordinator','-X','quit'],capture_output=True,text=True)
with (p/'coordinator_retirement.json').open('x') as stream:
 json.dump(dict(retired_at=datetime.datetime.now().astimezone().isoformat(),screen='ikdv2_C1_coordinator',
  exit_code=r.returncode,reason='Paused predecessor; interpreter import failed before training. Replacement campaign formal_C1_gpu5_attempt2 owns all seeds.',
  trained_batches=0,old_results_preserved=True,replacement_touched=False),stream,indent=2)
print('Retired old paused coordinator only; replacement training untouched.')
'''
subprocess.run(['ssh','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=code.encode(),check=True)
