"""Materialize the three authorized C1 runs after actual technical acceptance."""
import json
from pathlib import Path
import subprocess

B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907')
REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
PY=str(REPO/'environments/sn6-int8-kd/bin/python')
REL=B/'release_gpu5'
CFG=B/'formal_C1_configs_gpu5'
canary_y=json.loads((B/'C1_y_canary_attempt1/completion_receipt.json').read_text())
if canary_y.get('status')!='canary_completed' or canary_y.get('optimizer_updates',0)<24:
    raise ValueError('The C1_y path has not completed its required canary')
companions=B/'companion_queue_gpu5/C1_companion_canary_a1_resource_profile.json'
profile=json.loads(companions.read_text())
if profile.get('companion_measurement',{}).get('verified') is not True:
    raise ValueError('This shared-training queue requires real companion measurement')
compatibility=[B/('compat_'+arm+'_s'+str(seed)+'_attempt'+('2' if arm=='N' and seed==42 else '1'))/'compatibility_receipt.json'
               for seed in (42,0,123) for arm in ('N','C0')]
command=[PY,str(B/'prepare_c1_stage_gpu5.py'),'--stage','formal','--release',str(REL),
    '--draft',str(B/'configs_draft_v1/drone_C1.yaml'),
    '--calibration',str(B/'c1_calibration64_attempt1/calibration_receipt.json'),
    '--canary',str(B/'C1_canary_attempt1/completion_receipt.json'),
    '--review',str(B/'c1_aggregate_code_review_gpu5/review_receipt.json'),
    '--tests',str(B/'cpu_gpu5_receipt.json'),'--output',str(CFG),
    '--compatibility',*map(str,compatibility)]
subprocess.run(command,check=True)
subprocess.run([PY,str(B/'formal_campaign_gpu5.py'),'prepare',
    '--campaign',str(B/'formal_C1_gpu5'),'--release',str(REL),'--repo',str(REPO),
    '--formal-config-dir',str(CFG),'--canary-profile',str(companions),
    '--evaluation-profile',str(B/'admission_queue_attempt5/evaluator_profile_a2_resource_profile.json'),
    '--python',PY],check=True)
print('Actual C1 readiness/configs and persistent campaign prepared, not launched by this script.')
