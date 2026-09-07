"""Prepare C1/C1_y canaries only after the measured 64-batch coefficient qualifies."""
import json
from pathlib import Path
import subprocess
import sys

B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907')
PY=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python')
REL=B/'release_gpu5'
CFG=B/'calibrated_canary_configs_gpu5'
subprocess.run([str(PY),str(B/'prepare_c1_stage_gpu5.py'),'--stage','canary',
    '--release',str(REL),'--draft',str(B/'configs_draft_v1/drone_C1.yaml'),
    '--calibration',str(B/'c1_calibration64_attempt1/calibration_receipt.json'),
    '--output',str(CFG)],check=True)
jobs=[]
for arm in ('C1','C1_y'):
    output=B/(arm+'_canary_attempt1')
    config=CFG/(arm+'.yaml')
    jobs.append(dict(id=arm+'_canary_a1',kind='other',stage='canary',formal=False,
        bootstrap_profile=True,profile_key='drone-'+arm+'-train-B32-W4',vram_mib=16000,rss_mib=32768,
        config=str(config),result_receipt=str(output/'completion_receipt.json'),
        command=[str(PY),str(REL/'train_independent.py'),'--config',str(config),'--output',str(output),
            '--arm',arm,'--source','paired','--seed','42','--max-steps','24']))
with (B/'canary_manifest_gpu5.json').open('x') as stream:json.dump(dict(jobs=jobs),stream,indent=2)
print('Canary queue prepared; formal readiness not yet created.')
