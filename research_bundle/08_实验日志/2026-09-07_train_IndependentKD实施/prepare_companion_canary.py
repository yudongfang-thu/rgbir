"""Measure C1 alongside one existing project task using its actual single-run peak."""
import json
import math
from pathlib import Path

B=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907')
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
profile_path=B/'canary_queue_gpu5/C1_canary_a1_resource_profile.json'
profile=json.loads(profile_path.read_text())
receipt=json.loads(Path(profile['result_receipt']).read_text())
if (profile.get('status')!='COMPLETED' or profile.get('measurement_valid') is not True or
    receipt.get('status')!='canary_completed' or receipt.get('arm')!='C1' or receipt.get('optimizer_updates',0)<24):
    raise ValueError('Actual single-run C1 canary did not pass')
resources=profile['resources']
vram=math.ceil(max(resources['per_gpu_peak_vram_mib'].values()))+2048
rss=max(8192,math.ceil(resources['peak_rss_mib'])+4096)
config=B/'calibrated_canary_configs_gpu5/C1.yaml'
output=B/'C1_companion_canary_attempt1'
job=dict(id='C1_companion_canary_a1',kind='other',stage='canary',formal=False,
    require_project_companion=True,requires_profile=str(profile_path),profile_key=profile['profile_key'],
    vram_mib=vram,rss_mib=rss,config=str(config),result_receipt=str(output/'completion_receipt.json'),
    command=[PY,str(B/'release_gpu5/train_independent.py'),'--config',str(config),'--output',str(output),
             '--arm','C1','--source','paired','--seed','42','--max-steps','24'])
with (B/'companion_canary_manifest_gpu5.json').open('x') as stream:
    json.dump(dict(jobs=[job],reservation_basis='Actual C1 single-canary NVML peak +2048MiB for batch-dependent native assignment growth; actual process-tree RSS +4096MiB, min8192'),stream,indent=2)
print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',vram_mib=vram,rss_mib=rss)))
