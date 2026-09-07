"""Sequential real-cadence throughput profiles in the existing shared lease."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REL = str(BASE/'rgbir_independent_kd_v2_20260907/release_gpu5')
sys.path.insert(0, str(BASE/'rgbir_task_conditional_v1_20260907/release_v8'))
from resource_dispatch import run_job

queue = ROOT/'queue_attempt1'
queue.mkdir(exist_ok=False)
jobs = []
for arm in ('N', 'C0', 'C1'):
    command = [PY, str(ROOT/'short_screen_draft/short_profile.py'),
        '--reference-dir', REL,
        '--config', str(ROOT/('short_screen_draft/configs/drone_%s_s42_E20_DRAFT.yaml' % arm)),
        '--output', str(ROOT/('profile_' + arm))]
    if arm == 'C1':
        command += ['--candidate-source', str(ROOT/'performance_candidate/selected_only_v1.py')]
    jobs.append(dict(id='short_profile_'+arm, kind='train', formal=False,
        vram_mib=8192, rss_mib=32768, command=command))
(queue/'manifest.json').write_text(json.dumps(dict(jobs=jobs,
    scope='Sequential N/C0/C1 actual-statistics-cadence throughput only, each 24 updates',
    initial_bounded_ceiling_not_formal_measured_reservation=True,
    dev_or_ap_computed=False, existing_training_modified=False), indent=2))
for job in jobs:
    run_job(job, queue)
    arm = job['id'].removeprefix('short_profile_')
    receipt = json.loads((ROOT/('profile_'+arm)/'short_profile_receipt.json').read_text())
    if receipt['status'] != 'SHORT_SCREEN_THROUGHPUT_PROFILED':
        raise RuntimeError('Invalid profile receipt')
(queue/'completion.json').write_text(json.dumps(dict(status='COMPLETED', time=time.time(),
    arms=['N','C0','C1'], production_training_admitted=False), indent=2))
