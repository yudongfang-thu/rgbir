"""Fresh old/selected-only update comparison through the original shared lease."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REL = str(BASE/'rgbir_independent_kd_v2_20260907/release_gpu5')
CFG = str(BASE/'rgbir_independent_kd_v2_20260907/formal_C1_configs_gpu5/C1_s42.yaml')
sys.path.insert(0, str(BASE/'rgbir_task_conditional_v1_20260907/release_v8'))
from resource_dispatch import run_job

queue = ROOT/'queue_attempt1'
queue.mkdir(exist_ok=False)
job = dict(id='c1_selected_update24_diagnostic', kind='train', formal=False,
    vram_mib=8192, rss_mib=32768,
    command=[PY, str(ROOT/'performance_candidate/update24_selected_only/compare_24_selected_only.py'),
    '--reference-dir', REL, '--config', CFG,
    '--candidate-source', str(ROOT/'performance_candidate/selected_only_v1.py'),
    '--output', str(ROOT/'comparison_attempt1')])
(queue/'manifest.json').write_text(json.dumps(dict(jobs=[job],
    scope='Fresh old/selected-only C1, each 24 updates; all thin learning batches explicitly checked',
    two_raw_bundles_numeric_pass=True,
    prior_block16_trajectory_failure_retained=True,
    initial_bounded_ceiling_not_formal_measured_reservation=True,
    formal_training_admitted=False, existing_training_modified=False), indent=2))
run_job(job, queue)
receipt = json.loads((ROOT/'comparison_attempt1/receipt.json').read_text())
(queue/'completion.json').write_text(json.dumps(dict(status='COMPLETED', time=time.time(),
    comparison_status=receipt['status'], formal_training_admitted=False), indent=2))
