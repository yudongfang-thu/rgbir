"""Two stored real bundles, exact original per-object pool on selected content."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REL = str(BASE/'rgbir_independent_kd_v2_20260907/release_gpu5')
BUNDLES = BASE/'rgbir_throughput_20260908/real_probe_attempt1/capture_attempt1'
sys.path.insert(0, str(BASE/'rgbir_task_conditional_v1_20260907/release_v8'))
from resource_dispatch import run_job

queue = ROOT/'queue_attempt1'
queue.mkdir(exist_ok=False)
jobs = []
for index in range(2):
    jobs.append(dict(id=f'c1_selected_bundle_{index}', kind='eval', vram_mib=4096,
        rss_mib=8192, command=[PY, str(ROOT/'performance_candidate/benchmark_selected_only.py'),
        '--reference-dir', REL, '--mode', 'bundle', '--device', 'cuda',
        '--bundle', str(BUNDLES/f'raw_batch_{index:02d}.pt'),
        '--output', str(ROOT/f'replay_{index:02d}_attempt1'), '--warmup', '2', '--iterations', '7']))
(queue/'manifest.json').write_text(json.dumps(dict(jobs=jobs,
    scope='Same stored raw data; selected S/T content and original per-object pool',
    diagnostic_fields_skipped_are_missing_not_zero_measurements=True,
    formal_training_admitted=False), indent=2))
for index, job in enumerate(jobs):
    run_job(job, queue)
    receipt = json.loads((ROOT/f'replay_{index:02d}_attempt1/receipt.json').read_text())
    if receipt['status'] != 'PASS_NUMERIC_EQUIVALENCE_ONLY':
        raise RuntimeError('Thin path numeric comparison failed; retain attempt')
(queue/'completion.json').write_text(json.dumps(dict(status='COMPLETED', time=time.time(),
    formal_training_admitted=False), indent=2))
