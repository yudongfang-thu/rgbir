"""Replay unchanged real bundles with a rank-2 reduction candidate."""
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
    jobs.append(dict(id=f'c1_rank2_bundle_{index}', kind='eval', vram_mib=4096,
        rss_mib=8192, command=[PY, str(ROOT/'performance_candidate/benchmark_candidate_v2.py'),
        '--reference-dir', REL, '--mode', 'bundle', '--device', 'cuda',
        '--bundle', str(BUNDLES/f'raw_batch_{index:02d}.pt'),
        '--output', str(ROOT/f'replay_{index:02d}_attempt1'), '--warmup', '2', '--iterations', '7']))
(queue/'manifest.json').write_text(json.dumps(dict(jobs=jobs,
    scope='Same stored bundles; unchanged fixed numeric tolerance; no optimizer',
    previous_probe='real_probe_attempt1/replay_00_attempt1 FAIL_EQUIVALENCE retained',
    formal_training_admitted=False), indent=2))
for index, job in enumerate(jobs):
    run_job(job, queue)
    receipt = json.loads((ROOT/f'replay_{index:02d}_attempt1/receipt.json').read_text())
    if receipt['status'] != 'PASS_NUMERIC_EQUIVALENCE_ONLY':
        raise RuntimeError('Candidate still differs; preserve failure and do not admit training')
(queue/'completion.json').write_text(json.dumps(dict(status='COMPLETED', time=time.time(),
    formal_training_admitted=False), indent=2))
