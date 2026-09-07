"""Bounded real-flow capture and paired operator replay; no optimizer updates."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
CAMPAIGN = BASE/'rgbir_independent_kd_v2_20260907'
PY = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REL = str(CAMPAIGN/'release_gpu5')
sys.path.insert(0, str(BASE/'rgbir_task_conditional_v1_20260907/release_v8'))
from resource_dispatch import run_job

queue = ROOT/'queue_attempt1'
queue.mkdir(exist_ok=False)
capture = ROOT/'capture_attempt1'
student = '/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/weights/last.pt'
jobs = [dict(id='c1_real_flow_capture', kind='eval', vram_mib=8192,
             rss_mib=32768, command=[PY, str(ROOT/'performance_candidate/capture_real_batch.py'),
             '--reference-dir', REL, '--config', str(CAMPAIGN/'formal_C1_configs_gpu5/C1_s42.yaml'),
             '--student-checkpoint', student, '--coverage-dir', str(CAMPAIGN/'coverage_drone_attempt1'),
             '--output', str(capture), '--batches', '2'])]
for index in range(2):
    jobs.append(dict(id=f'c1_real_bundle_{index}', kind='eval', vram_mib=4096,
                     rss_mib=8192, command=[PY, str(ROOT/'performance_candidate/benchmark_candidate.py'),
                     '--reference-dir', REL, '--mode', 'bundle', '--device', 'cuda',
                     '--bundle', str(capture/f'raw_batch_{index:02d}.pt'),
                     '--output', str(ROOT/f'replay_{index:02d}_attempt1'),
                     '--warmup', '2', '--iterations', '7']))
(queue/'manifest.json').write_text(json.dumps(dict(jobs=jobs,
    reservation_semantics='Initial bounded diagnostic ceilings, not asserted measured peaks',
    optimizer_updates=0, formal_training_admitted=False), indent=2))
for index, job in enumerate(jobs):
    run_job(job, queue)
    receipt_path = capture/'receipt.json' if index == 0 else ROOT/f'replay_{index-1:02d}_attempt1/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    expected = 'COMPLETED_DIAGNOSTIC_CAPTURE_NOT_TRAINING_ADMISSION' if index == 0 else 'PASS_NUMERIC_EQUIVALENCE_ONLY'
    if receipt['status'] != expected:
        raise RuntimeError('Diagnostic prerequisite failed: '+job['id'])
(queue/'completion.json').write_text(json.dumps(dict(status='COMPLETED', time=time.time(),
    scientific_training_admitted=False), indent=2))
