"""One bounded pool measurement, through the existing global resource lease."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
REL='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5'
sys.path.insert(0,'/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/release_v8')
from resource_dispatch import run_job

queue=ROOT/'queue_attempt1'
queue.mkdir(exist_ok=False)
job=dict(id='c1_pool_block16_measurement',kind='eval',vram_mib=4096,rss_mib=8192,
         command=[PY,str(ROOT/'performance_candidate/benchmark_candidate.py'),
                  '--reference-dir',REL,'--mode','synthetic','--device','cuda',
                  '--warmup','2','--iterations','7',
                  '--output',str(ROOT/'synthetic_attempt1')])
(queue/'manifest.json').write_text(json.dumps(dict(jobs=[job],
    reservation_semantics='Initial bounded canary ceiling, not an asserted measured peak',
    optimizer_updates=0,formal_training_admitted=False),indent=2))
run_job(job,queue)
receipt=json.loads((ROOT/'synthetic_attempt1/receipt.json').read_text())
if receipt['status']!='PASS_NUMERIC_EQUIVALENCE_ONLY':
    raise RuntimeError('Candidate numeric equivalence failed')
(queue/'completion.json').write_text(json.dumps(dict(status='COMPLETED',time=time.time(),
    scientific_training_admitted=False),indent=2))
