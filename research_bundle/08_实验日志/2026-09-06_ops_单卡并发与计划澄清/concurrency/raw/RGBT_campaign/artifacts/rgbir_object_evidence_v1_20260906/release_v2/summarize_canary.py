"""Read-only source extraction plus a compact engineering summary on the dataset disk."""
import json
from pathlib import Path
ROOT=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906')
RUNS=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906')
summary={'arms':{},'comparison':json.loads((ROOT/'canary_comparison.json').read_text()),
         'scope':'Engineering validation only; no AP inference','physical_gpu':4}
for arm in ('paired','weight0'):
    run=RUNS/f'canary_{arm}_s42_attempt1'
    r=json.loads((run/'completion_receipt.json').read_text())
    keep=['status','optimizer_updates','optimizer_update_attempts','amp_skipped_updates','ema_updates',
          'batches','selected_objects','gradient_checks','gpu_allocated_peak_mib','gpu_reserved_peak_mib',
          'resources','seconds','checkpoint']
    s={k:r[k] for k in keep}
    rows=[json.loads(line) for line in (run/'kd_batches.jsonl').read_text().splitlines()]
    s['native_total_first_last']=[rows[0]['native_total'],rows[-1]['native_total']]
    s['kd_unweighted_first_last']=[rows[0]['loss_unweighted'],rows[-1]['loss_unweighted']]
    s['selected_total_over_base_total']=sum(x['selected_count'] for x in rows)/sum(x['base_count'] for x in rows)
    s['steady_last_10_batch_seconds']=(rows[-1]['time_seconds']-rows[-11]['time_seconds'])/10
    summary['arms'][arm]=s
summary['formal_reservation']={'vram_mib':10000,'rss_mib':49152,'free_safety_mib':2048,
    'reason':'Above observed NVML/RSS peaks; two arms execute serially on one GPU'}
(ROOT/'canary_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
