"""Persist this read-only follow-up and a non-authorizing actual adapter probe."""
import datetime
import json
from pathlib import Path
import sys

LOG=Path(__file__).resolve().parent
sys.path.insert(0,str(LOG/'posthoc_class_adapter_v1'))
from posthoc_class_adapter import module
analyzer=module(LOG.parents[1]/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/analyze_independent.py')
out=LOG/'heartbeat_20260907_1653'
out.mkdir(exist_ok=False)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
live=read(LOG/'formal_live_20260907_165319.json')
snapshot=read(LOG/'snapshots/2026-09-07T165357.033246_0800/snapshot.json')
leases=read(LOG/'snapshots/2026-09-07T165357.033246_0800/resource_leases_raw.json')
actual=read(LOG/'posthoc_class_adapter_v1/actual_analysis_attempt2.json')
harm=analyzer.harm_review(actual['records'],'C0')
comparison=analyzer.paired_comparison(actual['records'],'C0','N')
summary=dict(created_at=datetime.datetime.now().astimezone().isoformat(),
    observation_time=live['read_at'],c1=[dict(seed=r['seed'],progress=r['progress.json'],
    process_present=str(r['launched'][0]['pid']) in snapshot['project_processes'],
    completion_present='completion_receipt.json' in r,failure_present='failure_receipt.json' in r,
    lease_present=r['launched'][0]['lease_id'] in leases['leases']) for r in live['seeds']],
    old_controls={k:snapshot['runs'][k]['progress'] for k in ('S42','M42')},
    gpu_snapshot=snapshot['gpus'],project_rss_gib=snapshot['project_rss_kib_readonly']/2**20,
    posthoc_adapter=dict(status='DRAFT_AWAITING_INDEPENDENT_REVIEW',records=len(actual['records']),
        classes_per_record=[len(r['per_class_percent']) for r in actual['records']],
        C0_minus_N=comparison,C0_harm_review=harm,automatic_expansion=False),
    external_review=dict(job_id='36d75bab4efb42a5b4d4d2632372e323',status='FAILED',
        error='API Error: 402 Insufficient Balance',scientific_review_received=False,
        alternative='Existing classification reviewer resumed independent CPU/code review'),
    gpu_tasks_launched=0,training_changes=0,new_C1_AP_available=False,
    localization='BLOCKED_GEOMETRY_EVIDENCE',original_results_modified=False)
assert all(r['process_present'] and r['lease_present'] and not r['completion_present'] and not r['failure_present'] for r in summary['c1'])
assert harm['status']=='REVIEW_REQUIRED' and not harm['missing']
assert comparison['complete_three_seed_pairing']
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(out/'record_heartbeat_1653.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(path=str(out),c1_epochs=[r['progress']['epoch'] for r in summary['c1']],
    C0_harm=harm['status'],missing=harm['missing'],project_rss_gib=summary['project_rss_gib'])))
