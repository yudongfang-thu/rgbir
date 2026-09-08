"""Read only completed lease receipts and verify immutable probe snapshots."""
from pathlib import Path
import json
root=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
early=root/'evidence_1601';final=root/'evidence_1602_final';q=final/'queue'
same=[]
for p in sorted((early/'probe').rglob('*')):
 if p.is_file():
  rel=p.relative_to(early)
  if p.read_bytes()!=(final/rel).read_bytes():raise ValueError('Completed producer output changed: '+str(rel))
  same.append(rel.as_posix())
status=read(q/'raw_dfl_attempt1_status.json');profile=read(q/'raw_dfl_attempt1_resource_profile.json')
completion=read(q/'completion.json');admission=read(q/'raw_dfl_attempt1_admission.json')
producer=read(final/'probe/completion_receipt.json')
assert status['status']==profile['status']=='COMPLETED' and status['exit_code']==profile['exit_code']==0
assert completion['status']=='RAW_DFL_SINGLE_BATCH_QUEUE_COMPLETED' and not profile['monitor_errors']
assert status['minimum_free_mib']>=2048
assert admission['four_gpu_exception'] and len(admission['empty_gpus_after'])>=2
maximum_RSS=max(s['project_rss_mib'] for s in profile['samples'])
assert maximum_RSS<=300*1024
r=dict(status='COMPLETED_RESOURCE_AND_COLLECTION_CHECK',producer_seconds=producer['seconds'],queue_seconds=completion['seconds'],
 GPU=producer['resources']['gpu_ids'],four_gpu_exception_used=True,admission=admission,
 observed_whole_card_minimum_free_mib=status['minimum_free_mib'],observed_project_maximum_RSS_mib=maximum_RSS,
 producer_peak=producer['resources'],framework_allocated_peak_mib=producer['gpu_allocated_peak_mib'],framework_reserved_peak_mib=producer['gpu_reserved_peak_mib'],
 monitor_errors=[],early_and_final_probe_byte_exact=same,original_running_snapshot_retained=True,
 one_measured_batch_only=True,additional_training=False,raw_host_profiles_excluded_from_publication=True,new_hash_computed=False,
 limits='Sampled NVML/RSS observations and existing dispatcher guards; not a claim of continuously observed microsecond peaks.')
with (root/'RESOURCE_COMPLETION.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
print(json.dumps(r,indent=2))
