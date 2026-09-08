from pathlib import Path
from datetime import datetime,timezone
import json
HERE=Path(__file__).resolve().parent;ENTRY=HERE.parent
path=HERE/'TRAINING_SOURCE_REVIEW_v2.json';review=json.loads(path.read_text(encoding='utf-8'))
deployed=json.loads((ENTRY/'TRAINING_DEPLOYMENT_v2.json').read_text(encoding='utf-8'))
evidence=json.loads((ENTRY/'PINNED_TRAINING_CPU_ACCEPTED_v2.json').read_text(encoding='utf-8'));r=evidence['receipt']
assert r['status']=='PASS' and r['tests']==36 and r['wrapper_and_queue_tests']==22
assert r['L3']['status']=='PASS' and r['L3']['tests']==14 and r['L3']['failures']==r['L3']['errors']==0
assert r['calibrator_cli_exit']==0 and r['source_changed'] is False and r['GPU_used'] is False and r['new_hash_computed'] is False
assert 'Ran 22 tests' in evidence['log'] and 'Ran 14 tests' in evidence['log'] and evidence['log'].count('\nOK\n')==2
index={x['path']:x for x in deployed['files']}
for source in review['reviewed_source_files']:
    assert Path(source['source_path']).read_bytes()==Path(source['snapshot_path']).read_bytes()
    assert index[source['relative_path']]['byte_exact'] and index[source['relative_path']]['bytes']==source['bytes']
with (HERE/'TRAINING_SOURCE_REVIEW_v2_PRE_PINNED.json').open('xb') as f:f.write(path.read_bytes())
review.update(date=datetime.now(timezone.utc).isoformat(),status='READY_FOR_GPU_PREFLIGHT',
 execution_status='V2 deployed bytes and 36 actual pinned CPU tests plus calibration CLI accepted; no v2 GPU outcome is established by this review.',
 pinned_CPU_evidence={'receipt':str((ENTRY/'PINNED_TRAINING_CPU_ACCEPTED_v2.json').resolve()),'deployment':str((ENTRY/'TRAINING_DEPLOYMENT_v2.json').resolve()),'release':deployed['release'],'tests':36,'calibrator_cli_exit':0,'all_18_reviewed_files_bound_to_deployment':True,'new_hash_computed':False},
 next_step_contract='May start new attempt2 using release_v2 and this exact v2 snapshot. Start from the same first batch and run the fixed eight-batch calibration under unchanged resource and dose gates; then only if all encoded gates pass proceed to the three actual canaries and guarded matched FT3/evaluation. No resume of attempt1, no changed limits or batches, and no claim that v2 VRAM is fixed before actual measurements.',
 remaining=['Actual v2 fixed8 allocation/resource evidence and valid shared-dose calibration.','Actual matched canary gates before any FT3 training.','Actual terminal training/evaluation evidence before scientific conclusions.'])
path.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
mdpath=HERE/'TRAINING_SOURCE_REVIEW_v2.md';original=mdpath.read_text(encoding='utf-8')
(HERE/'TRAINING_SOURCE_REVIEW_v2_PRE_PINNED.md').write_text(original,encoding='utf-8')
mdpath.write_text('**最新状态：READY_FOR_GPU_PREFLIGHT（仅release_v2与新attempt2）。** 实际固定运行环境36项CPU和calibrator CLI已通过；18项运行源码/配置/AMP输入与v2独立快照及部署记录逐字节绑定。可从同一首批重跑固定八批，后续仍受队列真实canary和训练门约束。尚未证明v2实际显存问题已解决。\n\n'+original,encoding='utf-8')
print(json.dumps({'status':review['status'],'release_version':2,'reviewed_files':18,'pinned_tests':36,'calibrator_cli_exit':0}))
