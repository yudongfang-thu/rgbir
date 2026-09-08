from pathlib import Path
from datetime import datetime,timezone
import json
HERE=Path(__file__).resolve().parent;ENTRY=HERE.parent
p=HERE/'TRAINING_SOURCE_REVIEW.json';review=json.loads(p.read_text(encoding='utf-8'))
deployment=json.loads((ENTRY/'TRAINING_DEPLOYMENT_v1.json').read_text(encoding='utf-8'))
initial=json.loads((ENTRY/'PINNED_TRAINING_CPU_v1.json').read_text(encoding='utf-8'))
accepted=json.loads((ENTRY/'PINNED_TRAINING_CPU_ACCEPTED_v1.json').read_text(encoding='utf-8'))
r=accepted['receipt'];old=initial['receipt']
assert (r['status'],r['tests'],r['failures'],r['errors'],r['wrapper_tests_passed'],r['queue_tests_passed'],r['L3']['tests'])==('PASS',36,0,0,15,7,14)
assert r['GPU_used'] is False and r['new_hash_computed'] is False and r['source_changed'] is False
assert (old['tests'],old['failures'],old['errors'])==(23,0,1)
assert 'SystemExit: 2' in initial['log'] and 'test_l3_cpu (unittest.loader._FailedTest)' in initial['log']
assert 'Ran 14 tests' in accepted['log'] and accepted['log'].rstrip().endswith('OK')
index={x['path']:x for x in deployment['files']}
for source in review['reviewed_source_files']:
    assert Path(source['source_path']).read_bytes()==Path(source['snapshot_path']).read_bytes()
    assert index[source['relative_path']]['byte_exact'] and index[source['relative_path']]['bytes']==source['bytes']
backup=HERE/'TRAINING_SOURCE_REVIEW_PRE_PINNED.json'
with backup.open('xb') as f:f.write(p.read_bytes())
review['date']=datetime.now(timezone.utc).isoformat();review['status']='READY_FOR_GPU_PREFLIGHT'
review['execution_status']='Pinned CPU 36/36 accepted; GPU calibration/canaries/FT3 have not been executed by this review.'
review['pinned_CPU_evidence']={'deployment':str((ENTRY/'TRAINING_DEPLOYMENT_v1.json').resolve()),'release':deployment['release'],
 'accepted_receipt':str((ENTRY/'PINNED_TRAINING_CPU_ACCEPTED_v1.json').resolve()),'original_discovery_failure':str((ENTRY/'PINNED_TRAINING_CPU_v1.json').resolve()),
 'tests_passed':36,'breakdown':{'L3':14,'wrappers':15,'queue':7},'reviewed_source_files_matched_to_deployment':len(review['reviewed_source_files']),
 'first_failure_explanation':'CLI-argument-requiring L3 module was imported by discovery; 22 other tests passed. Same unchanged L3 source subsequently passed all 14 tests via its declared CLI. Original failed attempt retained.',
 'source_changes_between_pinned_attempts':False,'GPU_used':False,'new_hash_computed':False}
review['next_step_contract']='May start the exact deployed guarded queue only: fixed eight-batch calibration with per-batch resource gates and valid shared-dose requirements; then all three actual canaries with >=24 successful updates and matched initialization/config/first30 streams; only then each fresh three-epoch 192-batch arm and full-dev evaluation. Every runtime gate must pass in actual outputs. This readiness does not claim any canary or scientific effect has occurred.'
review['remaining']=['Actual eight-batch calibration/VRAM/RSS/gradient evidence.','Actual three-arm canaries and all encoded runtime gates before full short training.','Actual training/evaluation terminal evidence before any result claim.']
review['limits']=[x for x in review['limits'] if not x.startswith('Local tensor tests used')]
review['limits'].append('Local mathematical tests and actual pinned-runtime CPU suites passed; CUDA numerical/runtime behavior still requires the real guarded preflight/canaries.')
p.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md='''# Pinned CPU acceptance addendum

**READY_FOR_GPU_PREFLIGHT** for the exact deployed release_v1. All 18 reviewed operational Python/config/AMP input files still match `training_reviewed_source/` directly byte for byte and are included in the deployment receipt as byte-exact. No new hashes were computed.

The actual pinned Python completed 36 CPU checks: L3 14, wrappers/criterion 15, queue 7. The first discovery attempt passed 22 tests and failed only to import the L3 test module without its required CLI arguments; the original failure remains preserved. The unchanged L3 source then passed its 14 tests through the declared CLI. Both original logs/receipts were read independently. This is not a hidden production-source repair.

Readiness permits starting only the queue's encoded sequence: fixed8 calibration with actual per-batch resource/dose gates, all three actual canaries, then guarded fresh three-arm FT3/evaluation. It does not say those CUDA stages have already run or passed. Failed runtime gates retain their attempt and stop the candidate according to the frozen protocol. No physical-registration or KD-effect claim follows from source/CPU acceptance.

Authoritative current status and exact paths are in `TRAINING_SOURCE_REVIEW.json`; the earlier pending-pinned record is retained as `TRAINING_SOURCE_REVIEW_PRE_PINNED.json`.
'''
(HERE/'TRAINING_PINNED_CPU_ACCEPTANCE.md').write_text(md,encoding='utf-8')
print(json.dumps({'status':review['status'],'pinned_checks':36,'reviewed_source_files':len(review['reviewed_source_files'])}))
