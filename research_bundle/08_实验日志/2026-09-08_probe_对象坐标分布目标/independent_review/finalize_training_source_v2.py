from pathlib import Path
from datetime import datetime,timezone
import json,difflib
HERE=Path(__file__).resolve().parent;ENTRY=HERE.parent;RELEASE=ENTRY/'trainer_release'
previous=json.loads((HERE/'TRAINING_SOURCE_REVIEW.json').read_text(encoding='utf-8'))
snapshot=HERE/'training_reviewed_source_v2';snapshot.mkdir(exist_ok=False)
records=[];changed=[]
for old in previous['reviewed_source_files']:
    rel=old['relative_path'];source=RELEASE/rel;dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    before=Path(old['snapshot_path']).read_bytes();raw=source.read_bytes();dest.write_bytes(raw)
    assert raw==source.read_bytes()==dest.read_bytes()
    if raw!=before:
        changed.append(rel)
        (HERE/'CALIBRATOR_V1_TO_V2.diff').write_text(''.join(difflib.unified_diff(before.decode('utf-8').splitlines(True),raw.decode('utf-8').splitlines(True),fromfile='reviewed_v1/'+rel,tofile='reviewed_v2/'+rel)),encoding='utf-8')
    records.append({'relative_path':rel,'source_path':str(source.resolve()),'snapshot_path':str(dest.resolve()),'bytes':len(raw),'mtime_ns':source.stat().st_mtime_ns,'direct_byte_identity':True,'unchanged_from_reviewed_v1':raw==before})
assert changed==['calibrate_object_dfl.py']
proof=json.loads((HERE/'CALIBRATION_LIFETIME_INDEPENDENT_CPU_attempt2.json').read_text())
wrapper=json.loads((RELEASE/'WRAPPER_CPU_LIFETIME_V2.json').read_text())
assert proof['status']=='PASS' and wrapper['status']=='PASS' and wrapper['tests_run']==15
data=dict(previous)
data.update(date=datetime.now(timezone.utc).isoformat(),status='READY_FOR_PINNED_CPU',release_version=2,source_verdict='pass',
    execution_status='V1 actual calibration failed at batch2 resource gate; no v2 GPU stage has run in this review.',
    source_snapshot_directory=str(snapshot.resolve()),reviewed_source_files=records,changed_source_files=changed,
    previous_review=str((HERE/'TRAINING_SOURCE_REVIEW.json').resolve()),
    previous_failed_attempt_evidence=str((ENTRY/'training_evidence_1703').resolve()),
    failure_review=str((HERE/'CALIBRATION_ATTEMPT1_FAILURE_REVIEW.md').resolve()),
    reviewed_source_diff=str((HERE/'CALIBRATOR_V1_TO_V2.diff').resolve()),
    v2_affected_checks={'independent_lifetime_CPU':proof,'wrapper_CPU':wrapper,'calibration_CLI_and_AST':'Reported PASS in CALIBRATION_LIFETIME_V2.md; pinned production execution pending.'},
    inherited_unchanged_evidence={'L3_CPU_tests':14,'queue_CPU_tests':7,'source_byte_identity':True,'original_pinned_receipt':str((ENTRY/'PINNED_TRAINING_CPU_ACCEPTED_v1.json').resolve()),'original_independent_source_review':str((HERE/'TRAINING_SOURCE_REVIEW.json').resolve())},
    next_step_contract='Deploy exact v2 bytes to new release_v2, run affected wrapper/CLI/AST checks in pinned runtime, retain explicit inheritance of untouched L3/queue evidence. Then a new v2 READY may authorize new attempt2 from the same first batch through all eight calibration batches under unchanged resource/gate/lambda rules. Old v1 READY must not authorize changed v2 bytes.',
    remaining=['Pinned affected CPU checks and exact v2 deployment identity.','New attempt2 actual eight-batch allocation/resource/dose evidence.','All encoded real canary gates before any fresh FT3 training.'])
data.pop('pinned_CPU_evidence',None)
data['v2_correction_scope']={'mechanism':'Final for-loop scalar loss retained the previous shared student graph after deleting losses; delete final loss/stats/native_items plus prior graph references.',
 'diagnostics':'CUDA allocated at batch start/before cleanup/after cleanup plus reserved after cleanup, with synchronization. No empty_cache.',
 'unchanged':['fixed first8 batch flow','full model reset','temperature','selection gates','base denominator','gradient arithmetic','DFL-derived shared lambda','minimum4 valid batches','8192 VRAM/32768 RSS ceilings','global lease and subsequent canary/FT3 sequence'],
 'claim_limit':'The defect and release mechanism are verified; successful production VRAM resolution is not established until actual new fixed8 measurements.'}
with (HERE/'TRAINING_SOURCE_REVIEW_v2.json').open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
md='''# Training source v2: calibration lifetime correction

**READY_FOR_PINNED_CPU**, only for the new v2 source snapshot. Of all 18 running/source/config/AMP entries, only calibrate_object_dfl.py changed; 17 remain byte-identical to the previously accepted version. V1 history and attempt1 failure are retained.

The source diff removes the final loop `loss`, `stats` and explicit `native_items` references together with the original graph cleanup and adds synchronized batch allocation diagnostics. It does not change losses, gates, fixed8 flow, dose calibration or resource limits. Independent saved-tensor-context CPU evidence verifies release before next forward with identical gradient norms. The affected 15 wrapper tests pass. This is a technical correction, not an outcome-based retuning.

Exact files are in training_reviewed_source_v2/ and TRAINING_SOURCE_REVIEW_v2.json. The separate v1 snapshot remains unchanged. Next is exact new release_v2 deployment and pinned affected CPU checks, then a new readiness record for attempt2. Only the actual same fixed8 replay can verify whether production memory growth is resolved; canary and short-training gates remain required afterwards.
'''
(HERE/'TRAINING_SOURCE_REVIEW_v2.md').write_text(md,encoding='utf-8')
print(json.dumps({'status':data['status'],'version':2,'changed_files':changed,'reviewed_files':len(records)}))
