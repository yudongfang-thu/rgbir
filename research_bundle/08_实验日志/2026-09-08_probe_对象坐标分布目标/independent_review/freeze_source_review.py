from pathlib import Path
import json
from datetime import datetime,timezone
root=Path(__file__).resolve().parent
src=root.parent/'transport';snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=False)
records=[]
for name in ('probability_transport.py','run_cached_transport.py','test_transport_cpu.py','PLAN.md'):
    raw=(src/name).read_bytes();(snapshot/name).write_bytes(raw)
    assert raw==(src/name).read_bytes()==(snapshot/name).read_bytes()
    records.append({'source_path':str((src/name).resolve()),'snapshot_path':str((snapshot/name).resolve()),'bytes':len(raw),'direct_bytes_equal':True})
truth=json.loads((root/'source_small_truth_result.json').read_text())
assert truth['pass']
data={'date':datetime.now(timezone.utc).isoformat(),'auditor':{'agent':'/root/object_transport_review','model_identity':'not exposed to reviewer; inherited parent settings','independent_from_executor':'/root/ap_error','cross_model_claim':False},'status':'READY_FOR_CPU_CACHE_RUN','scope':'Source and independent synthetic truth only; actual cache execution not yet accepted','independent_checks':len(truth['checks']),'all_checks_passed':True,'source_identity':records,'new_hash_computed':False,'audited_input_hashes':[],'hash_policy':'User requested no new hashes; exact paths and direct full-byte snapshots used.','remaining':'Run fixed existing cache once, then independently recompute all 80 GT x 2 inputs before execution verdict.','limits':['No physical registration proof','No old L1 geometry admission','No KD utility or AP claim','FP64 T=1 softmax from raw logits, not cached FP32 bitwise equality']}
(root/'SOURCE_REVIEW.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md='''# Independent source review

**READY_FOR_CPU_CACHE_RUN** — source and independent synthetic mathematical truths pass. The real cached-input execution is not yet accepted.

Reviewed `transport/probability_transport.py`, `run_cached_transport.py`, `test_transport_cpu.py`, and `PLAN.md`; frozen exact-byte copies are in `source_snapshot/`. The runner declares all four original input files and verifies the common completed-forward contract, fixed 80 objects / 317 distributions, and role/GT/anchor identities. It emits both prelisted paths for every object and preserves missing roles.

The operator uses all 16 raw logits for explicit FP64 stable T=1 softmax, with cached FP32/native vectors retained by the runner. This is not a claim of bitwise equality to cached FP32 mass. Exact rational annotation geometry computes all mapped distances; every strictly positive out-of-support mass rejects the whole object. No tail removal, clipping, target normalization, or anchor substitution occurs.

Independent `review_source.py` completed 37 checks, including identity, translation, scale/stride, compression, all-bin probability and expectation closure, integer/last bin, tiny positive out-of-support mass, zero mass, invalid inputs and zero label extent. The separate all-destination-bin hat-basis oracle matches the operator with maximum probability and distance difference 0. Executor tests `CPU_attempt2.json`: 10/10 passed. Adding the fourth contract input was a source-review correction before the real cache ran; earlier attempts remain preserved.

Next: execute the fixed existing cache, then independently recompute every 80 GT × 2 path and all available side/bin values. This source review alone does not pass execution, establish physical registration, release old L1 geometry, or demonstrate KD utility.

Identity follows the explicit no-new-hash scope: resolved original paths, frozen copies, and direct full-byte equality. Reviewer `/root/object_transport_review` is separate from executor `/root/ap_error`; exact model identity is not exposed, so no cross-model claim is made.
'''
(root/'SOURCE_REVIEW.md').write_text(md,encoding='utf-8')
print(json.dumps({'status':data['status'],'independent_checks':data['independent_checks']}))
