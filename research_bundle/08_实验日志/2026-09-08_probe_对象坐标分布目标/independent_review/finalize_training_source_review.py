from pathlib import Path
from datetime import datetime,timezone
import json
HERE=Path(__file__).resolve().parent;RELEASE=HERE.parent/'trainer_release'
snapshot=HERE/'training_reviewed_source';snapshot.mkdir(exist_ok=False)
files=list(RELEASE.glob('*.py'))+list((RELEASE/'configs').glob('*'))+[RELEASE/'validated_amp_prior.json',RELEASE/'validated_amp_prior_config.yaml']
records=[]
for source in sorted(files):
    if not source.is_file():continue
    relative=source.relative_to(RELEASE);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    raw=source.read_bytes();dest.write_bytes(raw);assert source.read_bytes()==dest.read_bytes()
    records.append({'relative_path':relative.as_posix(),'source_path':str(source.resolve()),'snapshot_path':str(dest.resolve()),'bytes':len(raw),'mtime_ns':source.stat().st_mtime_ns,'direct_byte_identity':True})
checks={}
for name in ('TRAINER_PURE_CPU_TESTS_attempt2.json','TRAINER_TENSOR_CPU_TESTS_attempt2.json','TRAINER_WRAPPER_CPU_TESTS.json','TRAINER_CRITERION_CPU_TESTS.json'):
    data=json.loads((HERE/name).read_text(encoding='utf-8'));assert data['status']=='PASS';checks[name]=data
prior=HERE.parent.parent/'2026-09-08_probe_DFL真实信息读出'/'evidence_1602_final'/'probe'/'completion_receipt.json'
assert prior.read_bytes()==(RELEASE/'validated_amp_prior.json').read_bytes()
assert (HERE.parent/'transport'/'probability_transport.py').read_bytes()==(RELEASE/'probability_transport.py').read_bytes()
data={'date':datetime.now(timezone.utc).isoformat(),'auditor':{'agent':'/root/object_transport_review','model_identity':'not exposed; inherited parent settings','cross_model_claim':False},
 'status':'READY_FOR_PINNED_CPU','source_verdict':'pass','execution_status':'CUDA calibration/canaries/FT3 not executed by this review',
 'reviewed_source_files':records,'source_snapshot_directory':str(snapshot.resolve()),'new_hash_computed':False,'audited_input_hashes':[],
 'identity_method':'All operational Python, all configs and both AMP prior inputs copied and directly compared byte for byte; exact source paths retained. No new hashes per user instruction.',
 'prior_amp_receipt_bytes_equal_to_completed_raw_DFL':True,'transport_bytes_equal_to_accepted_CPU_operator':True,
 'fixed_protocol':str((HERE.parent/'SHORT_SCREEN_PROTOCOL.md').resolve()),
 'reviewed_behavior':[
  'Frozen R coarse base and R reliability/gap gates retained; source is same R index only, owned uniquely among all native teacher GT.',
  'T=1 teacher expectation quality gates precede complete positive-mass support; no student output enters mask.',
  'DFL T=2 softmax is applied to teacher logits before exact object-coordinate transport. FP64-to-FP32 positive underflow rejects the whole object and leaves base denominator unchanged.',
  'GT uses exactly the same mask/anchor/base and lambda, with adjacent RGB GT bin mass then sqrt-normalization; no RGB-only control claim.',
  'KL is four-side mean, target detached, T squared, selected sum divided by coarse base; criterion adds actual B times lambda once to complete native sum. N keeps auxiliary selection at lambda zero.',
  'Only DFL gradient ratios on fixed first eight reset batches determine shared lambda=min(1, median(.1 native_norm/B_unit_DFL_norm)); at least four valid ratios needed. GT gradient dose/cosine recorded without a second calibration.',
  'Fresh full student initialization/EMA/optimizer and frozen BN running buffers checked; parameter names identify P3/P4 pre-head gradient scope.',
  'Calibration resource checks and first-batch receipt precede continuation. Global lease dispatch retained, 8192 MiB VRAM/32768 MiB RSS ceilings, measured canary plus margin determine full FT3 reservations.',
  'Scoped AMP check binding reuses completed same-model actual AMP evidence and restores original binding; it does not claim setup equivalence.',
  'Queue requires three real canaries with at least 24 successful updates, matched first30 stream and initialization/config identities before each fresh 192-batch FT3/complete-dev endpoint. Technical failures stop without adaptive retries.',
  'Evaluation differs from copied prior evaluator only in new method/scope/arms/identity binding; native AP computation and full LLVIP dev population remain unchanged.'
 ],'executed_local_CPU_checks':checks,
 'inspection_corrections':['Added first-batch and per-batch measured resource checks before proceeding.','Added explicit target FP32 positive-mass-underflow rejection.'],
 'review_attempt_history':['Pure CPU attempt1 failed only because the reviewer used default Windows GBK with a Chinese temporary path; preserved. -X utf8 attempt2 passed without production source changes.',
  'Independent tensor attempt1 used an arbitrary 3e-8 tolerance against double GT sqrt/division. Observed FP32 error 4.4061e-8; preserved. Attempt2 uses machine epsilon for the two FP32 operations and passes. No production calculation or scientific protocol changed.'],
 'next_step_contract':'Deploy these exact bytes, run the same small suites in the pinned production runtime on CPU, then after those pass and deployment identity matches the reviewer may issue READY_FOR_GPU_PREFLIGHT. That authorizes starting only the encoded guarded queue: fixed8 calibration -> all three guarded real canaries -> new three-arm FT3/eval. It never claims unexecuted canaries have already passed.',
 'remaining':['Pinned production Python/Torch CPU execution and exact deployment identity.','Actual initial calibration/VRAM/RSS/gradient evidence.','Actual matched three-arm canaries and all runtime gates before full short training.'],
 'limits':['No physical registration, old L1 release, KD AP gain or shape-only benefit established.','Local tensor tests used torch 1.8.0+cu111 on CPU, not the pinned 2.10.0+cu128 runtime.','No GPU, SSH job, new inference or training was started by the reviewer.']}
for r in records:assert Path(r['source_path']).read_bytes()==Path(r['snapshot_path']).read_bytes()
with (HERE/'TRAINING_SOURCE_REVIEW.json').open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
md='''# L3 training source review

**READY_FOR_PINNED_CPU** — new L3 source, loss/mask/temperature contracts, calibration, criterion and guarded queue pass scoped independent inspection and local synthetic CPU checks. This does not report CUDA calibration, canary or FT3 completion.

The reviewer independently executed 13 pure control truths, 5 mathematical loss/gradient truths, all 14 authored selection/loss truths, all 7 queue simulations and 12+3 wrapper/criterion truths. KL analytical gradient maximum error was 7.45e-9; T2-before-transport matched the independent hat-basis oracle within 9.59e-9. GT FP32 sqrt/division differed from double reference by 4.41e-8 (within FP32 epsilon). Early reviewer harness failures are retained and explained in JSON.

All running Python files, every config and both AMP prior files are frozen under `training_reviewed_source/` and directly byte-compared to current source. Transport equals the independently accepted CPU operator; AMP receipt equals the completed prior original receipt. Complete paths and identities are in TRAINING_SOURCE_REVIEW.json.

Source review corrections added explicit first-batch/per-batch resource checks and positive-mass FP32-underflow object rejection. Fixed R base denominator, same-anchor teacher identity, T2-before-transport, same-mask GT/shared lambda, complete native loss plus one B*lambda multiplier and absence of student-dependent selection were inspected. Native evaluation arithmetic is inherited unchanged; the new method identity is explicit.

Next is exact-byte deployment and pinned-runtime CPU checks. Once those pass, the next review status can authorize only the already encoded guarded queue: fixed8 calibration, then all three real canaries, then three fresh matched FT3/evaluation arms. Each runtime gate must actually pass; source readiness is not a canary result. The scoped review establishes neither physical registration nor KD utility.
'''
(HERE/'TRAINING_SOURCE_REVIEW.md').write_text(md,encoding='utf-8')
print(json.dumps({'status':data['status'],'reviewed_files':len(records)}))
