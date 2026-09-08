from pathlib import Path
from datetime import datetime,timezone
import json

HERE=Path(__file__).resolve().parent;ENTRY=HERE.parent;OUT=ENTRY/'transport'/'output_attempt1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,o):
    with p.open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,ensure_ascii=False,allow_nan=False)
def lineno(p,text):return next(i for i,l in enumerate(p.read_text(encoding='utf-8').splitlines(),1) if text in l)
recomp=read(HERE/'recomputation_summary.json');summary=read(OUT/'summary.json')
assert recomp['passed']
byte_checks=[]
for r in recomp['input_source_evidence']:
    p=Path(r['path']);q=Path(r['snapshot'])
    ok=p.read_bytes()==q.read_bytes()
    if 'executed_copy' in r:ok=ok and p.read_bytes()==Path(r['executed_copy']).read_bytes()
    assert ok
    byte_checks.append({'path':str(p.resolve()),'snapshot':str(q.resolve()),'bytes':p.stat().st_size,'direct_bytes_equal':ok})
for name in ('objects.jsonl','summary.json','README.md'):
    p=OUT/name;q=HERE/'result_snapshot'/name;assert p.read_bytes()==q.read_bytes()
    byte_checks.append({'path':str(p.resolve()),'snapshot':str(q.resolve()),'bytes':p.stat().st_size,'direct_bytes_equal':True})
op=ENTRY/'transport'/'probability_transport.py';runner=ENTRY/'transport'/'run_cached_transport.py'
oploc=str(op.resolve())+':'+str(lineno(op,'def scatter_complete'))
runloc=str(runner.resolve())+':'+str(lineno(runner,'def run'))
checks={
 'gt_provenance':{'status':'pass','details':'Uses the declared paired native RGB/IR annotation boxes from the already accepted original 32-frame LLVIP train batch; every result preserves stable GT/frame/role/anchor/forward identity. Annotation-defined correspondence is not physical registration evidence.','evidence':[str(Path(recomp['input_source_evidence'][0]['path']))+' (all 80 object records)',str(OUT/'objects.jsonl')+' (all 160 records)']},
 'score_normalization':{'status':'pass','details':'No AP or efficacy score computed. FP64 stable T=1 softmax is explicitly derived from raw logits; cached FP32/native vectors retained separately. All positive mass is transported by a fixed annotation affine and hat basis without clipping, tail deletion or target renormalization. Mass and first moment close; exact support is tested before exposure of a whole-object target.','evidence':[oploc,str(HERE/'recomputation_summary.json')]},
 'result_existence':{'status':'pass','details':'Actual executor output_attempt1 exists with completed summary, 160 object/path rows, 4 executed source copies, and runtime %.9f seconds. Independent recomputation executed after production.'%summary['seconds'],'evidence':[str(OUT/'summary.json'),str(OUT/'objects.jsonl'),str(HERE/'recomputation_rows.json')]},
 'dead_code':{'status':'pass','details':'Actual entrypoint run() -> validate_inputs() -> analyze() -> transport_logits() -> scatter_complete() is exercised by executed rows. All 160 rows and 628 available side vectors were independently rebuilt directly from original inputs. Executed source copies match pre-execution v2 snapshots and current source byte for byte.','evidence':[runloc,oploc,str(HERE/'source_snapshot_v2'),str(HERE/'recomputation_summary.json')]},
 'scope':{'status':'pass','details':'Exactly 80 GT and two prelisted interfaces. same_R_primary: 79 supported / 1 missing, all 79 supported are affine identity. native_T_pressure: 15 supported / 63 positive-mass-outside / 2 missing, all 15 supported are affine identity. No new batch, inference, anchor matching, GPU, training, AP, old L1 admission, or efficacy claim.','evidence':[str(OUT/'summary.json')+' variants',str(HERE/'recomputation_rows.json')+' all 160 rows']},
 'eval_type':{'status':'pass','details':'Mixed: real_gt for annotation-defined geometry on cached actual model logits; synthetic_proxy for independent mathematical small truths. This is a target-interface integrity evaluation, not a detection-performance or physical-registration evaluation.','types':['real_gt','synthetic_proxy'],'evidence':[str(HERE/'source_small_truth_result.json'),str(OUT/'summary.json')]},
 'input_and_source_identity':{'status':'pass','details':'Four original input files, all four source files, and final result files rechecked by exact path and direct full-byte equality immediately before this verdict. Per-object identities were also checked. No new content hashes, per explicit user scope.','evidence':[str(HERE/'input_snapshot_manifest.json'),str(HERE/'source_snapshot_v2'),str(HERE/'result_snapshot')]},
 'independent_recomputation':{'status':'pass','details':'37 independent synthetic checks; actual 80 x 2 path and all 628 available side vectors x 16 bins inspected. Source/target probabilities and mapped distances differ from independent oracle by 0. Maximum independently reconstructed edge-coordinate difference 1.1368683772161603e-13; FP64 source versus saved FP32 difference 2.443613299485392e-07 is disclosed, not counted as transport error.','evidence':[str(HERE/'review_source.py'),str(HERE/'independent_oracle.py'),str(HERE/'audit_cached.py'),str(HERE/'recomputation_summary.json')]}
}
claims=[
 {'id':'complete_probability_target_interface','impact':'supported','affected_evidence':[str(OUT/'summary.json'),str(HERE/'recomputation_summary.json')],'rationale':'For this frozen cache, the complete-mass operator and its explicit support/missing outcomes agree with independent original-input recomputation.'},
 {'id':'same_R_primary_coverage','impact':'needs_qualifier','affected_evidence':[str(OUT/'summary.json')+' variants.same_R_primary'],'rationale':'79/80 objects supported; every supported map is identity because labels and the fixed anchor geometry coincide. This validates the declared target definition, not a nontrivial cross-anchor warp.'},
 {'id':'native_T_pressure_coverage','impact':'needs_qualifier','affected_evidence':[str(OUT/'summary.json')+' variants.native_T_pressure'],'rationale':'Only 15/80 supported, 63/80 support-rejected and 2/80 missing. All supported cases are identity; the rejected tail mass cannot be discarded or repaired to raise coverage.'},
 {'id':'physical_registration_or_old_L1_admission','impact':'unsupported','affected_evidence':[str(ENTRY/'README.md'),str(OUT/'summary.json')],'rationale':'Shared annotation geometry and interface correctness provide no independent physical registration proof and do not unblock old L1 geometry.'},
 {'id':'KD_utility_AP_or_shape_gain','impact':'unsupported','affected_evidence':[str(OUT/'summary.json')],'rationale':'There is no training intervention or detection evaluation here. The audit does not establish student learning, AP gain, or shape value beyond expectation-box quality.'}
]
result={'date':datetime.now(timezone.utc).isoformat(),'auditor':{'agent':'/root/object_transport_review','independent_from_executor':'/root/ap_error','observable_model_identity':'not exposed; inherited parent settings','cross_model_review_claim':False},'overall_verdict':'pass','integrity_status':'pass','reason_code':'FIXED_CPU_TARGET_INTERFACE_EXECUTED_AND_INDEPENDENTLY_RECOMPUTED','checks':checks,'claims':claims,
 'declared_input_set':[r['path'] for r in recomp['input_source_evidence'][:4]],'audited_input_hashes':[],
 'hash_exception':'User explicitly requested no new hashes. The skill default of SHA-256 is superseded by full-byte snapshots and direct equality with resolved paths; no hashes were computed.',
 'new_hash_computed':False,'final_direct_byte_comparison':byte_checks,
 'run_id':str(OUT.resolve()),'current_forward_id':summary['current_forward_id'],'code_revision':'No new commit used; four reviewed/executed source files matched byte for byte to source_snapshot_v2.',
 'data_version':'Accepted evidence_1602_final/probe, fixed LLVIP train first 32 frames, seed42, 80 GT, 317 cached distributions.',
 'evaluator_version':str((HERE/'audit_cached.py').resolve()),'recomputation_commands':['python independent_review/review_source.py','python independent_review/audit_cached.py','python independent_review/finalize_audit.py'],'recomputation_outputs':[str(HERE/'source_small_truth_result.json'),str(HERE/'recomputation_summary.json'),str(HERE/'recomputation_rows.json')],
 'review_trace':[str(HERE/'SOURCE_REVIEW.json'),str(HERE/'SOURCE_REVIEW.md'),str(HERE/'SOURCE_REVIEW_ADDENDUM.md'),str(HERE/'REVIEW_TRACE.md')],
 'inaccessible_inputs':[],'inspection_coverage':{'GT_objects':80,'interfaces':2,'result_rows':160,'available_side_vectors':628,'source_and_mapped_bins_per_side':16,'independent_synthetic_checks':37},
 'limits':['Does not re-run or expand the earlier 317-distribution forward audit.','Does not access GPU, train, or use test data.','No physical registration or KD utility conclusion.','An integrity pass is scoped evidence, not a guarantee against every possible defect.']}
dump(HERE/'EXPERIMENT_AUDIT.json',result)
md='''# 对象坐标完整概率目标：独立执行审计

**PASS（限固定CPU目标接口）**。实际执行完成后，从原始四文件全量重算80 GT × 2路径、628条可用边分布和全部16-bin；输出与独立hat-basis参考一致。未将源码READY当成已执行通过。

|固定输入|全部GT|可用|支持|越界拒绝|缺失|支持且恒等|
|---|---:|---:|---:|---:|---:|---:|
|教师同R索引（主）|80|79|79|0|1|79|
|教师native anchor（压力）|80|78|15|63|2|15|

主路径支持的79对象全部退化为标签和anchor相同下的恒等变换。压力路径63对象因正质量越界而完整拒绝，不能丢尾、截断或挪anchor补齐。通过仅说明本次目标接口的计算与记录可靠，不说明非恒等运输可用、物理配准成立、旧L1解锁或KD有效。

源概率、目标概率和未截断距离的独立重算最大差均为0；边坐标期望最大差1.1368683772161603e-13。实际源定义明确为原logits派生FP64 T=1 softmax，缓存FP32/native原向量完整保存；与缓存FP32最大差2.443613299485392e-07，不冒充逐位相同。生产CPU耗时%.9f秒；没有新推理、训练、GPU或新增样本。

独立数学小真值37项通过，作者CPU测试10项通过。原四输入、最终四源码及执行结果在结论前重新逐字节核对，路径和GT/角色/anchor/forward身份齐全；按用户本轮范围未计算新hash。源码旧快照和补丁历史均保留。

审阅者为 `/root/object_transport_review`，与执行者 `/root/ap_error` 分离；可观察环境未提供确切模型标识，不宣称跨模型复核。输入数据属于真实模型缓存与标签坐标（real_gt），数学小真值属于synthetic_proxy。全部字段、证据路径和逐对象核对见 [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json)、[recomputation_summary.json](recomputation_summary.json) 与 [recomputation_rows.json](recomputation_rows.json)。
'''%summary['seconds']
(HERE/'EXPERIMENT_AUDIT.md').write_text(md,encoding='utf-8')
print(json.dumps({'overall_verdict':'pass','final_byte_comparisons':len(byte_checks),'claims':len(claims)}))

