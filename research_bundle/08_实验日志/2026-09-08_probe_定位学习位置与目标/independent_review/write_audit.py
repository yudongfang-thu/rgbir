"""Write the scoped independent verdict and a non-hash evidence manifest."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ENTRY=HERE.parent
LOG=ENTRY.parent


def location(path):
    p=Path(path).resolve()
    try:return str(p.relative_to(HERE)).replace('\\','/')
    except ValueError:return str(p).replace('\\','/')


def main():
    anchor=ENTRY/'anchor_join/output_attempt3'
    target=ENTRY/'target_audit/output_attempt1'
    old=LOG/'2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/llvip'
    witness=LOG/'2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe'
    inputs=[old/x for x in ['calibration_batches.jsonl','calibration_receipt.json','direction_config.yaml']]
    inputs += [witness/x for x in ['witness_objects.jsonl','witness_frames.jsonl','identity_contract.json','witness_contract.json','export_contract.json','completion_receipt.json','first_batch_stream.json']]
    inputs += [LOG/'2026-09-08_probe_训练侧定位覆盖'/x for x in ['output_attempt1/objects.jsonl','bridge/output_attempt1/objects.jsonl','bridge/output_attempt1/original_first_batch_L2_stats.json','bridge/output_attempt1/summary.json']]
    inputs += [ENTRY/'source_identity'/x for x in ['localization_box_v2.py','direction_criterion.py','calibrate_direction.py']]
    inputs += [ENTRY/'SOURCE_IDENTITY.json',ENTRY/'check_source_identity.py']
    inputs += [anchor/x for x in ['summary.json','objects.jsonl','completion.json','README.md','input_manifest.json']]
    inputs += [ENTRY/'anchor_join'/x for x in ['analyze_anchor_join.py','CPU_attempt1.json','output_attempt1/failure.json','output_attempt2/failure.json','SOURCE_DISPLAY_ENCODING_FIX.json','source_utf8_attempt1.py']]
    inputs += [target/x for x in ['summary.json','objects.jsonl','audit_targets.py','localization_box_v2.py']]
    inputs += [ENTRY/'target_audit'/x for x in ['README.md','PLAN.md','CPU_attempt1.json']]
    inputs += [HERE/x for x in ['recompute_independent.py','CPU_RECOMPUTATION.json','INITIAL_RECOMPUTATION_FAILURE.json','compare_author_outputs.py','AUTHOR_COMPARISON.json']]
    manifest=[]
    for p in inputs:
        s=p.stat()
        manifest.append(dict(path=location(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256='not computed'))
    def check(details,evidence):return dict(status='pass',details=details,evidence=evidence)
    code=location(ENTRY/'source_identity/localization_box_v2.py')
    criterion=location(ENTRY/'source_identity/direction_criterion.py')
    calibration=location(ENTRY/'source_identity/calibrate_direction.py')
    report=dict(
        date='2026-09-08',auditor=dict(agent='/root/loc_target_review',model='unavailable',independent_agent=True,cross_model_review_claim=False),
        overall_verdict='pass',integrity_status='pass',reason_code='SCOPED_CACHE_DIAGNOSTIC_ACCEPTED_WITH_EXPLICIT_INFERENCE_LIMITS',
        audit_scope='Existing LLVIP first32 training witness and frozen initial-state eight-batch calibration only; no new model/GPU/training, no test data, no hashes.',
        checks=dict(
            gt_provenance=check('First32 GT identities are native validated dataset-label rows transported through augmentation; 79 historical base records join exactly to the 80-row witness universe. Later seven batches preserve augmented global GT rows, not stable physical-object identity. RGB/IR GT equality is an annotation fact, not independent physical-registration evidence.',
                                [location(witness/'identity_contract.json'),location(witness/'witness_objects.jsonl'),'CPU_RECOMPUTATION.json: inspected_original_bridge_gt_records=79']),
            score_normalization=check('Actual L2 loss is four-edge SmoothL1 beta=.1 on GT-relative decoded box coordinates, summed over selected then divided by pre-teacher base M. Caller adds B*lambda*KD to complete native loss. Eight batches keep 649 base/30 selected; no selected-denominator substitution. Normalized-coordinate and pixel-coordinate derivatives are explicitly separate from parameter gradients.',
                                     [code+':35-50, 155-197, 217-230',criterion+':37-40','AUTHOR_COMPARISON.json: 30 target rows passed']),
            result_existence=check('Original calibration has 8 completed batch rows and zero optimizer/EMA updates. Final anchor result is output_attempt3 with 80 rows; target result output_attempt1 has 30 selected rows. Author outputs match independent arithmetic. Anchor attempts1/2 are retained startup parse failures, not additional experiments or successful runs.',
                                  [location(old/'calibration_receipt.json'),location(anchor/'completion.json'),location(target/'summary.json'),'AUTHOR_COMPARISON.json']),
            dead_code=check('Inspected executed source path: criterion.losses calls localization_box_v2.compute; compute uses selected_anchors reference index for decode_selected(student, ids[:,0], ids[:,1]); gradients exist through selected DFL expectations. Selection/teacher targets are detached. Three locally inspected sources equal captured current 94 release bytes. Historical logged selected tuples and loss scalars support execution; this review does not rerun the historical model graph.',
                           [code+':53-65, 89-90, 204-230',criterion+':25-29',calibration+':44-66',location(ENTRY/'SOURCE_IDENTITY.json'),'AUTHOR_COMPARISON.json: source_byte_comparisons']),
            scope=check('Accepted as single-seed mature-initialization training-cache diagnostics: first32/80 GT and eight batches/256 unique image paths, 664 augmented GT occurrences, 649 base, 30 selected across 26 images. No AP, KD efficacy, causal explanation of old FT3 differences, all-training generalization, original-L1 admission, or cross-model robustness is inferred. Missing student raw fields and parameter Jacobians remain unavailable.',
                        ['CPU_RECOMPUTATION.json',location(target/'summary.json')+': scope and false claim flags',location(anchor/'summary.json')+': missing_fields']),
            eval_type=check('real_gt for native detection/GT identities and teacher-box-vs-annotated-GT residuals; R-output derivative analysis is a deterministic model-output proxy, not a fresh independent performance evaluation. The three reviewer toy checks are synthetic_proxy. GT controls are annotation-derived, and teacher coordinates are model-derived; neither is relabelled as new physical ground truth.',
                            ['CPU_RECOMPUTATION.json: toy_checks_passed=3',location(target/'README.md'),code+':45-50, 219-221'])
        ),
        claims=[
            dict(id='C1_ANCHOR_IDENTITY',impact='supported',claim='All 11 first32 native localization-opportunity objects have historical R candidate anchor equal to native S/R matched anchor. Four are actually L2-selected and use that R index for student learning; their four teacher target anchors differ from native teacher matched anchors.',evidence=['CPU_RECOMPUTATION.json: opportunity_objects','AUTHOR_COMPARISON.json'],rationale='Compared image, GT, prediction ID, anchor ID and original tuples rather than inferring identity from close boxes.'),
            dict(id='C2_SELECTION_DENOMINATORS',impact='supported',claim='First32 has 79 base/7 selected, with 4 of the 11 localization opportunities selected. Eight batches have 649 base/30 selected, 256 unique image paths and 26 selected image paths.',evidence=['CPU_RECOMPUTATION.json: eight_batch_stage_totals',location(anchor/'summary.json')],rationale='Direct original-record counts; selected learning index is null for rejected objects. Eight-batch stable physical-object deduplication is unavailable.'),
            dict(id='C3_COORDINATE_DERIVATIVES',impact='supported',claim='At saved R output, 120 teacher/GT derivative edges split into 47 equal same-sign saturated, 63 same-sign different magnitude, and 10 opposite-sign edges; unscaled stack cosine is .958663, base-scaled .959681 and pixel-scaled .980442.',evidence=['CPU_RECOMPUTATION.json','AUTHOR_COMPARISON.json'],rationale='Independent standard-library formula matches all 30 author rows. High cosine coexists with real target and derivative differences; output coordinate choice changes the cosine.'),
            dict(id='C4_PARAMETER_GRADIENT',impact='needs_qualifier',claim='Existing nonzero-batch L2-box/shared-native gradient norm ratio median is 1.261627%, with native cosine signs 4 positive/3 negative; this does not establish teacher-vs-GT parameter-gradient alignment.',evidence=['CPU_RECOMPUTATION.json: active_batch_shared_gradient_ratio_median',location(old/'calibration_receipt.json')+': parameter_names',calibration+': gradient and cosine functions'],rationale='Norms/cosines originate from actual prior autograd on 24 named tensors in model.16/model.19. This cache-only audit reads and recomputes their ratios, not full gradients. R proxy loss agreement within 5.354e-9 cannot prove student tensor identity.'),
            dict(id='C5_MAPPING_IDENTITY',impact='needs_qualifier',claim='All 649 RGB/IR base GT boxes agree, so the axis map is mathematically identity; 2 of 32 computed mapped targets are not byte-exact copies of IR boxes.',evidence=['INITIAL_RECOMPUTATION_FAILURE.json','CPU_RECOMPUTATION.json: teacher_mapping_float_nonexact_records'],rationale='Observed max reconstruction difference is 1.52587890625e-5 px. Author final output preserves both cases and uses original mapped values. Shared annotation does not prove physical registration.'),
            dict(id='C6_COMPUTE_DECISION',impact='needs_qualifier',claim='Deferring further expansion of this L2 version by only loosening gates or increasing dose is a defensible conservative action; the diagnosis does not prove either change cannot help.',evidence=[location(target/'README.md')+' final interpretation',location(target/'summary.json')+': stored_base_records_without_teacher_target=617'],rationale='617 base records have no computed teacher target, and there is no new controlled intervention. The audit supports a boundary on present evidence, not a causal claim about why previous FT3 performed poorly or a rejection of all localization distillation.')
        ],
        audited_input_hashes='not computed: explicit user-authorized no-hash scope; stat/byte comparisons do not substitute for a cryptographic manifest',
        declared_input_set=manifest,
        recomputation=dict(commands=[
            'D:/Anaconda/envs/KGJ_proj/python.exe '+location(HERE/'recompute_independent.py'),
            'D:/Anaconda/envs/KGJ_proj/python.exe '+location(HERE/'compare_author_outputs.py')],
            outputs=['CPU_RECOMPUTATION.json','AUTHOR_COMPARISON.json'],
            coverage='79 source GT bridge records; 426 positive native (.5/.75 x S/R/T) match links; 80 author anchor rows; all 30 selected teacher-target rows; three algebraic toy obligations; three captured-source byte comparisons; author display-encoding AST equivalence.',
            initial_failure='INITIAL_RECOMPUTATION_FAILURE.json',GPU_used=False,model_loaded=False,new_forward=False,new_training=False,official_test_accessed=False),
        unavailable_inputs=['Historical selected student DFL/logit/probability tensors and decoded boxes','Historical pixel tensors and complete raw-output equality across separate runs','Later seven batches stable native GT identities','Parameter Jacobians and teacher-vs-GT full parameter-gradient vectors','Teacher targets after early-gate rejection'],
        traces=['REVIEW_PROMPT.md','REVIEW_RESPONSE.md','REVIEW_TRACE.json'],
        final_verification='Read final author output_attempt3/output_attempt1, their displayed reports, actual source captures and key original inputs. Source/result hashes deliberately not computed. Stat manifest is a read-time record and not a guarantee against future edits.'
    )
    with (HERE/'EXPERIMENT_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    trace=dict(date='2026-09-08',agent='/root/loc_target_review',observable_model='unavailable',cross_model_claim=False,
               independent_prompt='REVIEW_PROMPT.md',raw_response='REVIEW_RESPONSE.md',parsed_findings='EXPERIMENT_AUDIT.json',
               sequence=[
                   'Read workspace instructions, index, prior original source and cache evidence before receiving author outputs.',
                   'Independently recomputed source identities, cardinalities, targets, SmoothL1 derivatives and ratios using Python standard library.',
                   'Initial exact mapped-vs-IR assertion failed on float reconstruction; failure preserved, user-facing authors notified and original values retained.',
                   'Read final target output_attempt1 and anchor output_attempt3; independently compared 30 targets/80 rows and captured source bytes.',
                   'Independently verified equivalent Unicode display source AST after two retained pre-entry parse failures.',
                   'Received root proposed conservative compute decision only after initial independent results; evaluated as bounded policy choice, not preferred scientific conclusion.'
               ],hash='not computed')
    with (HERE/'REVIEW_TRACE.json').open('x',encoding='utf-8') as f:json.dump(trace,f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
