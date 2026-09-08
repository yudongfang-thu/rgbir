"""Persist the final evidence audit; never changes execution artifacts."""
import json,pathlib,datetime
p=pathlib.Path(__file__).resolve().parent.parent
scope='OBJECT_DFL_FT3_BNFROZEN'
o=p/'independent_review/final_matrix'
r=json.loads((o/'RECOMPUTATION.json').read_text(encoding='utf-8'))
m=json.loads((o/'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
assert r['status']=='PASS' and all(x['direct_bytes_unchanged_at_end'] for x in m['files'])
def ev(s):return str((p/s).resolve())
def ck(details, evidence, status='pass'):return dict(status=status,details=details,evidence=evidence)
limits=[
 'Single dataset LLVIP, seed42, mature visible42 initialization, fixed FT3/2048 training images. No variance, statistical significance, E200 ranking validity, cross-dataset or general DFL claim.',
 'All arms have 192 batches, 96 optimizer attempts, 5 AMP skips, 91 successful optimizer updates; EMA is called 96 times. Equal update counts do not imply no AMP skips.',
 'Full dev actual native receipts/profile/captured population are traceable; accepted_endpoint_claim=false and independent_endpoint_accepted=false remain in original receipts. This audit accepts scoped provenance/comparison arithmetic, not a new native-vs-capture parity experiment or a fresh AP computation from raw predictions.',
 'The stored post-NMS boxes/classes/confidences are finite for all 7218 image records and 23637 GT occurrences. This does not prove every dense model tensor/gradient/checkpoint tensor finite; complete weights were not downloaded or reloaded.',
 'Only the first30 batches have full saved sample-flow records; mask/base diagnostics are 15 saved batches per arm, covering1156 base and54 selected occurrences. Total673 selected occurrences per arm is the actual trainer counter, not673 unique objects or independently observed masks for all192 batches.',
 'Shared RGB/IR labels define annotation-coordinate targets, not independent physical registration. No release of prior L1 BLOCKED and no separation of distribution shape from its mean.',
 'GT and DFL use the same lambda and nominal mask/base dose; actual calibration gradient ratios differ (DFL median0.1 versusGT0.12283740486876371). This is not strict gradient-dose matching.',
 'Each results.csv has column counts[15,15,15,8]; training-loop validation is disabled and its zero/short metric columns are not AP evidence. Full-dev AP comes from separate completed evaluation receipts. Original CSVs remain untouched.',
 'Runtime remote source_copies are retained remotely. Existing execution manifests record direct-byte copy identity; deployment/launcher record remote byte equality to the independently reviewed18-file snapshot. This audit directly rechecked local source/snapshot and read those runtime manifests without a fresh remote download.',
 'Initial mature endpoint is an existing separately completed native readout, with matched checkpoint path/stat, actual settings/evaluator and canonical/actual loader rosters; it supplements the current matchedN and does not replace it.',
 'Final report numerical tables, stage durations, update accounting and fourth-GPU example were checked. Its contextual historical C0/C1/L2 research statements were not newly audited here.'
]
checks={
 'gt_provenance':ck('Real dataset person labels. All three actual dev capture GT payloads match exactly, each2406 unique images and7879GT. Actual loader/canonical roster and settings equal initial reference; local alias filenames match while runtime contract records resolved canonical paths. Originaltrain/dev split only; no official test access reported.',[ev('training_evidence_final/evaluations/llvip/'+a+'/direction_evaluation_contract.json') for a in ['N','L3-DFL','L3-GT']]+[ev('initial_endpoint_reference/initial_evaluation_contract.json'),'RECOMPUTATION.json#capture_coverage']),
 'score_normalization':ck('Native ultralytics8.4.115 DetectionValidator undertorch2.10.0+cu128, complete dev, FP32 effective kwargs:640/B32/workers4/conf.001/iou.7/max_det300/recttrue/augmentfalse/halffalse/quantizenull. Five raw metric fractions copied exactly; display100x, differences100x(rawA-rawB), onepersonclass AP matches macro. Independently recomputed all15 contrast metrics and15 changes frominitial; no AP regeneration.',[ev('trainer_release/evaluate_object_dfl.py')+':222',ev('analysis/output_attempt1/summary.json'),'RECOMPUTATION.json#differences_pp']),
 'result_existence':ck('All10 fixed stages have actual cleanexit0, empty monitor_errors, valid receipts and correct order. Calibration8 then all3canary24successful then eacharm train192/evalfull-dev. Matrix actual666.8562324047089seconds. Initial8/canary files match previously audited snapshots; originalfailedattempt1 preserved.',[ev('training_evidence_final/queue/llvip_completion.json'),ev('training_evidence_final/queue/completion.json'),'RECOMPUTATION.json#stage_resources_and_seconds']),
 'dead_code':ck('Executed release_v2 manifest links to reviewed/deployed source; all18reviewed local files still byteexact. Effective configs differ from frozen only by completed calibration receipt and fixed coefficient. Actual positive unitKD gradient, selected counts and nonzero added losses confirm auxiliary path ran. NativeN total is exact in saved rows. Head-inclusive499tensor warmstart/freshoptimizerEMA/243frozenBNbuffers and detachedT/R recorded.',[ev('independent_review/TRAINING_SOURCE_REVIEW_v2.json'),ev('training_evidence_final/pinned_source/deployment.json'),ev('training_evidence_final/runs/llvip/L3-DFL/completion_receipt.json'),'RECOMPUTATION.json#artifact_checks']),
 'scope':ck('Supports scoped execution and arithmetic plus the predeclared decision to stop this version because DFLmAP is belowN andGT. No paper gain, causal modality, allDFL failure, physical-registration or long-training conclusion. All three arms drift below mature reference; this run does not identify the cause.',[ev('SHORT_SCREEN_PROTOCOL.md'),ev('FINAL_REPORT.md'),'RECOMPUTATION.json#difference_from_initial_pp']),
 'eval_type':ck('real_gt for actual held-outdev detection AP. Annotation-coordinate L3 targets use the shared RGB/IRGT and teacher/reference outputs; target-coordinate diagnostics do not supply independent physical truth.',[ev('training_evidence_final/evaluations/llvip/N/direction_evaluation_contract.json'),ev('training_evidence_final/runs/llvip/L3-DFL/kd_batches.jsonl')]),
 'resource_admission':ck('All10stage status/resource records clean. Every sample retains>=2048MiB cardfree and projectRSS<=300GiB. At most4activeprojectGPUs; every fourth-GPU admission has>=2emptycards. Ntraining example active0/2/4/5 andempty1/3 agrees with report. Three training reservations6144MiB/20480MiB; NVMLpeak5694MiB,allocated4733.03564453125,reserved5162; RSS17968/17862/18022MiB.', ['RECOMPUTATION.json#stage_resources_and_seconds',ev('training_evidence_final/queue/object_dfl_attempt2_llvip_N_train_admission.json')]),
 'mask_and_model_boundaries':ck('Thirty sample rows byteexact acrossalltrainarms and eachcanary. All15saved diagnostic base/mask/anchor records equal, all54selected satisfy fixed gates. Zero-selected savedbatches3/80 retained; fullsupport andFP32positive-tail rejects0 withinsaved54. Nonempty predictions in2400/2399/2398images, allsavednumbersfinite; unavailable full dense/state checks explicitly limited.', ['RECOMPUTATION.json#mask_coverage','RECOMPUTATION.json#capture_coverage']),
 'nonendpoint_csv_format':ck('Minor known logging defect: finaltrainingCSV row8columns under15columnheader; missing/zero validation fields do not represent AP and were not used. No material effect on independently completed eval receipts or comparison arithmetic.',[ev('training_evidence_final/runs/llvip/'+a+'/results.csv') for a in ['N','L3-DFL','L3-GT']],status='warn')
}
claims=[
 dict(id='fixed_matrix_executed',impact='supported',evidence=checks['result_existence']['evidence'],rationale='10actualstages completed with scoped provenance; not inferred from source readiness.'),
 dict(id='stop_current_DFL_FT3_version',impact='supported',evidence=['RECOMPUTATION.json#differences_pp',ev('SHORT_SCREEN_PROTOCOL.md')],rationale='ObservedDFL-N=-0.010679360492732437pp andDFL-GT=-0.043234480864812186pp on frozenmAP50-95; follows prospective screen rule.'),
 dict(id='all_three_below_initial',impact='needs_qualifier',evidence=['RECOMPUTATION.json#difference_from_initial_pp'],rationale='N/DFL/GT are-0.7071807864/-0.7178601469/-0.6746256660pp versus existingmatchedinitialreadout. Does not identifyBN/overfitting/alignment mechanism.'),
 dict(id='GT_gain_or_distribution_shape_benefit',impact='unsupported',evidence=['RECOMPUTATION.json#differences_pp'],rationale='GT-N+0.0325551204pp is a single-seed tiny observation; shapes andmeans notisolated, actualgradientdoses differ.'),
 dict(id='all_DFL_or_localization_distillation_invalid',impact='unsupported',evidence=[ev('FINAL_REPORT.md')],rationale='Onlythismatureinitialization/FT3/prespecifiedmask/coefficient versiontested. No generalnegativeinference.'),
 dict(id='full_per_batch_mask_or_all_dense_tensor_finiteness',impact='unsupported',evidence=['RECOMPUTATION.json#mask_coverage','RECOMPUTATION.json#capture_coverage'],rationale='Saved30sampleflow/15maskdiagnostics andpostNMSfiniteoutputs donotobserveall192maskrecords orallmodeltensors.')
]
audit=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),auditor=dict(agent='/root/object_transport_review',independent_of_implementer_and_executor=True,model_identity='Not exposed; inherited parent settings',cross_model_claim=False),overall_verdict='pass',integrity_status='pass',reason_code='FIXED_EXECUTED_MATRIX_PROVENANCE_AND_ARITHMETIC_ACCEPTED_WITH_EXPLICIT_SCOPE',checks=checks,claims=claims,scope=scope,endpoint='OBJECT_DFL_FT3_BNFROZEN_LAST_EMA',eval_type='real_gt',dataset='llvip',seed=42,n_seeds=1,standard_deviation=None,declared_input_set='INPUT_MANIFEST.json',audited_input_hashes=[],new_hash_computed=False,identity_method=m['identity_method'],inspection_coverage=dict(artifact_checks_passed=len(r['artifact_checks'])+1,declared_files=len(m['files']),existing_final_collection_files=163,new_tests=0,new_GPU=0,new_inference=0,read_existing_native_capture=True,new_AP_from_predictions_recomputed=False),recomputation_command='D:/Anaconda/python.exe -X utf8 '+ev('independent_review/audit_final_matrix.py'),recomputation_output='RECOMPUTATION.json',source_review=ev('independent_review/TRAINING_SOURCE_REVIEW_v2.json'),analyzer_review=ev('independent_review/ANALYZER_SOURCE_REVIEW.json'),limits=limits,unavailable_inputs=['Complete remote weights and all dense checkpoint/gradient tensors','Per-batch full masks for unsaved177batches','Fresh remote download of allruntime source_copies','New native/capture parity experiment or nativeAP re-computation'],raw_metrics_fraction=r['raw_metrics_fraction'],differences_pp=r['differences_pp'],initial_reference_fraction=r['initial_reference_fraction'],difference_from_initial_pp=r['difference_from_initial_pp'],queue_seconds=r['queue_seconds'],run_ids=[x['id'] for x in r['stage_resources_and_seconds']],original_failed_attempt_preserved=ev('training_evidence_1703'))
(o/'EXPERIMENT_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
md='''# 固定三臂实际终态独立审计

**PASS：LLVIP、seed42、成熟初始化的 FT3 固定矩阵实际完成，端点来源与比较算术闭合。当前 L3-DFL 未超过 N 或同掩码 GT，符合执行前冻结的停止本版本规则。此结论不否定全部 DFL 或定位蒸馏。**

| 臂 | mAP50–95（百分数） | 相对成熟初始化（pp） |
|---|---:|---:|
| 成熟初始化 | 32.878405 | — |
| N | 32.171224 | −0.707181 |
| L3-DFL | 32.160545 | −0.717860 |
| L3-GT | 32.203779 | −0.674626 |

DFL−N 为 −0.0106793605 pp，DFL−GT 为 −0.0432344809 pp，GT−N 为 +0.0325551204 pp。原 fraction、全部五项指标差值和初始差值均从实际回执独立重算，与已接受分析器相同。单 seed 不计算 SD 或显著性。

实际 10 阶段均 clean exit 0、monitor_errors 为空，顺序为固定 8 批校准→三臂 canary→各臂训练/完整 dev 评价，队列耗时 666.8562324 秒。每臂训练 192 batch、96 次 optimizer 尝试、5 次 AMP 跳步、91 次成功更新；EMA 调用 96 次。源和配置绑定 release_v2；18 个已审源文件仍与快照逐字节一致，运行配置只填入校准回执及固定系数。

初始化含检测头 499 个 state tensor，fresh optimizer/EMA，243 个 BN buffer 保持，T/R 隔离。前三十批数据与标签记录跨三臂、训练对 canary 均逐字节相同。15 条保存的诊断记录中，1156 个基础对象、54 次选中的 mask、anchor 与分母完全一致，均满足固定门；第 3、80 批零选中如实保留。673 是全程累计选择次数，不是唯一对象数，也不能代替未保存的 177 批完整 mask 观测。

独立 dev 评价每臂实际 2406 个唯一图、7879 GT。三臂 capture 的 GT 完全相同，actual kwargs、evaluator 版本、canonical/actual loader roster 与成熟参照一致：640/B32/workers4、FP32、conf=.001、IoU=.7、max_det300、rect=True、augment=False。原数据标签是真实 GT；共享 RGB/IR 标签不提供独立物理配准真值。保存的 7218 图记录内所有 GT、预测框及置信度数值有限；三臂有预测的图数为 2400/2399/2398，没有全臂输出失败迹象。这不证明每个未保存的 dense tensor 或权重元素有限。

所有阶段资源记录符合原 lease：样本显存余量至少 2 GiB、项目 RSS 不超过 300 GiB；每次第四卡例外均保留至少两张空卡。三臂训练预算 6144 MiB VRAM / 20480 MiB RSS，NVML 峰 5694 MiB、allocated 4733.0356 MiB、reserved 5162 MiB；RSS 为 17968/17862/18022 MiB。报告 N 训练占卡 0/2/4/5、剩空卡 1/3 的示例与准入回执一致。

**证据边界：**原 eval receipt 的 accepted_endpoint_claim=false 及 profile 的 independent_endpoint_accepted=false 均保留。本审计接受固定比较的执行来源、实际人群/参数和回执算术；没有重新执行 native/capture parity 或从 raw predictions 重算原生 AP，没有下载/重载完整权重。运行源远端副本身份采用既有部署/launcher/运行 manifest 的逐字节回执，并直接复核本地已审快照；未新下载全部远端源码副本。

三臂 results.csv 最后一行是 8 列而表头为 15 列，训练内验证关闭产生的零/缺列不可读作 AP。这是非端点日志格式问题；真正 AP 取自独立完整 dev 评价回执，原 CSV 未改写。GT 与 DFL 共用 λ=0.6900524651944485，但校准实际梯度比例中位数分别为 0.1228374 与 0.1，不能称严格梯度剂量匹配。

三臂均低于成熟初始化，只能说明这个固定微调设置的结果；不能将下降单独归因于 BN、过拟合或教师错位，不能泛化成所有 DFL 无效、物理配准已验证、形状效用已隔离或长训练无效。FINAL_REPORT 的关键数值、十阶段耗时、初始参照和第四卡示例已核；其历史 C0/C1/L2 背景不是本审计新增范围。

本轮仅审阅现有产物：108 项检查通过，258 个声明输入在审阅结束时逐字节未变，没有新 hash、测试、forward 或 GPU 任务。详情见 [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json)、[RECOMPUTATION.json](RECOMPUTATION.json) 与 [INPUT_MANIFEST.json](INPUT_MANIFEST.json)。审计脚本位于上级 independent_review/audit_final_matrix.py；不覆盖任何原始结果或失败 attempt。
'''
(o/'EXPERIMENT_AUDIT.md').write_text(md,encoding='utf-8')
print(json.dumps({'status':'PASS','audit':str(o/'EXPERIMENT_AUDIT.json'),'checks':audit['inspection_coverage']['artifact_checks_passed']},ensure_ascii=False))
