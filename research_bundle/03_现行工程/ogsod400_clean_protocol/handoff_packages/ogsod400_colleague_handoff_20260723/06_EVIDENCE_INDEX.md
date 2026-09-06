# 6. 证据索引

## 6.1 快照范围

- 本地 source cutoff：读取至 2026-07-23 00:30 CST 左右；reset-v2 文件的主审计时间为 2026-07-22。
- L20 live snapshot：2026-07-23 00:48 CST。
- 远程只读范围：hostname/time/uptime、GPU、B2 queue state/process、目录/manifest、行数与 error 关键词；未读取 B2 outcome。
- 未复制：数据、权重、raw runs、日志全文、`.aris/traces`、SSH credentials。
- 未发起：训练、停止/重排、queue 修改、外部模型 review。

## 6.2 当前综合审计

优先阅读：

```text
handoff.md
refine-logs/research_reset_20260722_v2/README.md
refine-logs/research_reset_20260722_v2/OGSOD_EXPERIMENT_AUDIT.md
refine-logs/research_reset_20260722_v2/SIXIANG_EXPERIMENT_AUDIT.md
refine-logs/research_reset_20260722_v2/PAPER_METHOD_REASSESSMENT.md
refine-logs/research_reset_20260722_v2/NOVELTY_AUDIT.md
refine-logs/research_reset_20260722_v2/INTERNAL_AUDIT.md
refine-logs/research_reset_20260722_v2/REMOTE_CATALOG_README.md
```

`research_reset_20260722_v2` 是当前 synthesis index，不是不可修改真理。新 immutable primary evidence 优先于它。

## 6.3 Machine ledgers

原工作区：

```text
registry/runs.csv
registry/campaigns.csv
registry/claims.csv
refine-logs/research_reset_20260722_v2/REMOTE_RESULTS_MANIFEST.csv
refine-logs/research_reset_20260722_v2/REMOTE_CAMPAIGN_ARTIFACTS.csv
```

本包镜像：

```text
catalog/campaigns.csv
catalog/claims.csv
catalog/remote_results_manifest.csv
catalog/remote_campaign_artifacts.csv
```

`runs.csv` 较大且持续变化，没有复制进最小包；接管者在原工作区读取 canonical 文件。`remote_results_manifest.csv` 是 241 个 L20 顶层 entry 的快照，不代表它们都已注册或有效。

## 6.4 方法与理论

### 基础理论/机制边界

```text
DERIVATION_PACKAGE.md
refine-logs/research_reset_20260717/DERIVATION_PACKAGE_V2.md
refine-logs/research_reset_20260717/INTEGRATED_AUDIT.md
refine-logs/research_reset_20260717/LITERATURE_MECHANISM_MAP.md
refine-logs/research_reset_20260717/EVALUATION_RECOVERY.md
```

`DERIVATION_PACKAGE_V2` 的核心边界：shared/private 不可自动识别、reachability 不足、正确干预对象是 RGB-over-SAR、局部 optimizer compatibility 仍需 outcome 验证。

### LCSR/RIF

```text
method/README_CN.md
method/src/lcsr/
method/src/lcsr_detach/
methods/lcsr/README_CN.md
methods/rif_interaction/README_CN.md
methods/rif_interaction/docs/CLAIM_GATES_CN.md
findings.md
refine-logs/LCSR_GAPDETACH_FAILURE_ANALYSIS_20260713_CN.md
```

### CMD hybrid teacher

```text
comparison/ablations/cmdistill_hybrid_teacher/
comparison/runtime/mm_arcs_v2_r2a_hbb/src/teacher_student_decomposition_kd_hbb/hybrid_teacher.py
comparison/runtime/mm_arcs_v2_r2a_hbb/src/teacher_student_decomposition_kd_hbb/loss.py
refine-logs/HF_HS_THREE_SEED_ANALYSIS_20260717.json
```

### MM-ARCS R2A

```text
comparison/runtime/mm_arcs_v2_r2a_hbb/MM_ARCS_R2A_RUNTIME.md
comparison/runtime/mm_arcs_v2_r2a_hbb/src/teacher_student_decomposition_kd_hbb/mm_arcs.py
comparison/ablations/mm_arcs_v2_r2a/
comparison/ablations/mm_arcs_r2a_confirmation_v1/
tools/analyze_mm_arcs_r2a_five_arm_exact400.py
tools/analyze_mm_arcs_confirmation_exact400.py
tools/validate_mm_arcs_r2a_exact400_analysis_lock.py
refine-logs/mm_arcs_r2a_confirmation_20260718/EXACT400_ANALYZER_REVIEW_HISTORY.md
refine-logs/mm_arcs_r2a_confirmation_20260718/R2A_FIVE_ARM_ANALYZER_REVIEW_HISTORY.md
```

### SiXiang full-CMD attribution

```text
results/sixiang_rgb_attr_v1_20260722_a8_a9_composite/
tools/analyze_sixiang_rgb_attr_a8_a9.py
tools/validate_sixiang_rgb_attr_a8_a9_composite.py
findings.md
```

canonical commit-last artifact：

```text
results/sixiang_rgb_attr_v1_20260722_a8_a9_composite/
  sixiang_rgb_attr_a8_a9_final_COMMIT.json
```

### SX-APR

```text
refine-logs/sixiang_anchor_residual_20260722/EXPERIMENT_FREEZE.md
refine-logs/sixiang_anchor_residual_20260722/SCIENTIFIC_LOCK.json
refine-logs/sixiang_anchor_residual_20260722/B1_ENGINEERING_FAILURE_RECEIPT.json
refine-logs/sixiang_anchor_residual_20260722/B2_ENGINEERING_AMENDMENT.json
refine-logs/sixiang_anchor_residual_20260722/B2_ANALYSIS_SOURCE_LOCK.json
comparison/runtime/sixiang_anchor_residual_hbb_v1/
tools/sixiang_anchor_residual_queue_b2.py
tools/validate_sixiang_anchor_residual_campaign_b2.py
tools/analyze_sixiang_anchor_residual_b2.py
```

远程 live state：

```text
/private/projects/ogsod400_clean_protocol/logs/
  sixiang_anchor_residual_v1_20260722_b2/queue_state.json
/private/results/sixiang_anchor_residual_v1_20260722_b2/
```

## 6.5 Idea 与 novelty

```text
idea-stage/IDEA_REPORT_20260716_012320.md
idea-stage/IDEA_REPORT_20260722.md
refine-logs/research_reset_20260722_v2/NOVELTY_AUDIT.md
refine-logs/research_reset_20260722_v2/PAPER_METHOD_REASSESSMENT.md
```

这些文档包含候选 idea、反驳、优先级变化和当前“不要启动第三矩阵”的决定。早期报告不是 current authorization。

## 6.6 数据与外部评估

```text
refine-logs/sixiang_onboarding_20260719/
tools/audit_sixiang_dataset.py
tools/audit_sixiang_split_integrity.py
tools/build_sixiang_yolo_split.py
refine-logs/dronevehicle_*/
tools/*dronevehicle*
refine-logs/m4_sar_*/
tools/*m4_sar*
```

SiXiang shipped split invalid；scene-clean split 为当前开发 split。DroneVehicle/M4-SAR 当前主要是数据工程与审计基础设施，不是 outcome evidence。

## 6.7 旧审计与使用方式

```text
EXPERIMENT_AUDIT.md
EXPERIMENT_AUDIT.json
refine-logs/research_reset_20260717/EVIDENCE_LEDGER.csv
```

旧审计记录了 frozen runtime 的命名/CRC/provenance 问题，对理解失败很重要；但若数值或状态与 reset-v2/canonical artifact 冲突，采用后者。特别是 old CMD/LD、legacy LADD、FGD 名称边界，不要从旧表直接复制为正式结果。

## 6.8 代码与目录入口

```text
README_CN.md
docs/WORKSPACE_MAP_CN.md
MANIFEST.md
AGENTS.md
configs/protocols/
baseline/code/
comparison/runtime/
methods/
tools/
tests/
```

`shared/`、`baseline/code/`、`comparison/runtime/` 是冻结快照；需要更改时创建新 revision/runtime，不原地覆盖。

## 6.9 本包完整性

生成后根目录包含：

```text
PACKAGE_MANIFEST.sha256   # 包内文件 SHA256，不含其自身
PACKAGE_FILE_LIST.txt     # 文件名、字节数
```

校验：

```bash
cd ogsod400_colleague_handoff_20260723
shasum -a 256 -c PACKAGE_MANIFEST.sha256
```

该校验只证明交接包未变化，不替代原始实验 artifact 的内部 SHA/provenance validation。
