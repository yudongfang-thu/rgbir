# 首轮执行状态（2026-09-06）

| ID | 实验/工程门 | 状态 | 证据 |
|---|---|---|---|
| CPU19 | 独立标签加载、原生增强/RNG等价、损失/梯度/空候选等19项 | PASS | cpu_tests_attempt1.log |
| C-P | paired seed42，batch32，24实际更新 | PASS | canary_paired_receipt.json |
| C-N | weight0 seed42，同配置，24实际更新 | PASS | canary_weight0_receipt.json |
| C-EQ | P/N初始模型和首批输入逐张量对比 | PASS | canary_comparison.json |
| REV | 独立代码审查 | PASS | EXPERIMENT_CODE_REVIEW.md |
| P42 | paired seed42，E200，固定last/EMA后val | RUNNING | 94 full_paired_s42_attempt1/progress.json |
| N42 | 同代码weight0 seed42，E200，同端点评估 | QUEUED | 94 full_queue_status.json |

只有物理GPU4，screen `rgbir_oev1_queue_s42`，任何时刻最多运行本轮一个GPU任务。canary为工程检查，不计作额外完整科学实验。正式运行仍为用户授权的两个实验。

全部性能结论待定。没有三seed或四臂归因结果；本轮不进入论文增益claim。后续应先读固定预算last/EMA的两臂统一val结果，再决定是否扩seed及归因，不能提前用canary损失下降证明有效。

查询：`ssh 94` 后，查看 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906/full_queue_status.json` 与 `full_attempt1.log`，对应run下有 `progress.json`、`kd_batches.jsonl`、`completion_receipt.json`（完成后）、`evaluation_val.json`（评估后）及 `run_evidence/`。
