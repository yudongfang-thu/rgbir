# 结果分析器 v2 独立接受记录

> root 复核后，接受本版用于受约束的开发集端点比较和预注册 pilot 扩展判断；不据此自动升级论文主张。

日期：2026-09-07。源码入口：新实验模块 `analyze_results.py`，归档副本 `analyzer_source_v2/`。

复核范围包括实际训练/评估 receipt 的终态、arm/seed/model identity、原始指标与绑定副本、独立评价 roster、单位换算、缺失与重复 attempt、共同训练 recipe、固定 last/EMA 端点、逐 seed 配对与样本 SD。已读源码中的 `bound_recipe`、`load_endpoint`、`paired_comparison`、`pilot_decision` 和相应 20 项已通过 fixture；v2 修复了只信任 manifest protocol_id 的不足。

接受范围：现有冻结 E200 协议的描述性结果和 CL42−C42 ≥0.3 pp、CL42−CGT42 >0 的算力扩展判据。通过三 seed 仍不自动证明因果归因、消除负迁移或测试集泛化；须额外满足四臂、实现验收及测试暴露审计。

当前有效严格配对只有 seed42：C−N 的 mAP50–95 为 +0.144554 pp，AP75 为 −0.327243 pp。缺失 seed 不填补；跨旧协议的 CMDistill/CCLKD partial 保持独立身份。分析器接受不把未完实验变为完成，也不把单 seed 描述变为正式增益主张。
