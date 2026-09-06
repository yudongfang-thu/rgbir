# OEv1 同剂量随机选择对照（2026-09-06）

> **调度更新（23:57）**：经同卡双训练短测和独立审核，未启动seed0/123交由新worker接管：seed0在GPU2与42双开，seed123在GPU4做canary后等待N42完整评估再正式训练。只改调度，不改本条科学协议；实时回执及旧队列安全交接边界见[单卡并发条目](../2026-09-06_ops_单卡并发与计划澄清/README.md)。下文23:25串行队列描述为历史状态。

> 已启动：检验OEv1“教师可靠性＋相对质量选择”是否优于同基础集合、同K、同名义剂量的随机选择。17项CPU检查与seed42真实24更新canary通过；23:25已核验GPU2上seed42/E200运行，0/123串行排队并各自先做canary；原P/N运行源码不变，尚无本对照的性能结果。

## 目的

主线 P/N 检验整套蒸馏干预的净收益。本条目增加 `paired_random`，用于区分选择策略与同剂量对象蒸馏；不是文献方法对比，也不能单独归因于 q 排序。

## 设置

冻结协议见 `EXPERIMENT_PLAN.md`；隔离源码从 94 真实 `release_v2` 只读下载，原副本在 `source_baseline/`，两行变更在 `source_delta.patch`。训练配置和 loss/loader/evaluator 原样复用。

## 结果

尚无E200性能结果。CPU 17项测试全部通过：原loss算子13项，新增真实criterion路由/梯度与随机选择科学性质4项；详见`CPU_VALIDATION.json`。

seed42真实canary与原P各完成24次成功optimizer update、6次AMP跳步；initial_student.pt、first_batch.pt全部张量相等。30个共同batch的学生文件顺序、base/eligible/K/分母/名义剂量一致，30批选择对象均不同。loss/loader字节相等，trainer恰为两个arm路由变更。GPU2单CUDA进程峰值6304MiB，RSS28774MiB，均在预约10000/49152MiB内。

正式队列`oev1_random_queue_3seed`于23:24启动，23:25已进入seed42第1轮且runtime_ready确认17990train/1469val、563batches、batch32/workers4。先42，再0，再123；0/123也须与各自原P canary做一致性验证，通过才可进E200。全部固定last/EMA独立val。

第一次本地canary校验包装脚本发现远端预先复制的validator与最终审查版不一致，在执行比较前断言失败。没有改训练或重跑canary；最终审查版另存`validate_random_canary_reviewed.py`，使用同一canary产物完成校验，结果见`canary_comparison_s42_attempt2.json`。保留第一次本地空输出和错误事实，不将其误记成模型训练失败。

## 结论

本对照等量的是每个 batch 的选择数量 K、基础集合大小与名义剂量 K/|E|；不是强制相同的实际 loss 数值或梯度范数。随机选择来自 base E，允许抽到未通过教师可靠性或 q>0 的对象，因此检验的是整个可靠性＋质量选择。

## 产物路径

- 来源：94 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906/release_v2/`。
- 实际部署：94 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_oev1_random_20260906/release_v1/`。
- 实际训练输出：94 `/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_oev1_random_20260906/`。
- `canary_launch.json`、`canary_comparison_s42_attempt2.json`、`full_queue_dispatch.json`；主调度器`random_worker.py`已独立审阅。
- 源码准备阶段文档中的`rgbir_object_evidence_random_20260906`是原拟路径，实际位置以上述launch receipt为准。

## 局限与下一步

source review、CPU及seed42 canary已通过。队列只重试明确未启动的guard资源拒绝，任何已启动训练/canary/评估失败均停止并保留attempt。每阶段入场重新检查用户AGENTS四卡放宽条件；不改全局guard。现有N/P继续，same-modal/shuffled/GT-only未实现。本条目是后续机制实验，冻结时已见过部分P结果，不追溯称与最初主实验同时预注册。
