# 方向筛选分析规则（新方向 AP 读取前冻结）

唯一输入为同一 collected campaign 下 `evaluations/{dataset}/{arm}/direction_evaluation_receipt.json` 及同目录小 failure 回执。不读 checkpoint、训练 CSV、test 或 swap scope。汇总检查固定 `DIRECTION_FT3_BNFROZEN_LAST_EMA`、seed42/3轮、BN buffers 未改变、完整 Drone dev1469/22462 和 LLVIP dev2406/7879，以及 fraction_0_to_1、类别 ID 完整性、逐类 AP 算术平均与总体 AP 一致。

固定矩阵为 Drone N/C1/C2/F-rel 和 LLVIP N/L2-box/L2-GT。固定差值为 Drone C1−N、C2−N、F-rel−N、C2−C1、F-rel−C1；LLVIP L2-box−N、L2-GT−N、L2-box−L2-GT。仅同数据集、共同 training_model、完整 dev YAML 和类别映射一致时计算。每个差均为 `(arm fraction − control fraction) × 100` pp，按 class_id 配对；不跨数据集汇总、不按最好结果挑比较。

每臂原值保留 fraction，展示百分数；差值正/负/零只按原值符号记录，无效果阈值、显著性或自动延长逻辑。单 seed 不算 SD，不声称正式增益或模态因果，不解除旧 C0 复核。根报告需结合真实梯度和覆盖率解释，分析器不从 AP 单独推断机制。

独立审阅补充：LLVIP L2-box 与 L2-GT 共享 teacher-mask 定标的同一 KD 系数；两臂均有完成回执时，系数必须 exact 相等，否则拒绝汇总。这是该固定对照的身份约束；不要求 N、C2、F-rel 或跨类别方向同系数，也不把同系数解释为相同梯度剂量。原源码/测试在 `*_before_l2_coefficient_check.py` 留存，首次 CPU 回执保留。

缺失/失败保留状态和 null；完成与失败同时出现记 CONFLICT，不择一。缺少 control 不计算该差。格式/口径错误的已完成回执直接拒绝。输出是 receipt-only 读数，不重新审计完整 GT 框身份、权重内容或训练选择正确性，也不把 evaluator 的 accepted_endpoint_claim=false 升级。

CLI：`python analyze_direction.py --campaign COLLECTED_CAMPAIGN --output NEW_OUTPUT`。输出 `summary.json`、`README.md` 和分析源副本；不覆盖旧输出。新方向真实 AP 在实现和合成检查冻结前未读取。`test_analyze_direction_cpu.py --output NEW_CPU_RECEIPT` 为七组已知真值，不需要 torch/GPU；不计算新 hash。
