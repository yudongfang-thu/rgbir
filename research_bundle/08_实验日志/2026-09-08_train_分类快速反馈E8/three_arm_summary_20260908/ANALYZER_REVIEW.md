# E8 描述性分析器独立限定审阅

**PASS / READY：`analyze_e8.py` 可用于本次固定的三个已审 E8 端点，输出同 seed 原值、百分点差及逐类表。** 2026-09-08，loc_stress。本次只读源码、调用 `self_test()` 和独立合成小样例；没有调用 `main()`，没有运行真实端点比较或写正式 summary/table。

- 主入口要求三份 endpoint review 为 PASS 且全部检查通过，训练完成、训练/评估 checkpoint stat 一致、8 轮/4504 批、step+skip=attempt，以及三臂 development roster 字节一致。本次三端点的共同配置/实际评估 contract 已在 C1 端点报告独立核对。
- 计算前要求 N/C0/C1 齐备，arm/seed42/E8 horizon/固定 last-EMA/Drone population/共同 native binding、fraction 单位及 SHORT_SCREEN 范围一致；拒绝非有限或越界指标、缺类/重复类/类名不一致及逐类 AP 均值不闭合。class_id 排序后再配对，逐类表不会依输入行顺序错配。
- 百分制原值为 `100 × fraction`，主差值为 `100 × (A−B)`，比较顺序固定 C1−N、C1−C0、C0−N；逐类 pp 与 macro 差闭合。额外 relative_mAP_percent 独立命名，零分母返回 null，不混充 pp。
- 输出明确 `n_seeds=1`、sample_sd=null、formal_gain_claim=false、automatic_expansion=false，不计算显著性、跨 seed 方差或调度决策。只服务这一已接受输入范围，不作为通用科学验收器或 E200 比较器。

作者 15 项已知真值检查独立重跑通过；另 12 项检查通过，包括非均匀逐类原值/有符号差/均值闭合、类别对应、相对差与 pp 区分、零分母、E200/论文增益/test 标志及非有限类别拒绝、审阅前后源码字节不变。见 [analyzer_review_receipt.json](analyzer_review_receipt.json)，合成检查入口 [review_analyzer_cpu.py](review_analyzer_cpu.py)。没有 GPU、权重读取、新 hash 或真实效应计算。

运行后的真实表仍需据原始 receipt 解释，保留单 seed/E8 早期学习的限制和 C1 最后 4 批未记录薄路径最终计数的局限。训练 CSV 末行短列不进入该分析器的 AP 来源。正式运行由根任务统一执行。
