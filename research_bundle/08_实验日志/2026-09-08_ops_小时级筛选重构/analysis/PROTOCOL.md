# 小时级三臂描述性汇总协议

在读取本次 FT 新 AP 前编写并以合成真值测试冻结。只读 `runs/{N,C0,C1}/hourly_training_receipt.json`、`evaluations/{arm}/hourly_evaluation_receipt.json`、对应完整 `development_roster.txt` 与 `queue/completion.json`。实际 evaluator 将指标原值内嵌 evaluation receipt，没有额外要求不存在的 metrics.json。

限定 Drone、seed42、共同成熟 RGB 初始化、2048 图 train、3 轮/192 批、固定末 EMA、完整 dev1469 图/22462 GT。检查训练/评价身份、checkpoint stat、训练计数闭合、共同 teacher/reference/native evaluator 与逐臂 full roster 字节一致。队列须已完成三 canary 后 N train/eval → C0 train/eval → C1 train/eval 的固定九阶段；未完成或失败输入不出三臂结果。

复用已接受 E8 分析器的纯计算规则：fraction 指标乘100为百分制；C0−N、C1−N、C1−C0 的差为100×原始fraction差，单位pp；五类按class_id对齐，原生宏AP等于五类AP均值，绝对容差1e-12（fraction）。拒绝缺项、布尔数值、NaN/Inf、越界、缺类/重复类、单位或身份不匹配。报告 mAP50–95/AP50/AP75，保留 P/R 原值但不把它解释成固定阈值损伤计数。

耗时分别使用各训练/评价回执的 seconds，以及整个队列 completion.seconds。队列差额含 canary、等待、启动与入口外处理，不能全称训练开销或代码加速。保留各臂实际 optimizer/AMP/EMA 计数；相同额外 epoch/sample 预算不要求事后补足相同 successful updates。

训练 `results.csv` 不进入此分析器。根任务已指出禁用训练内评价时末行列集合变化，不能按表头把末行数值自动解释为 AP 或 LR；完整3轮/192批与计时使用执行回执，本分析器没有自动 LR 解析，也不因 CSV 布局问题重训。

输出只允许 `DESCRIPTIVE_SINGLE_SEED_FT3_ONLY`，n_seeds=1、sample_sd=null、formal_gain_claim=false、automatic_expansion=false。不计算跨seed SD/显著性，不使用E200升级门槛，不自动解除C0复核，不判定从头训练收益或非目标类因果。不读权重、预测大文件、不运行GPU/SSH、无新hash。本分析器不替代已执行训练/评价入口的技术验收。

CLI：`python analyze_hourly.py --campaign <本地已收齐的campaign根> --output <不存在的新输出目录>`。输出 summary.json、COMPARISON_TABLES.md、README.md、inputs.json 与执行源码副本，拒绝覆盖。独立审阅由 loc_stress 执行，正式新 AP 汇总由根任务调用。
