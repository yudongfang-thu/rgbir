# LLVIP 置信度分析器独立限定验收

**PASS_RECEIPT_ONLY_CONFIDENCE_ANALYZER：固定新 N/C0 配对回执可用于单 seed FT3 原值与 C0−N 的 pp 汇总。** 独立源码/producer 字段核对及 6 项 CPU 合成检查（作者 4 + 独立 2）通过。未读真实新 AP、运行 GPU/SSH、导入 torch 或计算新 hash；当前源码前后字节一致。

fraction[0,1]、单 person 类 AP 与总量闭合，原值乘100/差乘100的单位正确；0 与负差不会被筛掉。严格 scope/endpoint、seed42、完整 dev2406/7879、train2048、BN running frozen/affine trainable、同初始化/三份训练数据来源及同 full dev YAML；N=0/C0=.1 固定剂量，必须不同实际训练配置和 completion 路径。两个新完成回执不齐、与 failure 冲突、旧 L2 N scope、近似而非精确 .1 的剂量、重复训练 run 都拒绝，不填零或选择旧控制。

实际 evaluator 已提供 `expected_train_images`、`training_subset_identity`、projection 和 BN 字段；输入布局与 mini queue 一致为 `evaluations/{N,C0}/direction_evaluation_receipt.json`。输出不计算 SD/显著性、自动扩展/E200、模态因果或旧 L2-N 对比。本审阅限小回执消费者，不重新加载模型或证明 GT 字节身份，也不把已通过 CPU 当真实两端点已完成。

证据见 [INDEPENDENT_ACCEPTANCE.json](INDEPENDENT_ACCEPTANCE.json)，独立脚本见 [independent_review_cpu.py](independent_review_cpu.py)。
