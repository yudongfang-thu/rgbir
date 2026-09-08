# Hourly FT3 分析器独立限定验收

**PASS_FOR_FIXED_HOURLY_DESCRIPTIVE_ANALYZER。** 冻结源码只获准汇总本次固定 N/C0/C1、seed 42、共同成熟 RGB 初始化、2048 图/3 轮的描述性原值、百分点差和独立计时。此验收不接受实际结果或科学增益主张。

2026-09-08，独立只读审阅 `analyze_hourly.py` 与合成测试；重跑作者 14 项并补 4 项独立已知真值，18/18 PASS。源码 13880 bytes，mtime_ns 1788833610308997400；运行前后直接字节一致，未计算新 hash。回执见 [analyzer_independent_acceptance.json](analyzer_independent_acceptance.json)，独立复算入口见 [review_analyzer_cpu.py](review_analyzer_cpu.py)。未读取新 AP、权重或实际结果，未调用 GPU/SSH，未导入 torch，未执行正式分析。

- 输入必须是三臂完整 FT3 LAST_EMA 单 seed 身份、192 批训练与完整 dev 1469 图/22462 GT。共同模型/T/R、评估绑定和训练/评估 checkpoint stat、canary 路径需一致；实际加载时另要求三臂 dev roster 字节一致且 1469 唯一项。固定分类系数为 0 / 0.1 / 0.09227393550836771，定位系数为 0。
- 原值严格接受有限的 [0,1] fraction，百分制为乘 100；差值为指定方向的 fraction 差乘 100，单位 pp。五类按 ID 对齐，固定名称且每类 AP 宏均值与汇总闭合；P/R 保留 evaluator 原生汇总含义。
- AMP 与 EMA 计数按实际值闭合，不要求各臂成功更新数人为相等；不生成跨 seed SD、显著性、自动扩展、E200 决策或解除 C0 REVIEW_REQUIRED。
- 计时保留队列总耗时；训练与评估入口耗时是其内含区间，余量包含 canary、等待、启动及入口外工作。不会把队列与内含区间再次相加；不把该计时当严格代码加速。

审阅发现的固定臂系数缺口已由作者修复并加负例；初版 synthetic fixture 浅拷贝导致 checkpoint 负例不独立，也已改为深拷贝。原 FAIL 回执和 attempt1 源码保留，正确的 checkpoint 比较器未放宽。

范围边界：这是已接受本次训练/评估入口的小回执消费器，不独立重建数据集、权重内容或重新评价 AP。实际输入仍须由根任务收取并执行；本次审阅不包含其原值。成熟初始化已经见过完整训练集，不能将输出称为只用 2048 图从头训练的效果。
