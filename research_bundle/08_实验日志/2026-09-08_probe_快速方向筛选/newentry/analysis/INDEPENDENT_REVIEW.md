# 固定方向分析器独立限定验收

**PASS_FOR_RECEIPT_ONLY_DIRECTION_ANALYZER：当前冻结版可读取本次固定 7 臂完成/失败回执，输出原值和 8 项描述性 pp 对比。** 未读取真实新 direction AP，未执行正式汇总；不接受科学增益、E200 准入或自动扩展结论。

独立只读源码与 producer `evaluate_direction.py` 的实际 receipt 字段核对，并重跑作者 7 项、补充 4 项 CPU 真值，11/11 PASS。源码在检查前后字节一致；未导入 torch、加载模型、使用 GPU/SSH 或计算 hash。证据见 [INDEPENDENT_ACCEPTANCE.json](INDEPENDENT_ACCEPTANCE.json)，独立入口见 [independent_cpu_review.py](independent_cpu_review.py)。

- 有限 fraction[0,1] 原值乘 100 显示为百分数；固定方向差乘 100 为 pp。按 class_id 对齐逐类值，宏均值与总体 AP 闭合；非均匀逐类变化的独立真值检查通过。
- Drone 固定 N/C1/C2/F-rel，LLVIP 固定 N/L2-box/L2-GT；固定 8 对比，不跨数据集汇总。只接受 seed42、FT3 固定 last/EMA 身份、BN buffers unchanged、完整 dev1469/22462 或2406/7879。同组完成臂的初始化、full dev YAML、class mapping 必须一致。
- 只有所有 7 臂实际完成输入存在且验证通过时才标 COMPLETE_DIRECTION_READOUT。缺失/失败/完成与失败冲突保留 MISSING/FAILED/CONFLICT 与 null；缺 control 时不产生差，不填 0。错误口径的完成 receipt 会报错，不冒充有效低 AP。全缺失的独立真值仍为 PARTIAL。
- 数据和 checkpoint 的运行时验证由已绑定评价入口完成，分析器只消费其小回执，不重复重建 GT、加载权重或解释源模态因果。输出单 seed SD=null，自动扩展/正式增益/E200 标记均 false。

本次审阅发现并由作者修复唯一确定缺口：L2-box 与 L2-GT 必须共享固定 teacher-mask 系数。修后两臂都完成时要求 kd_coefficient exact 相同；`.1` 与 `.10000000000000002` 的负例已拒绝。C2/F-rel 的独立定标系数可不同，未误加跨方法相等限制。修改前源码/测试与 attempt1 回执均由作者保留。
