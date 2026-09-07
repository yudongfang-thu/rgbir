# 定位目标平移诊断独立审阅

**限定范围 PASS：接受修正后的 outputs_attempt2 作为固定共同 anchor、固定对象集合上的 CPU 合成平移诊断。attempt1 的 worst gate 成功数字段及旧 README 对应表格 FAIL，不可引用；原始逐对象数组未受影响。此审阅不授权 L1/L_GT 训练。**

## 独立性与检查范围

审阅者为 `baseline_feature_analysis`，原执行者为另一代理 `loc_stress`；本次未参与原压力测试实现。原始审阅请求来自根任务，要求读取协议、执行代码、保存输出与真实源缓存并核验质量搬运、固定分母、零扰动和 train/dev 身份。没有单独可观察的不同模型身份，因此仅称独立代理审阅，不称跨模型审阅。

读取本次 `PROTOCOL.md`、当前和 attempt1 快照源码、README、两次输出；追溯 `2026-09-07_probe_Baseline蒸馏机会重诊断/remote_exports/{llvip,dronevehicle}_full_attempt1` 的真实 `objects.jsonl`、`logits.npz`、模型/summary，以及旧定位 readout roster/预测和 `new_probe_analysis/*_analysis_v1/dfl_per_object.csv`。只用本地 CPU，无新推理、训练、SSH、GPU、哈希或原产物修改。

## 发现、修复与再核验

attempt1 在 worst 汇总先写 `all_directions_gate_n=成功数`，随后对同名布尔事件调用通用 statistics，其 `n=固定分母` 覆盖成功数。这使两数据集全部 36 个 worst 汇总行的该字段错误，旧 README 的六格 gate 数显示全存活，和正文/保存数组矛盾。独立审阅当时直接通知根任务与作者，未接受该表。

作者保留 `outputs_attempt1` 和 `attempt1_source_snapshot`，新 attempt2 将布尔成功数、分母、比例独立命名为 `*_success_n`、`*_denominator_n`、`*_fraction`，不再进入连续统计器，并加入“5 对象分母/2 成功”的真值回归。独立复核确认两 attempt 的**全部逐对象数组逐元素 exact**，所有新 worst 成功数/分母均可直接由保存布尔数组 exact 重建。本次是汇总缺陷修复，没有变更对象、阈值、扰动、拟合或选择器。

## 实际 CPU 验证

| 检查 | 实际结果 |
|---|---|
| 原始对象重建 | LLVIP 3276=2728 train+548 val；Drone 17840=15019+2821；source_row、object_id、顺序、GT、anchor/stride、split 与原缓存和旧 readout exact |
| 独立全 25 条件重算 | 未 import 原计算函数，另用三角插值矩阵搬运概率；两数据集全部保存 CE/KL/head 梯度/IoU/支持域/逐边流失/gate 重建通过，最大绝对误差均为 2.274e-13，来自条件框坐标的浮点顺序 |
| 零扰动回溯 | 对旧已保存逐对象 CE、KL、cosine 最大差 0；对旧 native DFL 期望距离回验通过；raw 解码与源 same-anchor JSON 框相差不到 5e-5 输入像素 |
| 全部连续汇总 | 每数据集 33648 次数值比较，涵盖 n、有限/非有限计数、均值、总体 SD、min/max、p05/p50/p95 及全部 worst 连续向量；最大差 0 |
| 真值测试 | 重跑作者当前 11 项检查通过，包含 64 维 KD/GT 有限差分、温度不交换、边界守恒、严格门和新增成功数回归 |
| 文件保留 | 审阅窗口内记录的源代码和两 attempt 输出大小/mtime 均未变化；原始源缓存 stat 与已执行记录一致，无新哈希 |

`verify_independent.py` 是独立源到数组重建；`verify_summary_independent.py` 是全部保存统计重建。`author_truth_reexecution.json` 与 `author_verify_saved_reexecution.json` 是作者测试/自检的再次执行，后者仅重定向输出路径。独立计算结果分别在 `independent_cpu_receipt.json`、`independent_summary_receipt.json`，不可只把作者自检称为独立证据。

## 数学与分母语义

教师框在输入 640 坐标中平移 (dx,dy)，共同 anchor 不动；LTRB/stride 改变量为 `[-dx,-dy,+dx,+dy]/stride`，符号正确。几何主读数直接平移原 T=1 期望框，不裁画布或按支持域删除对象。1/2/4 指 L∞ 幅度，四个对角方向的 L2 为该值乘 √2，不是同半径误差。

每个温度的原概率以线性质量重采样平移，区间外 underflow/overflow 单列且不夹到 0/15。保留质量归一化得到 CE/KL/梯度所用条件分布。T=1 与 T=2 独立搬运正确实现冻结定义；它们通常无法解释为同一新 logits 在两个温度下的分布，不能称网络在真实平移图像上的响应。条件分布框与直接精确平移框差异已保留，不能用较低 CE 掩盖边界质量流失。

KL 为四边平均 `KL(pT2 || pN2)*4`；对 N raw logits 的梯度为 `2*(pN2-pT2)/4`，GT-DFL 梯度为 `(pN1-qGT)/4`，温度与边均值因子正确。64 维 cosine 是头部局部方向，不是含 CIoU、TAL、得分权重和共享 backbone 的真实优化梯度。零范数定义和非有限统计显式保留；本实际缓存没有非有限 CE/KL/cosine。

三个 fixed mask 均在零扰动时建立，所有幅度/方向沿用原分母；扰动后 gate 仅统计存活/新增，不对幸存者重算主表。worst 按每对象八方向先取极值，再对固定群体汇总。修复后的 dev gate 八方向存活为 LLVIP 60/61、51/61、34/61；Drone 44/47、31/47、9/47。固定分母没有因为结果变差而缩小。

## 对象身份和可支持结论

基础 cohort 使用 GT 关联出的 N 参考候选 anchor，虽没有重新取 GT ROI，仍有 GT 特权选择；它不是可部署检测器输出集合。base 仅要求 paired GT IoU≥.5、N 参考候选与 RGB GT 的 DFL 支持域；`quality_gate070` 是其共同 anchor 上的 N/T 正类与 confidence、N<.70、T 对 RGB/IR IoU和 margin 条件。它未实现原 L1 的 pair≥.8、唯一 owner、独立 geometry、实际训练增强流/对象选择器，因此不能叫“原 L1 selected”或“原 D2 通过”。LLVIP RGB/IR 共享标签使配对与双 GT 读数高度重复，不提供物理几何真值。

LLVIP 的 23 个 train 门内对象属于 **1024 张固定 train 导出中的 2728 个基础对象**，分布于 22 图/10 source group，全部 stride16。61 个 dev 门内对象属于 **200 张固定 dev 导出中的 548 个基础对象**，分布于 49 图/5 个不同 dev group，stride16/8 为 59/2。两者不是同对象重复测量，也不是完整 9619/2406 数据集统计。冻结 E200 端点在训练图上的 gate 稀少，不能换算成自然增强前 64 batch 的有效 KD 剂量。Drone dev 的 source_group 为 `unavailable:val`，不能从这个占位值推断来源多样性。

可以引用：在这两组已保存模型/固定对象层下，LLVIP 条件层在预定人工目标平移中保留较多教师目标优势；例如 4 输入像素八方向，LLVIP 61 个 dev 固定门对象中 55 个教师 CE 始终更低，但 gate 仅 34 个全方向存活。该数字是局部合成诊断，不是全 LLVIP 稳健性或跨数据集因果比较。不得据此推出真实配准容忍阈值、全网梯度可学性、实际 AP/KD 收益，或 L1/L_GT 可启动。既有几何、自然训练流覆盖、校准、匹配 N 与同 mask GT 对照要求保持原样。

`EXPERIMENT_AUDIT.json` 保存结构化 verdict/checks/claims 与输入说明。按用户明确约束未生成 SHA-256；`audited_input_hashes` 留空并记录原因。当前接受对象是 attempt2，历史 attempt1 错误字段继续标记不可引用。
