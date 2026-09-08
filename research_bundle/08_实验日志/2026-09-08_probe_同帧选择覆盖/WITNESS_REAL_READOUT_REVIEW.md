# 同帧检测见证：真实产物独立复算

**PASS_FOR_OBSERVED_SINGLE_BATCH_READOUT：原先“低置信度”的单候选状态明显低估了这一批的检测正确对象数；原生 NMS 后只剩 1 个 T 正确/S 错误对象，且它已被当前选择器选中。** 这是诊断定义的澄清，没有模型学习或检测结果改善，不能写成“修复了错误”或“消除了负迁移”。

## 输入与独立核对范围

只读 [witness_evidence_1315_final/probe](witness_evidence_1315_final/probe/) 的 `objects.jsonl`、`witness_objects.jsonl`、`witness_frames.jsonl`、完成回执和见证合同，对照 [原 probe 对象](evidence_1255_final/probe/objects.jsonl) 与 [已接受汇总](witness_analysis_final/summary.json)。本次直接用 Python 标准库逐对象重新计数，没有调用已接受分析器的聚合函数，没有新模型前向、GPU、训练、参数更新或文件 hash，没有修改生产者及原产物。

独立检查结果：原 80 条对象记录与上一 probe 逐字段 exact；新见证行去掉 `detector` 后与原记录 exact；四定义计数、18 张两两混淆表的 72 个单元，以及四定义下各桶对象数、selected 数和对象 ID 共 144 项断言通过。另将 232 个原生匹配见证（S 77、R 77、T 78）逐一绑定到实际帧预测，anchor、confidence、class、box、TP 标志 exact，逐模态/逐帧 prediction index 一对一，全部满足 confidence > .25、正确类别、IoU ≥ .5；帧内全部 TP 数也闭合。

本次仍为 LLVIP 固定首 32 张增强后训练图、80 个 RGB GT，全部有配对 IR GT；S/R 相对 RGB own GT，T 相对 IR own GT。不是全 dev，也没有估计 AP。运行回执记录 S/T/R 各一次前向、无 backward/optimizer/EMA update、学生完整 state 不变；历史流的路径与双标签 exact，但 `historical_pixel_tensor_comparison=false`，不能称与历史训练像素张量逐元素核过。[队列完成回执](witness_evidence_1315_final/queue/completion.json) 为 COMPLETED，78.268389 秒；probe 内计时为 27.334819 秒。

## 四种定义的真实对象计数

分母均为这 80 个 own-GT 对象。

|模型|旧 GT 辅助一对一 assigned 正确|FP32 dense-any 正确|native pre-NMS any 正确|native post-NMS 匹配正确|
|---|---:|---:|---:|---:|
|S|64|77|77|77|
|R|64|77|77|77|
|T|63|78|78|78|

后三列不仅总数相同，**每个对象的正确/错误布尔值也完全相同**。本批没有出现 dense 正确对象在 native 解码或 NMS/GT 匹配后失去正确见证。此结论限于当前记录，不能推断这些算子一般等价。

[见证合同](witness_evidence_1315_final/probe/witness_contract.json) 明确区分旧 FP32 decode 与原生 head decode：S/R 框坐标最大差 0.126129 像素，T 为 0.097534 像素；旧 dense confidence 用 ≥ .25，原生用 > .25。原生 NMS 固定 IoU .7、max_det 300、agnostic=false、multi_label=true，匹配 IoU .5；匹配来自 pinned 原生 validator，实际 pairs 与 TP 的闭合由 producer 回执记录。本次只读复核保存结果，未重新运行原生 matcher。

## 旧 low-confidence 到原生检测的映射

|模型|旧 low-confidence 数|其中 native 匹配正确|仍未匹配正确|旧 no-candidate 数/仍未匹配|
|---|---:|---:|---:|---:|
|S/R 各自|15|13|2|1/1|
|T|16|15|1|1/1|

S/R 的 13 个、T 的 15 个变化对象，原生正确见证的 anchor **全部不同于旧 assigned anchor**。S/R 旧 assigned confidence 范围为 0.059647–0.249447，实际原生见证 confidence 为 0.260498–0.816895、IoU 为 0.579477–0.930564；T 分别为 0.050705–0.245085、0.412109–0.858398、0.632792–0.918813。原来那一个被分配候选低于置信度阈值，并不等于同一检测器对该 GT 没有正确预测。

原 actual teacher gate 与 T 的 dense-any 布尔值在全部 80 对象上 exact。因此此前“actual teacher gate=true，却被旧 T assigned 状态记为错误”的 15 个对象，在本批都已有原生正确匹配。此前记录在 assigned 定义下正确；将其称为“教师检测错误被门放过”则不成立。

## 机会、风险和 selected 的重新解释

|旧 assigned 桶|旧对象/selected|转为 native 双方正确，对象/selected|保留其他 native 状态，对象/selected|
|---|---:|---:|---|
|T 正确/S 错误|9/7|8/6|T 正确/S 错误：1/1|
|S 正确/T 错误|10/3|10/3|0/0|
|双方正确|54/19|54/19|0/0|
|双方错误|7/3|5/3|双方错误：2/0|

最终 native 桶为：T 正确/S 错误 **1/80**，selected **1/1**；S 正确/T 错误 **0/80**；双方正确 **77/80**，selected **31/77**；双方错误 **2/80**，selected **0/2**。原 selected 共 32 个，31/32=96.875% 已是双方正确，1/32=3.125% 属于当前阈值的机会桶。上述占比是对象计数，不是 KD 损失、梯度或更新份额。

唯一剩余机会为 `130313.jpg::native_gt:3`（RGB global row 26）：S 无 dense/native 正确见证，T 原生见证 confidence 0.579590、IoU 0.895119，当前 selected rank 2。两个双方未正确对象为 `100439.jpg::native_gt:0`（row 17，无原粗候选、非 base）与 `060331.jpg::native_gt:1`（row 64，有 base，但 teacher gate=false）；均未 selected。这里的 native_gt 编号是原生已验证标签表的行号，不是原始 annotation 文本行号。

由此可说：这一批的当前选择器已经覆盖唯一的原生检测机会，主要选中双方已正确对象。不能由此认定“选择器无需改进”“双方正确对象没有蒸馏价值”或“风险为零”；固定 confidence/IoU 的命中不衡量定位精度余量、置信度校准和参数梯度冲突，更不代表后续训练不会造成负迁移。也不能把本次 1/80 与旧 200-dev 修复率直接比较。

## 一个后续有界诊断（建议，尚未执行）

固定现有历史自然流的**紧接第 2–4 批**（再 96 张图），沿用完全相同模型初始化、增强流、阈值、四定义、实际 selector 和无更新前向；逐批报告原生机会桶及其 selected 覆盖、双方正确占 selected 比例，同时保留这第 1 批。这样能检查“首批接近检测饱和、可纠错对象很少”是否只是这一批的特性。不得按难度或新结果挑图，不改门或阈值，不训练，不将四批结果升级为全训练集/AP结论；启动仍由 root 统一资源队列决定。
