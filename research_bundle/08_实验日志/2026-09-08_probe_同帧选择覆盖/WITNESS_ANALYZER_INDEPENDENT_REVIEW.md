**ACCEPTED — 仅接受这 32 张训练图、80 个增强 GT、原 selected=32 的同批描述性见证读出；没有发现需要修正的实际计算或表述问题。**

独立读取 `witness_analyzer.py`、实际 `witness_evidence_1315_final/probe` 和 `witness_analysis_final`。当前分析器与执行保存源码逐字节相同；旧 80 对象的身份、状态和 C 门逐字段 exact，重新运行已冻结 summarize 与正式 summary 除输入路径统计外完全相同。另用独立 Counter 重算四定义状态桶与 selected 数闭合。

逐图、逐 S/R/T 共 **96 个模型×图像**，从实际保存的原生 NMS 检出和 own-GT 像素框重新调用 pinned 原 `box_iou` 与 `match_predictions(use_scipy=False)`，直接捕获原函数匹配对。所有 GT/预测索引对、完整逐预测 TP 布尔位及对象 witness 绑定 exact；保存 IoU 与独立 CPU 重算最大绝对差 **0**。检出数 S/R/T=82/82/80，正确一对一匹配数=77/77/78。T 始终对应 IR own GT，没有把它改成 RGB 物理定位真值。

四种定义明确不同：旧 assigned 是 GT 辅助、粗阈值候选的空间一对一分配后判正确；FP32 dense-any 是任意原代理解码 anchor 对 own GT 满足 confidence≥.25/IoU≥.5；native pre-NMS 使用原 head 解码及 confidence>.25；native post-NMS 才使用固定 NMS 和实际 native 一对一匹配。结果为 S/R：64→77→77→77，T：63→78→78→78。不能将旧 assigned 的错误桶称为模型完全不存在正确检出。

**本次计数转换没有被检测到的解码精度或 NMS 伪差异。** 原 FP32 与 native 的最大框坐标差确实不为零：S/R 0.1261291504 输入像素，T 0.0975341797；raw 为 FP16、native 输出为 FP32。可是这 80 GT 上，dense-any 与 native pre-NMS 的逐对象布尔值和每对象正确 anchor 数全部一致，native pre/post 的逐对象正确布尔也全部一致。因此本次 13/13/15 个旧错误→dense正确来自 assigned 定义差异，不能归因为此次精度或 NMS 改变了正确性。这个结论仅限本批次与固定阈值，不能推广为解码精度或 NMS 永远没有影响。

原生匹配定义下，T 正确/S 错误 **1 对象且原 selected=1**；S 正确/T 错误 **0**；双方正确 **77 对象、selected=31**；双方错误 **2、selected=0**。原旧 assigned 定义相应“机会/反向风险”桶是 9/10 对象、selected=7/3，二者不能混用。分析器保留全部四定义和 ID，正文明确这些是固定定义下的状态模式：机会不等于已修复或可获得 KD 收益，风险不等于发生负迁移，零反向风险也不表示负迁移被消除。

计算边界：本次没有保存全量 raw 用于再次穷举所有 dense anchors，故 dense false/总数沿用已接受原 producer 的实际输出及门闭合；独立重算覆盖已保存的完整 NMS 检出、全部 GT 身份匹配和所有汇总，而非重新推理。未计算 AP、学习效果、全 dev 覆盖或跨数据集结论。没有 GPU、训练、权重加载或新 hash。

可复核小产物：`witness_review/review_actual_witness_cpu.py` 与 `witness_review/ACTUAL_WITNESS_REVIEW_RECEIPT.json`。原始 probe、旧定义结果和失败尝试均保留。
