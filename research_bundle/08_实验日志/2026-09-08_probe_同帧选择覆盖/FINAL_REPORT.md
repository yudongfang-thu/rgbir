# 同帧选择覆盖与原生检测见证：结果及证据修正

**本次发现的是诊断定义的偏差：被旧分析器分配到低置信 anchor，不等于检测器没有检出该对象。固定 LLVIP 首批 32 张训练图、80 个增强后 GT，旧定义的“教师正确、学生错误”有 9 个；原生 NMS 和一对一 GT 匹配后只有 1 个。不能继续用旧低置信计数直接论证置信度蒸馏空间大。**

两次有界推理均已完成，没有新增训练。首探针入口 38.802 秒、整个队列 166.045 秒；补充见证入口 27.335 秒、整个队列 78.268 秒。队列时间含启动、资源等待及清理；这些是单批诊断耗时，不是完整开发集推理或训练耗时。

## 目的和固定设置

本轮首先检验“教师能补的对象是否被现行 C 选择门挡住”。使用已完成 LLVIP 短训的首个自然训练批次，batch32、640、原双模态增强和随机流；S 加载短训前的成熟 RGB baseline，R 与这个初始 S 是同一 checkpoint，T 为既有 IR seed42。S 沿用 train 模式但冻结 BN 统计，T/R 为 eval，保留原 AMP。没有用短训后的 C0 权重，也没有使用 test。

原始 GT ID 取自 native 验证后的 dataset.labels 行，随真实增强及过滤布尔掩码传递，不是按近邻框猜测关联，也不声称是原 XML 文本行号。85 个 RGB/IR GT 各过滤 5 个，留下 80/80；双模态增强后 GT 坐标本批完全相同，这只是标签事实，不证明物理配准。首批图像路径、类别、框、batch 索引和双标签与原 sample_stream 一致；旧运行未保存像素张量，不能声称与历史像素逐项复验。

复用 trainer setup 是为了维持正式 loader 流；虽建立了 optimizer/EMA 对象，实际为 0 次 backward、0 次 optimizer/EMA update。学生参数和 buffers 前后不变，教师/参考无梯度。补充见证对前次 80 行原对象状态、门、质量和 selected 做 exact 比较，通过后才增加诊断字段。原生 Detect 解码可以刷新非 state_dict 的 shape/anchors/strides 缓存，不把这一缓存变化写成参数更新。

## 四种读出定义

|定义|实际含义|S 正确|R 正确|T 正确|
|---|---|---:|---:|---:|
|旧 assigned|GT 辅助空间一对一分配的那个 raw anchor 正确|64|64|63|
|FP32 dense-any|全部 raw anchor 中至少存在一个类别/IoU/置信度正确的候选|77|77|78|
|native pre-NMS|原生检测头解码后，NMS 前存在正确候选|77|77|78|
|native post-NMS|原生 NMS 后，原生一对一 GT 匹配正确|77|77|78|

旧状态/FP32 候选用 confidence≥0.25、类别正确、IoU≥0.5；原生 NMS 为 confidence>0.25、NMS IoU=0.7、max_det=300、multi_label=True、非 agnostic，随后原生 GT 匹配 IoU=0.5。LLVIP 单类下多标签选项不产生额外类别。T 始终对自身 IR GT 判断，S/R 对 RGB GT 判断。

原生匹配调用固定环境 BaseValidator.match_predictions，捕获实际 GT/pred 对并与 native TP 回验；没有用自行实现的常规 greedy 冒充原生匹配。补充见证复用同次 S/T/R raw 输出，没有额外主干或检测头卷积前向。旧 FP32 和原生解码的框存在至多约 0.126 像素差异，但本批三模态的 dense-any、native pre-NMS 和 post-NMS 正确 ID 均一致；本批定义变化不是该精度差或 NMS 造成的。

## 实际选择了哪些对象

原选择链保持：**80 matched → 79 base → 78 个 base 有教师正确候选 → 64 eligible → 32 selected**。分母仍为教师质量筛选前的 base 数；没有调整门槛、选择比例或损失。

|对象状态|旧 assigned 对象数 / selected|原生 post-NMS 对象数 / selected|
|---|---:|---:|
|教师正确、学生错误|9 / 7|1 / 1|
|学生正确、教师错误|10 / 3|0 / 0|
|双方正确|54 / 19|77 / 31|
|双方错误|7 / 3|2 / 0|

旧定义的 9 个机会都是学生 assigned 候选低置信，其中 8 个对象其实有原生正确检出。旧定义的学生错误共有 16 个，其中 13 个原生正确；教师旧错误 17 个，其中 15 个原生正确。选中集合没有变化，变化的是用于解释对象状态的诊断定义。

因此，本批不支持“主要教师机会在区域/R 粗候选门前被挡住”，也不能把旧风险桶的 3 个 selected 说成已经向学生传递错误目标。原生读出下唯一教师正确、学生错误的对象已入选；32 个 selected 中 31 个双方在 IoU0.5、confidence0.25 下都正确。

## 对方法判断的影响

1. **首先修正机会的测量。** “低置信 assigned anchor 多”与“真实漏检多”是不同命题。旧 LLVIP 200dev 的 146 个候选机会（90 低置信）、Drone 的 461 个（351 低置信），仍是原定义下可复算的对象表；其低置信部分不能直接改称原生漏检或可获得的 AP 增益。这里没有重新计算这两个 dev 样本，不能按本批比例修正其数字。
2. **当前成熟起点、这一训练批次的漏检修复空间很小。** 这给出了短训时选择大量已检出对象的一条具体线索，但一批无法代表训练总体，也不能证明它解释了 FT3 收益小。双方 IoU0.5 正确仍可能有置信度排序、IoU0.75+ 定位或泛化差异；没有据此判定分类、定位、特征蒸馏无效。
3. **已有完整 dev AP 差距与 TIDE 结果不因本次发现失效。** 它们来自独立低阈值预测评价，不使用这里的 assigned-anchor 状态当 AP。LLVIP 仍有实际 RGB/IR 模型差距，但差距不能等同于某种 KD 的可达收益。
4. **下一步先复用完整 dev 检测缓存，建立真实错误谱。** 同时核对两数据集，固定对象配对及匹配规则，报告真实检出、IoU0.5/0.75、排序/置信度和未检出对象。优先复用已有输出做 CPU 重分析。post-NMS 缓存不能恢复原 C 的全稠密窗口或实际 selected，不能再混用两者。

不会因这一单批结果修改原 C/L 门或启动新 E200；也不会重复已完成的九次短训。新的定向短训必须对应重新确认的实际错误对象，并先说明它在当前选择中的覆盖与所传递内容。

## 验证、资源和证据入口

映射 8 项、原选择导出 6 项 CPU 测试在 94 固定环境通过；补充原生检测见证 6 项 CPU 测试通过；两套分析器各通过 4 项已知真值，另有实际对象和独立源码审阅。分析器读出见下列入口，原始文件保留：

- [首批门链分析](analysis_final/README.md)、[首批真实对象复核](REAL_READOUT_REVIEW.md)、[候选差异具体见证](INTERPRETATION.md)。
- [四定义分析](witness_analysis_final/README.md)、[完整 ID 和混淆表](witness_analysis_final/summary.json)、[原生合同](witness_evidence_1315_final/probe/witness_contract.json)。
- [实际计数独立复核](WITNESS_REAL_READOUT_REVIEW.md)、[96 个模型×图像原生匹配独立复算](WITNESS_ANALYZER_INDEPENDENT_REVIEW.md)。
- [首探针完成回执](evidence_1255_final/probe/completion_receipt.json)、[补充见证完成回执](witness_evidence_1315_final/probe/completion_receipt.json)。
- [补充见证固定计划](WITNESS_FOLLOWUP_PLAN.md)及独立不可变 `release/`、`witness_release/` 源码。

两个队列均用既有 global lease，在 GPU4 执行并正常退出。补充见证监控采样的整卡最小空闲 14,375 MiB、项目总 RSS 最大 168,797 MiB，监控无错误；这两个是采样记录，不冒充连续精确峰值。完整原资源回执在本地/94 保留，公开目录提供安全摘录。未创建第二套资源池、未改停旧训练、未下载权重或计算新文件 hash。

完整 dev 缓存的后续 CPU 核对已另建[固定协议与执行条目](../2026-09-08_probe_开发集真实检出机会/README.md)；本报告不混入其未完成读数。两个数据集的缓存可用性与缺口见[输入审计](DEV_RECHECK_INPUT_AUDIT.md)。

94 原始产物根目录：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_selection_coverage_20260908/`，首探针 `attempt1/`，补充见证 `witness_attempt1/`。本地目录为 `E:/SHARE/光sar/08_实验日志/2026-09-08_probe_同帧选择覆盖/`。本批为训练图、成熟 S、单次读取、固定置信度，不是 AP 评估、泛化实验、负迁移观测或最终方法增益证据。
