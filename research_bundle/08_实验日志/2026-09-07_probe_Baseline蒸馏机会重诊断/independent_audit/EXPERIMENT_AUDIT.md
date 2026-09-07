# Baseline 蒸馏机会：独立原始证据审计

**结论：现有数据足以支持“Drone 优先研究置信度/前景判别，保留条件定位，局部特征仍缺直接效用证据”的研究排序线索；不能把 IR-only 当作纯分类错误，不能把 CKA 当作特征蒸馏收益，也不能把不同抽样、不同推理策略的错误率直接比较。**

审计时间：2026-09-07。审计者：`/root/baseline_opportunity_audit`，独立子代理，未接收执行者预先结论；实际模型名未由运行工具单独暴露，不声称跨模型审计。审计性质：只读原始代码与既有本地 JSON/JSONL/NPZ，另写本目录报告及确定性重算；无 SSH、GPU、新推理、训练、源目录修改或 hash 计算。用户明确禁止 hash，因此采用输入路径、大小、mtime 和字段级一致性，未执行技能的 hash 建议。

整体 verdict：**warn**。既有结果可追溯且可重算，但现有记录对“哪种知识更有蒸馏效用”只提供机会线索。

## 1. 模型、样本与推理身份

两轮早期 baseline probe 都使用历史 `formal_native` seed42 的 YOLO11n、E200、`last.pt`。Drone 的 RGB/IR 数据分别为 hbb_v1/rgb 和 hbb_v1/infrared；LLVIP 为 grouped_v1/visible 和 grouped_v1/infrared。`input_manifest.json` 中训练 data 与各自声明一致；D1/D2 summary 中 teacher/reference 路径与旧 probe 一致。本地未读取远端权重或影像实际内容，故权重内容等价依赖既有回执和路径身份。

|记录|图像|RGB GT|IR GT|标签配对|推理/匹配|
|---|---:|---:|---:|---:|---|
|旧 Drone probe|200 val|3116|3276|3083，同类 IoU≥.1|640方形；post-NMS conf≥.05；命中分析 conf≥.25、IoU≥.5|
|旧 LLVIP probe|200 dev|672|672|672，全部标签字节相同|同上|
|D1 Drone train|2048/17990|31931|35407|30765，同类 IoU≥.5|固定640方形 letterbox；pre-NMS conf≥.05；按几何进行类别无关的一对一分配|
|D1 Drone val|200/1469|3084|3423|2946|同上|
|D1 LLVIP train|2048/9619|5592|5592|5592|同上|
|D1 LLVIP val|200/2406|643|643|643|同上|

旧 probe 是固定 seed20260906 均匀随机抽样；D1 是来源组比例分配后取均匀 frame rank。按**完整 RGB 路径**，两套 val/dev 只重合 Drone **30图460框**、LLVIP **13图35框**。按 RGB GT 本地索引联表后，类别不一致为0；映射到相同640输入画布，GT最大坐标差分别为 **0.00004306 / 0.00003436 输入像素**，符合浮点舍入。旧 probe 与 D1 train 完整路径交集均为0。

**联表必须使用完整 image 路径＋rgb_gt_local_index，不能只用 stem。** Drone train/val 都有相同短文件名；本次只按 stem 的初始检查会产生28个伪交集，最终重算已使用完整路径。它不是训练验证泄漏的证据。

代码依据：

- [旧 probe 匹配和推理](../../2026-09-06_probe_RGBIR数据特性与可迁移知识/probe_rgbir.py)：`matched_labels` IoU≥.1；`gt_errors` 第144行按同类匹配，`predict_model` 第159行，预测参数第169行。
- [D1 实际执行源码](../../2026-09-07_probe_TaskConditional机会诊断/drone_train/run_evidence/source_snapshot/trainer/01_diagnose_opportunities.py)：`spatial_assign`、第315行 `reference_state`、第326行 `d1_records`。
- [重算结果](inventory_recomputed.json) 包含48个声明输入及每组完整交集。

## 2. 对分类/置信度和定位机会的独立复核

D1 的错误状态是**顺序互斥标签**：无候选 → 类别错 → conf<.25 → IoU<.70 → well_localized。它不是相互独立的类别、置信度和框质量三个维度。例如类别错也可能定位差，low_confidence也可能IoU低；“localization_gap”排除了低置信度对象中的定位问题。pre-NMS几何最优匹配还不是部署检测器的NMS后错误分解。

|D1对象状态|Drone train|Drone val|LLVIP train|LLVIP val|
|---|---:|---:|---:|---:|
|class_error|581|133|0|0|
|low_confidence|3606|442|730|130|
|localization_gap|452|106|117|162|
|well_localized|27022|2321|4716|292|
|no_coarse_candidate|270|82|29|59|

为了避免把框差当作分类机会，追加了描述性条件计数：RGB候选 IoU≥.5、教师类别正确且 conf≥.25、教师对自身和 RGB GT 都 IoU≥.5。

|满足上述条件的对象|Drone train|Drone val|LLVIP train|LLVIP val|
|---|---:|---:|---:|---:|
|RGB类别错|248|37|0|0|
|RGB类别对但低置信度|2719|321|578|77|

由此可说 **Drone中置信度/前景判别机会比纯类别混淆机会更突出**；“分类分支”必须解释它主要迁移的是检测类别分数与前景证据，不应写成大量车辆类别需要纠正。LLVIP单类别无法检验多类别区分能力，只能辅助验证前景分数/定位。

这些数字是描述性候选计数，不是蒸馏收益估计；teacher置信度并未校准，多GT/anchor竞争仍可能影响分配。train是baseline已训练数据，其机会率不能替代held-out效果；LLVIP train与val状态差异尤其大。

## 3. 定位 D2 的真实覆盖及字段语义

|D2|base对象|selected对象|selected / 全RGB GT|selected 教师−参考 GT-DFL CE均值|
|---|---:|---:|---:|---:|
|Drone train|23511|626|1.9605%|−0.470229|
|Drone val|2177|112|3.6316%|−0.636935|
|LLVIP train|5492|211|3.7732%|−0.505197|
|LLVIP val|559|131|20.3733%|−0.861513|

D2文件每个base对象恰好一条已选代表anchor，独立计数中 row_count=unique_object_count。它不是保存全部8400个anchor。base之前已经经过GT匹配、pair_iou、支持/inside、unique owner和reference候选条件；被早期过滤的对象没有DFL诊断字段。无几何合同版本把几何当成可用占位，`geometry_verified=False`，不能把其 `geometry_count` 解释为已验证配准。

`reference_logit_kd_native_dot/cosine` 来源是 **DFL box logits** 的局部概率梯度，不是类别logits，不是共享骨干梯度，也不是当前训练学生梯度。`teacher_gt_dfl_ce/reference_gt_dfl_ce` 是相对 RGB GT 的温度1两bin软标签CE；仅保存CE/entropy/dot/cosine标量，未保存4×16原始logits。selected本身按教师框质量优势筛选，CE优势不能独立证明筛选机制或蒸馏优势。

依据：[实际定位loss](../../2026-09-07_probe_TaskConditional机会诊断/drone_train/run_evidence/source_snapshot/loss/01_localization_loss.py)，第317–333行计算DFL CE、entropy和局部梯度。最小可复用项是按 `(image,rgb_gt_local_index)` 合并D1/D2，观察类别/分数和定位优势是否重合；不能从这些文件恢复完整类别Bernoulli分布、梯度训练结果或base之外的DFL机会。

## 4. 局部特征证据及缺口

旧 probe 保存同一200图的P3/P4/P5检测头输入特征诊断。前景区域是RGB与IR GT框并集，不是逐对象ROI；CKA在空间token上中心化，以5个固定无自配donor作为背景比较。

|前景 paired−donor CKA均值|Drone|LLVIP|
|---|---:|---:|
|P3|0.582264，n=200|0.306179，n=200|
|P4|0.574736，n=198|0.427636，n=200|
|P5|0.279906，n=179|0.045497，n=147|

`energy_maps.npz` 每模态仅有 `(200,80,80)`、`(200,40,40)`、`(200,20,20)`，分别为通道L2范数。**未保存完整通道特征张量**；它可用于对象/边界能量集中度和空间对应诊断，不能计算新的逐ROI通道CKA、类别线性可分性、特征重建误差或共享参数梯度。P5因前景token<8而缺失较多，跨层均值排名还有采样差异。

现有数据证明的是配对图像前景有共享空间结构。它没有建立“IR特征补充RGB缺失信息”“哪些channel有类别判别力”“local feature优于类别/定位输出”或“P3/P4应当蒸馏”的因果结论。不能因为局部特征证据不完整就宣判该方向无效。

## 5. IndependentKD 中现成 N/C0 完整对象预测

统一根目录：`E:/SHARE/光sar/08_实验日志/2026-09-07_train_IndependentKD实施/legacy_diagnostics_snapshot_20260907_161459/`

六套文件：`{N,C0}_s{0,42,123}_attempt1/predictions/objects.jsonl.gz`，每套1469图、22462 RGB GT。六套完整image/GT/画布逐字段精确一致。本次仅核实其可复用身份与计数，未重算其mAP或重复接受全部历史桥接。

|端点|mAP50_95（0–1）|存储预测数|
|---|---:|---:|
|N0|0.5434618521|88236|
|N42|0.5451360845|86843|
|N123|0.5440743649|86322|
|C0_0|0.5477218640|112513|
|C0_42|0.5465816217|105783|
|C0_123|0.5463684721|106929|

字段：`image, canvas_shape, original_shape, gt_boxes, gt_classes, pred_boxes, pred_classes, pred_confidence`。真实画布全部 **544×672**；合同 `imgsz=640, rect=True, conf=.001, iou=.7, max_det=300, batch=32, half=False`。这不是旧probe的640方形预测。N是OEv1同trainer的weight0 E200，**不是**旧formal_native RGB checkpoint。没有同一次矩形完整IR输出或中间特征；可用于N/C0误差转移和三seed稳健性，不可直接拿来拼旧IR推断严格蒸馏机会。

应使用 [rect_v2 对象分析器](../../2026-09-07_train_IndependentKD实施/object_error_analyzer_rect_v2/object_error_analysis.py) 的实际画布处理；根目录旧 `object_error_analysis.py` 仍有 max(shape)==640 前提，不适用于这批672宽数据。

## 6. 最小补充分析建议

1. **先CPU统一旧200图**：从 `dronevehicle_full/prediction_records.json`、`llvip_full/prediction_records.json` 重做类别无关GT↔预测匹配；按完整image+GT索引，联合报告预测类别是否正确、分数、RGB GT坐标IoU。教师框必须对RGB GT再评分，不能直接相减模态各自GT IoU。输出类别/分数-only、定位-only、二者重合、二者均无优势及未可靠配对。
2. **按相同对象连接现有特征信息**：先用旧200图已有energy maps做ROI/边界能量统计，使用同图局部偏移和donor比较，按错误桶分层。图像级CKA关联只能标为图像层面的相关性；不要把它贴到每个对象上伪装为对象级特征效用。需要通道证据时，最小缺口是同名单、同模型、同letterbox的一次中间特征与完整score向量导出，不能由现有NPZ重建。
3. **D1/D2保留为单独验证层**：按base限制和顺序标签显式交叉统计，检查train/val与来源组、尺度、亮度proxy稳定性。旧probe与D1只有30/13图交集，不可拼成完整同样本矩阵。阈值敏感性属于诊断，不回改既有冻结方法门。
4. **不把新N/C0成绩当作baseline机会证明**：它们说明一个已执行类别分支的实际后果；机会排序仍需上述baseline联合诊断。现阶段没有定位/局部特征同预算长训负结果。

## 产物与检查边界

- `recompute_inventory.py`：不导入或执行源实验脚本、不触发hash/GPU/network的独立重算。
- `inventory_recomputed.json`：48个声明输入、样本交集、GT坐标回验、D1/D2计数、六端点身份及预测数。
- `EXPERIMENT_AUDIT.json`：结构化checks/claims。
- 本目录是独立审计子产物；主实验README和总索引由根代理维护，未触碰其他实验目录。

未审核：远端权重内容、训练数据实际像素、完整mAP重算、新C1效果、实际训练梯度、真实几何配准。现有真实GT与代理指标须分别解释，未访问test。
