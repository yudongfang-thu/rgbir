# probe_rgbir.py 独立只读审查

> 2026-09-06。结论：中心化 CKA 公式、固定抽样、显式 donor null、独立归一化热图注释总体符合描述性 probe 定位；发现一个会改变核心共同目标命中矩阵的实质问题：检测匹配最大化 IoU 总和后再阈值化，不保证最大一对一命中数。建议 full 前修正并保存完整预测以便 CPU 重算。其余边界主要涉及前景粗池化、类映射身份、分布解释与证据可追溯性。

审查对象：本目录 `probe_rgbir.py`，读取时 SHA256 `305d98d6546545bbe8016d753da79bd2340aa3e1239aeaab93914d6144b25b85`；同时读取 `probe_config.json`、`run_probe.sh` 与冻结 README。下文行号对应此版本。未修改脚本、未运行 GPU，执行了一次只用 NumPy 的两目标/两预测框 CPU 反例计算。此审查不等价于实际服务器执行成功或数据身份完整审计。

## 1. 实质数值问题：GT 命中分配

**P1，L143–152，`gt_errors`。** 当前先对所有同类 IoU 分数做 `linear_sum_assignment(-score)`，再把被分到的 IoU≥0.5 记 hit。最大总 IoU 分配与最大 IoU≥0.5 一对一命中分配并不相同。

已用真实、合法归一化 HBB 构造反例（同类、预测置信度均≥0.25）：

```text
GT 1 = [0.10, 0.10, 0.60, 0.90]
GT 2 = [0.40, 0.10, 0.90, 0.90]
P  1 = [0.10, 0.10, 0.70, 0.90]
P  2 = [0.00, 0.10, 0.39, 0.90]

IoU = [[0.8333333333, 0.4833333333],
       [0.3750000000, 0.0000000000]]

对角分配：sum IoU = 0.8333333333，1 hit
交叉分配：sum IoU = 0.8583333333，0 hit ← 当前 Hungarian 选择
```

两预测之间 IoU 低于脚本 NMS=0.7，因此不能依赖 NMS 保证这个问题不出现。它会将本可命中的对象记为漏检，进而改变 both/RGB-only/IR-only/neither 的数量和其机制解释。

**建议修复定义：** 对描述性 fixed-threshold hit，先限定同类且 IoU≥0.5 的有效边，优先最大化有效匹配数，再用 IoU 打破同命中数方案的平局；例如每个有效边赋予大于全矩阵 IoU 总和上界的 cardinality bonus。若选择 confidence-order greedy 以贴近某一正式 evaluator，也须明确记录该定义。不要把两种不同定义的矩阵混用为“标准检测命中”。

**定位误差应单独保留。** 当前 `best` 实际是某次一对一分配的 IoU，不是每个 GT 的 best-IoU。更适合同时存：是否固定阈值命中、命中预测索引/置信度、分配 IoU，以及单独 class-aware/class-agnostic 最大候选 IoU。后者允许重复引用候选，因此只能作诊断，不能作 TP 数。这样才能区分漏检、置信度不足、错类和定位不够准。

`wrong_class_overlap` 当前表示“存在错误类别预测与 GT 相交≥0.5”，不是互斥的错类错误：一个 GT 可以已有正确命中，同时还有错误类别重复预测。图表/报告应沿用这个准确名称，不要直接汇总成“错分类率”。

## 2. full 前值得补上的可追溯性

**P2，L260–262：类数相同不保证类顺序相同。** 代码仅检查 `len(names)==nc`，未验证模型 name/index 与处理标签类表一致，也没有将 `names` 落盘。建议记录两模型实际 `net.names`，与各自训练/评估 YAML 的类别映射逐项核对。当前路径检查很有价值，但同样类数的 VEDAI 或 DroneVehicle 变体可能使用不同排列；如已有外部 inventory 独立确认，可在 receipt 引用该证据。

**P2，L270–295：保存所有预测后再汇总。** 当前 `pred` 只在内存，输出仅保留 matched 对象的已分配 IoU。这样将来修正匹配规则、做置信度扫描、class-agnostic 分解或检查 FP/重复，必须重新用 GPU 推理。建议把 per-image normalized `xyxy/conf/class` 与 GT、模型映射一起保存为小 JSON/NPZ；即使本轮仅用 conf≥0.25，也保留实际推理得到的 conf≥0.05 预测。已选的 200 对很小，不需要存原始大特征。

**P2，L157–168、L296：记录实际身份而非默认值。** 建议在 manifest 保存捕获的三层原始 `[C,H,W]`、stride 与模型实际 seed；`single_checkpoint_seed:42` 当前硬编码，三个配置文件名虽都写 seed42，但最好与 args `seed` 一致性断言后写入。保存原始 feature shape 也能证明 P3/P4/P5 名称确实对应预期层，而不是仅凭 detection head 的最后三个输入顺序推断。

这些建议不要求增加模型训练或扩大 probe 样本。

## 3. CKA：公式正确，解释须限定粗空间结构

**通过的项：** L133–141 沿 spatial samples 去均值，分子 `||XcᵀYc||F²`、分母 `||XcᵀXc||F||YcᵀYc||F` 正确；中心化方差近零与少于 8 token 记 N/A；不同模型通道数允许不同。它修复了旧 P2 未中心化的主要数值问题。

**需要在报告显式写出的限制：**

- 三层全部 `adaptive_avg_pool2d→20×20`（L168），因此测量的是三层均经过粗池化的空间表征。P3 的 80×80 细节被平均为 20×20，P5 通常原生就是 20×20。不能将 CKA 大小排序解释成各层原生细节/频率可迁移性排序。
- FG mask 以 20×20 cell 的中心是否落 GT 内定义（L122–130），并非 cell 内像素前景占比。小车即使在 P3 上占数个原始 cell，也可能在池化网格中完全没有 FG token。VEDAI 小目标 FG 有效图数可能很低；这应报告为测量分辨率限制，不能说“该数据集没有前景共享信息”。所有图 N/A 时，不要对空 boxplot 的视觉结果作结论。
- 即使中心点落在框内，池化 cell 仍混入四周背景；`union FG` 又包括两侧不重合的框区域。所以 FG CKA 是“前景附近粗网格的结构相似”，并非纯粹的像素级共同物体表征。
- `.half()` 保存再转 float 会先量化小激活差异。大体相关通常仍可用，但精度小于该量化误差的 delta 不宜解释；低方差 N/A 阈值也仅是数值保护，不保证每个剩余值统计稳定。

当前协议可继续作为粗空间 probe。若因小目标 FG 有效图过少而另做 native-resolution/对象 ROI probe，应登记新版本/补充分析，不能无记录地换口径追求漂亮相关。

## 4. shuffle/null 定义与同一统计量比较

L264–268 确实产生没有固定点的随机 derangement；五次重复可出现重复 donor，这不破坏随机抽样定义，但不是每目标强制五个不同 donor。**两对 canary 只有唯一 derangement，所以五次 null 全是同一个 donor**，不能从 canary 的五次均值谈稳健性；脚本 `canary` 标志已足以区分用途。

L281 与 L282 的 mask 存在条件差别：paired 使用当前图 `mask`，donor 使用 `mask & donor_valid`。如果数据集各图的 aspect ratio/letterbox valid mask 相同，则两者一致；若不同，则 token 数、位置分布会变化，`paired−donor` 混入采样区域差异。建议从实际 shape 断言同一数据集 valid masks 一致；如不一致，应每个 donor 使用共同 mask 计算其 paired 与 null 两个值，再作差，并保存 donor 有效 token 数。

FG donor 使用的是 recipient 的 union FG 位置，不要求 donor 自身也为前景。这是合法的“正确图像配对相对随机图像”null，但它混合了目标存在/位置/类别/场景差异；不能把正 delta 专门归给“同一实例细粒度语义”。后续需要类/尺度匹配 donor 时应另立该 null，不用本轮 global donor 代替。

## 5. 标签与几何：不是硬错误，但不能越界解释

L77–91 标签匹配先最大化同类总 IoU，再保留 IoU≥0.1，是明确的描述性关联规则；结果图也标注 `matched >=0.1`。它不是所有真实共同目标的完整对应，尤其严重偏移/密集同类对象会漏配或错配；报告应同步给 matched 数/两侧总 GT 数。筛后的 IoU 分布天然排除了<0.1，不能代表所有对象的配准质量。

`center_shift_over_rgb_sqrt_area` 使用归一化 x/y 后的欧氏长度与面积（L84–89）。对非正方形图像，它不等价于像素欧氏距离除以目标像素面积平方根；应直接称“归一化图像坐标中的位移/归一化目标尺度”，不转换成像素偏差叙述。当前同数据集统一尺寸下可作相对描述，跨不同宽高比数据集直接比较要谨慎。

`gray_edge` 先 resize 到 320×320（L94），保留的是归一化方形坐标代理；shape 不同的横纵缩放比例不同。图像 phase shift 单位应沿用“归一化坐标”，不把单一二维平移估计说成已经测出真实配准误差。窗口函数和低响应警告是合理的。共享白边/letterbox 在 image_metrics 中没有剔除原始图白边；对 DroneVehicle 要依赖当前 hbb_v1 已去边的外部 receipt 核实。

`masks` 的 square letterbox 计算与 `predict(...rect=False,imgsz=640)` 方向一致，未发现明显宽高/xy 互换；但需 canary 从预测输入实际尺寸与捕获层 shape 验证。两侧 shape 不同且视场不同，即使 normalized label 坐标数值相同也不保证同一物理射线。

## 6. 图解释与是否过度

图上已经明确独立归一化、annotation correspondence 非 pixel truth、CKA 非 KD utility、低 phase response 位移歧义，这是合适的克制表述。固定亮度分位选择没有按方法胜负挑样本。

热图 L169 实际是先计算通道向量 **L2 norm** 再池化，严格说是 activation magnitude，而不是平方能量；命名为“通道激活强度（L2 norm）”最准确。其视觉峰值经过 20×20 池化再插值以及 2%/98% 分位裁剪，不能拿它测几像素框边界或称红亮点是预测因果贡献。

当前图不展示 GT 对应连线或跨模态边缘叠图，只靠并排图与柱状图不宜下“配准很好/很差”定论；可以依据已有标签/phase/edge 统计作多证据描述，并把图定位为实例说明。不同数据集图统一写 `IR/NIR` 不会误称 thermal，但 VEDAI 面板单独标 NIR 更清晰。

## 7. 本轮审查建议的优先级

1. 修复 fixed-threshold hit 分配；在 canary 确认输出 before/after 与保存全部预测。
2. manifest 写实际类映射、feature shapes、seed；如外部 inventory 已核实则引用其 hash/路径。
3. full 结束后核对 FG 有效图数/每图 token 和 valid mask 是否一致；按限定口径解释粗池化 CKA。
4. 保留 donor 规则、标签阈值、图像几何代理与热图规范；不扩展到新方法或新模型。

完成上述事项后，本 probe 可支持“这些 baseline 在这些开发样本上的对应结构与错误互补”的描述性结论。仍不能以单 checkpoint 热图/相关性或固定阈值命中矩阵宣称方法增益、学生信息上限或所有负迁移的根因。

## 8. v2 修订后只读复核（2026-09-06，追加保留初审）

复核脚本 SHA256：`df41641e242df727a29cf5e52380ce1ac6f8481920fae9e3c579fd0d584df281`。根任务报告 v2 canary 已完成；本段独立核验的是最新本地源码和已拉回的 full manifest/文件结构，不代替运行者的显存 canary 回执。

**原 P1 检测匹配问题已修复。** `gt_errors` 使用 `eligible=score>=0.5`，匹配权重 `eligible*(min(score.shape)+1+score)`。令最大匹配边数 `k=min(score.shape)`；一条有效边的 bonus 为 `k+1`，大于所有已选边 IoU 和的最大变化 `k`，因此先最大化有效边数，再最大化 IoU。非法边权重零且不记 hit，原反例不再导致 0 hit。`matched_labels` 对 IoU≥0.1 也使用同一定义，标签覆盖统计的版本变更应以 v2 为准。代码额外保存 `best_candidate_iou`；输出 `iou` 对命中者为 assigned IoU，未命中者为 best candidate IoU，二者混合时不是统一回归误差，both-hit 定位差才拥有一致的一对一口径。

**完整预测与类名改进已实施。** `prediction_records.json` 保存各图 GT、两模态预测的归一化 xyxy/conf/class；`sample_files.json` 保存图像及标签路径和 SHA256。`class_names.json` 保存两模型映射，并用 `model_names[0]!=model_names[1]` 拒绝两侧类名或顺序不一致。尚未在本脚本内逐项与 data YAML names 比较，但该新检查已经阻止本次指出的师生同 nc 错类序风险；与处理标签映射一致性仍可由外部数据 inventory 支持。

**原生特征分辨率变更已实施。** `predict_model` 直接保留 `z[0]`，能量图同样直接使用通道 L2 norm，已删除对 P3/P4/P5 的统一 20×20 平均池化。`masks_by_level` 从每层捕获特征的网格尺寸重新计算 valid/FG/BG，paired 和 donor 各使用本层 mask。对本次方形 640 输入的标准三层 head，预期网格为 80/40/20；v2 源码读取实际最后一维决定网格，而非硬编码这些数。初审关于“全部层共同 20×20”的具体局限不再适用于 v2 结果。剩余边界是 P5 小目标 token 稀少、中心点前景掩码与通道/空间维度随层变化，不能由 CKA 排名推出蒸馏潜力排名。

原生 feature 张量仍以 FP16 存 CPU，再 FP32 计算 CKA；原文中的量化边界保留。脚本也仍假设每层特征方形、同层师生网格相同、同一数据集跨图 grid 相同；这些对当前配置合理，但未做全面 shape 断言，不应直接复用到矩形预测或异构层级模型。

**复核结论：** 初审确定的命中分配硬错误已消除；类名顺序、原始预测保存、原生层分辨率三项修订可从源码确认。本轮 full 可按 v2 口径作描述性报告，避免把 v1 的 20×20 canary 数字与 v2 full 的层级统计混在一起。保持 W1/旧 P2/新 baseline probe 的身份与证据用途区分。
