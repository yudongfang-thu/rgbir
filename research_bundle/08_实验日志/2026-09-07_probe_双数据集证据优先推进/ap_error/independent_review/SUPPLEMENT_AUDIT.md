# 类别 oracle 与 baseline 对象桥接独立审阅

**PASS，限事后描述性补充：逐类Cls oracle的AP平均与主表闭合；全部3084个raw对象的状态、修复/损伤及共同eligible比较独立重建一致。两份数据不能互相替代为同一完整AP匹配。**

审阅者 `/root/loc_stress`，2026-09-07。只写本目录，未启动TIDE大规模评估、GPU、SSH、训练或hash计算。原补充源码、旧失败attempt与结果未修改。复算脚本 [recompute_supplements.py](recompute_supplements.py)，实际CPU回执 [supplement_recompute_receipt.json](supplement_recompute_receipt.json)，执行约2.6秒。

## class_oracle_v1

读取`class_oracle_supplement.py`及执行副本，字节相等。源码调用官方`fix_errors(isinstance(ClassError))`，保持原GT分母、同一class ID和原score；之后用官方macro Cls结果交叉检查。其class差值允许逐类正负，宏main值遵循官方最终非负截断，不能先对每类截断再平均。

六端点×AP50/AP75的每一份均核验：

- 五类GT总和22462，类别ID与原主表一致；每类原AP与主表保存的独立AP结果一致。
- `Cls_oracle_AP−AP=Cls_oracle_dAP`；五类原AP均值与原macro AP一致，五类oracle AP均值减原macro AP与主表Cls dAP闭合（容差1e-9pp）。所有本次macro差值为正，不触发非负截断歧义。
- 全部20行按臂/类/阈值的均值、样本SD和C0−N三个seed差值，与README格式化表逐项一致。
- Cls混淆矩阵为**预测类别行、关联GT类别列**，不是每个GT只计一次的分类混淆矩阵。每行总数在all/score≥.25/score≥.50下均与主表对应预测类别的Cls错误数一致，矩阵总和与main Cls计数闭合，对角线全为0。
- 另对N_s0完整raw框/score，以独立NumPy score-greedy匹配和官方错误判定先后顺序重新构造Cls矩阵，AP50/AP75及三个score区间全部exact；这一步未调用TIDE引擎。

N臂AP50下freight car/truck/van占五类Cls dAP总和的三seed均值比为 **94.2624011%**，对应报告约94.3%。它表示该macro oracle算术贡献集中在三类；不是三类GT对象修复率，也不意味着三类都缺少IR知识或都适合相同分类蒸馏。

逐类oracle受官方“修正最佳错误预测或抑制额外预测”语义影响，不能将所有Cls dAP理解成把一个GT的错误类别改对，也不能与special FP等独立oracle相加。这一点源码与README均已交代。

## baseline_class_bridge_v2

读取`BRIDGE_PROTOCOL.md`、`bridge_baseline_classes.py`及执行副本、原`analyze_baseline_probe.py`与字节相等副本。独立实现raw assigned候选状态判定，**没有调用原object_view或opportunity_summary来验证它们自身**；保持conf≥.25、对RGB GT IoU≥.50、标签pair≥.50和low-confidence优先状态顺序。

实际读取原raw JSONL全部3084个非背景val对象，200个dev图中194图有GT、6图仅有背景窗口，名单与summary exact。v1的“所有200图均有非背景GT”断言失败已保留，v2纠正的是库存检查，不是删选GT或改变阈值。

对所有3084行、N42/T42/N0三模型，独立从assigned框再计算对RGB GT的IoU：与缓存scalar的最大差分别 **3.1775e-7/2.5215e-7/2.5832e-7**，差异为原浮点舍入量级，.50阈值判定不一致数为0。Teacher判断使用的是RGB GT IoU，没有误用可能基于IR标签的`assigned.iou`。每行完整bridge对象ID、类别、pair、N/T/N0状态、repair和harm布尔值与保存JSONL完全一致。

|类别|GT|N错误|共同eligible|共同修复|T-only|N0-only|
|---|---:|---:|---:|---:|---:|---:|
|car|2662|495|2540|137|232|25|
|freight car|81|30|75|7|8|0|
|truck|176|100|175|18|28|5|
|bus|66|18|66|8|5|0|
|van|99|55|90|5|13|4|
|总计|3084|698|2946|175|286|34|

全部五类GT/N正确/N错误、每个互斥state、T/N0按state修复、T/N0总体修复和潜在损伤、共同eligible中的所有字段均加和闭合。共同eligible上 **T修复461=175+286，N0修复209=175+34**；N0对全部RGB GT的修复221使用更大eligible集合，不能直接将221与461解释成同分母差值。T总体潜在损伤193、N0总体潜在损伤192也已回算。

van 55个N错误由 **28 class_only、26 low_confidence、1 localization_only** 构成，T在这三类分别修复9/8/1个；该结论为固定GT关联候选的对象状态，未包含全部假阳性分数排序。`no_candidate`只表示原粗候选阈值.05后无关联候选，不表示网络没有输出。

## 接受范围与未核验项

接受两份补充的当前描述性数值与已声明口径。class-oracle使用完整1469图的native rect544×672 post-NMS输出；bridge使用200图的640方形letterbox、GT辅助assigned候选，并且只比较现有单seed模型组合。这些差别已披露，不能按图像ID重叠就冒充同一检测匹配或同一AP估计。

共同eligible内T-only正确只提供候选知识可用性线索，不是可达KD收益上界，不证明可迁移/可学性；teacher/native recipe差异和独立RGB自然互补仍在。此次没有重新推理或重新运行全部每类oracle，仅对其官方源码路径、全部真实产物算术以及完整raw N_s0 Cls矩阵独立复核。LLVIP新输入和L1准入不在本次接受范围。
