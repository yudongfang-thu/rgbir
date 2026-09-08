# LLVIP 训练侧定位机会与原 C 覆盖

**首32图 CPU 读出已完成：双方 IoU≥.5 的77个对象中，11个仅T达到.75，反向1个；前者原C selected覆盖8/11。这是训练侧定位候选及C覆盖，不是L覆盖或蒸馏收益。** 没有重复首批前向或启动训练。前置协议为[上一阶段下一项有界工作](../2026-09-08_probe_开发集真实检出机会/NEXT_BOUNDED_ACTIONS.md)，本条保存实际实现、输出及验收。

输入固定为 `../2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe/` 的 witness_frames、witness_objects、GT身份和完成回执。原80对象/32图，S/R/T、score>0.25、原native NMS参数不变；CPU独立按IoU0.5/0.75重匹配，0.5必须与已有见证GT/pred绑定闭合。

按稳定GT ID统计双方粗检出而仅T精确、反向、T-only等状态，交叉原C的base/eligible/selected，报告对象与图像数、逐桶分母。不得把C选择覆盖冒称实际L1/L2覆盖；若追加L门检查，必须使用原实现及其真实anchor索引，另行记录接口。

若需要扩展，范围最多原自然流前4batch共128次图像出现，新增第2–4批前向需保持原流并通过唯一lease；首批缓存复用，不重算/重复计数。当前没有新增GPU执行结果。旧L1物理几何合同不因本条被解除。

任何结果仅描述固定训练图和成熟S，不能与完整dev计数拼接成过拟合结论或KD增益。源码、原值、独立验收及服务器镜像在本条续记。

## 实际设置与闭合

输入是上一 probe 原32图、增强后80个RGB GT和80个IR GT（全部已配对，1张空图）。S/R使用RGB own GT，T使用IR own GT；稳定ID沿用原生已验证标签表行号。score > .25、native NMS IoU .7、max_det 300、multi_label=true、agnostic=false及原预测顺序保持不变。读取既有 post-NMS 框，不重新运行 NMS。

[native_cached_match.py](native_cached_match.py) 与开发集条目已接受helper逐字节一致，执行保存的原生 `box_iou`、`DetectionValidator._process_batch`、`BaseValidator.match_predictions` 函数体，捕获原函数真实匹配pairs并检查TP/一对一关系。分别调用 IoU .5 与 .75；没有把 .5 的配对结果直接筛成 .75。本批两阈值实际没有发生GT重新绑定预测，也没有 .75正确但.5错误对象；小真值仍覆盖了“提高IoU后必须换预测”的情况。

IoU .5 对原见证闭合：240个逐GT正确标志、232个正匹配GT/prediction ID以及244个逐预测TP全部通过；正匹配的anchor、confidence和box exact，CPU IoU数值差要求小于1e-6。[CPU_attempt1.json](CPU_attempt1.json) 记录4/4真值PASS、CUDA未初始化。真实输出与运行源码副本在 [output_attempt1](output_attempt1/)，本次没有加载权重或使用GPU，没有新forward、AP估计、文件hash或实际L选择器调用。

## 原始结果

|模型|IoU≥.5 正确对象/80|IoU≥.75 正确对象/80|
|---|---:|---:|
|S|77|61|
|R|77|61|
|T|78|72|

以下为互斥对象桶。`对象/图`中的图是该桶涉及的不同frame数，各桶图数不能相加。C三列保持此前真实selector的base/eligible/selected布尔值，没有按本次IoU门重新选择。

|对象桶|对象/图|C base 对象/图|C eligible 对象/图|C selected 对象/图|selected/本桶|
|---|---:|---:|---:|---:|---:|
|双方≥.5，双方≥.75|60/29|60/29|48/28|20/15|33.33%|
|双方≥.5，仅T≥.75|11/8|11/8|10/7|8/6|72.73%|
|双方≥.5，仅S≥.75|1/1|1/1|0/0|0/0|0%|
|双方≥.5，双方未到.75|5/5|5/5|5/5|3/3|60%|
|仅T≥.5|1/1|1/1|1/1|1/1|100%|
|仅S≥.5|0/0|0/0|0/0|0/0|NA|
|双方未到.5|2/2|1/1|0/0|0/0|0%|
|全部|80/31|79/30|64/29|32/20|40%|

双方粗检出的定位优势桶为11/77=14.29%，反向1/77=1.30%；这里的优势仅指本次固定阈值下的own-GT检测状态。11个对象全部进入原C base，1个因原quality_q=0未eligible；余10个eligible中2个排名34/38，未进入本批前32个selected。全部稳定ID及三门保留ID见 [summary.json](output_attempt1/summary.json)，逐对象框/匹配与原C门见 [objects.jsonl](output_attempt1/objects.jsonl)。

原32个selected中，8个属于“双方≥.5，仅T≥.75”，占25%；20个双方≥.75，3个双方粗检出但均未到.75，1个仅T≥.5。上述都是对象份额，不是损失、梯度、学习权重或有效更新份额。

## 解释与下一步界限

先前 IoU .5 下仅1个T正确/S错误对象，不能推出首批没有定位精化候选；本次在相同缓存中发现11个双方粗检出但仅T达到.75的对象。它们与原C选择确有交集，尚未证明现有L1/L2的anchor、几何、支持域或质量门会纳入这些对象，也未证明学习后能改善定位。两侧各自GT的正确性及标签配对不认证旧L1物理几何合同。

本批可作为后续检查具体定位学习位置/目标的证据。若root需要判断这8张图的定位候选是否只是首批特例，下一步仍限于已冻结原流第2–4批、原阈值、无更新见证；本次没有替root启动扩展，不能以单批11:1直接推广整个训练集或对齐完整dev数量。成熟模型、训练图、增强态、单seed和阈值指标的边界全部保留。

## 复跑入口

在本目录运行以下入口，必须使用新输出目录，已有attempt禁止覆盖。`NATIVE_SOURCES` 为本地旧LLVIP接受评价的 `2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/remote_completed_attempt2/N42_full_attempt1/sources`，也可传包含同一原生三文件的实际镜像目录。

```text
D:/Anaconda/envs/KGJ_proj/python.exe test_training_localization_cpu.py --native-source-dir NATIVE_SOURCES --output CPU_new.json
D:/Anaconda/envs/KGJ_proj/python.exe analyze_training_localization.py --input ../2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe --native-source-dir NATIVE_SOURCES --output output_new
```

本次方法与真实结果已冻结交独立核验；验收产物由审阅者写入 `independent_review/`，不改本次原输出。服务器镜像由root统一处理，本文件不预称已同步。

## 追加：11对象的 native-box 门槛代理

按root在核心完成后的独立CPU请求，从相同保存匹配读出原阈值 `.70` 与领先 `.05`，没有改门或筛选范围：[NATIVE_BOX_PROXY_11.json](NATIVE_BOX_PROXY_11.json) 保留11对象两侧IoU、预测ID、anchor、confidence、框和原C selected；[read_native_box_proxy.py](read_native_box_proxy.py) 可从核心逐对象输出重建该小表。

11个对象中，S原生匹配IoU < .70有4个，T相对S的own-GT IoU领先 > .05有10个，二者同时满足4个；这4个均在原C selected中。其余7个S已达到.70，包括1个T领先不超过.05。这仅说明：不能从“仅T达到.75”直接推断全部11对象满足一种“.70且领先.05”的条件。

这里使用的是各自原生post-NMS匹配框，**不是现有L2的confidence-first参考anchor、独立教师候选或映射后的目标框**，也没有检查其DFL支持、有效尺度、唯一owner等真实门。因此这4个绝不能写成“实际L2可选4个”或“L2漏选多少”；与实际L2旧记录的连接必须先通过同一流、同一对象与学习anchor身份核验。根据root最新决定，本批已证明训练侧存在定位差异，不自动扩展第2–4批GPU前向。
