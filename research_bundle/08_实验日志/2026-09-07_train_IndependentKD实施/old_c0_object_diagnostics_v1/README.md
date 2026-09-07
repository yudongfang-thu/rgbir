# 旧 C0−N 三 seed 补充对象诊断

**旧 C0 的三seed对象诊断已完成：固定操作点净正确对象变化为+4/−3/+101，背景误检增加+3/+5/+46；小幅AP收益伴随可观察的对象损伤，不能声称负迁移已消除。** 本目录仅是旧OEv1/C0相对N的开发集补充诊断，不是新C1结果，不签发总AP桥接接受。首轮画布读取失败保留；独立接受的rect v2在新attempt2成功。

已核对本地 `legacy_diagnostics_snapshot_20260907_160828` 中实际N42/C042补评。原对象分析器独立接受回执仍有效，metric/config/eval receipt保留原weight0/paired身份，E200/640/nc5符合读取接口。本地CLI读取失败原因是metric中的objects/contract仍为94绝对路径，不是标签或实验身份冲突。依根授权，六份齐备后在94原路径执行同字节接受的分析器，原metric/contract不改写。

固定规则仍为confidence≥0.25、同类一对一IoU≥0.50、背景对任意GT最大IoU<0.10。修复分母是N错误GT，损伤分母是N正确GT；逐seed报告计数及比例，三seed汇总用样本SD。类别、640画布尺度使用全1469图。

可复用亮度元数据来自此前冻结的Drone dev200 `2026-09-07_probe_TaskConditional机会诊断/drone_val/images.jsonl`，重用原low<78.283/high标签，不重新读图/调阈值。它只覆盖完整dev中的200图，另1269图保持UNKNOWN；此前来源字段全部unavailable:val，规范表示为UNKNOWN。亮度结果只描述该固定子样本，不充当全dev或真实昼夜结论。

历史第一版计划输出为94 `.../rgbir_independent_kd_v2_20260907/legacy_object_analysis_v1/`，使用原v1接受代码；该attempt因下述画布限制失败并保留。当前成功输出与接受版本见下一节，未覆盖旧目录。

## 完成的设置与证据

实际成功目录为94 `.../rgbir_independent_kd_v2_20260907/legacy_object_analysis_attempt2/`，screen `ikdv2_oldC0diag_attempt2`。使用独立接受的`legacy_object_analyzer_rect_v2`，三seed依次0/42/123；只读六个真实last/EMA补评及其bound objects、metric、config、contract。原始方法身份保留weight0/paired，报告层称N/C0，不重标成C1。六次补评的五汇总指标均与旧端点精确相等；其逐类及对象记录是补充的新产物。

完整开发集1469图、22462个RGB GT，三seed图像、GT原数组及顺序、实际544×672画布完全一致。置信度≥.25、同类一对一IoU≥.50；背景误检要求对任意GT最大IoU<.10。此匹配不是AP评估匹配，不能用原生best-F1汇总recall替代。

三次CPU耗时7.81/7.64/7.59秒，采样进程树RSS峰值152.03/149.38/149.52MiB（含runner及子进程，0.25秒采样），CUDA_VISIBLE_DEVICES为空、无模型推理。下载的三份analysis_source与本地接受rect v2三文件逐字节相同。见[实际完成回执](attempt2_outputs/completion.json)、[seed0结果](attempt2_outputs/seed0/pair_summary.json)、[seed42结果](attempt2_outputs/seed42/pair_summary.json)、[seed123结果](attempt2_outputs/seed123/pair_summary.json)。

汇总脚本[tabulate_old_c0_diagnostics.py](../tabulate_old_c0_diagnostics.py)复用已接受`analyze_independent.py`的单位转换与mean/样本SD函数，保留每seed原值。它不构造已接受的正式端点，不签AP bridge，不自动扩展实验。完整数字见[summary_tables.json](tables_attempt2/summary_tables.json)。

## 修复与损伤

|seed|N正确 / N错误GT|修复 / N错误GT|损伤 / N正确GT|净正确变化|正确对象比例变化|
|---|---:|---:|---:|---:|---:|
|0|19698 / 2764|473 / 2764 = 17.113%|469 / 19698 = 2.381%|+4|+0.0178pp|
|42|19646 / 2816|472 / 2816 = 16.761%|475 / 19646 = 2.418%|−3|−0.0134pp|
|123|19570 / 2892|517 / 2892 = 17.877%|416 / 19570 = 2.126%|+101|+0.4496pp|

跨seed修复487.33±25.70个、损伤453.33±32.47个；净正确变化34.00±58.13个，即+0.1514±0.2588pp。修复率17.250±0.570%，损伤率2.308±0.159%，两率分母不同，不能直接以17%大于2%声称净收益很大。seed0/42修复和损伤几乎相抵，seed123净收益较大；这个操作点不是三seed稳定增加正确对象的证据。

## 背景误检

|seed|N背景FP|C0背景FP|数量差|FP/image差|相对数量差|
|---|---:|---:|---:|---:|---:|
|0|940|943|+3|+0.002042|+0.32%|
|42|914|919|+5|+0.003404|+0.55%|
|123|871|917|+46|+0.031314|+5.28%|

平均FP/image增加0.012253±0.016521，三seed同向，满足冻结计划的专门损伤复核信号。绝对增量前两个seed较小，第三个更大；这不是统计显著性结论，也不是停止C1既定E200的理由。不能将整体mAP上升解释为所有背景/类别均获益。

## 逐类AP与固定操作点差异

下表AP均指mAP50–95，单位为百分数；Δ单位为pp，SD为样本SD。宏平均对各类等权，不按GT数量加权。

|类别|GT数|N mean±SD|C0 mean±SD|Δ seed0 / 42 / 123|Δ mean±SD|
|---|---:|---:|---:|---:|---:|
|car|18965|65.425±0.218|65.453±0.093|+0.203 / +0.167 / −0.286|+0.028±0.272|
|freight car|710|39.314±0.304|40.198±0.621|+0.818 / +0.029 / +1.808|+0.885±0.892|
|truck|1336|51.164±0.180|50.948±0.388|+0.280 / −0.147 / −0.779|−0.215±0.533|
|bus|751|72.737±0.457|72.879±0.385|+0.253 / −0.268 / +0.438|+0.141±0.366|
|van|700|43.472±0.319|43.967±0.402|+0.575 / +0.943 / −0.034|+0.495±0.493|

五类Δ均值再平均为+0.266655pp，与旧整体结果一致。freight car是唯一mAP三seed全正的类别；truck平均下降，但不是三seed都下降。没有任何类别的mAP50–95三seed全负，不能把单类平均下降写成已确认稳定退化。

固定操作点的净正确对象变化则为：car +41/+47/+86，freight car +3/−31/+10，truck −16/+7/+2，bus −8/−4/+12，van −16/−22/−9。特别是van的平均AP上升同时伴随固定conf=.25正确对象三seed减少。这支持继续检查分类分数与排序/操作点的得失，不能仅凭这组现象确定是置信度校准、类别混淆或定位导致。

辅助AP75中，car的逐seed差为−0.041/−0.031/−0.860pp，均值−0.311pp；truck均值−0.955pp但方向不一致。分类蒸馏没有保证严格定位指标提高，仍需独立定位证据，不能由此跳过L几何准入或认为L必定有效。全部逐类AP50/AP75及各seed数值保留在汇总JSON。

## 尺度、来源与冻结亮度代理

面积使用原对象记录的实际输入坐标，不缩放框；名义imgsz640对应`input640_*`标签，实际padding不会改变box面积。

|分组|GT数|净正确变化 seed0 / 42 / 123|
|---|---:|---:|
|area<32²|3923|+17 / +26 / +43|
|32²≤area<96²|17949|−11 / −25 / +65|
|area≥96²|590|−2 / −4 / −7|

小目标固定操作点三seed正向，大目标三seed负向，但后者只有590个GT且绝对净损失较少。这是后续检查损伤分布的线索，不是从结果反调尺度门控的依据。

来源可靠映射未找到，22462个GT全部保留UNKNOWN。旧200图亮度子集含low98图/1586GT、high102图/1498GT；其余1269图/19378GT为UNKNOWN。low组净正确−7/+1/+10，high组+3/+11/+20；low组背景FP差−1/−6/−3。该子样本没有显示“低亮度一定获得更多净修复”，但样本覆盖和分母不同，也不能据此否定低亮度蒸馏机会。标签是既有灰度均值代理，不能称真实昼夜。完整组内修复/损伤分母及背景FP/image均保留在各seed原JSON中。

## 对下一步的约束

本轮补充诊断说明原C0有对象交换与分组得失，背景FP需要专门复核；它没有证明完整跨模态因果归因、负迁移被消除，或C1更好。已启动C1三seed保持原协议完成E200后，使用同一操作点及完整开发集对比；C0的本次结果不用于重调C1系数、门控或中途停训。L几何证据不足的状态不变。

旧总AP桥接仍单独保留DRAFT。新补评提供逐类与objects，不自动把原六端点的所有历史运行证据补成已接受。四臂、内容对照与最终论文主张仍按冻结计划逐项收齐。

## 第一次实际运行

只读确认六个reevaluation_receipt均completed、调度completion.json为COMPLETED后，按授权在screen `ikdv2_oldC0diag_all`启动同字节accepted CLI，空CUDA_VISIBLE_DEVICES，三seed计划串行。seed0在载入真实objects时触发`validate_row`的画布限制，后两seed未执行。原目录/日志/资源回执保留，已复制到本地[失败CPU attempt](failed_cpu_attempt1)。没有生成或伪造成功对象数值。

只读查看真实N42/C042原objects：两者1469图的canvas_shape均为[544,672]，original_shape为[512,640]，与原评价imgsz640/rect合同并不矛盾。94实际`ultralytics/data/build.py:308`评价pad=0.5；`data/base.py:392`按`ceil(shapes*imgsz/stride+pad)*stride`给画布。该aspect=.8、stride32得到544×672。分析器把名义imgsz误当实际最长边，合成fixture没有覆盖pinned native额外padding。

该失败发生时未改原accepted分析器、NEW或原metric/contract/objects。随后按技术修复授权创建rect v2副本，经根独立19项测试及源码复核接受，再以新attempt2完成分析；旧v1三文件及接受回执保持不变。没有把原框或画布强制缩成640，也没有将读取失败解释成蒸馏无效。
