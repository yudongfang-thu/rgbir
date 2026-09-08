# L3 对象坐标分布短筛：执行前冻结协议

**本协议定义新实验L3-DFL及同掩码L3-GT，不修改原L1/L2。仅在本条目CPU目标接口、实际数据接口和资源/梯度canary通过后运行N/DFL/GT三个seed42短臂；无E200自动扩展。冻结时尚无新L3训练AP、校准系数或实际GPU梯度结果。**

## 目标与范围

检验：基于标签对象坐标定义的教师完整定位分布，是否比同学习位置额外GT监督提供更好的短程反馈。该目标空间不宣称物理配准，不解除原L1几何合同；不单独声称分布形状独立于教师期望框有效。

数据仅LLVIP，继续原已冻结2048张自然训练子集、2406张完整dev、seed42和双标签/增强流。学生/R从原visible42成熟last权重初始化，教师原IR42；N/DFL/GT全部重新开始同一3轮微调协议。原L2或历史N结果不能代替本次匹配控制。

## 选择与目标

- 原L2的匹配与R基础集合定义保持：同类GT一对一IoU≥.5，再标签IoU≥.8；R的P3/P4、native几何唯一归属、未clamp GT支撑[0,14.99]、confidence≥.05与RGB GT IoU≥.1，按confidence→IoU→anchor ID选择一个学习anchor。
- 分母是上述基础对象数，位于R可靠性和所有教师质量门之前；空基础分母使用max(1,base)，不按最终通过数量归一。
- R类别正确、confidence≥.25、RGB GT IoU<.70保持，**不放宽旧R门**。
- 新版教师固定读取同一R学习anchor的分布，不搜索另一个更好的教师anchor。该anchor须在IR全部GT的native候选规则下唯一归属对应IR对象；类别正确、confidence≥.25、教师期望框对IR GT IoU≥.50。
- 通过对象坐标运输后，教师期望框对RGB GT IoU≥.60，且领先R> .05；原T=1完整16-bin分布所有正质量必须可表达在学生[0,15]支撑，越界整对象拒绝。质量判断使用T=1期望框，不使用平滑温度下改变的框。
- 教师的anchor、完整支撑及同位置质量条件是L3的新定义；不把旧L2独立教师点的selected mask直接套给同索引新目标。当前S输出、fg_mask或实时检出状态不参与选点/门控。

L3-DFL温度固定2：先对教师实际logits/T做softmax，再用已接受的对象坐标运输算子形成学生域q；运输后不再开方或重归一。学生用log_softmax(logits/T)。单对象损失为四边平均KL(q||S_T)×T²，选中对象求和后除基础对象数。教师、参考及目标均detach。仅一个KD标量加入`native_total.sum()+B_actual*lambda*KD`。

L3-GT与L3-DFL使用完全相同的对象、anchor、教师质量/运输支撑掩码、分母及lambda。目标换成RGB GT的相邻两bin分布，再使用`sqrt(q)/sum(sqrt(q))`作为温度2目标；不称RGB-only。N执行同辅助/选择路径但lambda=0，检查总损失等于native。

## 固定预算和准入

训练沿用已完成的短筛recipe：3 epochs、每轮64个B32 batch（共192个batch）、nbs64、workers4、640、SGD lr1e-4恒定、warmup0、原deterministic/增强；BN运行统计冻结、BN affine可学，fresh optimizer/EMA。192 batch不是192个成功optimizer update；报告真实尝试、AMP skip和成功次数。端点固定第3轮last/EMA，完整dev独立原生评价。

本次新三臂与校准统一使用已执行DFL读出的局部AMP setup自检替代：绑定同模型/环境已验证AMP=True，setup期间返回该值并finally恢复原check_amp，实际autocast不变。不运行或下载无关AMP自检模型；明确setup变更，不宣称与旧短筛完整轨迹等价，因此本次N重新运行并核对三臂首30批数据流。

校准只用原自然流首8个batch，学生副本每批恢复完整初始参数与buffers，BN冻结，教师/R eval；无optimizer/EMA更新。固定P3/P4进入head的两个来源模块，保存参数名。使用

`lambda=min(1, median(0.1*norm(g_native)/norm(g_B*DFL_unit)))`。

至少4/8批有限非零；否则检查代码，非实现问题就阻塞该候选，不换批、不扫lambda、不用epsilon。GT复用同lambda并报告实际梯度剂量和与native夹角，不宣称两臂梯度相同。一次校准同时记录GT单位梯度，不另校准其系数。

新路径的第一实际GPU阶段是这8批校准/显存测量，首批实测峰及进程树RSS必须在预留上限内才继续固定流；不是正式训练。随后N、DFL、GT各至少24次成功更新的canary，记录首30batch身份和非零KD梯度、完整初始化及辅助隔离。全通过才排三臂完整FT3。所有阶段仅原global lease、screen、动态GPU，显存/整卡余量/RSS和项目并发规则不变。正式短训预约采用本路径实际canary峰加现有余量，不能用旧L2峰直接放行。

只因技术无效、资源越界、非有限损失/梯度、数据或评估错误停止对应attempt，保留原始产物。不据中途AP停训或修改门。

## 结果判断

仅报告DFL−N、DFL−GT及每臂相对原成熟初始化的差，主mAP50–95、辅助AP50/AP75/召回。三臂按同固定末端评价，不挑best。

- DFL未超过两项匹配控制：结束该版本短筛，不自动扩展。
- 超过两控制但仍低于原初始化：记为可能减轻微调漂移，不延长/升级。
- 同时超过两控制和原初始化：保留单seed短程候选；仍不自动E200，也不当三seed或四臂论文结论。

本轮不为形状单独扩充均值控制矩阵、不增加数据集、不重新选择训练子集。旧C1/归因任务和其他任务的C1提速切换继续由原负责人处理。
