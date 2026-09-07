# 固定 anchor 定位目标平移压力诊断 v1（2026-09-07）

> Outcome-blind 冻结：本文件先于真实扰动结果写入。仅 CPU，对 LLVIP 先运行、DroneVehicle 随后运行；不拟合、不扫参数、不重新推理、不改 cohort、anchor、正式选择器、0.70 门或原 L1 准入。结果等待独立审阅。

## 输入和固定分母

只读上一阶段 `2026-09-07_probe_Baseline蒸馏机会重诊断/remote_exports/{llvip,dronevehicle}_full_attempt1/{objects.jsonl,logits.npz,summary.json,model_identity.json}`，及已独立接受的 `feature_review/localization_readout_v1/outputs/{dataset}/cohort_and_donors.csv`。raw NPZ 有 N42/T42 四边 16-bin DFL 和完整类别 logits；JSONL 有 RGB/IR GT、共同 anchor/stride、same-anchor 框/置信/类别和原对象身份。LLVIP 无 N0 不补造。

基础 cohort 完全复用定位 readout：非背景、paired_gt_iou≥0.5、anchor_has_reference_candidate=true、RGB GT 相对共同 anchor 的四边未 clamp 距离全部在 [0,14.99]。须独立重建且与旧 CSV 的 source_row、object_id、顺序 exact；任一不一致中止。train 与 val 分开，val 即原 dev，不接触 test。所有扰动保留每个基础对象，不按扰动后表现删对象。

固定分层均由零扰动定义：

1. `base`：上述全部基础对象。
2. `n_iou_lt_070`：base 中 N42 同 anchor 期望框对 RGB GT IoU 严格 <0.70，仅任务门分层。
3. `quality_gate070`：base 中 N42/T42 同 anchor 均预测正确类别且 confidence≥0.25，N42 IoU<0.70，T42 对 RGB GT IoU≥0.60、对 IR GT IoU≥0.50，T−N RGB IoU>0.05。阈值沿用旧定位 quality gate；配对门与唯一 owner、几何准入不混入此层，不把该层命名为原 L1/D2 selected。

另外报告 pair≥0.8、IR GT 未 clamp 支持域、原生候选唯一 owner 的可重建性/计数作为上下文；这不将 readout anchor 重新选择为原选择器 anchor，也不补造正式增强流或几何证据。每层分别记录对象数、唯一图数、来源组、stride；主要结论以 dev 为准，train只检查可观察覆盖。

## 预定扰动

共 25 个固定条件：零扰动 (0,0)；对于 a∈{1,2,4} 输入像素，各取 (±a,0)、(0,±a)、(±a,±a) 共八方向。对角的欧氏幅度是 a√2，同时记 L∞ 和 L2 幅度，不称作 a 像素径向误差。无需随机数。

只把教师预测框平移 (dx,dy)，GT、N42、anchor、stride、类别 logits 均保持固定。未 clamp 的 teacher LTRB 距离平移量为 [-dx,-dy,+dx,+dy]/stride。几何主读数直接将原始 T=1 DFL 期望框加 [dx,dy,dx,dy]，不裁画布/内容区，不删除支持域外对象。报告教师四边距离是否仍全在 [0,14.99]、RGB GT 与 IR GT 支持域上下文、教师框对固定 RGB GT/IR GT IoU、T−N RGB IoU和其正比例。零扰动 raw 解码与 JSONL same_anchor 框仅容许既有 FP32 舍入量级（最大绝对差≤1e-3输入像素）。

## DFL 离散概率质量搬运近似

这不是网络重新推理，也不模拟真实图像配准误差。对每个边在 bin 坐标中平移每个原始 bin 的概率质量：原 bin k 的质量移动到 k+δ，按相邻整数 bin 的线性权重分配。小于0、大于15的份额分别记 underflow/overflow，绝不 clamp 到边界 bin。保留区间 [0,15] 内的质量仅在计算常规 CE/KL/梯度时按每边剩余质量重新归一化，明确这些是**截断后条件分布的近似量**，不是完整移位分布。逐对象/逐边保存丢失质量，零扰动必须无丢失且概率逐元素 exact。

T=1 的 p=softmax(z) 和 KD T=2 的 p=softmax(z/2) 分别独立搬运。搬运与温度化一般不交换；此处定义的是给定温度概率的直接重采样，不从搬运后的 T=1 概率反推真实 logits。零扰动走原始 log_softmax 分支，CE/KL/head 梯度与旧分析数学定义 exact，浮点重复计算要求 atol≤1e-12。

按固定 RGB GT 构造原生两-bin qGT，距离不 clamp，四边同权。报告：

- teacher/native GT CE（T=1），teacher−native CE（负为 teacher 目标更好）、teacher 更好比例。遇到正 GT 质量落在零预测概率上，CE=+inf 如实记；不加伪计数，不 silently drop。各统计均附有效/非有限数。
- KL(teacher_shift_T2 || native_T2)×4，四边平均，零 teacher 质量项贡献为0。
- 对 N42 raw DFL logits 的 KD 梯度 `gKD=2*(pN_T2−pT_shift_T2)/4`、GT梯度 `gGT=(pN_T1−qGT)/4`；逐对象64维 cosine、dot、两个 norm和norm ratio。零范数 cosine=NA，单列；正 cosine 比例同时给固定分母与有定义分母。
- T1/T2 每边 underflow、overflow、total lost；条件分布期望框相对“直接精确平移框”的差异，用以暴露边界截断影响。

所有 gradient 只在 DFL 检测头 raw logits 层：没有 CIoU/native完整损失、score权重、TAL正样本更新、backbone或共享特征梯度，也不是实际KD剂量校准。

## 固定掩码、gate存活与汇总

所有主指标均用上面三个零扰动固定层统计。另在每个固定层报告扰动后 quality_gate070 是否满足、原 quality_gate070 成员存活数/比例以及加入新成员数；只作 attrition 统计，不用幸存者重算替代主表，不回补对象。分类置信和 N<.70 固定，变化只来自教师平移后的 RGB/IR IoU和margin；DFL支持域单列，不暗中纳入gate。

每数据集每split每固定层保存25条件原始汇总表；另对每幅度八方向逐对象取 worst-case：最小T−N IoU、最大teacher−native CE、最小cosine、最大丢失质量、八方向均存活。所有幅度均预定报告，不挑最佳方向。没有独立seed/真实误差抽样，报告对象分布的均值、SD、分位数和固定分母，不宣称统计显著、三seed增益或真实残差容忍阈值。

## CPU真值与产物

执行真实数据前：测试LTRB/box逆变换与平移符号、零扰动exact、内域整数/分数mass搬运、两侧边界流失守恒与无clamp、T1/T2分开搬运、KL已知一致与非负、GT-DFL两bin、KD/GT梯度有限差分、zero-norm NA、严格0.70和margin边界、固定mask不随扰动重选。遇实际检查失败保留 attempt 和说明，不覆盖失败。

本目录保存代码、PROTOCOL、真值回执、小汇总CSV/JSON、同序全部cohort与逐对象全部25条件NPZ、执行回执与README。只新增本目录文件，不改历史产物、训练、共享索引；主代理负责综合条目。最终只能标 `COMPLETED_PENDING_INDEPENDENT_REVIEW`，是否接受由独立审阅者决定。
