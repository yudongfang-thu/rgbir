# 定位线性readout独立审阅

**接受本次CPU描述性结果。共同cohort、行顺序、DFL距离/框解码及所有保存IoU/MSE独立回验通过；不把相对较弱ridge基线的改善解释为定位载体优越或KD收益。**

审阅者：`/root/baseline_opportunity_audit`；2026-09-07。只读本目录PROTOCOL.md、localization_readout.py、outputs和两数据集真实remote_exports；没有SSH、GPU、hash、重新拟合或修改执行代码。此文件由独立审阅者新增。

## 代码与协议

- 所有比较臂共用RGB GT、IR标注IoU≥.5、真实N42 P3/P4参考候选及四边GT距离在[0,14.99]的集合；背景和GT-center fallback被排除。没有按某模型的预测表现分别过滤。
- 输入只读取`*_anchor_P3/P4`固定patch，未读取GT自适应ROI feature。块投影固定128维/层；同维输入复用投影；标准化只用该共同cohort的train数据。
- Ridge采用对象平均平方损失加α=1的L2，四边等权，target只中心化，截距不惩罚。保存MSE单位为stride归一化距离的平方；不是像素平方。
- 原生DFL是softmax后16-bin期望，按相同anchor/stride解码；不是ridge拟合值。
- 负预测距离不截断，任一负边使对象IoU=0且记invalid；非负预测即使超过14.99也按原值解码。这与冻结协议一致。
- shuffle在train/val各自的共同cohort内按object_id排序、固定seed打乱后循环位移；同split双射、无自配，同图donor数单列。
- geometry meta只含anchor_x/640、anchor_y/640、log(stride)，未把GT宽高/类别/IoU作为输入。GT关联仍属于诊断特权。

## 独立实际核验

重新执行6项小型真值检查，全部通过。另从真实对象表独立重建筛选条件，并使用保存预测独立计算每个split/臂的IoU和MSE。

|检查|DroneVehicle|LLVIP|
|---|---:|---:|
|共同cohort总对象|17840|3276|
|train / val|15019 / 2821|2728 / 548|
|source row、object_id、GT框顺序|全部精确一致|全部精确一致|
|所有保存IoU/MSE重算最大差|0|0|
|保存native距离 vs raw DFL期望|精确一致|精确一致|
|native解码框 vs原导出N42 same_anchor框最大差|0.000049504输入像素|0.000049189输入像素|
|donor同split、双射、无自配|通过|通过|

解码框误差属于FP32原导出和独立float64重算的舍入量级。全部指标的背景排除与共同分母都已检查。

## 结果解释必须保留的边界

Drone dev原生N DFL的mean IoU为 **0.848451**，高于本次所有ridge臂；N_DFL ridge为0.742626，N_DFL+T_DFL为0.771577，N_DFL+N_feature+T_feature为0.763415。LLVIP原生为0.726483，N_DFL+T_DFL为0.732449，feature组合为0.712817。

因此同一固定ridge函数族内可以比较是否有教师辅助信息，但较高维feature相对低估的N_DFL ridge的差值不证明feature比原生DFL更有用。所有feature读出仍低于同cohort原生DFL；不能把其结果叫作部署定位收益、AP或蒸馏效果。LLVIP没有N0，两数据集模型recipe也非完全相同，这些限制不因指标核验通过而消失。

独立核验脚本与机器可读原值：

- [verify_supplemental_readouts.py](../../independent_audit/verify_supplemental_readouts.py)
- [supplemental_readout_verification.json](../../independent_audit/supplemental_readout_verification.json)
