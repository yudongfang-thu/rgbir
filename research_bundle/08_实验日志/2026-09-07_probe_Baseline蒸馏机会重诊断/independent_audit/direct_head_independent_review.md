# 原生head直接读出：独立审阅

**通过。直接head脚本按原生固定conf=.25拒识背景、其余argmax，没有拟合或阈值搜索；两数据集、所有模型、两cohort的完整指标字典均由真实logits独立重算精确复现。**

2026-09-07，审阅者`/root/baseline_opportunity_audit`。只读 `new_probe_analysis/direct_head_baseline.py`、`direct_head_baseline_v1/`、冻结probe的cohort掩码和真实remote_exports。无SSH、GPU、hash或修改原始probe。

独立实现没有照抄sigmoid运算，使用等价规则：`max raw logit >= log(.25/.75)`保留argmax，否则类别设为nc。阈值边界保留`>=`。先核对raw对象y顺序与冻结probe缓存完全相同，再复用相同的ROI共同valid掩码；直接head没有读取ROI特征。

|主fixed-anchor的前景macro recall|N42|T42|N0|
|---|---:|---:|---:|
|DroneVehicle|0.6014269184|0.5434904602|0.5682084852|
|LLVIP|0.7060653188|0.6485225505|未导出|

全部分组、confusion matrix、前景/背景/all accuracy与macro recall均逐字段一致，不只核对上述汇总值。相应原值见 [supplemental_readout_verification.json](supplemental_readout_verification.json)。

这项sanity说明固定ridge可能低估原始logits所含信息；它没有给原probe补上更强的训练结论。不能因feature比偏弱ridge高就宣称logits缺信息、feature载体优越或KD预期收益。相同GT关联anchor上的直接head指标不是常规NMS检测评价、AP或对象分配召回率；IR在未配对位置的分数也不证明同对象教师信息。
