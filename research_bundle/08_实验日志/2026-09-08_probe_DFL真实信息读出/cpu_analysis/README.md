# 真实DFL的CPU读出入口

**实际读出已完成并通过独立复核：32图、80对象、317唯一分布；79共同合法同索引对象上T的GT CE和期望误差更低，但未隔离分布形状的额外学习价值。** 详见 [RESULT_READOUT.md](RESULT_READOUT.md)、[实际汇总](output_attempt1/summary.json)及[共同79对象断言](COMMON79_ASSERTION.json)。原数据经[实际核验](../independent_review/ACTUAL_JSON_CPU_attempt1.json)，派生记录经[独立全量复算](../independent_review/DERIVED_CPU_attempt1.json)通过；CPU分析没有新增GPU或forward。

[PLAN.md](PLAN.md) 保留原先固定的对象、角色、GT合法范围及统计口径。核心数学5项和记录/分母2项真值分别通过 [CPU_METRICS_attempt1.json](CPU_METRICS_attempt1.json)、[CPU_WRAPPER_attempt1.json](CPU_WRAPPER_attempt1.json)。

入口：`python analyze_dfl.py --input <已完成且独立验收的新probe> --output <新目录>`。生产者保存 `objects.jsonl`、`anchor_distributions.jsonl`、`dfl_contract.json` 和完成回执；输出目录存在就拒绝，不改生产数据。不同model/role分别汇总，不将重复role视为新的唯一分布，也不把80对象、11定位候选、1反向对象的不同分母混合。

完成回执、contract、每对象和每分布的`current_forward_id`必须一致，派生输出继续保留。历史frame/GT身份仅作稳定对象关联，不能替代新forward身份。

主量由真实logits稳定log-softmax复算，使用float64以避免直接log下溢概率。分布描述为期望bin、熵nat、方差bin²。DFL CE相对本role的own GT、未clamp LTRB/stride，在每边 `0≤d<15` 时按相邻bin插值；非法边明确null/reason，四边均值仅在四边全合法时计算。尤其`14.99<d<15`读出保留真实d，**与native训练内部可能clamp到14.99不同**；本程序不是重建实际训练DFLoss。

CPU复算与保存FP32数值的固定闭合限为概率2e−6、期望1e−5 bin、由保存FP32期望重建框1e−4像素，用于算术舍入检查。native概率/DFL卷积期望及native解码单独记录：不把低精度概率偷偷归一化，不将native/FP32差宣称为同一个值。producer的原native/FP32解码exact合同仍分别保留。

输出逐对象role记录与唯一distribution读出，同时保存运行源码副本和输入stat，不计算hash。`mean4_DFL_CE_nat_valid_objects` 是同model/role内、四边GT合法对象的读出均值；它不包含真实训练分配、类别、IoU或其他损失权重，不能称KD梯度或AP收益。

各model/role可能具有不同的合法对象集合，分析器分别描述各集合，不直接相减T/S集合均值。实际报告另用[ID清单](READOUT_DENOMINATOR_APPENDIX.json)与[显式断言](COMMON79_ASSERTION.json)核对了79共同合法同索引对象；S/T各自native角色的76/77合法集合仍不相同，未将其全体均值当作配对结果。

合成真值证明相同期望可对应不同熵/方差/GT对数概率。新实际向量能显示分布形状究竟是什么，但非零熵、较高方差或某个GT CE差都不能证明更可靠、可迁移或有额外学习收益。本入口不计算跨anchor KL、不重映射概率、不修改旧L1/L2，不据本批数据自动扩训。
