# CPU聚合口径与输入合同

本分析器只统计已固定的LLVIP首个32图真实训练批次，一次S/T/R无梯度前向。它不执行模型、反向、训练或AP评价，也不与旧200图dev的修复比例混合。实现与真值为 [analyzer.py](analyzer.py)、[test_analyzer_cpu.py](test_analyzer_cpu.py)，最新4组通过见 [ANALYZER_CPU_attempt2.json](ANALYZER_CPU_attempt2.json)，原attempt1保留；attempt2同时覆盖真实export contract与阈值保护。

```text
python analyzer.py --input NEW_PROBE_OUTPUT --output NEW_READOUT_DIRECTORY
```

输入必须包括 `objects.jsonl`、`frames.jsonl`、`completion_receipt.json`、`export_contract.json`；要求32图/一批、实际S/T/R各一次raw forward、无训练/反向/optimizer/EMA更新、学生状态未改、原首批流exact、稳定GT身份通过，以及实际selector导出的full诊断与原selector crosscheck通过。未完成或同目录failure冲突时拒绝。输出保留原始对象ID和输入stat，不算hash；原JSONL不改。

## 状态桶和分母

主桶为 T正确/S错误（机会）、S正确/T错误（风险）、两者正确、两者错误、teacher未配对，五桶恰好划分全部增强后RGB GT。S/R相对RGB own GT；主T相对IR own GT。未配对时教师未知而非错误，不能计入风险桶。T同一框相对RGB GT的状态另作次级表，不做物理几何变换、配准认证或实际C门替换。

状态阈值与原baseline诊断保持一致：粗候选confidence≥.05、IoU≥.1，正确confidence≥.25、正确类、IoU≥.5；互斥优先级为no_candidate→low_confidence→class_and_localization→class_only→localization_only→correct。这里是GT辅助的一对一空间分配状态，**不是AP、post-NMS独立预测或实际C门**。原C教师条件是any-candidate判断，可能与分配后状态不同；输出单独保留两者的2×2交叉计数，绝不强行令其一致。

分母分别为：全部增强后RGB GT；其中原selector实际配对的RGB GT；机会/风险等桶自身对象数；每道累计门的进入对象数。被原增强裁掉的source GT不是此处S错误，已在稳定身份trace中另外记录；32图中的空图仍保留图数分母。分母为零时比例为null/NA，不填0。计数不表示梯度份额、损失份额或可修复AP。

## 实际门与首次流失

累计表固定次序为 matched→region_valid→reference_candidate→base→teacher_correct_own→quality_positive→eligible→selected。base与eligible是原集合的显式核对点，可能与前一步不产生额外流失。所有matched行，包括nonbase，都来自同批原helper重建并与实际full selector核验；未matched后续字段为null。

原定义逐对象核对为 `base=region_valid AND reference_candidate`，`eligible=base AND teacher_correct_own AND (q>0)`，selected只能来自eligible。P3/P4共同valid原样保留。分析器另以整批eligible数量的ceil(.5×N)和q降序、精确tie保留matched行顺序复算selected及1起rank，禁止逐图重新配额。

原 `reference_candidate_count` 在全部matched上计数，可能包含region不合格行；它与累计表“已过region后的R候选保留数”有不同分母，二者分别报告。首次流失对象ID逐门保留，各门流失与最终selected之和闭合该桶总数。顺序用于记账，不把每门流失量解释成可加因果效应或改门之后必得的收益。

四组小真值覆盖已知机会/风险和门流失、空对象/零分母、stable ID/scope/原计数异常拒绝，以及整批ceil配额与q精确tie；不会以只对计数自洽替代原producer的真实ID和selector映射审阅。

## 可支持的结论

只能描述这批成熟初始化train对象中，哪些状态进入当前固定C门、在哪个条件处流失，以及进入selected的状态构成。T正确/S错误是一种候选机会，不证明KD会修复；S正确/T错误是一种风险诊断，不证明实际负迁移已经发生。GT关联、教师IR own GT正确和稀疏选择均不能证明物理几何已核或负迁移已消除。没有全数据覆盖率、dev AP、梯度、更新、泛化或E200效果主张。
