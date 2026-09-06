# 已有对比方法复用核对

> **保留明确标注的适配参照；CCLKD partial评估与OEv1有相同1469张有序roster，但历史训练与辅助IR标注差异仍存在，不能混成当前同代码净效应。**

2026-09-07 02:30:10通过SSH只读重取54个小产物，CPU直接重算，没有重新训练或评估。单位为mAP50–95百分数，SD为样本SD。

| 方法 | seed0 | seed42 | seed123 | mean±SD | 身份 |
|---|---:|---:|---:|---:|---|
| 历史native | 54.357648 | 53.815556 | 53.687433 | 53.953546±0.355778 | 历史主要recipe匹配参照 |
| CMDistill corrected | 53.900286 | 53.244840 | 53.669168 | 53.604765±0.332435 | PROTOCOL-ADAPTED |
| CCLKD partial | 54.066006 | 54.493295 | 54.332596 | 54.297299±0.215820 | PROTOCOL-ADAPTED，仅LLD+CCL |

对历史native：CMD −0.348781±0.291793pp，逐seed全负；CCLKD partial +0.343753±0.550510pp，逐seed负/正/正。不说CMD原论文被否定，也不说partial稳定正收益。

## 设置与可比性

本轮逐run args.yaml比较E200、640、batch32/nbs64、SGD、学习率/动量/衰减/warmup、几何/颜色增强、AMP、deterministic，当前OEv1与旧方法仅workers4对8有差异。初始模型和RGB数据yaml路径相同。主要显式recipe一致不是完整训练器或逐batch数据流等价。

| 证据维度 | OEv1 | CMDistill corrected | CCLKD partial |
|---|---|---|---|
| 学生/预算 | RGB YOLO11n E200 | 相同主要设置 | 相同主要设置 |
| 教师 | 冻结IR seed42 | 同一教师路径 | 同一教师路径 |
| KD直接使用IR GT | 是 | 否 | 否 |
| 训练workers | 4 | 8 | 8 |
| 独立endpoint | 新回执绑定last/EMA | metrics_record固定last | 新评估绑定历史last/EMA |
| 有序val roster核对 | 基准 | 旧record缺绑定roster，不冒充逐项验证 | 三seed均与OEv1逐项相同 |
| 训练源码 | 执行时snapshot | 旧适配receipt，非新合同完整链 | 旧v2未完全绑定，当前v3不能覆盖历史 |
| 完整论文复现 | 不适用 | 否 | 否，缺FLD/RLD |

CMD旧metrics_record无显式metric_units；本次按已核对的历史`rgbt-detector-metrics-record-v1`格式解释为0–1，不让新分析器猜单位。历史native completion_receipt没有terminal status，仅记录arm/config/seed/output。这是证据等级限制，不补写新合同COMPLETED。

CCLKD新评估receipt与绑定metric JSON一致，三份roster与OEv1一致；能支持旧权重的评估端点，不能补齐历史训练源代码。CMD旧权重可后续补同入口/roster/环境独立评估回执，训练workers与IR辅助标注差异仍应明列；重评估不能消除训练差异。

## 复核入口

`compare_existing.py`读取`comparator_snapshot/`并生成`comparator_analysis.json`；原路径与mtime见source_manifest。FGD/LD新增适配不在本历史复用审计内，没有伪造新baseline结果。
