# 同帧检测见证：定义转换与实际selected覆盖

原80对象、状态、C门和selected保持exact；仅改变诊断读出。T始终相对IR own GT。没有新AP、学习更新或全dev结论。

|模型|旧assigned正确|FP32 dense-any正确|native pre-NMS正确|native post-NMS匹配正确|
|---|---:|---:|---:|---:|
|S|64|77|77|77|
|R|64|77|77|77|
|T|63|78|78|78|

## 原生NMS后匹配的机会与风险

|状态桶|对象数|原selected|selected/桶|占全部selected|
|---|---:|---:|---:|---:|
|T_correct_S_error|1|1|1.000000|0.031250|
|S_correct_T_error|0|0|NA|0.000000|
|both_correct|77|31|0.402597|0.968750|
|both_error|2|0|0.000000|0.000000|
|teacher_unknown|0|0|NA|0.000000|

所有模态的两两2×2混淆表、四种定义下的机会/风险与完整ID见[summary.json](summary.json)。原FP32与native decode/置信度边界可能不同；native pre-NMS列用于分开记录这种差异。NMS后的匹配为固定confidence .25的一阈值诊断，不能当作AP评价。风险桶不表示负迁移已经发生，机会桶也不表示已修复。
