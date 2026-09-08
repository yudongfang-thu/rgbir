# 固定首32图同帧状态与实际C门覆盖

仅真实首批训练图的一次无梯度前向；对象分母为增强后保留RGB GT。T主状态相对IR own GT，T框对RGB GT另列。状态诊断不代替实际选择门；没有AP、训练效果或负迁移消除结论。

32图，增强后RGB GT 80，实际配对 80；空图 1。

|主状态桶|对象数|占全部RGB GT|实际selected|selected/桶|
|---|---:|---:|---:|---:|
|T_correct_S_error|9|0.112500|7|0.777778|
|S_correct_T_error|10|0.125000|3|0.300000|
|both_correct|54|0.675000|19|0.351852|
|both_error|7|0.087500|3|0.428571|
|teacher_unpaired|0|0.000000|0|NA|

### T_correct_S_error

|实际累计门|进入|保留|本门流失|保留/进入|
|---|---:|---:|---:|---:|
|matched|9|9|0|1.000000|
|region_valid|9|9|0|1.000000|
|reference_candidate|9|9|0|1.000000|
|base|9|9|0|1.000000|
|teacher_correct_own|9|9|0|1.000000|
|quality_positive|9|8|1|0.888889|
|eligible|8|8|0|1.000000|
|selected|8|7|1|0.875000|

### S_correct_T_error

|实际累计门|进入|保留|本门流失|保留/进入|
|---|---:|---:|---:|---:|
|matched|10|10|0|1.000000|
|region_valid|10|10|0|1.000000|
|reference_candidate|10|10|0|1.000000|
|base|10|10|0|1.000000|
|teacher_correct_own|10|10|0|1.000000|
|quality_positive|10|9|1|0.900000|
|eligible|9|9|0|1.000000|
|selected|9|3|6|0.333333|

完整对象ID、S/R/T错误类型、T对RGB次级状态、原C0计数与全部逐门分母见[summary.json](summary.json)。区域/R候选是并列primitive条件；表按固定顺序作首次流失分解，不能把前后门顺序当因果贡献。32图属于train，不与旧200图dev修复率混用。
