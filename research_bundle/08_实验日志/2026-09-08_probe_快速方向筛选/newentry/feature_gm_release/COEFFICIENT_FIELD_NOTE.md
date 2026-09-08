# 配置兼容字段与实际损失

本独立 F-rel-GM 配置中的 `classification_coefficient` 是复用训练配置、回执一致性检查的历史字段，与 `kd_coefficient` 保存同一个固定值 14.438521129817886。这里的字段名称不代表运行分类蒸馏。

实际运行只读取 `kd_coefficient`，`feature_gm_criterion.py` 将唯一 arm `F-rel-GM` 映射为 `F-rel`，只计算一个局部特征关系损失。没有另外计算或累加 C0/C1 分类损失，定位系数为零。总损失为原生检测损失之和加实际 batch_size × 固定系数 × F-rel。原生检测本身的分类损失保持原有定义。

在首次 F canary 和 AP 之前补充此字段说明；没有修改损失、系数、选择或训练协议。
