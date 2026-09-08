# F-rel-GM 协议复核与固定读出

限定结论：新方案在数学和单位表述上没有确定阻断；它是**读过8批校准后、首次F AP前固定的协议修订**，不属于原数字上限协议的事前准入。原F-rel的BLOCKED回执和“没有F AP”状态继续保留。这里没有启动GPU、重新校准、读取F AP或修改训练源码。

固定 λF=14.438521129817886、λC1=0.09227393550836771。只解除跨损失数字 λ≤1 的上限；F算子、原selection/base分母、P3/P4、3×3局部Gram关系损失、数据与增强、初始化、BN running buffer冻结且affine可训练、恒定SGD lr1e-4/warmup0、2048图/3轮192批/seed42及fresh optimizer/EMA不变。不同损失缩放/归一化下，λ的数值不能独立表达梯度剂量。因此取消数字上限有合理单位依据，但不因此宣称“大λ一定安全”或“已经证明特征有效”。

已有真实8批的独立CPU复算见 [FIXED8_DOSE_READOUT.json](FIXED8_DOSE_READOUT.json) 和 [dose_readout.py](dose_readout.py)。输入是 `../../results_1156_partial/calibration/drone/` 下原receipt和JSONL；保持原blocked原值。原记录测量共享参数集合上 `B×unit-KD` 的参数梯度范数；native用同一集合的native.sum()梯度。正标量λ只缩放范数，不改变方向。

|固定λ投影比值|最小|中位数|均值|最大|
|---|---:|---:|---:|---:|
|F / 已加权C1|0.49062902|1.00867493|1.17067873|2.29769542|
|F / native|0.21837910|0.23687698|0.27688243|0.38409355|
|已加权C1 / native|0.12990083|0.26750170|0.27442871|0.47237529|

8批原始目标系数的中位数就是固定λF，但偶数样本的倒数变换不保持中位数严格互逆，因此实际F/C1比例中位数**不是1**，不为此重调λ。F/native余弦6批正、2批负，中位数0.08883839、范围−0.01553915至0.21689270；这不是F与C1梯度方向的比较。这里也不把全模型、后续每批梯度或实际optimizer更新当作已匹配。canary只能验证该运行窗口的技术条件，不能认证学习收益或全程安全。

独立复跑新source的6组CPU检查全部通过，见 [INDEPENDENT_SOURCE_CPU.json](INDEPENDENT_SOURCE_CPU.json)：F算子与原源码byteexact、唯一arm别名映射、旧/新N实际criterion函数在明确toy依赖下的loss/梯度/选择exact、配置/剂量/新scope、原blocked保留及eval人口接口。接受范围为进入有界真实canary的代码准备；没有证明真实训练轨迹相同。

## 首次F AP前固定的三项读出

[analyze_feature_gm.py](analyze_feature_gm.py) 读取新F-rel-GM及原主队列N/C1三个完整eval小回执与各自真实保存config。保留原N/C1的 `DIRECTION_FT3_BNFROZEN`，不改其scope来伪装同一运行。新F必须为 `FEATURE_RELATION_GM_FT3`，endpoint对应3轮last/EMA，完整dev1469图/22462GT，五类macro AP、fraction单位、λ、模型初始化、全部子集/mapping、优化器/LR/BN/增强及3E必须匹配。输出三臂raw AP50/AP75/mAP/precision/recall和F−N、F−C1百分点差，不选择最好checkpoint、不扫λ、不自动扩展。

另外强制要求queue的 `matched_control_projection.json` 对N/C1均显式通过common config、初始化stat、Fcanary first30与主流、主完整训练first30证明；缺一个即拒绝三项汇总。该projection依赖queue验证真实流，本分析器只核字段、绑定及配置，不伪称重新读取所有训练图。完整回执未齐时拒绝，不复用旧F、不同BN/LR的FT3或其他N。

```text
python analyze_feature_gm.py --main-campaign SCREEN_ATTEMPT1 --feature-campaign FEATURE_GM_ATTEMPT1 --output NEW_READOUT
```

主回执布局 `evaluations/drone/{N,C1}/`；新回执布局 `evaluations/F-rel-GM/`。4组小真值通过见 [CPU_CHECKS_attempt2.json](CPU_CHECKS_attempt2.json)：已知raw/百分点、错误scope/缺失projection/剂量拒绝、配方或指标拒绝、极小负差不抹零及倒数中位数反例。attempt2补候选配置projection的path/stat bytes绑定，并要求新queue完成及候选完整训练first30证明；原attempt1保留。没有读取新F结果。单seed探索，SD/显著性、E200效果、跨模态因果和论文增益均不在接受范围。
