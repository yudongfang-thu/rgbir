# native 差距、蒸馏载体与定位门控审计（2026-09-07）

**结论：当前weight0没有非零IR蒸馏，但与历史native存在真实工程/随机增强轨迹差异；C只传GT类别的池化标量证据，L已是4×16定位分布。绝对R IoU<0.70明显压缩机会，局部特征值得作独立载体对照，但已有相关文献，不能将加特征或C/L对称组合直接宣称原创。**

## 目的和执行范围

回答用户对历史native差距、类别/定位logits、严格门控与更大局部特征载体的疑问。三条并行只读审计（历史工程、实际载体与过滤、特征文献），主线程核验分类/定位文献并整合。无GPU任务、无新AP评估、无长训或在跑配置修改；新增CPU检查及研究结论全部保留脚本和小产物。

正式端点沿用2026-09-07 11:19快照。历史native明确指CGKD W1三seed标准native，不混用其他formal_native教师目录。

## 1. native差异：先区分观察与原因

| seed | 历史native | 当前N(weight0) | C | N−历史 (pp) | C−N (pp) |
|---|---:|---:|---:|---:|---:|
|0|54.358|54.346|54.772|−0.011|+0.426|
|42|53.816|54.514|54.658|+0.698|+0.145|
|123|53.687|54.407|54.637|+0.720|+0.229|
|mean|53.954|54.422|54.689|+0.469|+0.267|

C−历史的平均观察差+0.736，可代数拆成N−历史+0.469与C−N+0.267；前一项的具体成因尚未量化。N−历史为2/3正，不是3/3；C−N才是3/3正。

- 真正变更：workers8→4，标准native trainer→paired trainer，逐轮val/plots开启→关闭并独立末次评估。主要训练预算、SGD、增强参数和通用初始化文件相同。
- 新CPU实证：真实Drone loader同文件顺序，worker4/8前4批检查一致，第5–10批所检查的首张增强图像及标签集合不再逐元素相等。worker改变了具体增强轨迹，不只是速度；但未证明它单独解释+0.469pp。
- 当前pinned代码重建旧/新建模入口，三个seed各499/499初始张量及Torch RNG后继精确相同。不是恢复历史未保存的初始化快照，因此保留证据范围。
- C/N都执行相同T/R和paired计算，N最后乘0.0，无非零教师梯度。它读取IR以保持工程对照，不是运行时完全不读取IR的RGB-only实现。
- 当前loader generator不含实验seed；三seed不等于三套独立增强流。三seed同向仍有意义，但很小的SD不能代表所有训练随机性。

完整配置、更新步数、原始结果和源代码见 [native_audit.md](native_audit.md)。历史结果不再作为C正式增益分母；现有新旧C/N短测等价只证明Task-Conditional与OEv1实现等价，未证明旧标准native整程等价。

## 2. 载体事实与质量门诊断

C：只取GT类别raw logits，分别在各模态自己的GT前景和局部背景做LME，差除T=2，再平均P3/P4，得到每对象一个连续标量，用SmoothL1蒸馏。Drone原检测头有5类，但当前没有传全部类间关系。

L：CPU核验实际教师/参考reg_max=16；每选中anchor的四边各16-bin logits经温度softmax，KL(T||S)，已传分布形状而非仅四坐标。

只对已有记录做单条件反事实，保留E、既有anchor、所有其它质量条件及教师领先>0.05：

| 训练诊断 | 原R IoU<0.70 | 仅改<0.80 | 仅取消绝对上限 |
|---|---:|---:|---:|
|Drone|626 (1.96% GT)|2115 (6.62%)|4415 (13.83%)|
|LLVIP|211 (3.77% GT)|598 (10.69%)|1431 (25.59%)|

逐条复现31,739条E记录现行选择。上述是未核验几何诊断，不是长训覆盖率；绝对R<.70确实挡住不少R已经较好、T仍进一步占优的anchor。把上限放宽到.8时教师GT-DFL CE更差比例未升，但只是代理。不能从代理推断学生共享参数梯度、AP或负迁移改善。

质量稀疏与几何零准入独立：当前可信物理点范围未覆盖完整目标，verified E仍为0。阈值改变不能创造几何证据。几何政策本身可按载体重新论证，但不能把放松政策称为补齐配准真值。

这是已有AP可见后的探索，不是outcome-blind阈值发现；现行配置未变。详见 [payload_gate_audit.md](payload_gate_audit.md) 与CSV/脚本/回执。

## 3. 文献与建议

- 分类：BCKD提示保留YOLO独立sigmoid语义；DKD、2025 RLD/LDRLD提示目标类以外关系有价值；C²KD提示跨模态类间关系也可能冲突。
- 定位：LD已有分布蒸馏与任务区域选择；TBD、Task Adaptive Regularization有按任务质量指导输出或特征的先例。
- 局部特征：GID、FGFI、FGD、CMDistill/CMKD-net有充分先例；CrossKD可作为让教师任务头约束特征的折中。更高维度不保证学生可用信息更多。
- LLVIP单类不能用跨类softmax增加信息：输出恒为1。需要独立置信度、前背景、层级或空间响应载体。

最有解释力的下一步是固定选择，比较原C标量→全类别池化证据→单一局部ROI特征；另行研究L门，而非同时改变门控与载体。新增载体需重新冻结梯度剂量，沿用相同数值λ不等于剂量公平。配准不足时优先对象级语义/关系；这不等价于绝对坐标定位迁移。所有新实验均应有独立版本，当前在跑实验保持原身份。

文献主表与原文入口：[logit_literature.md](logit_literature.md)、[feature_literature.md](feature_literature.md)。本次还纠正旧笔记对CGDet类数、FreqKD调参和领域原创性范围的过强表述，勘误保留在feature_literature中。

## 产物与局限

- 原始小证据：`native_raw_evidence/`；脚本：`assemble_native_evidence.py`、`native_cpu_replay.py`、`audit_gate_sensitivity.py`、`audit_head_layout.py`。
- 新结果：native comparison / CPU replay JSON与stdout、gate sensitivity CSV与receipt、head layout receipt。
- 独立措辞复核：[independent_wording_review.md](independent_wording_review.md)。
- 94只读来源与本地原始结果路径详见子报告；新CPU检查未新增服务器训练目录。
- 原文短页抽取仅供本地核验，不作为新研究原创内容或公开发布材料。
- 不将旧native观察差归因给单个因素；不将静态GT-DFL CE当作AP；不将相关文献结果当本项目复现。当前缺口为统一native轨迹验收、L几何覆盖和新载体实际训练证据。
