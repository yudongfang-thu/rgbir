# C1 训练薄路径可行性独立源码审查

结论：**在原 C0 选择、全 base 分母、全 GT 背景排除和数值有效域不变的条件下，只为 selected 对象计算 S/T 全类 pooling、按原日志时点采集完整 R/base 统计，保持 C1 目标函数与实数域梯度是可行的。** 这属于实现优化候选；未证明浮点逐位等价、未测得提速、未准入在训任务切换。直接把 selected 张量交给现有 loss 函数会改变分母，不能这样实现。

2026-09-08，独立审阅者 loc_stress。仅阅读现行源码和固定 helper，不执行实验，不修改生产源码、冻结门/系数或运行任务，不计算 hash。本记录为本次有界审阅结论落盘。

## 1. 数学上可删的计算

现行 `classification_logit.py:_classification_terms` 的总损失为

\[
L=\frac{T^2}{\max(1,M)}\sum_{i\in S}\frac{1}{\max(1,V_i)}\sum_{\ell:v_{i\ell}=1}\left(KL_{i\ell,y_i}+\frac{\eta}{C-1}\sum_{c\ne y_i}KL_{i\ell,c}\right).
\]

其中 M 是全部 pre-teacher base 对象数量，S 是冻结 C0 选中集，V_i 是该对象原 common-valid level 数；C=1 时非目标项定义为零，C1 的 eta=.25，C1_y 的 eta=0，T=2，teacher raw clip=16。没有对象间 softmax、对象间均值更新、批级对比分母或依赖未选对象的训练归一化。

因此，在所有中间量有限的合法输入上，未 selected 的 S/T 全类 delta 不贡献损失或梯度。`reference_delta` 不进入 `_classification_terms` / `classification_loss_components`，只进入 class mean / base record 统计。S/T 的逐对象 pooling 彼此独立，可将 C0 选择完成后再计算 selected 的 S/T 全类值。**R 模型 forward、R GT-only evidence 与 decoded candidate 必须继续运行**，因为它们决定 reference 粗候选、q、base/eligible 和选择；不能把“R 全类 delta 可删”说成“R 可删”。

最小风险顺序是先保留原 C0 evidence/候选/选择整段算术及身份顺序，再减少后续全类 pooling 和未使用统计。学生 GT-only C0 evidence 在现代码仅供 c0_loss/诊断，不决定 C1 q，但进一步删它或去图是额外优化范围，本次不把它混入最小候选。

## 2. 必须保持的依赖

- **分母 M 必须显式保留。** 当前 `ClassificationSelection.base_count` 从 `student_delta.shape[0]` 推导，现有 `_classification_terms` 又以 m 归一化。把第一维裁成 K 后直接复用，会把 M 换为 K，提高实际剂量。selected/base/matched 三层映射、全 base labels / eligible / class 计数也不能随内容裁剪。
- **每对象尺度权重不变。** `valid_levels` 来自原 C0 的 `vs & vt & vr`；selected 的 S/T pooling mask 几何必须与原始对应行相同，至少在实际计算行维持原 validity 一致断言。不能按剩余计算行或全 batch 有效尺度总数重新归一化，也不能把只在一方有效的 level 纳入。
- **背景仍排除全部同模态 GT。** `region_masks(boxes, all_boxes, ...)` 的 `all_boxes` 必须包括未匹配、非 base、未 selected GT；仅第一参数 boxes 可以变为 selected。否则背景锚点、delta、validity 及梯度都会改变。前景允许与其他 GT 重叠，原 overlap fraction 是诊断，不是额外门，不能为了稀疏化做互斥划分。
- **原选择顺序与 tie policy 保留。** q 来自 T/R GT-only 多尺度 evidence，rho 取 ceil 后按原稳定排序；C1 delta 不参与门。保留原 batch/GT/base 顺序，不能把 q 顺序误当 base 顺序。
- **梯度域与训练缩放保留。** 仅 S score 前景和背景保留梯度，T/R detached，DFL 没有直接 KD 梯度；总损失仍由 trainer 加 native＋actual_B×冻结 coefficient×KD。不能省略 selected 的背景梯度或改 teacher clip/temperature/off-target mean。
- **空集仍需可反向。** M=0 和 M>0、K=0 都要返回连到真实学生 score 的有限零梯度标量，不是无 grad 的常量；不允许为了形成有效梯度重选对象。C=1/C1_y 的零非目标项也要保持既有组件与观察口径。

可以考虑专用 selection/loss 接口，显式区分 `base_count=M` 与 `computed_selected_count=K`；不能伪装成现有“第一维覆盖所有 base”的 public kernel。若为了保持旧 reduction 布局使用稀疏 scatter/零载体，它们只是计算占位，日志不能把未计算行写成测量值零。

## 3. NaN 与浮点边界：不构成无条件等价证明

原 `_layout` 检查所有 S/T/R raw scores 和 DFL 有限；`_labels` 检查全部 GT，不能因某对象未选而跳过这些验证。原 pooling 还要求合法 mask、无前景/背景相交及正 minimum；这些检查仍需保持。

`_classification_terms` 在乘 selected mask 前检查**全部 base S/T delta 有限**。有限 raw 不代表任意极端 FP32 输入的差值都有限：例如约 +3e38 的前景与 -3e38 的背景可在相减时溢出。原路径会拒绝该未 selected base 行；薄路径若根本不算它，就可能继续。`0×NaN/Inf` 也不能按实数零项直接消去。因此需要保留等价的有效域/异常检查，或对存在风险的输入退回原路径；不能声称“全 raw 有限就已覆盖全部旧断言”。不能通过把非法未选值写零掩盖差异。

即使正常输入数学等价，删零贡献图、改变张量形状/归约树或多个对象共享锚点时的梯度累加顺序，仍可能导致浮点非 exact。沿用原逐对象 pool 能减少此次 block16 分组归约带来的额外差异，但不能由源码直接担保最终 loss / raw gradient / shared parameter gradient 位级相等。这不是改变科学门或重校系数的理由；需用冻结容差与原路径比较，并保留失败。

## 4. 日志和 observer 不可与 loss 一并裁掉

`classification_loss_from_selection` 当前每 batch 都构建统计，尽管 `IndependentCriterion` 仅在 **sanity、前三批或 calls % cfg['log_every_batches']==0** 写 `kd_batches.jsonl`。训练薄路径可以只在这些已定时点计算完整 R/base delta 和重统计；不能把“每 100 批”简化成删除前三批或 sanity。

非采集批必须仍有真实 selected/base/normalizer、当前 loss/组件及每步计数供 `selected_total`、`last_stats`、进度和完成回执使用。未采集的 base delta mean、reference selected mean、完整 base records 等写 null，并附 `statistics_collected` / population / batch identity 等明确语义；不能写零、继承 C0 同名但不同语义字段、沿用上次值冒充当前批，或把 selected-only 均值标为 base 均值。当前 schema/analyzer 是否允许 null 需作为实现接口审查，不应静默变更。

完整诊断必须读取同一次 forward 的 raw tensors，不能额外做 student train-mode forward（会改变 BN/RNG/后续训练）。也不应让“日志批”与“普通批”使用两套未比较的 loss 算术；宜始终走同一个训练 loss，完整统计另用 detached tensors，或证明两种模式的 loss/梯度一致。

固定 epoch 0/10/50/100/199 首个训练 batch 的 `FixedEpochGradientObserver` 与 log_every 无关，**不能跳过或挪到下一个有 selected 的 batch**。其读取的是 native、KD、target、off-target 四个实际 shared-parameter 梯度；薄路径仍须提供同一梯度图的 `target_loss` 和 `off_target_loss_unit`。首轮 canary 的 raw-score / DFL gradient composition 检查也不能跟着日志降频。

## 5. 可执行范围与未证据化部分

允许将其作为独立实现候选评估：全 C0 选择不动，selected-only S/T 原 pooling，显式 M 分母，完整 R/base 统计按原采集策略拆出。它跳过的内容比单纯 block16 更多，值得检查；本次没有运行计时，不能给出训练提速倍数或承诺“大幅”。

独立接受前应覆盖 M>0/K=0、M=0、混合有效尺度、重叠 GT/背景排除、单类与多类/C1_y、非法值与极端有限值、选中/非选中映射等已知真值，并使用同批原 raw 验证选择、分母、loss 各分量和 score/shared-parameter 梯度，检查普通/完整统计两模式一致。现有冻结 coefficient、门、样本不需要因该候选而修改。是否部署是另一项准入，不能直接修改正在训练的冻结源码或据此无损迁移。

审阅源码：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/{selection_adapter.py,classification_logit.py,independent_criterion.py,gradient_observation.py,train_independent.py,runtime.py}` 及其 `task_conditional_reference/legacy_oev1/object_evidence_loss.py`。本次结论不覆盖历史 N/C0、定位 L1/L_GT 或未来 content-control 分支。
