# OEv1 方法与实际实验设置核对（2026-09-06）

> 当前主线是 DroneVehicle 上的 IR→RGB 对象判别证据蒸馏：训练时利用 IR 教师与 IR 标注，部署时只使用 RGB YOLO11n。实际运行的是两个干预臂 paired/weight0、三个学生 seed；目前证据能够检验整套干预的净效果，尚不能单独证明跨模态内容、质量选择或负迁移抑制有效。

本文只读核对本地保存的 94 release_v2、真实训练 run 的 source_snapshot、args、completion/evaluation receipt；没有修改方法，没有新增训练或 GPU 评估。运行状态使用已保存的 21:46 快照；同目录总报告若提供更新快照，以其明确时间为准。

## 一、为什么现在选择这个方向

目标不是简单让 RGB 网络的全部特征变得像 IR，而是回答：对于 RGB 已经具有弱线索、IR 判断更可靠、两边目标能够对应的对象，哪些类别判别信息值得传递？

数据诊断给出的直接动机是：DroneVehicle 抽样的 3,083 个共同目标中，IR 独有命中 360 个，RGB 独有命中 78 个；其中 118 个 IR 独有命中目标，在 RGB 侧其实已有位置正确、类别正确但分数不足的候选。相对地，双方已经命中的 2,505 个目标上，IR 定位 IoU 更好的比例只有 49.46%，中位差为 −0.00122。因此首先验证“目标与背景分得是否更清楚”，比假设 IR 普遍能教会 RGB 更精确的框更符合当前观测。这些是诊断对象计数，不是 AP 增益或可达上限。

本版方法的简要解释是：**在一部分符合条件的车辆上，让 RGB 学生学会 IR 教师对“这个位置确实比周围背景更像该类别”的判断强度，同时保留 RGB 原生检测监督。**

它是一个有明确失败条件的初版假设；当前没有证据表明这套设计已达到论文级新颖性或稳定效果。

## 二、训练图中有哪些模型和信息

| 对象 | 输入与状态 | 实际作用 |
|---|---|---|
| RGB 学生 S | 当前 RGB 训练图；可训练 YOLO11n | 接受 RGB 原生 GT 检测监督，以及选中对象的判别证据蒸馏 |
| IR 教师 T | 同一场景 IR 图；已训 IR native seed42 的 last/EMA；冻结 | 提供对象判别证据，并对 IR 自身 GT 做可靠性代理检查 |
| RGB 参考 R | 同一 RGB 图；已训 RGB native seed42 的 last/EMA；冻结 | 判断目标是否已有 RGB 候选线索，并作为教师相对优势的参考 |
| RGB GT | 学生模态原生标注 | 原生 box/cls/DFL 监督；界定 RGB 前景和背景 |
| IR GT | IR 模态独立标注 | 跨模态目标对应、IR 前景和背景、教师正确性代理 |

教师和参考都采用 `eval/no_grad`，不参加优化器、EMA 或部署模型。推理时只有 RGB 学生，不需要 IR 图、两侧 GT、参考模型或门控模块。

学生从已有通用 `yolo11n.pt` 初始化，随后训练 200 轮；**不是从已训 RGB 参考权重继续微调，也不是所有参数从零随机初始化**。更换学生 seed 后，实测 499 个初始 state 张量中有 12 个不同，位置均在任务分类头 `model.23.cv3.*`。

IR 真实标注属于新增的训练期辅助信息，不能把本版称为“完全不使用 IR 标签的蒸馏”。旧方法 receipt 中的 `teacher_labels_used_by_kd=false` 不适用于本版。

## 三、实际算法逐步解释

### 1. 保持学生原生数据流，再重放几何变换

RGB 图和自身 GT 先执行一次原生变换；保存变换前后随机状态，IR 图和其独立 GT 重放相同的几何随机状态，最后恢复学生变换后的状态。这样添加教师支路不会额外推进学生的数据增强随机序列。

保留水平翻转、缩放、平移；关闭 mosaic、mixup、cutmix、HSV、Albumentations、旋转、错切与透视。它没有做学习式图像配准，也没有把 RGB 的框直接复制成 IR 标注。

### 2. 找到真正可以对应的目标

同图、同类别、两侧 GT 框 IoU≥0.5 才允许对应。使用一对一匹配，首先最大化有效匹配数，再用总 IoU 破平局；不是把两个标签列表按索引一一相连。

每个模态随后仍在自身 GT 框里取证据。这样可以容纳一定框位置差异，但不能解决大视差或没有同类目标对应的情形。

### 3. 提取目标相对局部背景的类别证据

在 YOLO 的 P3、P4 输出网格上，用正确类别的原始 logit 定义前景与背景：

- 前景：该模态自身 GT 框内的原生 anchor 中心。
- 背景：同中心、宽高各放大至 2 倍的外框中，排除该模态**全部** GT 的网格点，包括其他类别和未配对对象。
- 每层至少 1 个前景点、4 个背景点才有效；只对两侧共同有效的层求平均。

令 `LME(z)=log(mean(exp(z)))`，对象 i、模态 m 的证据为：

`e_i^m = mean_valid_levels[(LME(z_fg,c) − LME(z_bg,c)) / 2]`。

这里的 T=2 是在两项 LME 作差之后相除，**不是先把每个 logit 除以 T 再计算 LME**。它把一片区域压成一个正确类别的前景—背景相对判别标量，不是完整类别分布 KL，也不是全图特征 MSE。加入同一个常数到前景和背景 logit 不改变差值；这不等于不同模型的证据尺度已经校准。

输入 640 时，P3/P4 分别为 stride8/16 的 80×80、40×40 网格。P5 不进入蒸馏证据。DFL 框会在候选检查时被解码，但**没有 DFL、框位置或特征对齐的蒸馏损失**；共享骨干仍可能使类别蒸馏间接影响定位。

### 4. 定义基础集合 E：可对应、有区域、有 RGB 线索

一个对象进入 E，需满足前面的跨模态 GT 对应与有效区域要求，并且冻结 RGB 参考在 NMS 前存在：

`任意类别置信度≥0.05，且预测框与 RGB GT 的 IoU≥0.1`。

这一条件很宽松，允许类别尚错、位置尚粗或分数尚低的候选。它来自已训的**冻结参考**，不是当前正在训练的学生。

E 依赖 IR 标签和对应关系，不能称作“纯 RGB-only 候选集”；没有候选也只能表示当前参考未检出，不能证明 RGB 中物理上不存在可学信息。

### 5. 判断教师是否可靠，以及相对参考是否有优势

IR 教师的正确性代理要求在 NMS 前存在一个候选：argmax 类别与 GT 相同、该类别置信度≥0.25、与 IR 自身 GT 的 IoU≥0.5。这是训练期 GT 引导的候选检查，不是独立验证集标签，不是经过 NMS 和一对一评估得到的 TP。

相对质量排序值为：

`q_i = max(softplus(−e_i^R) − softplus(−e_i^T), 0)`。

正 q 表示按照这个局部判别代理，教师证据优于冻结 RGB 参考。它不是两侧原始 confidence 相减，也不是信息量或校准误差的测量。教师和参考已经见过训练集，因此该量不是独立、无偏的“可迁移性”估计。

### 6. 只选一部分对象，并保留分母

先得到 `eligible = E ∩ teacher-correct ∩ {q>0}`，然后在整个 batch 上按 q 降序取：

`K = ceil(0.5 × |eligible|)`。

相等 q 由稳定的 batch/GT 顺序破平局。没有按类别、目标尺度或亮度分层取 K；也不是“所有 E 中固定选择一半”。

因此，即使 rho 固定 0.5，真正参与蒸馏的比例和总剂量仍受教师正确性、q 和基础集合大小影响。作为明确例子，P42 最后一个 batch 的 receipt 记录 E=160、eligible=124、selected=62，名义剂量为 62/160=0.3875。这个单 batch 仅说明分母规则，不能充当整个训练的平均选择率。

### 7. 只增加一个判别证据损失

教师目标截断到 [−8,8]；使用 beta=1 的 SmoothL1：

`KD = Σ_selected SmoothL1(e_i^S, stopgrad(clip(e_i^T))) / max(1, |E|)`。

训练器实际总损失为：

`total = native_total.sum() + 0.1 × batch_size × KD`。

原生 `native_total` 已包含 batch 尺度，KD 只加一次。原生检测损失的权重为 box=7.5、cls=0.5、DFL=1.5；新增 KD 的 lambda=0.1。weight0 依然计算相同辅助支路、候选、选择和 KD，最终只把新增项系数设为 0，以便形成同代码对照。

这一版没有可训练 gate，没有亮度门，没有动态学生正确性门，也没有一旦学生超过教师便停止蒸馏的保证。因此“减少负迁移”是需要后续对照和失败子集分析检验的目标，不能由机制描述直接宣布实现。

## 四、所有当前 OEv1 长训共用的设置

| 设置项 | 实际冻结值与解释 |
|---|---|
| 数据 | DroneVehicle HBB 处理协议；RGB train 17,990，统一开发 val 1,469；本版不访问 test |
| 类别 | 5 类车辆；RGB/IR 类别顺序核验一致 |
| 模型与输入 | YOLO11n；640×640；通用预训练起点，任务头按框架构造 |
| 学生 seed | 0、42、123；各 paired/weight0 一次 |
| 教师/reference seed | 全部固定 42；没有覆盖教师训练随机性 |
| 固定预算 | 200 epochs；每轮 563 batches，总计 112,600 batches |
| batch / nbs / workers | 32 / 64 / 4；warmup 后通常约 2 batches 累积一次更新，不能把 nbs64 写成单次送入 64 张 |
| 优化器 | SGD；lr0=0.01，lrf=0.01，即最终学习率比例 0.01；momentum=0.937，weight decay=0.0005 |
| 调度 | 非 cosine；warmup 3 epochs，warmup momentum=0.8，warmup bias LR=0.1 |
| 几何增强 | translate=0.1，scale=0.5，fliplr=0.5；其余冻结关闭 |
| AMP/确定性 | AMP=true，deterministic=true；patience=0；禁止自动减 batch 或 NaN 恢复 |
| 训练中 val | 禁用；不按中间 AP 挑阈值、停臂或挑 checkpoint |
| 正式端点 | 固定 E200 的 last/EMA；完整 1,469 图开发 val 独立评估 |
| 主指标 | mAP50–95；配对差值使用百分点：100×(P−N) |
| 次指标 | AP50、AP75、precision、recall；不能用其中一个更好就替代主指标 |
| 新增知识参数 | P3/P4，T=2，局部背景宽高2倍，rho=0.5，lambda=0.1，target clip8，SmoothL1 beta1 |

相同 epoch 和 batch 预算不意味着 AMP 成功更新次数严格一样。P42 实际 receipt 为 56,722 次 update attempts、27 次 AMP skip、56,695 次成功更新；本轮应保留这些真实计数。

原生 DataLoader 的 generator 使用固定 seed。已做的 seed0/123 canary 中，首批输入及前 30 个 batch 样本顺序相同，而任务头部分初始化不同；三 seed 结果应解释为这个固定数据流条件下的初始化重复，不能声称已覆盖全部数据顺序/增强随机性，也不能由前 30 batch 推断全部 200 轮逐张量都相同。

## 五、具体有哪些实验臂，分别回答什么

| 实验/对照 | 设置变化 | 当前执行情况 | 能回答的问题 |
|---|---|---|---|
| OEv1 paired，P | 上述 IR 对象证据选择与 lambda0.1 | 已安排 seeds0/42/123，均固定 E200 | 整套干预有没有价值 |
| OEv1 weight0，N | 数据、辅助支路、选择不变，仅 KD 系数0 | 已安排 seeds0/42/123，均固定 E200 | 与 P 形成同代码净效果对照 |
| 同剂量随机，paired_random | 与 P 同 E、同 K、同 lambda 和分母，从整个 E 随机取 K | loss 算子和 CPU 检查已有；正式 trainer CLI 未开放，未长训 | 正确性/质量选择是否超过同剂量随机选择 |
| 全基础集合，paired_uniform | 不筛选，使用整个 E | loss 算子已有；未长训；剂量更高 | 全量基础集合迁移的结果；单独比较不能隔离选择与剂量 |
| same-modal | RGB 教师及 RGB 标签 | loss/loader 有局部支持；本轮没有冻结完整教师/内容/剂量协议，没有正式长训 | 跨模态内容是否超过同模态蒸馏 |
| shuffled / 实例 donor | 保留可比条件，再破坏配对内容 | 尚未冻结内容策略；代码显式拒绝 | 真正对应实例的信息是否有额外作用 |
| 同 mask GT-only | 保留选择，替换教师软内容为 GT 构造信号 | 后续需冻结；未运行 | 收益是否仅来自额外标签监督或难例加权 |

实际 `train_object_evidence.py` CLI 只允许 `paired` 和 `weight0`，不能因为 loss 文件列有更多 arm 就把它们写成已完成实验。尤其不能将普通全图 shuffled 导致 GT 无法对应、几乎零候选的结果，拿来“证明”配对知识有效。

现有六个长训任务的调度为：GPU4 上 P42→N42；GPU5 上 N0→P0；GPU6 上 P123→N123。同卡串行训练与终点评估，各 seed 的方法/对照共用同一配置；这个顺序兼顾了首批方法和对照覆盖，但不等于消除了共享服务器时段差异。

CPU 检查、真实 batch canary 是工程验证，不是新的科学效果臂。首轮 19 项 CPU 检查、两臂各 24 个成功更新 canary；扩展 seed0/123 也各做成对 24 更新 canary。它们验证初始化/首批学生输入成对一致、weight0 loss/score gradient 与 native 精确一致、KD 非零梯度且只加一次、教师/reference 不进 optimizer/EMA。

## 六、现有结果怎样对比

以下是 21:46 已保存的独立 E200 last/EMA 端点，全部单位为百分数。

| 臂/学生 seed | mAP50–95 | AP50 | AP75 | precision | recall |
|---|---:|---:|---:|---:|---:|
| P / 42 | 54.6582 | 77.0696 | 63.8393 | 77.4323 | 72.4305 |
| P / 123 | 54.6368 | 77.3046 | 64.2148 | 77.8713 | 72.9887 |
| N / 0 | 54.3462 | 76.9925 | 63.9329 | 77.6549 | 73.5822 |

此时 N42 完成135轮、P0完成28轮、N123完成20轮。上表是不同 seed 的已完成臂，**不能把两行 P 的均值减去 N0 当净收益，也不能用 N0 的 recall 更高来判定 P 降低召回**。真正主比较须逐 seed 收齐 P−N，再报告三个差值的 mean±样本SD与各自方向。

历史同 seed native 的 mAP 为 N0=54.357648、N42=53.815556、N123=53.687433；据此 P42 描述性高0.842606pp，P123高0.949415pp，而新 N0 与历史 N0 差−0.011463pp。两个 P 的历史参照正差值得继续观察；新 N0 接近历史 N0 也提供一个基线路径检查。但历史 native 不替代本版同代码 N42/N123，单个 N0 一致也不证明所有 seed 和完整数据流水线完全等价。

另一项已完成的历史结果是 CMDistill-adapted/corrected 相对专用 native 的三 seed 差值 −0.457、−0.571、−0.018pp，mean±SD为−0.349±0.292pp。它说明那个较全量的损失组合未带来净收益；不能单靠 OEv1 超过 CMDistill 就说新方法有效，仍须超过本版 weight0，并补归因对照。

即使六个 P/N 全部完成且三 seed 正向，也只足以支持“该冻结对象判别干预在当前协议下的净收益”。证明跨模态独特信息、选择价值、避免负迁移，仍需要 same-modal、shuffled/合理 donor、同剂量随机和必要的 GT-only 控制，以及亮度/大小/教师失败对象的改善与损伤分析。

## 七、实际版本与证据来源

本次直接字节比较确认：94 release_v2 本地副本的 `train_object_evidence.py`、`paired_rgbir_data.py`、`object_evidence_loss.py`、`config_drone.yaml` 与 P42 实际终态 receipt 中对应 source_snapshot 完全一致。未计算新的哈希。

- 冻结首轮计划：[EXPERIMENT_PLAN_20260906_frozen.md](../2026-09-06_train_RGBIR对象判别蒸馏首轮/EXPERIMENT_PLAN_20260906_frozen.md)
- 冻结三 seed 计划：[EXPERIMENT_PLAN_20260906_frozen.md](../2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/EXPERIMENT_PLAN_20260906_frozen.md)
- 实际发布源码：[release_v2]（未导出的工作区路径：../2026-09-06_ops_GitHub完整审计包/remote_snapshot_20260906/artifacts/rgbir_object_evidence_v1_20260906/release_v2/）
- P42 实际源码与终态证据：[full_paired_s42_attempt1](../2026-09-06_audit_RGBIR夜间结果与GitHub更新/oev1_snapshot/raw_runs/full_paired_s42_attempt1)
- 同期端点核验：[review_oev1_endpoints_complete.json](../2026-09-06_audit_RGBIR夜间结果与GitHub更新/review_oev1_endpoints_complete.json)
- 跨 seed 初始化/数据流检查：[cross_seed_realization.md](../2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/cross_seed_realization.md)
- 方法动机及诊断局限：[RGBIR数据特性与蒸馏方向诊断](../../07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md)

本地可编辑的 `EXPERIMENT_PLAN.md`、研究建议和代码中新增加的可选算子，不自动等于已部署设置。任何未来修订都应以新冻结记录、新 release/receipt 和新实验结果为依据，不倒写到正在运行的 OEv1 v1 结论中。
