# 给独立模型审计员的阅读指南

这份仓库供你复核研究过程、原始结果和当前方法代码。研究目标是：训练时使用配对的另一模态，提升部署时只输入一个模态的目标检测器，优先形成与 J-STARS 遥感应用相关、可归因的研究证据。请独立判断现有解释是否成立；维护者的分析、审查意见和方法名称都是待核对的材料。

本指南按 GitHub 仓库根目录编写链接。`research_bundle/` 保留本地材料的目录关系；仓库原有的 `00_project_context/` 至 `07_questions_for_auditor.md` 是较早审计包。历史原件保留用于追溯，**不能把旧入口中的强结论当作目前定论**。运行进度按各快照的采集时间读取；“排队”“运行中”和旧 epoch 数都不是实时状态。

## 1. 先修正阅读前提

| 容易沿用的旧表述 | 目前应采用的口径 |
|---|---|
| 当前研究单向 RGB→IR | 新 Object Evidence v1 主实验是 **DroneVehicle IR→RGB**；LLVIP 诊断也是 IR→RGB；VEDAI 诊断方向是 **RGB→NIR**。强弱模态按数据和任务定义。 |
| 监督 KD 从未净胜 native | 已核实的 DroneVehicle CMDistill bundle 在三 seed 开发评估中为负；HNEWA-inspired 的两个可配对 native seed 有小幅正差，但对 same-modal 的三 seed 差值不稳定。不能把有限方法矩阵扩成所有监督 KD 的否定结论。 |
| SpaceNet6 OS-SSL 是唯一强正例，因此移植必有效 | 它是一条有独立正面证据的历史路线；“唯一”取决于核查范围，跨数据集迁移要单独评估。OS-SSL-IR 已有独立实验记录，不能沿用“尚未开始”的旧状态。 |
| IR 整体 AP 更高，所以所有知识都适合蒸馏 | 教师整体强不代表每个对象的类别、定位或背景知识都更好，也不代表学生模态可重建其观测。 |
| 标签完全重合证明像素配准准确 | LLVIP 框由 IR 标注复制到 RGB；VEDAI 使用共享标签。两者的 GT IoU=1 不构成独立配准测量。 |
| 亮、暗两个桶都降点，所以所有门控都无效 | 这只描述某个方法、seed 和粗亮度划分的净结果，未检验对象选择机制。 |
| 新三 seed 已完成跨模态归因 | Object Evidence v1 当前设计只包含 paired/weight0 三 seed。最终 P−N 即使为正，仍需 same-modal、合理 shuffled 和与所声称机制相符的 null。 |

更正依据集中在[最新 RGBIR 诊断](research_bundle/07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md)和[现有结果复核](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/current_results_notes.md)。旧文件如果没有链接到最新勘误，请自行沿日期和源记录追溯，不能仅按文件名中的“final”“corrected”“accepted”判断可靠性。

## 2. 建议的阅读顺序

| 顺序 | 材料 | 希望你解决的问题 |
|---|---|---|
| 1 | [全项目复盘](research_bundle/07_研究分析/全项目复盘与研究诊断_20260905.md)、[最新 RGBIR 诊断](research_bundle/07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md) | 过去失败的证据、已撤回的解释与当前问题分别是什么？ |
| 2 | [现有结果复核](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/current_results_notes.md)、[原始来源快照](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/current_results_sources.json) | 方法/对照/seed/端点/单位是否可比，均值和 SD 能否重算？ |
| 3 | [六 baseline 探针记录](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/README.md)、[模型身份](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/dataset_model_inventory.md)、[特征图册](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/特征图册.md) | 数据对应、目标互补和特征差异究竟支持什么，不能支持什么？ |
| 4 | [Object Evidence 冻结协议](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏首轮/EXPERIMENT_PLAN.md)、[方法代码](research_bundle/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_object_evidence_v1) | 文档定义与真实 loss、loader、trainer 是否一致？ |
| 5 | [框架核查](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏首轮/framework_notes.md)、[首轮工程证据](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏首轮/README.md)、[三 seed 扩展](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/README.md) | weight0 是否真正等价，运行和结果证据是否闭合？ |
| 6 | [端点汇总规则](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/ENDPOINT_ANALYZER_NOTES.md)、[跨 seed 实现核验](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/cross_seed_realization.md) | 已获得哪些完整端点；三 seed 实际覆盖了什么随机性？ |
| 7 | [文献调研](research_bundle/01_文献/RGB-IR_20260905新增/00_方法调研综述.md)、[精读笔记](research_bundle/01_文献/RGB-IR_20260905新增/文献精读笔记_20260905.md)、[OS-SSL-IR 记录](research_bundle/08_实验日志/2026-09-06_train_OS-SSL-IR迁移/README.md) | 当前方法与已知方法有何实质区别，哪条后续路径最值得投入？ |

文献笔记是研究过程材料，包含后来修正的解释。新诊断已指出 CCLKD 包含定位分布知识、CMDistill 包含响应 KD，不能据早期笔记主张“首次跨模态定位蒸馏”。若要判断新颖性，请回到出版原文或作者材料，区分论文主张、我们的实现和我们的推断。

## 3. 可直接复算的证据

### 现有完整评估

旧仓库的 [DroneVehicle 原始评估](02_raw_results_dronevehicle) 与 [HNEWA-inspired 评估](04_hnewa_eval_records) 仍是有效的原始输入；最新解释和补充来源见上述 `current_results_*` 文件。

请逐项核对 dataset、split、类别、checkpoint、训练预算、初始化路径、方法身份和指标字段，再按相同 seed 求差。原始 AP 若为 0–1，先乘 100 转为百分点；差值 SD 使用 `ddof=1`，不能用两臂 SD 相减。特别检查 precision 是否被误当作 mAP、`best` 是否混入固定 `last/EMA`、训练 CSV 是否被当成独立 final 评估。

可作为复算校验值，但不要只抄这些数字：

- DroneVehicle CMDistill−W1 native 的 mAP50–95 差值约为 −0.4574、−0.5707、−0.0183 pp，均值 −0.3488，样本 SD 0.2918。
- HNEWA-inspired paired−same-modal：DroneVehicle 约 +0.2068±0.3597 pp，LLVIP 约 +0.0909±0.5920 pp，均为 2/3 seed 正向。paired−shuffled 回答的是另一问题，不能替代该比较。
- HNEWA 的 native 独立 final 来源只有部分 seed 可明确对应；不能随意借 W1 或新 weight0 填满矩阵。

`b32a2` 与 `b32_e200` 等目录名不证明预算不同，实际 args 已显示主要 recipe 相同。历史 N/L 的精确运行契约仍有未闭合项，但不能因此把已有三 seed 比较一概废弃。

### baseline 特征与对象互补

全部新探针位于[探针目录](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识)：三个 `*_full/` 子目录各含 `summary.json`、`prediction_records.json`、`matched_objects.json/csv`、`image_metrics.*`、`feature_metrics.*`、样本清单和运行回执；[probe_analysis.json](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/probe_analysis.json) 与 [CPU 复算脚本](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/analyze_probe_outputs.py) 保存后续错误细分。请使用正式 `*_full`，不要将 canary 或失败 attempt 计入样本。

范围是 **三个数据集、六个已核实的 seed42 YOLO11n baseline、521 对开发图像**：DroneVehicle 200、LLVIP 200、VEDAI official_fold01 全部 121。不是所有候选 RGBIR 数据集，也不是三训练 seed 的机制统计。

| 数据集 | 值得核验的机会 | 必须保留的限制 |
|---|---|---|
| DroneVehicle，IR→RGB | 3083 个共同对象中 IR 独有命中 360；其中 RGB 有严格低置信正确位置候选 118。双方命中时 IR−RGB IoU 均值约 +0.00790、IR 更准比例 49.46%。 | 有对象判别/评分机会，未显示普遍精定位优势；两侧独立标注和 HBB 转换必须追溯。 |
| LLVIP，IR→RGB | 672 个共同对象中 IR 独有命中 107；双方命中时 IR−RGB IoU 均值约 +0.04197。 | 共享框不能独立验证配准；“IR 更贴近 IR 来源标签”与真实跨模态定位优劣需区分；学生缺观测不能靠高置信教师自动补齐。 |
| VEDAI，RGB→NIR | 364 个共同对象中 RGB 独有命中 41、NIR 独有 31；RGB 整体更强但双方命中定位不更强。 | NIR 不是热红外；只有 121 张 val；test 别名不是独立封存集；paper80 教师与该 val 重叠 95/121，已排除。 |

命中矩阵采用固定 conf/IoU、一对一匹配，它不是 AP、学生可达上限或预计 KD 增益。CKA 必须中心化并看有效样本量；paired−donor 同时受类别、场景和位置变化影响。独立归一化的激活图不能跨图比较绝对强度，同编号通道也不保证同语义。phase correlation 是跨模态几何代理，低响应时不能据其位移判定严重错配。

## 4. 当前代码真正检验的假设

Object Evidence v1 以 RGB 为部署学生，冻结 IR 教师和 RGB 参考；使用训练期两模态各自 GT 建立同类对象对应。在 P3/P4 提取正确类别 logit 的“前景相对邻近背景”证据，选择教师代理质量更好、参考学生存在粗候选的对象，用 SmoothL1 迁移这一标量。学生仍接受原生 RGB GT 检测监督。

请重点核查以下边界：

- **新增标注信息**：IR 独立训练标签参与对象匹配、背景排除和教师质量过滤。它是训练期辅助信息，不能沿用旧方法 `teacher_labels_used_by_kd=false` 的声明；需要能隔离标注贡献的对照。
- **基础集合 E**：它包含跨模态对应和区域有效条件，虽然还要求冻结 RGB 参考候选，但不是纯 RGB 定义的集合。
- **选择和剂量**：`q` 是局部判别 margin 的代理差，未证明校准或物理可学性；选择 `ceil(0.5 × eligible数)` 个对象，分母为 `max(1, |E|)`。因此有效 KD 剂量会随 eligible 变化；仅设相同 λ 不保证 random/uniform 对照剂量等价。
- **标量相对证据的内容**：仅使用 GT 正确类别，可能主要表现为额外前景/背景或难例监督。它是否保留 IR 特有、非 GT 可替代的信息，需要同 mask 的 GT-only/常数目标及同 K 随机选择等对照检验。
- **历史广播计量问题**：框架核查发现 pinned 原生检测 loss 是三元素向量，旧 criterion 将标量 KD 加到向量后再求和，会使 KD 有效系数成为名义值的三倍。新方法先 `native_total.sum()` 再只加一次 KD。代码事实不等于历史负结果已找到唯一原因；新方法的收益也不能只归因于新知识而忽略剂量修正。
- **实现范围**：正式 launcher 只跑 paired/weight0；loss 模块的其他算子存在，不等于其训练协议已冻结、完整实验已执行。shuffled 和 GT-only 策略需要先定义清楚。
- **随机性范围**：IR 教师与 RGB 参考固定为 seed42；学生 seed0/42/123 改变初始化的随机部分。已核查 seed0/123 的首批输入和前30批顺序相同，来自 pinned 原生 DataLoader generator。不能宣称覆盖了教师 seed 方差或完整数据流随机性，也不能仅据首批相同否定条件初始化重复。

已通过的 CPU/真实 batch canary 检查支持梯度、数据流、零权重等价和资源约束；它们不证明方法有效或已消除负迁移。完整 E200 必须等固定端点评估。训练期间关闭验证，CSV 的 AP=0 是占位，不能列入性能表。

## 5. 希望你的审计产生什么

请给出可核查的判断和按信息价值排序的最小后续实验，重点围绕以下问题：

1. **知识内容**：现有对象证据是否比绝对类分数、前景特征、关系或定位分布更契合已观测错误？它是否有超出训练 GT 的实际信息？
2. **选择机制**：冻结参考候选、教师正确性、相对质量排序分别排除了什么？如何用同 K、同 E、同分母和可比梯度剂量的 random/null 区分选择价值？
3. **负迁移**：如何同时量化 teacher-correct/student-wrong 的修复、student-correct/teacher-wrong 的损伤、双方正确对象退化和背景误检？总体 AP 正值不足以声称避免负迁移。
4. **配对归因**：同类/同尺度实例 donor 如何定义以免 shuffled 只是破坏几何？same-modal、GT-only 与 IR 标签辅助控制各自检验哪个替代解释？
5. **数据集选择**：DroneVehicle 能否作为遥感主线；LLVIP 的机制补充是否充分；VEDAI 方向反转是否值得做；FLIR/M3FD/KAIST 是否确有已冻结划分和 baseline，还是增加了无关工程成本？
6. **投稿贡献**：基于当前证据，最窄可支持的主张是什么？哪些需要补证据，哪些已被现有结果反驳？请不要把“组合多个已知 loss”“多数据集”“更严格实验”单独当作新算法贡献，也不要承诺投稿成功。

建议返回：结论、来源表、复算结果、按严重程度排序的代码/协议问题、替代解释、最多三个优先实验，以及明确的继续/停止条件。每个重要判断引用仓库文件路径或 JSON 字段，说明你实际读了什么、没读到什么。缺文件标为缺证据，不自行补数。

可直接把[独立审计提示词](REVIEW_PROMPT.md)与本仓库分支 URL 交给另一模型。
