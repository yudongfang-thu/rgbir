# RGB–IR 任务条件跨模态蒸馏：调研、研究策略与 Codex 实施规格

> **文档日期：2026-09-07｜设计版本：DRAFT-v1｜对象：研究负责人及本地 Codex coding agent**
>
> **目标：**保留当前 Object Evidence v1（OEv1），先验证一个受限的定位蒸馏分支是否提供额外价值；只有证据成立，再发展为对象级、分任务选择的统一跨模态蒸馏方法。
>
> **本文不是已实现方法或已批准的长训配置。**文中的新阈值、接口、目录和命令均为实施建议；示例参数没有经过性能验证。不得据本文改写历史实验、覆盖正在运行的配置，或宣称新方法已经有效。
>
> **代码依据：**`yudongfang-thu/rgbir` 的 `research/full-evidence-20260906` 分支；本轮对接依据固定在已审计提交 `c6073b9506058d9a12e4d9719614ca7e61f950b0`。这不是要求回退到旧提交。Codex 开工时应记录实际 HEAD，比较相关文件变化，以现场源码和协议为准，不能仅按本文行号机械打补丁。
>
> **执行边界：**默认交付代码、CPU 测试、只读诊断和待批准配置。真实 GPU canary 和正式训练必须遵守用户已有授权与现场资源规则；没有对应授权时，产出就绪状态和阻塞原因，不自动启动训练。

---

## 阅读与使用顺序

研究负责人优先阅读第 1–4、12、16–17 节；Codex 先阅读第 0 节，然后按第 5–15 节实现。第 18 节给出分批提交任务，第 19 节是可直接交给 agent 的任务文本。附录 A 给出定位分布损失核心算子；它不替代配准、对象关联和完整 trainer。

全文使用以下状态：**已有事实**表示来自项目协议、源码或原始诊断；**设计建议**表示本轮提出且尚待验证；**需另行验证**表示不能由现有材料推出。论文相关判断是研究建议，不是 J-STARS 的录用保证。

## 0. 给 Codex 的开工摘要

**第一步不是把两个损失加起来，而是确认真正要帮助的对象，以及它们在可对应的预测位置上是否有可靠教师优势。**

本轮工作分三层，顺序不得颠倒：

| 层级 | 交付物 | 暂时不做的事情 |
|---|---|---|
| A：诊断与接口 | 定位缺口报告、对象与 anchor 对应、几何契约、代码测试 | 不启动大范围超参搜索，不以探针直接宣称增益 |
| B：最小方法 | 原 OEv1 不变，新增独立 L 分支；支持 N/C/L/C+L | 不同时改 C 的选择规则、不加动态参考/分解网络 |
| C：论文方法 | 根据 B 的结果决定是否实现统一任务路由与关键控制 | 不把普通 C+L 包装成已证明的新路由方法 |

必须保持以下契约：

1. RGB 是当前部署输入；IR 教师、RGB 参考和辅助标注只在训练期使用。
2. 冻结教师和参考；学生保留完整 RGB 原生检测监督。
3. C 分支严格复用 OEv1 的知识定义、选择、温度、归一化和权重。
4. L 分支先限制为经过独立几何核验、同物理 anchor、同 stride、同 DFL 支撑的对象。
5. 先对原生 loss 求和，再单次加入每个 KD 标量，不能广播到三项原生 loss。
6. 不用当前学生预测决定首版 L 的选择集合；选择由冻结 T/R 与训练标签构造，并停止梯度。
7. 两个任务可以同时学习，也可以都不学；不强迫每个对象只能二选一。
8. C+L 只比 C 多一个 L 干预。所有额外路径对 RNG、数据、EMA、初始化和预算的影响都要核验。
9. 不把历史 native 或其他 campaign 的近似数字填进新矩阵。
10. 本文中的目录是建议新增目录，命令是待实现接口；不存在时先实现，不能假装已经运行成功。

**首批任务到 CPU 测试和只读诊断为止。**满足科学前提与工程验收后，才生成正式冻结配置；没有定位机会时，应如实报告，而不是放松所有门槛以保证能训练。

---

## 1. 我们正在做什么，拟增加什么

### 1.1 当前 OEv1 的准确身份

OEv1 是 **Object Evidence v1，对象级判别证据蒸馏**。当前主任务为 DroneVehicle IR→RGB：冻结 IR 教师 T、冻结 RGB 参考 R、训练 RGB 学生 S。学生从通用预训练权重开始，不是从 R 接着微调。[P1–P3]

每个对应对象使用正确类别 logit，分别汇总标注框内前景和邻近背景，构造一个相对证据标量。通过教师正确性与参考模型线索检查后，选择一部分对象，让 S 的证据靠近 T。原生 RGB 分类、框和 DFL 监督保留。

**当前 OEv1 不是“学生定位准确才蒸馏分类”。**其参考候选仅要求任意类别 conf≥0.05、IoU≥0.1，表示粗线索，而非准确定位。其目标也不是完整类别概率向量或校准置信度。[P1]

### 1.2 拟增加的 L 分支

L 要回答：**对参考模型类别判断可靠、定位有缺口的对应对象，教师在学生坐标系及实际监督位置上是否提供了更好的定位知识？**

这里的“类别准、定位不准”首先描述冻结参考 R 的状态。不能仅因为 R 的类别正确，就推断教师定位有用；还必须检查教师的几何优势和对应可靠性。

### 1.3 统一研究问题，而不是两个松散模块

可采用的主问题是：

> 跨模态教师的优势随对象和检测子任务变化。如何分别判断判别与定位知识的可用性，只传递适合该对象、该任务的知识，并避免覆盖学生原本更好的判断？

这只是研究问题，不是已经证明的新颖性。分类/定位解耦、分任务选样、按教师质量加权及定位分布蒸馏都有先例。[R1–R10]

| 参考 R 的状态 | 教师需满足的条件 | 候选干预 |
|---|---|---|
| 框较好，类别错误或前景证据弱 | 判别更可靠，对象对应成立 | C |
| 类别可靠，框较差 | 同坐标及实际 anchor 上定位更准 | L |
| 两项都弱但有可用线索 | 两项知识分别可靠 | C 与 L 可同时使用 |
| 参考已较好、教师更差或对应不可靠 | 无可靠任务优势 | 不加该任务 KD，保留原生 GT 监督 |

**阶段 B 与阶段 C 必须分开命名：**`oev1_plus_loc_v1` 只验证增加定位知识的边际价值；`task_conditional_v2` 才可能重新定义两分支选择规则。不得在阶段 B 中偷偷修改 OEv1，再用旧 C 结果当消融。

---

## 2. 文献调研与创新边界

### 2.1 与本项目最接近的工作

| 文献 | 已有内容 | 与我们的重合点 | 实施与写作要求 |
|---|---|---|---|
| **Task Adaptive Regularization，2020** [R1] | 根据任务先验选择性迁移特征、分类和回归知识；不盲目复制所有教师信息 | “学生哪里弱就补哪里”已有先例 | 不能把任务缺口思想称首创；进一步比较参考定义、共享 proposal 和选择依据 |
| **TDKD，ACM MM 2020** [R2] | 分类与回归蒸馏解耦，对不同任务使用不同样本 | 分任务蒸馏和分任务选样已有先例 | 不能把两个 loss 或两个 mask 单独称为两个创新点 |
| **LD，CVPR 2022** [R3] | 定位分布蒸馏与有价值定位区域 | DFL/边界分布模仿已有成熟基础 | LD 应作为定位算子与强基线来源，不是本文的新算子 |
| **TBD，预印本 2022／出版版本见原报告** [R4] | 分类分数与定位质量协调，任务解耦特征蒸馏 | 分类好不代表定位好不是新观察 | 需说明跨模态对象对应与任务优势选择的额外区别 |
| **CoLD，TGRS 2023** [R5] | 光学教师指导 SAR，类别导向候选划分、教师 IoU 加权定位知识 | 跨模态＋类别引导定位＋定位质量权重高度接近 | 重点比较“教师绝对质量”与“教师对参考的任务相对优势”、坐标与标签处理 |
| **CrossKD，CVPR 2024** [R6] | 跨检测头预测模仿，处理 GT 与教师预测监督的冲突 | 多加教师输出损失可能与原生监督冲突 | 需检查梯度与错误变化；不能声称双头 loss 自然互不干扰 |
| **CMDistill，J-STARS 2025，在线 DOI 为 2024** [R7] | 特征、关系、框和类别响应的联合跨模态蒸馏 | 同刊已有框＋类别跨模态知识迁移 | 项目版本为 adapted，不能冒充精确复现；错误剂量版本不能是唯一对手 |
| **TID，Neurocomputing 2025** [R8] | 综合分类与回归输出估计学习状态，选择重要与薄弱区域 | 根据学生双任务状态选知识已有直接先例 | 不能仅凭“联合分析学习状态”主张新颖 |
| **Protocol-consistent Sigmoid-τ，IVC 2026** [R9] | Sigmoid/BCE 一致分类蒸馏；教师/学生任务分配差异构造分区，监督框与 DFL | 与现代 YOLO 中“分类KD＋分区定位KD”非常接近 | 优先审查原生 assigner 分区；不能把 Softmax 类别 KL 偷换进 Sigmoid 头 |
| **CCLKD，2026，出版信息见原报告** [R10] | 跨模态多层与类别约束；上轮正文调研指出包含候选框定位分布知识 | “首次跨模态定位分布蒸馏”不成立的风险 | 本轮未重新取得完整正文；不得未经逐式核查就把其分布等同逐边 DFL bins |

**本轮核验深度：**[R1/R3/R4/R6] 重新核对作者预印本摘要；[R2] 核对 ACM 官方目录与作者说明；[R5/R8/R9] 核对出版平台可访问摘要/引言；[R11] 核对期刊官方作者说明。[R7/R10] 保留上一轮报告的文献线索，本轮未重新获取其完整正文，不能把它们的具体公式作为直接实现规格。所有文献均未在本轮重新训练复现。

### 2.2 哪些不能算新贡献

以下内容可以是必要技术，但单独不足以支撑方法新颖性：

- 增加分类损失和定位损失；
- 用 IoU 或置信度筛选教师输出；
- 使用 KL 匹配定位分布；
- 使用分类/回归两个不同 mask；
- 在 YOLO 上把已知 KD 换到 RGB–IR 数据；
- 将“质量门控”“任务感知”“解耦”等词拼成新名称。

### 2.3 值得争取的具体差异

建议围绕三个可证伪问题组织技术，而不是先确定论文标题：

**对象对应：**独立模态标注、原始视差和增强后坐标会影响定位监督的意义。先构造可靠的共同对象与物理参考点，而非默认任意师生 anchor 可以对应。

**分任务优势：**同一个教师可以在某对象的判别上更强、定位上更弱。使用分任务的质量比较，分别决定哪些知识值得传递，而不是用同一个总体置信度给两任务统一加权。

**拒绝无益监督：**教师没有对应任务优势、学生在该任务已足够好或定位对应不可靠时，不施加相应 KD。拒绝机制必须通过同剂量控制检验，不能把“蒸馏得少”当成“选得对”。

这三点组成一个研究方向，**仍不等于未被已有工作覆盖的创新组合**。代码实现前后都需与 CoLD、Task Adaptive/TID 和分区 DFL 方法逐项对照。严格实验是证明贡献的手段，不自动成为算法贡献。

### 2.4 精读与复现优先级

先精读 CoLD、LD、Sigmoid-τ，再核对 Task Adaptive/TID 的选择依据。第一轮不复现全部文献；优先形成一个现代 YOLO 上、数学与代码一致的简单分类KD＋LD基线，以及一个真正有效的同模态基线。跨架构或原始数据转换做不到完全一致时，明确写 `adapted`，保留适配差异。

---

## 3. 现有证据说明什么，不能说明什么

### 3.1 当前定位统计不能直接否定新分支

既有诊断在 DroneVehicle 的 2505 个双方命中对象上得到 IR−RGB IoU 均值约 +0.00790、中位数 −0.00122，IR 更准比例 49.46%。它说明 IR 不是对所有已检出对象都更擅长定位。[P4]

但“双方命中”本身要求两侧同类 IoU≥0.5，因此许多拟帮助的“类别正确、IoU<0.5”对象没有进入这组统计。**必须测拟作用人群，而不是用双方已经定位尚可的对象替代定位缺口人群。**

LLVIP 的双方命中定位优势更大，但其复制标签来源会影响解释；VEDAI 是 RGB–NIR，不能写成热红外，也不能把现有 test 指向 val 的别名当独立测试。[P4]

### 3.2 现有 OEv1 正面线索不是 L 的证据

上次结果审计采用 2026-09-06 21:46 的快照，OEv1 两个 paired 端点约为 54.658/54.637 mAP50–95，但当时完整同 seed 对照尚未齐备。[P5]

这段只说明本文设计时的背景，不代表服务器现在的实时状态。本轮没有再次读取全部最新端点。Codex 应先读取实际 `LATEST_RESULTS.md` 与独立评估；即使 C 已被新结果证明有效，也不能由此推断 L 必然有效。

### 3.3 三个待检验假设

| 编号 | 主张 | 需要什么证据 |
|---|---|---|
| H1：有机会 | 类别可靠、定位不足且可对应的对象上，T 在学生坐标和实际监督 anchor 上有优势 | 冻结 T/R 的对象级、anchor 级诊断 |
| H2：可利用 | L 比 N 好，或者 L 在 C 上提供正的边际收益 | N/C/L/C+L 的同 seed、同预算独立端点评估 |
| H3：有独特机制 | 任务选择和教师具体内容优于简单联合 KD、额外 GT 监督和强同模态 | 针对主张的关键控制，而非仅 paired>weight0 |

通过 H1 只是获得方法动机；H2 只是证明该设置下的干预效用；H3 才关系到“任务条件跨模态知识”的归因与论文差异。

---

## 4. 总体策略与明确不做的事情

### 4.1 主路线

`只读定位机会诊断 → CPU 与真实 batch 功能验收 → N/C/L/C+L → 关键归因 → 方法冻结后的独立测试`

首轮任务名建议为 `RGBIR-OEV1-PLUS-LOC-v1`。不要直接用已经具有结论含义的名字，例如“无负迁移蒸馏”“最优任务路由”。

### 4.2 首轮非目标

不做共享/私有特征分解、不加新的 projector、不训练配准网络、不在线训练门控器、不引入生成模型、不同时引入 OS-SSL 初始化、不动态切换教师，不因验证集 AP 调整阈值或换 best checkpoint。

OS-SSL 是独立研究线。它的初始化一致性、RGB-only 控制及有效随机性问题应单独处理，不能与本轮 C/L 的因果比较混合。

### 4.3 对称不是互斥，也不是天然一致

首轮 C 沿用 OEv1，L 单独选样。同一对象允许 C-only、L-only、C+L、neither。两个 loss 均可影响共享骨干；检测头输出分开不意味着参数梯度完全隔离。

“RGB 物理上没有信息”和“当前参考没有预测”不是同一结论；没有候选时标记 `no_reference_candidate`，不能据此宣称不可学习。

---

## 5. 工程与数据契约

### 5.1 先确认实际工程，不在证据仓库盲目新建训练栈

GitHub 中 `research_bundle/` 是证据镜像，不保证可以离开原训练工程直接运行。现场训练工程历史位置是 `SpaceNet6_OTD_official_reproduction`，方法子目录为：

```text
experiments/rgbir_object_evidence_v1/
    object_evidence_loss.py
    paired_rgbir_data.py
    train_object_evidence.py
    evaluate_object_evidence.py
    config_drone.yaml
```

Codex 需要先定位实际可导入的模块和环境，记录 `module.__file__`、实际包版本、工作树状态与有效配置。不要只修改镜像中的同名代码却运行另一份。

新代码建议放在独立目录 `experiments/rgbir_task_conditional_v1/`，复用旧模块时采用明确导入或版本化快照，不直接修改历史 OEv1 文件。禁止升级 Ultralytics/PyTorch 来“顺便修复”接口，再把新旧结果并表。

### 5.2 原始预测接口

当前对接约定为：[P2–P3]

```text
student_raw / teacher_raw / reference_raw:
  scores: [B, C, A]             # Sigmoid/BCE 类别头的原始 logits
  boxes:  [B, 4 * reg_max, A]   # DFL 原始 logits，不是 xyxy
  feats:  (P3, P4, P5)         # 用于核对空间网格和 anchor 排列

RGB batch:
  img, batch_idx, cls, bboxes  # bboxes 为 normalized xywh
IR batch:
  strong_img, teacher_batch   # 独立 IR GT，经相同几何规则处理
```

不得按名字把 `boxes` 当作解码框。`scores` 的类别维不能直接套分类任务的 Softmax KL；定位分布的 Softmax 是在每条边的距离 bins 上，与类别 Softmax 不是一回事。

所有接口必须检查形状、dtype、device、类别顺序、有限性以及 anchor 数。禁止用 `nan_to_num` 把真实数据/损失错误变成零损失后继续训练。

### 5.3 一对一对象身份

匹配对象时优先使用数据提供的可信实例对应；没有实例 ID 时，使用同类双模态 GT 的几何对应。延续项目已验证的规则：先最大化达到阈值的有效匹配数量，再用总 IoU 破平局，而不是先做总 IoU 匹配后才删低 IoU 边。[P2]

一个实现方式是对有效边赋值 `M + IoU`，无效边赋零，其中 `M > min(n_rgb, n_ir)`，再做最大权匹配并保留有效边。类不同的边必须无效；空标签应合法返回空集合。

对象 ID 必须包含 `image_id + rgb_gt_index + ir_gt_index`。不能在过滤/排序后用数组当前下标当持久 ID。增强导致框删除后，需保留原始对象 ID 或明确保存增强后 ID 的对应表。

### 5.4 几何转换与两类“可靠”

必须分开：

- **对象关联可靠：**两侧标签指向同一对象。
- **定位分布可比较：**用于 KD 的预测位置、坐标单位与离散支撑具有相同物理含义。

前者成立不保证后者成立。标签 IoU 高、甚至标签文件相同，也不能独立证明双传感器像素配准准确。

若原始映射为 `W_raw`，RGB/IR 各自从原图到训练输入的几何变换为 `A_rgb` 和 `A_ir`，则在齐次坐标下：

$$
W_{aug}=A_{rgb}\,W_{raw}\,A_{ir}^{-1}.
$$

该式包含 resize/letterbox、平移、缩放及翻转。只重放相同增强随机数，不代表原始 `W_raw` 为单位映射。

**首版 DFL 正式路径只支持经过证据确认的同坐标网格。**配置必须显式写 `geometry_mode=verified_identity_grid` 并提供几何核验记录。没有几何证据时，允许继续做诊断/CPU 算子测试，但禁止自动启动直接逐 bin KD。

标签框 IoU、中心相对偏差可以作为额外保守筛选，但不能代替独立几何检查。不得逐对象用两套 GT 拟合一个恰好把教师预测拉向 RGB GT 的变换，再把收益全归为教师知识。

若使用一般仿射/投影映射，解码框可先映射四个角再得到对应的 HBB；应记录包围框化带来的近似。DFL 距离分布还需要参考点、边方向和尺度的重参数化，**不能仅插值 logits 或 reshape**。这一拓展不属于首版。

### 5.5 标签辅助的诚实声明

OEv1 和新 L 都使用 IR 独立标注进行对应和质量检查。训练回执必须记录：

```yaml
student_native_gt_only: true
teacher_labels_used_by_kd: true
deployment_input: rgb_only
```

`student_native_gt_only` 仅表示原生检测监督来自 RGB 标签，不表示整个训练方法没有使用额外标注。

---

## 6. 实验一：定位机会诊断的具体定义

### 6.1 两份诊断，不能相互替代

**D1：检测对象级机会。**从固定 T/R 输出判断“RGB 类别可靠但定位不足”的对象上是否有教师优势。它回答是否有研究动机。

**D2：实际可蒸馏 anchor 级机会。**在准备施加 KD 的同物理预测位置重新判断优势。它回答首版算子能否利用该动机。

必须同时报告 D1 与 D2。教师在别的 anchor 上有好框，不证明当前选中的 teacher anchor 是好目标。

### 6.2 D1 的关联规则

使用冻结 T/R，原生模态输入及各自 GT。为了让“类别错误”仍能被统计，**预测与 GT 的诊断关联先按空间，不按正确类别筛掉预测**。固定低置信阈值、NMS 参数、最大候选数和一对一关联规则，记录原始 anchor ID（可获取时）。

建议空间关联采用：IoU≥0.1 的有效边，最大有效匹配数优先，再最大总 IoU；不使用后续蒸馏门控的结果来决定关联。关联完成后再读取该预测的类别、正确类分数和 IoU。

阈值首先用于描述，不是已知最优参数。原生 NMS 下的关联与 pre-NMS 候选必须分别标注；不要混用结果后把一个 pre-NMS 候选称为一对一 TP。

每个 GT 仅占一个主诊断条目。不存在候选时记录 missing，不能把缺失当作一个“分类正确、IoU=0”的观测。若另算 GT 周围最佳正确类别框作为 oracle，必须单列，不进入正式主诊断。

单类别 LLVIP 上，`argmax==person` 本身几乎没有筛选作用；还必须检查前景分数或另一项明确的可靠性条件。

### 6.3 状态分桶

建议诊断桶如下，全部由同一冻结参考 R 定义：

| 桶 | 条件（诊断建议） | 主要问题 |
|---|---|---|
| `class_ok_loc_poor` | 关联预测类别正确、前景响应达到固定阈值；IoU∈[0.1,0.5) | 教师是否能修复明显定位缺口？ |
| `class_ok_loc_mid` | 同上，IoU∈[0.5,0.75) | 教师是否有精定位优势？ |
| `class_ok_loc_good` | 同上，IoU≥0.75 | 是否应避免不必要定位 KD？ |
| `class_wrong_loc_good` | IoU≥0.5，但关联预测类别错误 | 判别知识是否有机会？ |
| `weak_response` | 有粗空间候选，响应低于可靠阈值 | 是否为评分/前景判别问题？ |
| `no_candidate` | 无预定义空间候选 | 单列缺失，不推断物理不可见 |
| `unreliable_geometry` | 对象/坐标核验不通过 | 不用于直接定位 KD |

这些诊断分桶不改变 OEv1。训练门控可以与分桶不完全相同，但必须事前说明和冻结。

### 6.4 D2 的首版 anchor 选择

首版推荐每个对象只选 **1 个**由 R 决定的 anchor，以减少多点归因与重归一化复杂性：

1. 取配置允许的尺度，首版建议 P3/P4，即索引 `[0,1]`。这只是为了与 C 对接、控制范围，不是基于“P5无价值”的断言。
2. anchor 中心位于该 RGB GT 内；在同坐标 DFL 模式下，同时核验相应 IR 对象内的合法支撑。
3. R 在该位置有粗候选，例如 `max sigmoid(score)≥0.05`、解码框对 RGB GT `IoU≥0.1`。
4. 对满足条件的 anchor，先处理拥挤对象归属：每个 anchor 只能分给一个对象，默认分给其 R 解码框 IoU 最大的 GT；精确平局依次按较小 GT 面积、持久 GT ID 处理。
5. 每个对象从自己拥有的候选中按 R 的最大类别分数降序选第一名；平局按 R IoU、全局 anchor ID 处理。**这个排序不依赖当前 S，不按教师收益挑 anchor。**
6. 未得到 anchor 的对象标记 `no_usable_reference_anchor`；不能为了补齐数量退化成任意位置。
7. 在最终这一位置读取 T 的类别与框，再做 L 质量筛选。教师别处的最好预测只作为 D1 描述。

这是一个保守的**实施草案**，不是对所有检测器通用的最优选点方式。若该方案覆盖率太低，应报告 D1 有机会但 D2 未找到可利用位置；后续改选点需登记为新版本，而非静默用教师最佳位置替换。

### 6.5 必需诊断字段与产物

```text
image_id, scene_or_sequence_id, rgb_gt_id, ir_gt_id, class_id
rgb_gt_xyxy, ir_gt_xyxy, image_size, coordinate_frame
gt_pair_iou, center_shift_relative, geometry_mode, geometry_valid
reference_candidate_exists, reference_pred_class, reference_true_class_score
reference_iou_to_rgb_gt, reference_anchor_id, reference_level
teacher_object_best_iou_to_rgb_gt                 # D1 描述项
teacher_same_anchor_iou_to_rgb_gt                 # D2 判定项
teacher_same_anchor_iou_to_ir_gt
teacher_same_anchor_pred_class, teacher_true_class_score
teacher_minus_reference_iou, localization_gate
brightness_bin, relative_area_bin, failure_reason
```

输出建议：

```text
artifacts/<campaign>/diagnosis/
  protocol.json
  sample_roster.json
  object_opportunities.csv
  anchor_opportunities.csv
  summary.json
  report.md
  fixed_case_ids.json
```

报告至少包括：拟帮助人群数量、教师优/参考优/双方弱数量、实际同 anchor 通过比例、几何筛选前后覆盖、类别/尺度/亮度分布、ΔIoU 的均值/中位数/分位数。图像相邻帧存在相关性时按场景/序列看分布，不能把每个目标当独立实验重复。

### 6.6 继续或停止

诊断没有一个通用的“至少多少对象就必然有效”门槛。至少需要确认：非空机会不只来自一两张异常图，实际监督位置有可信优势，主要类别或目标范围具有明确覆盖，几何核验没有系统性错误。

如果 D1 机会很大、D2 几乎为零，优先修正“监督位置是否取对”的设计，不要马上加大学习率或 KD 权重。如果可对应且有优势的对象确实极少，停止本版 L 的长训计划，保留负诊断，不对整个跨模态领域作否定结论。

---

## 7. C 分支：必须保持的 OEv1 定义

对对象 i、模态 m、有效尺度 l：

$$
e_i^m=\frac{1}{|V_i|}\sum_{l\in V_i}
\frac{\operatorname{LME}(z_{i,l,fg,c}^m)-\operatorname{LME}(z_{i,l,bg,c}^m)}{T_C},
\quad \operatorname{LME}(x)=\log\left(\frac{1}{n}\sum_j e^{x_j}\right).
$$

其中 `T_C=2`；温度是在两个聚合值相减后除，不是将 logits 先除 T 后再池化。两者一般不等价。[P1–P2]

前景为本模态 GT 内部；背景为同心两倍框内排除**本模态所有 GT**的区域。每个尺度至少 1 个前景点、4 个背景点，仅对两模态及参考共同有效尺度平均。

基础集合 `E_C` 包含真实对象对应、有效区域及 R 的粗候选。教师正确性和

$$
q_i=[\operatorname{softplus}(-e_i^R)-\operatorname{softplus}(-e_i^T)]_+
$$

决定 eligible 集合；按 q 稳定排序取 `ceil(0.5 * eligible_count)`。教师 target 截断到 `[-8,8]`；使用 SmoothL1(beta=1)，按 `max(1, |E_C|)` 归一化。C 权重保持 0.1。[P1–P3]

首版扩展中不得顺便修正 C 的 q、截断、采样、池化或区域定义。旧版本的潜在风险——例如未截断选择与截断目标的差别、冻结参考不等于当前学生、池化位置未必是原生正样本——先记录为诊断项。若以后修改，另开版本并重新做 C-only 对照。

---

## 8. L 分支：首版的完整算法契约

### 8.1 基础集合与质量比较

设 `E_L` 为**教师质量门控前**已经具有可靠对象对应、合法共同坐标支撑和预选 R anchor 的对象集合。`E_L` 与 `E_C` 不必相同，日志与分母必须分开。

每个对象 i 的预选 anchor 集合记为 `A_i^0`，首版大小为 1。R、T 与 S 在位置 a 的预测框分别是 `b_ia^R`、`b_ia^T`、`b_ia^S`。质量比较使用：

$$
u_{ia}^R=\operatorname{IoU}(b_{ia}^R,g_i^{RGB}),\quad
u_{ia}^T=\operatorname{IoU}(W_{aug}(b_{ia}^T),g_i^{RGB}).
$$

同一 RGB 目标框是两侧“适不适合给 RGB 学生做定位监督”的共同评判坐标。教师对自身 IR GT 的 IoU 另记为 `v_ia^T`，作为教师本模态正确性的辅助检查，不能用它直接替代 `u_ia^T`。

### 8.2 二值门控：先把研究因素减少到可解释

上一轮讨论过连续优势权重；**本文建议首版先用二值 mask**，避免同时引入“选哪些对象”和“优势如何连续加权”两个变化。连续权重保留为后续单独消融，不混入首轮。

$$
w_{ia}^{L}=r_i\cdot
\mathbf1[\hat c_{ia}^R=y_i,\ p_{ia,y_i}^R\ge\tau_R]
\cdot\mathbf1[\hat c_{ia}^T=y_i,\ p_{ia,y_i}^T\ge\tau_T^{cls}]
\cdot\mathbf1[u_{ia}^R<\tau_S^{loc}]
\cdot\mathbf1[u_{ia}^T\ge\tau_T^{loc}]
\cdot\mathbf1[v_{ia}^T\ge\tau_T^{own}]
\cdot\mathbf1[u_{ia}^T-u_{ia}^R>\delta_{loc}].
$$

`r_i` 是通过几何与对象对应检查的二值标记。所有类别、质量、mask 来自冻结 T/R 与训练 GT，不对它们反向传播。严格 `>` 或 `>=` 的边界行为要在测试与配置中固定。

**仅作 canary/设计草案的参数：**`τ_R=0.25`、`τ_T^cls=0.25`、`τ_S^loc=0.70`、`τ_T^loc=0.60`、`τ_T^own=0.50`、`δ_loc=0.05`。它们不是验证出的最佳值，也不是自动批准的正式配置。

教师分类正确是首版减少错误关联的保守要求，并不意味着“分类错的教师绝对没有定位知识”。日后取消这条要求是单独的研究改变。

### 8.3 首版定位知识：优先测试已对齐的 DFL 分布

对每边 `d∈{left,top,right,bottom}`，用同温度 `T_L` 在距离 bins 上计算：

$$
p_{iad}^{S}=\operatorname{softmax}(z_{iad}^{S}/T_L),\quad
p_{iad}^{T}=\operatorname{softmax}(z_{iad}^{T}/T_L),
$$

$$
\ell_{ia}^{L}=\frac{T_L^2}{4}\sum_d
D_{KL}(\operatorname{stopgrad}(p_{iad}^{T})\Vert p_{iad}^{S}).
$$

`T_L=2` 可以作为首版数值实现默认，不因验证 AP 搜索温度。它与 C 的温度作用位置不同，配置键必须分开。定位分布蒸馏有 LD 等先例，此处不主张 KL 算子本身新颖。[R3]

### 8.4 DFL 的硬性校验

必须同时满足：同物理 anchor 中心、同层 stride、同边次序、同 reg_max、同距离单位、同 bins 支撑、同增强后坐标。仅张量 shape 一样远远不够。

- 不可以将教师最佳 anchor 的 bins 与学生另一个 anchor 的 bins 直接相减。
- 不可以把对所有空间位置的 Softmax 当成四条边的分布。
- 不可以用 teacher expectation box 的 IoU 损失冒称 DFL 分布蒸馏。
- 不可以让不同网格输入自动 resize 后继续运行。
- 需要检查 GT 边界距离在当前 anchor 的可表示范围内；超出支撑不能静默 clamp 成一个看似有效目标。

同坐标条件下，共享随机翻转后的两侧仍应校验增强后网格。一般映射造成参考点偏移或边方向变化时，首版 DFL 应拒绝，不自动切换到另一损失。

### 8.5 框 KD 是独立备选，不是静默 fallback

为了先核验选点和梯度，可以实现 `loss_kind=box_giou`：对映射后的教师框与学生在预定位置解码的框做 GIoU 类损失。选择依据仍是冻结 T/R，学生解码必须保留梯度。

此版本只蒸馏点估计。若采用它做正式试验，应以独立方法 ID、独立 L/C+L 标记。**同一 run 内不允许因某个对象 DFL 对齐失败而自动改用 box KD**，否则无法清楚解释到底验证了什么。

### 8.6 对象均衡与分母

首版采用预选集合不随教师过滤重归一化的约定：

$$
L_L=\frac{1}{\max(1,|E_L|)}
\sum_{i\in E_L}\frac{1}{|A_i^0|}
\sum_{a\in A_i^0}w_{ia}^{L}\ell_{ia}^{L}.
$$

一个对象越大、anchor 越多，不应自动得到更多总权重。当前每对象 1 个 anchor 时公式简化；未来扩到多个 anchor 必须保留对象内平均。教师过滤后不除以新的通过数，也不除以 `sum(w)`，避免“选得少”被自动重新放大。

在密集张量实现中，`anchor_weight[b,a]=w_ia^L/|A_i^0|`，全局 `normalizer=max(1,|E_L|)`。每个 anchor 必须只有一个对象所有者。

### 8.7 总损失与梯度路径

$$
L_{total}=\operatorname{sum}(L_{native})+B(\lambda_C L_C+\lambda_L L_L).
$$

这里原生 loss 已按 pinned 框架包含 B 的尺度；不得再整体乘 B。当前 C 的 λ 保持 0.1；L 的正式 λ 见第 11 节。最后一个小 batch 使用实际 B，而不是固定配置的 32。

梯度应满足：T/R/标签/mask 均无梯度；L 的原始 DFL logits 有梯度；L 对学生 scores 的直接梯度应为 None 或零（选择已 detach）；C 的直接监督保持原定义。共享参数上的总梯度可同时受到两分支影响，不作“天然解耦无冲突”的承诺。

空对象、空通过集合时返回可导零，如 `student_boxes.float().sum()*0.0`；不能返回 Python 数字后破坏接口。输入非有限值时失败，而不是把 NaN 乘零当有效。

---

## 9. 与现有 trainer 的对接

### 9.1 已有接口的两个陷阱

在对接依据的 `train_object_evidence.py` 中：

- `combine_loss` 已先 sum 原生 loss 后加一个 KD 标量；新版本应扩展为两个标量，不能退回广播写法。
- `EvidenceCriterion.__call__` 内部调用 `object_evidence_loss(..., arm='paired')`，再通过外部权重区分 paired/weight0。因此仅给 CLI 增加 `L_only` 或 `C_plus_L`，并不会自动实现新分支。[P3]

新增明确的 `ArmConfig`，对每个实验臂分别定义 `lambda_C/lambda_L`、C 内容模式、L 内容模式和 selector 模式。禁止“选了某个字符串，但内部仍固定调用旧 paired”的静默失效。

### 9.2 建议的实验臂语义

| 新臂 ID | C 路径 | L 路径 | 最终权重 |
|---|---|---|---|
| `N` | 计算诊断 | 计算诊断 | C=0，L=0 |
| `C` | 原 OEv1 | 计算诊断 | C=0.1，L=0 |
| `L` | 计算诊断 | 新定位 KD | C=0，L=冻结值 |
| `CL` | 原 OEv1 | 新定位 KD | C=0.1，L=与 L 相同 |

首版 N/C 保留 L 的只读辅助路径，便于证明额外计算不改变学生输入、RNG 与原生监督。后续优化掉零剂量的昂贵路径，须先证明数值和数据流契约不变，不在主试验途中改。

### 9.3 伪代码

以下是待实现逻辑，不是已运行程序：

```python
native_total, native_items = native_criterion(student_raw, batch)

# Models are outside the exported student's module tree.
with torch.no_grad():
    teacher_raw = teacher(ir_images)
    reference_raw = reference(rgb_images)

# Preserve the exact OEv1 knowledge/selection definition.
loss_C, stats_C = original_oev1_loss(
    student_raw, teacher_raw, reference_raw, batch,
    config=frozen_oev1_config, arm="paired",
)

# Geometry, ownership, base population and masks use frozen outputs only.
selection = build_localization_selection(
    reference_raw=reference_raw,
    teacher_raw=teacher_raw,
    rgb_labels=rgb_labels,
    ir_labels=ir_labels,
    geometry=geometry_contract,
    config=localization_config,
)
loss_L = localization_kd(
    student_raw["boxes"],
    teacher_raw["boxes"],
    selection.anchor_weight,
    normalizer=max(1, selection.base_object_count),
    temperature=localization_config.temperature,
)

total = native_total.sum() + actual_batch_size * (
    arm.lambda_C * loss_C + arm.lambda_L * loss_L
)
```

`geometry_contract` 必须是经检查的对象，不是默认 `True`。teacher 输入、reference 输入以及学生输入不能因为实验臂不同而切换。解码 reference/teacher 时可 detach；解码 student 用于 box KD 时不可误用同一个 detach helper。

### 9.4 训练期模块生命周期

继续将 T/R 放在训练专用对象中，不作为学生 `nn.Module` 的可保存子模块。保持教师/参考 `eval()` 与 `requires_grad_(False)`；不要每 batch 重建模型，不向优化器注册它们。

沿用项目对 EMA/deepcopy 的处理，并检查最终导出结构。若新增 head 或参数（本版不需要），必须在 optimizer/EMA 建立前注册，不能首个 forward 才 `add_module`。

### 9.5 无损复用 N/C 的验收

在同一初始学生状态和同一真实 batch 上，验证：

- 新 N 与原生学生 loss、scores/boxes 梯度一致；
- 新 C 与旧 OEv1 的 C 标量、selected IDs、分母、总 loss 和共享参数梯度一致；
- 调用 L 诊断前后，学生 batch、标签与 Python/NumPy/Torch/CUDA RNG 不变；
- 多步更新后，新 N/C 没有因模块导入、随机数消费或 BN 状态产生新的变化。

同环境确定性比较优先做直接 tensor equality；有限精度允许误差时事前固定容差，不能看差值后放宽。即使单 batch 相等，也不能立即证明全程完全等价；还应核查初始化、数据流、EMA、更新预算和端点。

**如果新 wrapper 改变了初始化或数据顺序，历史 N/C 不能继续复用。**要么修复 wrapper，使其保持对照契约，要么将 N/C 一并纳入新矩阵。

---

## 10. 后续统一任务路由：只预留，不抢先实现

只有 L/CL 的净效用结果值得推进时，才讨论 `task_conditional_v2`。它不是本轮必须完成的“第二个创新模块”。

### 10.1 与首版不同的地方

首版 C 使用旧 OEv1；后续 v2 才可能重新定义 C 的资格，使其更贴近“参考定位已经较好，但类别/前景证据不足”的诊断。L 仍根据定位缺口和教师同位置定位优势选择。

两个权重各自判断：

$$
w_i^C = \text{判别对应有效}\times\text{参考判别缺口}\times\text{教师判别优势},
$$

$$
w_i^L = \text{定位对应有效}\times\text{参考定位缺口}\times\text{教师定位优势}.
$$

这是设计框架，不是完整冻结公式。尤其“参考判别缺口”不能随意把低置信度、错类、局部 margin 低混成同一数值后宣称校准。

### 10.2 需要比较的简单替代方案

在提出复杂路由前，至少对比：原 OEv1+L、教师绝对质量门控、带参考缺口的分任务门控。必须保持损失内容、初始化、训练预算、对象基础集合和剂量记录一致。

若只改变 mask，还应说明两个任务的资格集合本来就不同。所谓共享 mask 应定义在合法集合上；不能强行把不支持定位的对象也喂给 DFL，制造一个注定失败的对手。

### 10.3 严格对称只是待检验的特例

“C 只用于定位准对象、L 只用于类别准对象”可以作为一个严格对称版本，但可能排除两项都差且本来有机会的对象。因此不要一开始就规定全部目标只能走一个分支，也不要让“对称”比实际可用信息优先。

### 10.4 目前不建议的增加项

可学习 gate、动态学生 EMA 参考、跨模态特征分解、配准网络、梯度投影算法及多教师融合，都应等简单基线有稳定边际收益后再决定。共享梯度冲突的存在是诊断，不自动授权加入新的梯度手术组件。

---

## 11. 配置、参数与正式冻结

### 11.1 草案与正式配置分离

下面是建议 schema，不是现成可启动的正式 YAML。`null` 必须阻止正式运行，不能由代码悄悄填默认值。

```yaml
schema: rgbir-task-conditional-spec-v1
method_id: RGBIR-OEV1-PLUS-LOC-v1
protocol_status: DRAFT
formal_training_authorized: false

implementation:
  evidence_repo_commit: c6073b9506058d9a12e4d9719614ca7e61f950b0
  execution_repo_commit: null       # 现场记录，不强制回退
  native_environment_verified: false
  old_oev1_contract_verified: false

models:
  student_initial_checkpoint: null
  frozen_ir_teacher_checkpoint: null
  frozen_rgb_reference_checkpoint: null
  deployment_modality: rgb
  teacher_labels_used_by_kd: true

data:
  dataset: dronevehicle
  student_data_yaml: null
  teacher_data_yaml: null
  train_pair_manifest: null
  fit_diagnostic_roster: null
  development_roster: null
  test_access_allowed: false

geometry:
  mode: verified_identity_grid
  verified: false
  evidence_path: null
  unsupported_mode: error
  object_match_iou: 0.5
  extra_label_pair_iou_filter: 0.8  # 草案保守筛选，不是配准准确的证明

classification:
  mode: oev1_exact
  coefficient: 0.1
  temperature: 2.0
  target_clip: 8.0
  rho: 0.5
  change_existing_definition: false

localization:
  loss_kind: dfl_kl
  levels: [0, 1]
  anchors_per_object: 1
  proposal_policy: frozen_reference_score_ranked
  anchor_owner_policy: reference_iou_then_area_then_id
  selector: binary_task_advantage
  coarse_reference_conf: 0.05
  coarse_reference_iou: 0.1
  reference_class_conf: 0.25
  teacher_class_conf: 0.25
  reference_iou_upper: 0.70
  teacher_iou_min_student_frame: 0.60
  teacher_iou_min_own_frame: 0.50
  minimum_iou_advantage_exclusive: 0.05
  temperature: 2.0
  coefficient: null               # 正式运行前必须固定
  coefficient_canary_only: 0.1
  normalize_by: pre_teacher_gate_base_objects
  within_object_denominator: pre_teacher_gate_anchor_count
  gate_grad: false
  automatic_box_fallback: false

training:
  inherit_verified_oev1_recipe: true
  epochs: 200
  batch_size: 32
  nominal_batch_size: 64
  workers: 4
  student_seeds: [0, 42, 123]
  preserve_existing_data_rng_policy: true
  change_lr_or_schedule: false
  intermediate_ap_for_selection: false
  automatic_nan_recovery: false
  automatic_batch_reduction: false

evaluation:
  primary_metric: mAP50_95
  endpoint: fixed_budget_last_ema
  independent_evaluation_required: true
  expected_drone_val_images: 1469  # 在该既有协议下核验；换数据集必须另配
  overwrite_existing_results: false

execution:
  phase: cpu_only
  resume_into_existing_run: false
  overwrite_existing_run: false
  resource_policy: inherit_current_project_AGENTS
```

不能让固定 DroneVehicle 的 `1469` 成为跨数据集通用常量。切换数据、标签版本、split 或教师，应生成新的协议 ID。

### 11.2 λ_L 怎么定，而不是怎样搜最佳 AP

首轮不要在开发 AP 上做大范围网格搜索。可在预先固定的训练 batch 上做一次梯度尺度校准，并登记为设计步骤。建议使用固定 R 状态的学生副本或预先登记的 warm-start 校准状态，仅用于梯度测量，正式学生初始化仍保持原协议；不能把用于校准的副本接着当正式学生。

例如，选择一组共同参数 `θ_A`，定义：

$$
g_{det}=\nabla_{\theta_A} \operatorname{sum}(L_{native}),\qquad
g_{L,unit}=\nabla_{\theta_A}(B L_L).
$$

在固定、未用于挑 AP 的训练 batch 上测每批 `||g_det||/||g_L,unit||`。可提出一次性目标比例，例如中位数 KD/原生梯度比为 0.1，再得到 λ_L。**0.1 是工程设计目标，不是文献证明的最优比例。**校准参数子集、batch 数、裁剪上限和无有效样本处理都要先登记。

如果全部 batch 的 L 梯度为零，返回 `NO_LOCALIZATION_SIGNAL`，禁止除以 epsilon 得到巨大 λ。若需要极端 λ 才达到目标比，应先审查覆盖率与归一化，不自动补偿到任意大值。

同一个 λ_L 用于 L 和 CL；不能分别调到各自更好的 AP。正式冻结后不自动随 epoch 调权；若以后研究日程或动态剂量，那是新实验。

### 11.3 “同剂量”的三个层次

同 λ、同入选数量与同实际梯度强度并不等价。记录三个层次：名义系数、基础/入选比例、实际共享参数梯度范数与方向。

用于比较内容或选择时，首先固定 λ、分母及集合/数量定义，避免机械改变优化目标。若额外做梯度匹配的对照，需要作为独立协议说明，它也会改变训练干预，不能悄悄通过动态归一化把所有方法变成一样。

---

## 12. 最少但能改变决策的实验与控制

### 12.1 实验组一：D1/D2 定位诊断

不训练新方法。先读取现有 200 对诊断样本；保存的输出若没有 raw anchor ID，则它只能用于 D1，D2 需要在允许的资源范围内重新做冻结模型前向。不得从后 NMS 的框反推出并不存在的 raw DFL 分布。

只使用训练/开发数据，不碰封存 test。训练数据上的教师质量属于 in-sample 代理，不能写成独立可学习性估计。开发诊断已经参与方法选择，也不能再伪装成最终未见数据。

### 12.2 实验组二：N/C/L/CL

| 比较 | 能回答的问题 | 不能据此宣布什么 |
|---|---|---|
| C−N | 原 OEv1 的整体干预是否有用 | 教师具体内容、IR 标签和选择的贡献已分离 |
| L−N | 本版定位干预是否有用 | 普遍所有跨模态定位都有效 |
| CL−C | 定位是否在当前判别方法上提供额外价值 | 任务路由已经优于全部现有方法 |
| CL−L | 判别分支在定位版本上是否仍有价值 | 两项一定超加性或互不干扰 |

N/C 完整、同代码契约可复用时，新增 L/CL 各 3 seeds，即 **6 个学生长训**。复用条件不成立则需重新获得新 N/C，不能为了节省预算接受不匹配比较。

同一 seed 的 L/CL 在相同 λ_L、基础 E_L 和固定 T/R 下比较。不要让 L-only 用另一种 mask，CL 用“与 C 去重后的 mask”；这样的改法同时改变了两个因素。

### 12.3 实验组三：只给成立的主张补关键控制

这不是要求一口气跑完所有组合。根据最可能的替代解释选择；但要写“跨模态独特信息”或“任务选择优于常规 KD”时，相应强控制不能长期缺失。

| 优先控制 | 保持什么 | 改变什么 | 解释范围 |
|---|---|---|---|
| **同 mask GT 定位** | E_L、位置、mask、分母、训练预算 | 教师定位 target 改为 RGB GT/GT-DFL | 区分教师定位内容与额外对象回归监督 |
| **有效 RGB-only KD** | 学生、数据和训练预算可比 | 教师与选样仅使用 RGB 信息 | 检验 IR 信息相对普通同模态教学的必要性 |
| **简单联合 KD／教师质量选择** | C/L 内容、预算、合法支撑 | 不使用分任务参考缺口选择 | 检验任务选择是否超出已知联合 KD |
| 同 mask 原型判别 | 原 C mask/分母 | 实例教师证据换为预先冻结的训练集原型 | 检验实例判别 target 是否必要 |
| 同 K 随机选择 | 合法基础集、每批 K、分母、λ | 随机取对象，不按优势排序 | 检验选择规则，不保证天然同实际梯度剂量 |
| 合法对象 donor | 类别、尺度、对应坐标和总剂量 | 换成非配对对象的教师内容 | 检验实例配对，避免用故意错几何的整图 shuffle 充当唯一归因 |

### 12.4 GT-DFL 控制的实现细节

对学生 anchor 的每条 GT 边距离 d（以 stride 单位表示），标准两邻近 bin 目标可写为：`p[floor(d)]=ceil(d)-d`、`p[ceil(d)]=d-floor(d)`；d 为整数时全部质量落在同一 bin。必须先验证 d 在合法支撑，边界不能越界。

若使用固定 GT 软分布与学生 logits 的 CE/KL，应明确温度和缩放。这是“同 mask 的额外 GT 监督”，不等于 teacher-target 与 GT-target 在统计意义上信息相同；两者差异正是要比较的内容。

**同 mask 的 GT-only 控制仍继承 IR 对应和教师筛选信息，所以不是完全 RGB-only。**它检验的是教师具体 target 在既定选择上的增量，而非所有 IR 信息的总增量。

### 12.5 同模态控制不能退化

沿用优势门控时不能直接令同模态教师等于 R：两者输出一样会令优势归零，从而变成零 KD。

应选一个有教学能力、独立训练或容量更强的 RGB 教师，或者使用明确声明的成熟同模态 KD 方案。比较教师容量、预训练数据、训练步数和额外算力；若用更大 RGB 教师，可以把它作为强性能基线，但不要声称与同容量 IR 教师完全隔离了模态因素。

同模态路径应使用 RGB 图像和 RGB 标签，不通过 IR 匹配来决定其基础集合。若为了检验 teacher 内容而刻意使用共同 IR mask，应另命名为 `same_mask_rgb_teacher`，不冒称无 IR 信息的基线。

---

## 13. Codex 建议代码结构与接口

### 13.1 新增目录建议

```text
experiments/rgbir_task_conditional_v1/
├── README.md
├── config.py
├── contracts.py
├── geometry_contract.py
├── object_matching.py
├── candidate_assignment.py
├── localization_selector.py
├── localization_losses.py
├── combined_criterion.py
├── diagnose_opportunities.py
├── calibrate_gradient_scale.py
├── train_task_conditional.py
├── evaluate_endpoints.py
├── analyze_task_errors.py
├── configs/
│   ├── draft_drone.yaml
│   └── frozen/                  # 验收后另存，不覆盖草案和旧协议
└── tests/
    ├── test_matching.py
    ├── test_geometry.py
    ├── test_selector.py
    ├── test_localization_losses.py
    ├── test_criterion_regression.py
    ├── test_rng_and_export.py
    └── test_endpoint_records.py
```

这是建议接口，项目已有相同能力时应复用而不是另造一套大框架。尤其 loader、资源 guard、EMA 导出、run receipt 与独立 evaluator 应尽量沿用已核验路径。

### 13.2 建议的数据合同

下面只定义字段含义，不要求机械使用相同类名。shape 注释必须转成实际断言。

```python
from dataclasses import dataclass
from typing import Any
from torch import Tensor

@dataclass(frozen=True)
class GeometryContract:
    mode: str                       # verified_identity_grid / box_mapping_only
    independently_verified: bool
    evidence_path: str
    coordinate_frame: str
    input_size: tuple[int, int]
    strides: tuple[int, ...]
    reg_max: int

@dataclass
class LocalizationSelection:
    # Dense, detached weights, containing per-object anchor-count normalization.
    anchor_weight: Tensor           # [B, A], floating, nonnegative
    owner_object_id: Tensor         # [B, A], -1 means unassigned
    base_object_count: int           # before teacher quality filtering
    base_anchor_count: int
    selected_object_count: int
    selected_anchor_count: int
    object_rows: list[dict[str, Any]]
    diagnostics: dict[str, Any]
```

`base_object_count` 不得用 `selected_object_count` 代替；`anchor_weight` 是 loss 输入，不是可学习参数。若旧 raw logits 在该版本中格式变化，写 adapter 并测试，不改损失语义来适应错误 shape。

### 13.3 各模块职责

| 接口 | 输入 | 输出与责任 |
|---|---|---|
| `validate_geometry_contract` | 数据/增强元信息、几何证据 | 验证同物理网格；不通过要显式失败 |
| `match_gt_objects` | 两模态独立 GT | 稳定一对一对象对应与 unmatched 原因 |
| `assign_reference_candidates` | 冻结 R、RGB GT、合法层 | 唯一 anchor owner、对象预选 anchor；不读取 S |
| `build_localization_selection` | T/R、独立 GT、几何 | E_L、教师质量前基础数、二值权重、诊断 |
| `aligned_dfl_kd` | S/T raw boxes、anchor weight | 无 λ、无 B 的标量定位 KD |
| `combine_detection_loss` | native vector、C/L 标量、B、权重 | 单次相加的总 loss |
| `analyze_opportunities` | 固定 D1/D2 记录 | 覆盖与任务优势报告，不输出预测 gain |
| `evaluate_endpoints` | 完整 run 与冻结评价协议 | 独立 last/EMA 指标与完整来源绑定 |

### 13.4 性能实现建议

冻结 T/R 只前向一次，C/L 共享其输出，但不得修改共享 tensor。计算 L 的参考框时不要额外跑第三次教师。对象匹配在 CPU 上处理的小矩阵可以接受，但应测同步开销；不要对全量 anchor 做无界 O(A²) 比较。

稠密 raw DFL logits 本来已存在；只对已选位置 gather 后计算 loss 可节省显存。gather 版本与附录密集参考实现必须通过数值/梯度等价测试。

诊断日志不需要每 batch 保存全部 raw logits。保存稳定对象 ID、关键标量和固定频率样本；随机抽样日志使用独立 generator，不消费学生数据或模型 RNG。禁止缓存完整训练集 teacher GPU tensor。

---

## 14. 测试与验收清单

以下是 **Codex 必须实现并运行的测试要求，不是本轮已经通过的完整验收**。附录算子单独的 CPU 测试状态见文末。

### 14.1 对象、几何与选择

| 测试 | 通过条件 |
|---|---|
| 最大有效匹配 | 构造总 IoU 最优却丢有效边的反例，算法仍优先有效数量 |
| 类别不一致 | 不同 GT 类别不能成为对象配对 |
| 空/单侧标签 | 返回可解释空集合，无越界、NaN 或伪造对象 |
| 拥挤归属 | 同一 anchor 不能被两个对象重复加权，平局规则稳定 |
| 同物理网格 | shape 相同但 anchor 偏移时拒绝；同 stride 不足以通过 |
| 增强一致 | resize/letterbox/平移/翻转后两侧独立标签与图像变换正确 |
| 一般映射 | DFL 不支持时显式拒绝，不能偷偷 resize/fallback |
| DFL 支撑 | 边距离越界单列排除，不静默 clamp 伪装有效 |
| 同位置教师质量 | 教师别处框很好、当前 anchor 很差时 L 必须拒绝 |
| 参考依赖 | 改当前 S 输出不能改变首版 E_L/mask/owner |
| 不满足优势 | R 定位好于 T，或差距不越阈值时对应 gate 为零 |
| 单类退化 | 只有 argmax 正确但分数极低时不能通过可靠性门 |

### 14.2 损失与梯度

| 测试 | 通过条件 |
|---|---|
| 相同定位分布 | KL 在数值容差内为零 |
| 教师与选择 detach | S 有梯度，T/R/gate 无梯度 |
| 零 gate | loss=0、学生该 KD 梯度=0 |
| 空集合 | 返回设备/dtype 合法的可导标量零 |
| 固定分母剂量 | 权重全部减半时 loss 与梯度减半 |
| 对象均衡 | 同一对象合法复制为更多等价 anchor 后不增加总对象权重 |
| 单次加入 | B=32、λ_C=0.1、λ_L=0.2 时对 C/L 标量导数为3.2/6.4，而非其三倍 |
| 温度轴 | softmax 仅在每条边 bins 维；不跨 anchor 或四条边混合 |
| 小 batch | 最后 B<32 时用实际 B，分母与对象数仍正确 |
| box 解码梯度 | box KD 下 S 解码保留梯度，T 解码停止梯度 |
| C 无变化 | 新 C 的标量、selected IDs、q、分母与旧 C 一致 |
| 直接任务梯度 | L 不通过分类 scores 选择反传；C/L 对共享参数影响可正确测量 |

### 14.3 Trainer、RNG、端点

| 测试 | 通过条件 |
|---|---|
| 初始状态 | 同 seed 四臂的学生参数与 buffers 对齐 |
| 数据流 | 同 seed 学生文件顺序、增强图像、标签、RNG 相同 |
| 多 seed 的真实变化 | 比较初始化与数据 RNG 的实际实现，不只读配置中的 seed 数字 |
| 冻结模型 | T/R 参数不进入 optimizer、EMA、state_dict 或部署导出 |
| 原生 weight0 | 同 batch 原生 loss 与共享参数梯度保持预先约定的等价 |
| 数值故障 | NaN/Inf/异常 shape 失败并保存 receipt，不自动修复继续长训 |
| CSV 字段 | 所有行列数一致；中途 AP 未评估用显式状态，不把零/学习率误读成 AP |
| 完整端点 | E200、固定 last/EMA、有效独立评价和文件清单都齐备才计入 |
| 缺失 seed | 不跨 seed 配对，不用旧模型填空；不完整时不输出三 seed 汇总 |
| 输出防覆盖 | 已有 run/eval 目录、配置、权重、receipt 不得静默覆盖 |

### 14.4 真实数据 canary

在符合现场授权和资源规则时，用固定真实 batch 做短程 canary；可沿用项目 24 次成功 optimizer update 的规模，但必须记录尝试数与 AMP skip，不能只数循环次数。

验收至少包括：L 非零对象覆盖、非零 DFL 梯度、native/C 回归等价、T/R 无梯度、峰值显存与 RSS、导出学生结构、吞吐及异常停止行为。canary 不按 AP 选择方法，也不能因为“loss 能下降”就标为科学成功。

需要保留 `TESTS_PASSED`、`CANARY_PASSED`、`FORMAL_TRAINING_COMPLETED`、`EVALUATION_COMPLETED` 四种状态，不能都写成一个含糊的 `completed`。

---

## 15. 日志、评估与统计

### 15.1 每个训练 run 的最小记录

```text
effective_config.yaml
launch_manifest.json
implementation_snapshot/
initialization_receipt.json
rng_realization.json
runtime_ready.json
training_metrics.csv
kd_batch_metrics.jsonl
object_selection_audit.jsonl
completion_receipt.json
failure_receipt.json                 # 仅失败时
weights/last.pt
evaluation_val.json
evaluation_val_roster.txt
eval_evidence/
```

保留实际生效配置而不只保存含默认 seed42 的模板。参考、教师、学生初始化的身份、类数、类别映射、数据 split、字节/内容身份应采用项目现行可复核机制；不要只有绝对路径。若现场规范不允许新增某类 hash，使用允许的快照和直接内容比较；不要为了记录机制改动研究干预。

### 15.2 分支诊断标量

至少记录：

- `C_base/C_eligible/C_selected` 与 `L_base/L_selected_objects/L_selected_anchors`；
- `C_loss_unweighted/L_loss_unweighted`、实际 λ、B、各分母及 weighted total；
- `L_teacher_iou/L_reference_iou` 的入选均值与分位数、几何拒绝数和 no-anchor 数；
- 类别/尺度/亮度入选组成；对象重复使用次数与唯一对象覆盖；
- 在固定审计 batch 上，S 超过 T 的比例、C/L 对共同参数的梯度比例与余弦；
- 成功更新、尝试更新、AMP skip、EMA update、已完成 epoch、耗时和资源峰值。

`selected_objects` 跨 batch 累计通常含重复训练观测，不能直接解释成唯一训练目标总量。

### 15.3 梯度诊断

在同一个参数子集上定义 `g_D`、`g_C`、`g_L`；C/L 使用已经乘 λ 与 B 的真实贡献。报告：

$$
\rho_L=\frac{\|g_L\|_2}{\|g_D\|_2+\epsilon},\quad
\cos(g_D,g_L),\quad\cos(g_C,g_L).
$$

梯度为零时余弦应标记无定义，不能补成零相关。混合精度下应注明计算 dtype/是否 unscale；不要将不同参数子集或不同 batch 的范数相除。

负余弦说明该局部更新方向可能冲突，不直接证明最终 AP 下降。该诊断只测固定少量 batch，不需要每步多次反向而显著改变吞吐。审计副本应恢复 RNG/BN 状态，避免诊断过程本身扰动正式训练。

### 15.4 独立端点

保留当前固定 E200、last/EMA 的口径，使用同 evaluator、同输入尺寸、同 NMS、同图像名单及同标签版本。原生 validation CSV 与独立评价分开；`progress.json` 的历史 `running` 字段不能覆盖合法完成 receipt。

eval 产物至少保存 mAP50–95、AP50、AP75、逐类 AP、precision/recall 及评价身份。缺少端点时明确 pending，不用 best 补 last，不用 CSV 近似补独立评估。

### 15.5 配对统计与机制分析

对每个有效 student seed s：

$$
\Delta_s^{CL-C}=100\left(AP_{s,CL}-AP_{s,C}\right).
$$

先按相同 seed 求差，再计算 mean±sample SD（ddof=1）。不要直接减两臂 SD，也不要把 epoch/batch/图像数当作训练重复数。

需要解释不确定性覆盖范围：固定 T/R 时三 seed 不覆盖教师训练不确定性；固定 DataLoader RNG 时可能主要覆盖初始化变化。若不同 nominal seed 的整个初始化和数据流都相同，先核验实际随机性，不能用近零 SD 宣称稳定。

机制子集用固定 R 和固定数据定义。不要针对每个新模型重新选择它表现有利的“难例”。同时报告定位改善和损伤：参考原本定位好/教师不如参考的组、错类组、低亮度组、小目标组和背景误检。

AP75 受分类分数与预测排序影响，不是纯定位指标。应补固定关联下的 IoU/中心与尺度误差、分类错误和重复检测。对子集评估时，未属于目标子集的 GT 应采用明确 ignore 语义；不能简单删除后把它们的预测当背景 FP，从而制造错误结论。

### 15.6 实验晋级建议

这些是研究决策建议，不是已经预注册的阈值，也不是 J-STARS 硬性要求：

- H1 不成立：不开本版 L 的正式长训。
- L−N 为正但 CL−C 近零/为负：定位可能与 C 冗余或冲突，不能把它写成联合方法已成立的第二贡献。
- CL−C 有可重复正差且定位错误减少：进入关键控制，不直接宣布新机制已证实。
- 同 mask GT 定位追平教师定位：收窄为额外监督/选样作用，不能继续声称教师具体框分布是必要来源。
- RGB-only KD 追平跨模态：不能主张 IR 的独特教学优势；保留工程收益与边界。
- 简单联合 KD 追平路由：不把路由描述为不可替代贡献。
- 只有某数据集/子集有效：如实限定适用范围，不临时换方法或删掉负 seed。

团队可以在读出新矩阵前另行冻结“实用边际增益”阈值，例如 0.3 pp，但不能把它当统计显著性标准。三个 seed 只能提供有限的重复性信息，必要时增加独立确认，而不是仅靠 3/3 正向下强结论。

---

## 16. 投稿 J-STARS 的方向与策略

### 16.1 范围契合与不能承诺的部分

J-STARS 官方作者说明要求应用遥感/地球观测相关、技术内容新颖且重要，并提供充分实验及条件描述。[R11] DroneVehicle 航拍场景下训练期利用辅助模态、部署期仅 RGB，具有合理的应用主线。

但“有两个创新点”“提升一个固定 AP 数字”都不是官方录用条件。本文的三 seed、强对照和机制分析是研究建议，不应写成期刊的强制条款。是否录用还取决于与先前方法的差异、实验质量、写作和审稿判断。

### 16.2 建议的一篇文章结构

**主线：**跨模态教师并非在每个对象、每个检测子任务上都更可靠；以对象与几何对应为前提，选择教师在相应任务上的有益信息，保留学生自身优势。

贡献可组织为：

1. 一项围绕拟作用对象、区分判别和定位的可复核诊断，说明为什么整体教师 AP 和统一质量权重不足以判断可用知识；
2. 一个统一的、具有拒绝条件的任务条件蒸馏机制，两个知识通道为判别证据与定位分布；
3. 一套证明净效用、相对强控制的增量和适用边界的实验。

第二项才是方法贡献的核心；第一和第三项支撑它。不要把单独的分类 loss 与定位 loss 分别包装成新颖算法。

### 16.3 最重要的比较

主表至少应该包含同协议 native、当前 OEv1、定位分支、联合方法，以及一个有效同模态对照和一个有代表性的简单联合/区域定位蒸馏基线。CoLD 和现代 YOLO 分区定位是重要参考；适配与原论文不同之处要公开。

机制表再集中回答：双分支是否各自有边际作用、任务选择是否必要、教师具体内容能否被 GT/原型替代。不是所有控制都要交叉乘成巨大矩阵，但论文主张覆盖到的问题必须有相应证据。

### 16.4 数据策略

以 DroneVehicle 为当前主要开发和最终独立测试场景；LLVIP 是地面低照度机制补充，不应成为遥感论文唯一强结果。VEDAI 可作航拍光谱与教师方向反转补充，但要明确 NIR、合法 split 和当前样本边界。[P4]

若现有两个遥感场景的结果不足，再有目的地增加数据集，而不是同时打开 FLIR/M3FD/KAIST。新增数据集需要先核验 split、防相邻帧泄漏、标注来源和同模态 baseline。

如需补一个更大的学生容量，优先在方法冻结后做单一代表性扩展，检验效果是否仅对特定小模型成立；不先展开大范围架构搜索。

### 16.5 论文主张分级

| 实验证据 | 最窄可支持表述 |
|---|---|
| 只有机会诊断 | 指定对象/任务上存在可用于方法设计的教师优势 |
| CL 胜 C/N | 在该训练协议下，增加定位干预有边际效用 |
| 胜同 mask GT/原型 | 教师具体内容超出额外同类监督的作用得到支持 |
| 胜有效 RGB-only KD | 指定条件下 IR 辅助信息具有额外价值 |
| 胜简单联合/共享策略 | 分任务选择的必要性得到支持 |
| 独立测试与第二场景保持 | 对已覆盖场景的泛化获得支持 |

不得把某一层的通过自动升级为所有层成立。局部失败不等于项目没有价值；它决定论文应讲多窄、下一轮应投入多少。

### 16.6 明确的禁用措辞

不使用“首次分类与定位联合蒸馏”“首次跨模态定位分布蒸馏”“保证消除负迁移”“GT 已知所以定位必然能学好”“教师更强所以学生上限等于教师”等说法。任何“选择了有益知识”的主张都必须面对随机/简单选择与剂量替代解释。

---

## 17. 版本管理与资源策略

### 17.1 保护历史

开工先 `git status`，记录 HEAD 和实际代码根目录；不执行清理工作树、重置分支、删除历史 run 或覆盖原配置。新实现用独立分支/目录；实验不复用旧输出目录。

历史失败、旧 receipt、过时分析不删除。新增勘误、来源和失效原因，明确哪个结论由什么新证据修订。不因新方法效果不佳而回写原预注册。

### 17.2 资源边界

现场项目规则优先。此前项目规则涉及共享 GPU、单卡实测峰值与至少 2 GiB 余量、全项目并发上限和总 RSS 限制；Codex 应重新读取现行 AGENTS/资源 guard，不能把本文当固定分配 GPU 的许可。

CPU 测试不加载全量权重；真实 T/R 诊断尽量小 batch、顺序执行与保存小产物。新正式训练不抢占或终止其他会话任务。资源检查失败就保留待运行状态，不关闭其他人的进程。

本文不提供新长训耗时保证；预算按实际 canary 吞吐和已有队列评估。完成“新 L 是否有用”的最小比较，比同时复现十篇论文更优先。

### 17.3 恢复策略

正式运行中断后，resume 必须校验代码/配置/模型/优化器/数据 RNG/样本流身份。不得把重新开始的轨迹写成原 run 的自然延续。改变 selector、几何映射、λ 或温度后，一律新 run、新版本。

---

## 18. Codex 分批任务与 Definition of Done

### Batch A：只读盘点与定位诊断

**修改范围：**新增配置 schema、诊断脚本和报告，不改旧 trainer。

完成标准：定位 actual root/HEAD/environment；输出模型与 split 身份；复算 D1，能获取 raw 时做 D2；保存几何和 candidate contract；明确 H1 的支持与缺口。没有权重/数据时完成 CPU fixtures 与诊断 CLI，输出 `BLOCKED_MISSING_INPUT`，不编造统计。

### Batch B：纯算子与选择器

**修改范围：**新增对象匹配、anchor owner、L selector、DFL/box 独立算子、测试。

完成标准：第 14 节对象/几何/梯度测试通过；原版算子未被覆盖；mask 与 student state 无关；不支持的几何模式失败；生成 `implementation_notes.md`，逐条映射公式与代码。

### Batch C：trainer 对接与等价验收

**修改范围：**隔离的 combined criterion 和新四臂入口，复用现行基础设施。

完成标准：N/C 与原路径保持合同；新 L 确有梯度；两项 KD 单次加入；T/R 不进入部署模型；日志列数稳定；同 seed 与跨 seed 的实际随机性有证据。实际 GPU 未获授权则在此保存 `READY_FOR_CANARY`，不自动长训。

### Batch D：冻结协议与最小正式矩阵

**前提：**H1、工程验收、资源和训练授权均满足。

完成标准：生成不覆盖草案的正式配置；冻结 λ_L、几何证据、seed、评价与对照复用条件；执行登记的 L/CL，按需要补 N/C；每个端点独立评估；分析 CL−C 与错误改善/损伤。不得自动接着执行所有消融。

### Batch E：按结果决定论文方法

**前提：**Batch D 给出值得继续的证据。

完成标准：明确是否实现 task_conditional_v2，或者保留更窄的 C+L 方法；选择最有信息价值的控制；给出相对于 CoLD、LD、Sigmoid-τ 和 Task Adaptive/TID 的实质差异清单。若差异未成立，记录“尚无足够方法新颖性”，不靠更换名字掩盖。

---

## 19. 可直接交给 Codex 的任务文本

> 请在现有 RGBIR 工程中实施本规格，但本次默认只推进到只读诊断、代码、CPU 测试和 canary-ready；不要自动启动新长训。
>
> 先读取仓库最新入口、实际项目 AGENTS、现行 OEv1 协议与源码，记录 HEAD、模块真实路径、运行环境、模型及 split 身份。本文以 `c6073b9` 为对接依据，不要求回退；源码有变动时先说明差异，不能机械覆盖。
>
> 保留原 OEv1 的 C 定义与历史 run。建立独立 `rgbir_task_conditional_v1`：先实现冻结 T/R 的 D1 对象级及 D2 同 anchor 定位机会诊断，再实现教师质量前基础集合 E_L、唯一 anchor owner、二值任务优势门控和同物理 anchor 的 DFL KL。首版每对象一个 R 预选 anchor；T 在别处有好框不能替代实际监督位置的质量检查。DFL 对齐证据不足时显式阻断，不自动切换损失或插值 logits。
>
> 总损失严格为 `native_total.sum() + B * (lambda_C * loss_C + lambda_L * loss_L)`。C 仍是原 OEv1、权重 0.1；L 的正式系数尚未冻结。teacher/reference/selector 无梯度，空集合返回可导零。独立的 box KD 只作清楚命名的备选，不在同 run 内自动 fallback。
>
> 提供 N/C/L/CL 四臂入口，检查 CLI 路由确实改变权重与执行路径。核验新 N/C 对旧实现的损失、梯度、初始参数/buffers、学生输入与 RNG 等价，以及 T/R 不进入 optimizer/EMA/导出。至少完成第 14 节列出的单测，保存实际测试输出，不能把计划测试标为通过。
>
> 交付代码、测试、诊断 CSV/JSON/Markdown、待批准配置、实现与公式对照和下一步阻塞清单。只在定位机会、工程验收及资源授权满足后，另行冻结 L/CL 的正式矩阵；不要把所有消融和新数据集自动展开。
>
> 不用 train CSV 占位 AP 代替独立 last/EMA 评估，不跨 seed 配对，不用历史 native 填缺项，不改原预注册。研究目标先是判断 `CL−C` 的边际价值；它成立后，再决定任务路由、同 mask GT 定位和有效 RGB-only 基线，以支持可归因且不过度的新方法主张。

---

## 20. 来源与本轮边界

### 20.1 项目来源

以下链接固定到设计所依据的已审计提交，不代表仓库最新实时进度：

- **[P1] OEv1 冻结协议。**[EXPERIMENT_PLAN.md](https://github.com/yudongfang-thu/rgbir/blob/c6073b9506058d9a12e4d9719614ca7e61f950b0/research_bundle/08_%E5%AE%9E%E9%AA%8C%E6%97%A5%E5%BF%97/2026-09-06_train_RGBIR%E5%AF%B9%E8%B1%A1%E5%88%A4%E5%88%AB%E8%92%B8%E9%A6%8F%E9%A6%96%E8%BD%AE/EXPERIMENT_PLAN.md)。
- **[P2] OEv1 损失实现。**[object_evidence_loss.py](https://github.com/yudongfang-thu/rgbir/blob/c6073b9506058d9a12e4d9719614ca7e61f950b0/research_bundle/03_%E7%8E%B0%E8%A1%8C%E5%B7%A5%E7%A8%8B/SpaceNet6_OTD_official_reproduction/experiments/rgbir_object_evidence_v1/object_evidence_loss.py)。本轮沿用上一轮完整审计和文档中的定义。
- **[P3] OEv1 训练入口。**[train_object_evidence.py](https://github.com/yudongfang-thu/rgbir/blob/c6073b9506058d9a12e4d9719614ca7e61f950b0/research_bundle/03_%E7%8E%B0%E8%A1%8C%E5%B7%A5%E7%A8%8B/SpaceNet6_OTD_official_reproduction/experiments/rgbir_object_evidence_v1/train_object_evidence.py#L39-L142)。本轮重新读取该范围，核对 raw 输出、单次损失加入、固定 paired 调用和梯度检查。
- **[P4] 既有 RGBIR 对象诊断。**[RGBIR数据特性与蒸馏方向诊断_20260906.md](https://github.com/yudongfang-thu/rgbir/blob/c6073b9506058d9a12e4d9719614ca7e61f950b0/research_bundle/07_%E7%A0%94%E7%A9%B6%E5%88%86%E6%9E%90/RGBIR%E6%95%B0%E6%8D%AE%E7%89%B9%E6%80%A7%E4%B8%8E%E8%92%B8%E9%A6%8F%E6%96%B9%E5%90%91%E8%AF%8A%E6%96%AD_20260906.md)。本轮未重新推理 521 对图像。
- **[P5] 2026-09-06 21:46 结果快照。**[LATEST_RESULTS.md](https://github.com/yudongfang-thu/rgbir/blob/c6073b9506058d9a12e4d9719614ca7e61f950b0/LATEST_RESULTS.md)。历史背景，不作为当前最新进度。
- **[P6] 上一轮对话生成报告。**`RGBIR_Task_Conditional_Distillation_Research_20260907.md`。本轮完整读取其 230 行原文，并将研究建议细化为当前规格；并未把该报告中所有文献的全文核验重新执行一遍。

### 20.2 文献及可核验入口

- **[R1]** Sun et al., *Distilling Object Detectors with Task Adaptive Regularization*, arXiv:2006.13108, 2020。[作者预印本](https://arxiv.org/abs/2006.13108)。本轮核对摘要；不添加未确认的会议归属。
- **[R2]** Liang et al., *Task Decoupled Knowledge Distillation For Lightweight Face Detectors*, ACM MM 2020，DOI `10.1145/3394171.3414069`。[ACM SIGMM 官方目录](https://sigmm.org/opentoc/MM2020-TOC-1)；[作者研究说明](https://nlpr.ia.ac.cn/iva/homepage/jqwang/research.htm)。本轮核对分类/回归分任务与不同选样的公开说明。
- **[R3]** Zheng et al., *Localization Distillation for Dense Object Detection*, CVPR 2022。[作者预印本](https://arxiv.org/abs/2102.12252)。本轮核对摘要，算子公式来自已知分布 KD 定义与上一轮研究；不宣称本轮重做原论文完整复现。
- **[R4]** Tang et al., *Task-Balanced Distillation for Object Detection*。[作者预印本 arXiv:2208.03006](https://arxiv.org/abs/2208.03006)。出版版本的卷页信息沿用上一轮记录，本文不依赖卷页来推导实现。
- **[R5]** Wang et al., *Category-Oriented Localization Distillation for SAR Object Detection and a Unified Benchmark*, IEEE TGRS 61, 2023，DOI `10.1109/TGRS.2023.3291356`。[IEEE 出版入口](https://doi.org/10.1109/TGRS.2023.3291356)。本轮核对公开摘要/引言，未完整取得付费方法公式。
- **[R6]** Wang et al., *CrossKD: Cross-Head Knowledge Distillation for Object Detection*, CVPR 2024。[作者预印本](https://arxiv.org/abs/2306.11369)。本轮核对公开摘要。
- **[R7]** Tong et al., *CMDistill: Cross-Modal Distillation Framework for AAV Image Object Detection*, J-STARS，DOI `10.1109/JSTARS.2024.3479717`。[出版 DOI](https://doi.org/10.1109/JSTARS.2024.3479717)。卷期及联合蒸馏描述来自上一轮调研；本轮 DOI 抓取未成功，未重新核验全文。实施 exact reproduction 前必须取得原文。
- **[R8]** Su et al., *Classification and regression Task Integration in distillation for object detectors*, Neurocomputing 624, 129386, 2025，DOI `10.1016/j.neucom.2025.129386`。[出版摘要](https://www.sciencedirect.com/science/article/abs/pii/S092523122500058X)。本轮核对任务综合、学习状态和区域选择，未完整取得付费正文。
- **[R9]** Shi et al., *Protocol-consistent Sigmoid-τ logit distillation for lightweight object detection*, 2026，DOI `10.1016/j.imavis.2026.106045`。[出版摘要](https://www.sciencedirect.com/science/article/abs/pii/S0262885626001526)。本轮核对 Sigmoid/BCE 与任务分配分区的描述；作者公开仓库为 verification release，不能默认含完整论文生产 checkpoint。
- **[R10]** Ma, Ni, Luo, *Cross-modal contrastive learning-based object detection under incomplete modalities*, 2026，DOI `10.1080/10095020.2026.2633014`。[出版 DOI](https://www.tandfonline.com/doi/full/10.1080/10095020.2026.2633014)。本轮抓取未成功；保留上轮报告的需核对文献，不用不可见公式直接实现“原版 CCLKD”。
- **[R11]** IEEE GRSS, *J-STARS Information for Authors*。[官方作者说明](https://www.grss-ieee.org/publications/jstars-information-for-authors/)。本轮核对 Subject Matter 与充分实验的要求；本文所提实验矩阵并非期刊硬性模板。

上述是与当前方案直接相关的调研，不是完整系统综述，也不是“已检索所有先前工作”的证明。新颖性判断仍需在投稿前核对最终出版版本、邻近区域/质量选择方法及其实际公式。

### 20.3 本轮实际做了什么

完整读取上一轮 Markdown；读取对接 trainer 的关键范围；重新核对上述可访问的一手来源；编写本实施规格和可导出的参考算子；检查文档结构、配置示例及核心代码。没有修改 GitHub 仓库、连接训练服务器、重新评估真实模型或运行新 GPU 实验。

**第 14 节的完整工程验收仍属于 Codex 待执行任务。**附录只覆盖 loss 内核，不代表跨模态几何正确、选择有效或定位分支能提高 AP。


---

## 附录 A：已对齐 DFL 的参考实现

以下为从上一轮附件保留的独立参考内核，可用于验证 Codex 优化版本的数值与梯度。调用者必须先建立几何、对象及 anchor 契约；这个函数不能自行证明图像配准正确。`anchor_weight` 已包含每对象预选 anchor 数的归一化，`normalizer` 使用教师质量筛选前的基础对象数。

```python
"""Minimal DFL-KD kernel for already aligned teacher/student anchor grids.

This is NOT a complete RGB-IR trainer or a new published method. The caller
must establish identical physical coordinates, anchor locations/order, strides,
bin semantics and class/GT identities. Equal tensor shapes alone are not enough.

Object-balanced use: give each of object i's eligible anchors weight
w_i / number_of_eligible_anchors_i, and use the pre-selection base-object count
as normalizer. Each anchor must have one declared object assignment. Do not
normalize by the sum of the gates unless that is a separately declared design.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F


def aligned_dfl_kd(
    student_raw: Tensor,
    teacher_raw: Tensor,
    anchor_weight: Tensor,
    *,
    normalizer: float,
    temperature: float = 2.0,
) -> Tensor:
    """Return a scalar, without detector batch scaling or a global KD lambda.

    Args:
        student_raw, teacher_raw: raw DFL logits [B, 4 * reg_max, A].
        anchor_weight: nonnegative detached weights [B, A], already containing
            any per-object anchor-count correction and reliability gate.
        normalizer: positive, predeclared denominator (e.g. max(1, |E_loc|)).
        temperature: shared softmax temperature; use a fixed value per protocol.
    """
    if student_raw.ndim != 3 or student_raw.shape != teacher_raw.shape:
        raise ValueError('Expected equal teacher/student shapes [B, 4*R, A].')
    b, channels, a = student_raw.shape
    if channels % 4 or channels // 4 < 2:
        raise ValueError('DFL requires 4 * reg_max channels and reg_max >= 2.')
    if anchor_weight.shape != (b, a):
        raise ValueError('anchor_weight must be [B, A].')
    if teacher_raw.device != student_raw.device or anchor_weight.device != student_raw.device:
        raise ValueError('All tensors must be on the same device.')
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('normalizer must be finite and positive.')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive.')
    for tensor in (student_raw, teacher_raw, anchor_weight):
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Inputs must be finite floating-point tensors.')
    if bool((anchor_weight < 0).any()):
        raise ValueError('anchor_weight must be nonnegative.')

    r = channels // 4
    # [B, 4R, A] -> [B, A, 4, R]; compare bins on the last axis, in FP32.
    s = student_raw.float().reshape(b, 4, r, a).permute(0, 3, 1, 2)
    t = teacher_raw.detach().float().reshape(b, 4, r, a).permute(0, 3, 1, 2)
    log_s = F.log_softmax(s / temperature, dim=-1)
    p_t = F.softmax(t / temperature, dim=-1)
    per_anchor = F.kl_div(log_s, p_t, reduction='none').sum(-1).mean(-1)
    per_anchor = per_anchor * temperature**2
    return (per_anchor * anchor_weight.detach().float()).sum() / normalizer


def combine_detection_loss(
    native_total: Tensor,
    classification_kd: Tensor,
    localization_kd: Tensor,
    *,
    batch_size: int,
    classification_weight: float,
    localization_weight: float,
) -> Tensor:
    """For the repo's pinned native-loss convention, add each scalar only once."""
    if classification_kd.numel() != 1 or localization_kd.numel() != 1:
        raise ValueError('KD terms must each contain one scalar.')
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    if any(not math.isfinite(v) or v < 0 for v in (classification_weight, localization_weight)):
        raise ValueError('KD weights must be finite and nonnegative.')
    return native_total.sum() + batch_size * (
        classification_weight * classification_kd.reshape(())
        + localization_weight * localization_kd.reshape(())
    )
```

测试重点是：同分布近零、零 mask 可导零、teacher/gate detach、固定分母下线性剂量，以及两个 KD 标量只加入一次。完整选择器、真实 loader 和训练导出仍按第 14 节另验收。


## 附录 B：本轮文档与内核检查记录

本轮在本地 **PyTorch 2.10.0+cpu** 重新执行了附录 A 对应内核的 7 项检查，全部通过：学生有梯度且教师/gate detach；同分布近零；零 gate 的 loss/梯度为零；固定分母下 gate 减半则 loss 减半；两个 KD 标量只加一次；负 gate 被拒绝；形状不一致被拒绝。

同时检查了 Markdown 代码围栏、数学块配对、Python 示例语法、YAML 解析、章节与来源编号，以及默认配置禁止正式训练/测试集访问、正式 λ_L 未伪造填写、几何未默认通过。

**这不是新方法的完整测试报告。**真实对象对应、实际 anchor 几何、训练接口、N/C 等价、资源、导出和检测增益，仍须由 Codex 按正文验收。本文没有执行新训练或修改服务器与仓库。

---

**最终执行策略：先验证“有没有可靠定位知识可以传”，再验证“加入 L 是否比仅 OEv1 更好”，最后验证“这种选择与跨模态内容是否不可被简单控制解释”。不要反过来先做一个复杂双分支，再从结果中补故事。**
