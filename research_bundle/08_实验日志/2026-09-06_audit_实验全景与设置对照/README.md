# 实验全景与设置对照（2026-09-06）

> **主线已收敛为 DroneVehicle 的 IR→RGB 对象判别证据蒸馏 OEv1，并行检验 OS-SSL 配对预训练。六 baseline / 521 对数据诊断完成；OEv1 完成 3/6 个独立端点、0/3 个完整同 seed P/N 比较；OS-SSL 完成 3/9 个微调，新增 seed123 独立 last 结果 paired−shuffled = +0.708405 mAP / +0.533024 AP50 pp。两条新路线都尚未形成三 seed 净收益和完整归因结论。**

进度快照：北京时间 **22:26**；新增 OS-SSL 文件核查：**22:28**。本次只读服务器，未启动训练或 GPU 评估，未修改正在运行的方法/阈值/队列。详细说明与新原始证据已保存本地；本次没有另作 GitHub 推送，既有公开分支的上一版快照为 21:46。

## 目的与口径

解释当前研究问题、实际方法、每条实验线的设置、可比结果和剩余缺口。读取实际 release、run 配置、原始评估 JSON、训练 CSV 和冻结协议；复用 monitor-experiment / analyze-results 流程，并分别独立复核方法设置、历史统计和新增评估。

下文 mAP 均指 **mAP50–95**，AP50 另列；指标用百分数，差值用百分点（pp）。三 seed SD 为样本标准差。独立评估、训练 CSV、历史参照严格区分；不按 best 选结果，不混合不同 seed 凑方法对照。所有当前结果属于开发 val，尚非封存测试结论或 accepted analyzer 升级后的论文主张。

## 1. 当前在往哪个方向努力

研究目标是：**训练时利用另一模态，部署时只保留弱势模态检测器；识别教师具有、学生也可能利用的信息，检验能否获得净收益并减少损伤。** 当前正式新实验以 DroneVehicle 的 IR 教师辅助 RGB 学生为主。

这一目标分成两条不同的实验路线：

| 路线 | 跨模态信息进入训练的位置 | 传递的主要内容 | 部署 |
|---|---|---|---|
| OEv1，对象判别证据蒸馏，主线 | RGB 检测训练期间加入冻结 IR 教师监督 | 选中目标的正确类别前景—背景证据差 | 单 RGB YOLO11n |
| OS-SSL-IR，并行迁移检查 | 先跨模态自监督预训练，再用其骨干初始化 RGB 检测器 | 骨干表征；检测微调时不再加监督 IR KD | 单 RGB YOLO11n |

主线不同时叠加全图特征、关系、框、频域等多个损失。本轮先检验一种有数据动机的对象判别知识。IR 不是所有 RGBIR 数据集的默认强教师：VEDAI 是近红外，其现有模型显示 RGB 更强，方向应反过来检查。

当前阶段是“数据诊断完成、可复核实现完成、首批正式效果正在产生”。尚不能称为方法有效性和负迁移抑制已经证实，更不能据此判断已满足投稿条件。SAR 历史中的负结果也不能推成“SAR 普遍没有蒸馏空间”；SpaceNet6 的 OS-SSL 历史正例与监督检测 KD 需要分开看。

## 2. 数据集与 baseline 已做了什么

已核对并复用三个数据集、两种模态共六个有效 baseline 权重，均为 YOLO11n / seed42 / E200 / last。已生成完整预测、P3/P4/P5 特征诊断、12 组双模态特征图与 3 组局部配准图。没有为本次说明重训模型。

| 数据集 | 实际 train / 开发 val | 诊断样本 | RGB / IR或NIR baseline mAP，CSV | 角色 |
|---|---:|---:|---:|---|
| DroneVehicle HBB，5 类 | 17,990 / 1,469 | 200 对 | 53.798 / 59.659 | 当前主任务，IR→RGB |
| LLVIP grouped，person | 9,619 / 2,406 | 200 对 | 32.867 / 48.833 | 判别与定位的机制补充，IR→RGB |
| VEDAI512 paper8，official_fold01 | 1,089 / 121 | 全部 121 对 | 37.018 / 34.874 | 光谱小目标，优先 RGB→NIR |

Drone/LLVIP batch32/nbs64；VEDAI batch64；imgsz640、SGD、200 轮。表中是单 seed、单模态能力背景，不是新方法增益。不同模态标注可能不同，教师 AP 高出的部分不能直接视为学生可达提升上限。VEDAI 的 paper80 教师因 train 与当前 val 有 95/121 张重叠，已排除；现用正确 official_fold01 权重。

**对象互补性决定知识选择。** 命中定义为 conf≥0.25、同类 IoU≥0.5；先对应两侧 GT，再在各自 GT 上判断。probe 的 GT 对应阈值为 0.1，实际 OEv1 更严格地使用 0.5。

| 方向 | 共同对象 | 双方命中 | 教师独有命中 | 学生独有命中 | 双方命中时教师定位更准比例 |
|---|---:|---:|---:|---:|---:|
| Drone IR→RGB | 3,083 | 2,505 | 360 | 78 | 49.46% |
| LLVIP IR→RGB | 672 | 484 | 107 | 22 | 60.95% |
| VEDAI RGB→NIR | 364 | 222 | 41 | 31 | 45.50% |

Drone 的 360 个 IR 独有命中里，**118 个 RGB 已有类别和位置正确、只是分数不足的候选**。双方已命中对象上，IR−RGB 定位 IoU 平均仅 +0.00790，中位数还略负。由此优先检验“让 RGB 更能把目标与背景区分开”有明确依据；没有依据宣称 IR 普遍适合教精定位。最暗四分之一图像包含 320/360 个 IR 独有命中，但这没有控制场景和目标数量，不能直接证明亮度门有效。

LLVIP 双方命中时 IR 的 IoU 平均优势 +0.04197，后续定位问题比 Drone 更值得单独检验。VEDAI 是近红外，不能套用热红外夜视解释；其 RGB 独有命中里也有 NIR 低置信正确框和错类候选，值得研究类别/评分互补。

**配准和特征诊断的结论更有限。** Drone 对应框平均 IoU0.904，2.69% 对应框 IoU<0.5，说明有对象对应基础也有局部差异。LLVIP/VEDAI 标签两侧字节相同，IoU=1 不证明独立像素配准完美。三个数据集的配对前景 centered CKA 均高于随机 donor，例如 Drone P3/P4 差 +0.582/+0.575；这说明对应表征存在，不意味着强迫特征相等就会提高检测。VEDAI P5 只有 1 张图具备有效前景 token，不能据此作层级优劣判断。图中 L2 激活各自归一化，颜色多少也不是知识质量。

## 3. OEv1 实际方法：教哪些信息、怎样选择

训练中有三个模型：可训练 RGB 学生 S；冻结 IR 教师 T；冻结 RGB 参考 R。T/R 是各模态 seed42 的 E200 baseline。学生从通用 yolo11n.pt 初始化，任务分类头部分随 seed 改变，不是从已训 RGB baseline 继续训练，也不是全网络随机初始化。教师与参考不进入优化器、EMA 或部署网络。

**第一步，找可对应的对象。** RGB 与 IR 重放相同几何增强，保留各自真实标注；同图、同类、GT IoU≥0.5，一对一最大有效匹配数优先。没有复制 RGB 标签给 IR，也没有训练配准网络。

**第二步，把知识压缩为一个对象相对背景的类别证据。** P3/P4 上分别取自身 GT 框内前景，背景为宽高扩大 2 倍的外框、排除该模态全部 GT。每层要求至少 1 个前景和 4 个背景点，只平均两边共同有效层。对正确类别 logit 定义：

`e = mean_valid_levels[(LME(logit_fg) − LME(logit_bg)) / T]`，其中 `LME(z)=log(mean(exp(z)))`，`T=2`。

它表示“目标位置比附近背景更像这一类”的程度。不是直接抄教师置信度，也不是全类别 KL、全图特征 MSE 或框位置蒸馏。温度在前景/背景 LME 作差后相除。

**第三步，筛选候选。** 基础集合 E 除了 GT 对应和有效区域，还要求冻结 RGB 参考存在任意类别 conf≥0.05、IoU≥0.1 的粗候选。然后要求 IR 教师存在 argmax 类别等于 GT、该类别 conf≥0.25、IoU≥0.5 的候选。这里均指 pre-NMS 候选检查，不等于 probe 的一对一 TP；教师还需按照证据代理优于 RGB 参考：

`q = max(softplus(−e_R) − softplus(−e_T), 0)`。

在满足教师正确且 q>0 的 eligible 集合中，按 q 取前 `ceil(0.5×|eligible|)` 个对象；不是所有 E 固定取一半。q 用于排序筛选，不是连续乘入损失的质量权重。RGB 粗候选是可学线索代理，不是物理可见性证明；T/R 见过训练集，q 也不是无偏的迁移能力估计。

**第四步，增加一个损失。**

`KD = sum_selected SmoothL1(e_S, stopgrad(clip(e_T,−8,8))) / max(1,|E|)`。

实际实现 `total = native_total.sum() + 0.1 × batch_size × KD`，对应框架原生 batch 尺度，KD 只加一次；保留原生 box/cls/DFL 监督。分母用筛选前基础集合 E，避免每次选中少数对象后又全部放大；但总有效剂量仍随选中比例变化。

**当前只运行两个臂，各三个学生 seed。** P 是以上 λ=0.1 的 paired；N 是同代码 weight0，仍执行辅助支路、候选和选择，仅最终 KD 系数为 0。这比拿不同旧训练器的 native 直接作净效果对照更明确。首轮 CPU 与真实 batch canary 已验证配对初始化、首批学生输入、weight0 loss/梯度与 native 等价，P 有非零 KD 梯度且只加一次。工程检查不能代替科学效果。

这版使用独立 IR GT，属于额外训练期监督信息。没有当前学生正确性门，也没有学生超过教师后停止迁移的保证；“减少负迁移”仍是待检验目标。

## 4. 两条新路线的设置对照

| 设置 | OEv1 | OS-SSL-IR |
|---|---|---|
| 检测数据 | DroneVehicle RGB train17,990 / val1,469，5类 | 相同 |
| 部署模型 | RGB YOLO11n，640 | 相同 |
| 跨模态阶段 | 200轮检测训练中，IR教师提供对象证据 | 先 SSL 10,000 steps，再200轮 RGB检测 |
| 辅助标签 | 使用 IR 图与独立 IR GT | SSL 以 RGB/IR 图像为输入，不以 IR 检测 GT 作监督 |
| 模型初始化 | 通用预训练 YOLO11n + 任务头构造 | SSL骨干注入5类模板，再 RGB微调 |
| 检测 batch / nbs | 32 / 64，warmup后通常约2批累积 | 32 / 64 |
| 检测 workers | 4 | 8 |
| 检测优化器 | SGD；lr0=.01，lrf=.01，momentum=.937，WD=.0005 | 相同主要参数 |
| 学习率 | 非cosine；warmup3；最终LR比例.01 | 相同主要检测recipe |
| 检测增强 | translate.1、scale.5、fliplr.5；关闭mosaic/mixup/HSV等 | 相同主要增强 |
| AMP / early stop | AMP开启；patience0；固定200轮 | 相同 |
| 检测 seed | 0/42/123；教师/参考固定42 | 微调0/42/123；每臂SSL本身只做seed42一次 |
| 训练中验证 | 关闭，终点独立评估 | 每轮有CSV；正式比较优先独立last评估 |
| 当前新训练矩阵 | P/N ×3，共6个 | paired/shuffled/IR-only ×3，共9个微调；另复用旧native3个但有混杂 |

OEv1 特有超参：P3/P4，T2，外框宽高×2，rho.5，lambda.1，SmoothL1 beta1，teacher target clip8。所有 seed 使用冻结值，没有按中间 AP 调参。实际 canary 显示跨 seed 初始 499 个 state 张量里12个任务头张量不同；DataLoader固定generator使首批和前30批样本一致，不能把这些重复描述成覆盖了全部数据流随机性。

OS-SSL 预注册沿用 BYOL（在线分支预测配对目标分支表征；目标分支停止梯度/动量更新）的预训练路线，计划 projector/predictor 与 LARS 细节见预注册。实际预算可确认是每臂10,000步、batch32、seed42，使用17,990对训练图像。运行时来自恢复的旧pyc实现，最小config仅保留少数关键参数，**尚不能把完整源代码与所有预训练超参均写成执行时审计通过**。

三个 SSL 臂：paired 使用正确 RGB/IR 对；shuffled 用固定 seed42 donor 打乱对应；代码中的 sar_only 在这里实际是 IR-only 单模态 SSL，随后仍训练 RGB 检测。它不是 RGB-only 自模态对照。预训练产物向 detector 注入240个骨干张量，三个 SSL 臂其余259个张量已验证相同。末尾 SSL loss paired≈.124、IR-only≈.083、shuffled≈.049，目标难度不同，不能按这些 loss 大小排序检测能力。

## 5. 以前的实验告诉了什么

### CMDistill-adapted/corrected

RGB学生、冻结IR教师，同时匹配 P3/P4/P5 全图 Pearson相关（1−r）、最深层空间关系、解码框IoU和全anchor分类BCE；三个大项权重均1。检测主要recipe为E200/b32/nbs64/640/SGD，与专用N相同；属于协议适配版，不能标原作者精确复现。共享主要recipe已核实，不能根据目录名误判为不同batch预算；也没有宣称两种trainer完整逐batch等价。

| Drone，独立last mAP | seed0 | seed42 | seed123 | mean±SD |
|---|---:|---:|---:|---:|
| 历史匹配 native N | 54.3576 | 53.8156 | 53.6874 | 53.9535±0.3558 |
| CMDistill L | 53.9003 | 53.2448 | 53.6692 | 53.6048±0.3324 |
| L−N，pp | −0.4574 | −0.5707 | −0.0183 | **−0.3488±0.2918** |

该全量组合在主指标上三 seed 全负。AP50平均差−0.2768±0.7085，seed123为正，所以不能说所有指标都负。它否定这套已跑组合的净收益，不能否定所有选择性蒸馏。LLVIP CMD三seed mAP34.0603/33.4440/33.3168，mean33.6070±.3976；缺匹配的三seed专用native，不补造完整净差值。

### HNEWA-CMKD-MSE-inspired

先训练 **RGB+IR六通道融合教师**，再给RGB学生做P3/P4/P5全图MSE，alpha.5，E200/b32/640，教师seed42。h1 paired是融合教师读正确RGB+IR；h2 shuffled保持RGB、打乱IR半边；h3 same-modal换成单RGB教师；b0是native。KD三臂均有学生0/42/123，b0仅0/123端点身份明确，不能用别campaign的新N42补齐。

| 数据集，mAP mean±SD | paired | shuffled | same-modal | paired−shuffled | paired−same-modal |
|---|---:|---:|---:|---:|---:|
| Drone | 54.1359±.2317 | 53.7999±.1074 | 53.9291±.3337 | +.3360±.2894，3/3正 | +.2068±.3597，2/3正 |
| LLVIP | 34.6786±.8004 | 34.3287±.7854 | 34.5877±.3451 | +.3499±1.2305，2/3正 | +.0909±.5920，2/3正 |

这说明部分配对差为正，但超过强同模态对照的幅度小且不稳定。paired胜shuffled也可能有shuffled损伤成分，不能省略native和same-modal。旧“+12.7”来自precision误作mAP，已撤回。

### 其他已做尝试

| 项目 | 完成内容 / 主要设置 | 当前判断 |
|---|---|---|
| 早期P3 causal v1 | Drone/LLVIP，seed42，E200；native/paired/same-modal/shuffled/random-dose历史CSV | Drone mAP依次53.798/53.509/54.095/53.936/54.367；LLVIP32.867/33.532/35.209/35.398/33.308；paired未稳定超过强对照，单seed不足归因 |
| CCLKD-adapted | Drone三seed E200训练完成 | 既有核查快照缺对应独立终态检测JSON；不能用完成或loss判断有效/无效 |
| CGA-KD早期实现 | 有计划、canary、W1；W2冻结 | G/C等实现与主张不一致，I有通道问题；实现偏差不等于选择性思想被证伪 |
| 旧P2特征probe | 做过特征图/CKA | 曾混入错误VEDAI native且CKA未中心化；旧“拉力/饱和”等结论撤回，由本次六baseline诊断替代 |

SAR方法史和完整逐seed值见专门说明。旧FreqMix“+9”是错数据集比较，复核实际OGSOD主要recipe匹配native参照为负，不能继续保留该幅度来讲增强或跨模态故事。

## 6. OEv1 当前进度与结果

以下已完成轮数来自22:26快照，不把正在进行的轮计作完成。

| 学生seed | P：paired进度 / mAP | N：同代码weight0进度 / mAP | 本轮同seed净差 |
|---|---|---|---|
| 0 | 38/200，运行中 | 200/200，**54.3462** | 待P0 |
| 42 | 200/200，**54.6582** | 146/200，运行中 | 待N42 |
| 123 | 200/200，**54.6368** | 31/200，运行中 | 待N123 |

**3/6 个训练及独立评估端点完成，但 0/3 个完整同seed P/N配对。** 不能把两个P的平均减去N0，不能由跨seed recall差宣称增加或损伤召回。

| 已完成端点，独立E200 last/EMA | mAP | AP50 | AP75 |
|---|---:|---:|---:|
| P42 | 54.6582 | 77.0696 | 63.8393 |
| P123 | 54.6368 | 77.3046 | 64.2148 |
| N0 | 54.3462 | 76.9925 | 63.9329 |

为了判断是否值得等待，可以列出**历史参照**：P42比历史同seed native高+0.8426 mAP，P123高+0.9494；新N0与历史N0相差−0.0115。这是两个有希望的观察和一个基线路径检查，尚不替代本版同代码N42/N123。已观察的近+0.9不能写成正式三seed收益。

## 7. OS-SSL 当前进度与新增结果

三个SSL预训练均完成；九个RGB微调中3个完成、1个运行、5个排队：

| SSL初始化 | 微调seed0 | 微调seed42 | 微调seed123 |
|---|---|---|---|
| paired | 排队 | 排队 | E200完成，独立last评估已出现 |
| shuffled | 167/200 | 排队 | E200完成，独立last评估已出现 |
| IR-only | 排队 | E200完成，目前仅CSV | 排队 |

22:22–22:23新生成、22:28核验的两个独立端点：

| seed123，独立last | mAP | AP50 | AP75 |
|---|---:|---:|---:|
| paired SSL | 53.932152 | 75.856702 | 63.083680 |
| shuffled SSL | 53.223747 | 75.323679 | 62.740825 |
| paired−shuffled，pp | **+0.708405** | **+0.533024** | +0.342855 |
| 历史W1 native，仅混杂背景 | 53.687433 | 75.730143 | 63.306324 |
| paired−历史native，仅背景 | +0.244719 | +0.126560 | −0.222644 |

21:46旧CSV差为+.669mAP/+.495AP50；现独立last差为+.708/+.533，方向一致、值略不同。后续优先独立评估，同时保留CSV原件。单seed超过+.5数值并不意味着预注册三seed归因门通过；不能为了通过门而挑选口径。

IR-only seed42训练CSV mAP54.623/AP5076.822；没有独立metrics_record，且与paired123不同seed，不据此宣布IR-only优于paired。预注册要求paired−native至少+1.0 AP50且3/3正，以及paired−shuffled、paired−IR-only均至少+.5 AP50的三seed汇总，尚未获得这些完整比较。

**native初始化混杂仍未解除。** 旧W1 native由COCO80类权重转5类，实际加载451/499张量；三个SSL初始化已是5类模板，加载499/499张量。因此SSL对旧native差异不只有骨干SSL，也混入检测头起点。三个SSL臂非骨干259张量一致，使内部paired/shuffled比较更明确；但完整自监督收益仍需同模板零SSL native。目标是RGB部署，还缺RGB-only SSL；IR-only不能替代这个强自模态对照。三个微调seed也不等于三次独立SSL预训练。

**证据链限制。** 新指标明确last/data/seed/arm，但未完整保存执行时CLI、评估图像清单及源代码回执，低于OEv1端点证据完整度。服务器README另登记D3：这两次已有评估在GPU2绕过资源守卫执行。本次仅审计该事实，未执行它；偏差已保留，不把此端点说成全流程守卫通过，也不凭该偏差判数值必然有误。

## 8. 哪些结论已经成立、还要验证什么

**已有证据支持：** 教师优势与候选可迁移知识因数据集而异；Drone更值得优先研究对象判别。既有全量CMD组合没有净收益；融合教师全图MSE相对same-modal优势弱。新OEv1有两个超过历史native的正向观察，OS-SSL有一个独立同seed paired优于shuffled的观察。

**尚不能支持：** OEv1三seed净增益、跨模态内容独特价值、质量选择优于随机、已经避免负迁移、跨数据集泛化、无IR标签需求、论文级新颖性。OEv1与OS-SSL的绝对分数受初始化、训练阶段、辅助信息和验证流程差异影响，不能直接排序并作因果归因。

下一步应保持当前冻结实验完成，而非看到中间结果后加模块：

1. **先收齐OEv1本轮P/N×3**，逐seed独立last配对差，报告mean±SD和方向。P/N检验整套干预是否有净价值。
2. 如有净价值，再冻结最小归因矩阵：same-modal区分自模态蒸馏；同集合/同K/同剂量random区分选择价值；合理同类/同尺度donor或shuffled区分内容对应；同mask GT-only区分IR标签/额外监督和教师软信息。当前仅P/N完整可跑；loss中部分算子不等于这些实验已完成。
3. 对“避免负迁移”单独做收益与损伤分析：教师独有命中、学生独有命中、双方命中、双方失败对象；漏检减少、背景误检、学生原有优势是否受损；按预先固定亮度/尺度/类别分桶。不能只报总AP或阈值下TP增加。
4. OS-SSL完成冻结矩阵的同时保留初始化偏差。后续如扩展，先加同模板零SSL native和RGB-only SSL；不以补步数、改超参救本轮结果。旧native不能作为纯SSL效应终局判定。
5. 数据集扩展应检验假设差异：LLVIP查定位/暗光条件；VEDAI查RGB→NIR方向与小目标。FLIR/M3FD需要先冻结有效划分和baseline，不能因为已经有数据文件就写成完成诊断。面向J-STARS，后续还需要与遥感应用相符的独立数据证据。

## 产物路径与复核入口

- [方法设置逐项核对](oev1_method_settings.md)：实际代码、算法、全部阈值、未跑对照、版本身份。
- [数据与历史逐seed对比](dataset_and_history.md)：六baseline、521对probe、CMD/HNEWA原始表与历史范围。
- [22:26只读进度](progress_snapshot.json)、[远端采集脚本](refresh_progress.py)、[SSH封装](fetch_progress.py)。远端主要run在`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/`。
- [22:28新增OS-SSL审计](osssl_new_eval/README.md)、[原始小文件](osssl_new_eval/raw)、[复算summary](osssl_new_eval/summary.json)、[来源mtime清单](osssl_new_eval/source_inventory.json)。权重仅记录94路径，不下载。
- [当前矩阵CSV](experiment_matrix.csv)、[生成脚本](build_matrix.py)：每行保留route/arm/seed/进度/指标来源，不计算缺失配对。
- [既有21:46端点与GitHub回执](../2026-09-06_audit_RGBIR夜间结果与GitHub更新/README.md)。上一版公开分支为`research/full-evidence-20260906`，commit`c6073b9`；不能把本次新增本地内容冒称已经推送。

本次不改原始results、失败attempt、训练receipt或checkpoint；没有改冻结阈值、访问test或重新挑checkpoint。新结论以本目录文本与原始小证据为准。
