# RGB–IR 诊断指标、数据来源与最小方法假设

> 2026-09-06。结论：先寻找“同一目标上、学生已有粗定位、教师定位更准、跨模态坐标能够对应”的机会，再检验少量定位知识是否有净收益。教师独有命中、亮度差、特征热图或 CKA 都不能单独证明可蒸馏性。下文是本轮独立文献与诊断设计核查，不是新方法有效性结论。

本记录读取了项目 README、实验索引、2026-09-05 综合审计、W1 native 终态、P3 昼夜探针和本目录冻结协议；使用 research-lit 的本地原文优先与一手外部补核流程。未运行 GPU、未修改训练实现、未改动冻结探针协议。统计结果以本目录随后生成的原始产物为准。文献核查为定向范围，不构成完整 novelty certification；所有文献数字均未作为已复现效果引用。

## 1. 数据特性中最容易混淆的事项

| 数据集 | 一手来源能确认的性质 | 对本项目诊断与方法设计的含义 |
|---|---|---|
| DroneVehicle | 原作者仓库现行 release 写 28,439 对 UAV RGB–IR；昼夜、城区道路、住宅区、停车场；五类有向框，RGB/IR 各自标注数量不同；下载图带四周 100 px 白边，840×712，可去边成为 640×512。[作者仓库](https://github.com/VisDrone/DroneVehicle) | 必须先核验去边与框坐标是否同步、HBB 转换是否一致。模态间 GT 框差可能同时包括剩余配准偏移、热/可见轮廓不同、标注者偏差；不能都称配准误差。类别和实例不完全对称，不能直接将 IR 全部目标作为 RGB 密集监督。 |
| LLVIP | 15,488 对，低照为主，原论文声明时间与空间严格对齐；作者明确先在 IR 图标行人，再将框复制至 RGB。作者仓库 2023 年还更正过漏标与不准标注。[原论文](https://openaccess.thecvf.com/content/ICCV2021W/RLQ/papers/Jia_LLVIP_A_Visible-Infrared_Paired_Dataset_for_Low-Light_Vision_ICCVW_2021_paper.pdf)、[作者仓库](https://github.com/bupt-ai-cz/LLVIP) | 两份标签 IoU=1 主要反映标签复制，不能作为独立像素配准验证。RGB GT 存在也不能证明人眼/模型在 RGB 中有足够观测。它很适合研究“教师额外观测与学生可学习性脱钩”，但 IR 大幅领先不保证 RGB-only KD 大幅可学。按场景/序列划分开发集尤为必要。 |
| VEDAI | 原论文 §3.1/Table 2 明确为原始四通道航拍中的 RGB 与 **near infrared (NIR)**；1024 与 512 版本分别约 12.5/25 cm/px，512 是同幅影像降采样。§3.3/§4.1 给十折；§3.4 给中心、方向、四角、类别、遮挡/截断。原文初始描述为 1,210 幅，release 计数应读实际 manifest。[作者数据页](https://downloads.greyc.fr/vedai/)、[原论文 DOI](https://doi.org/10.1016/j.jvcir.2015.11.002) | 这不是热红外，不能用“夜间热目标”“温度”解释它。它是小样本、多类别、极小目标/丰富背景的光谱迁移补充；可保留遥感主线，但不能宣称第二个热红外泛化数据集。本项目 paper8 HBB/paper80/fold01 都是不同协议标识，教师训练 ID 与开发 ID 交集必须先查。 |
| FLIR-aligned | Teledyne 当前官方 ADAS release 与文献常用版本不等价。CFR 原文 §4.1 说明：从旧 FLIR 中人工删除错配图，保留 4,129 train/1,013 test，三类 bicycle/car/person；图示标签来自 thermal。[CFR 原文](https://arxiv.org/html/2009.12664v1)、[FLIR 当前官方数据说明](https://oem.flir.com/en-gb/solutions/automotive/adas-dataset-form/) | “aligned”并不自动表示逐幅重新做了精密标定；CFR 版首先是人工筛掉严重错配的子集。必须核对下载来源、图像数、COCO 类别、标签坐标系、train/test 列表。不能将当前官方新版统计填进旧 aligned 版主表。适合道路场景补充，且三类的 RGB/IR 优势可能不同。 |
| M3FD | 作者仓库说明同步 RGB/IR 系统、所有图已配准；可见光由相机内参校正、IR 使用 homography 变换；4,200 对用于融合/检测，另 300 对为独立场景融合集；六类 People/Car/Bus/Motorcycle/Lamp/Truck，作者承认可能漏标/错标。仓库结构只有一套 `labels/`。[作者仓库](https://github.com/JinyuanLiu-CV/TarDAL) | 场景/光照/天气更丰富，适合检验单一亮度门控是否足够。单应配准不能自动消除所有视差和非平面目标局部差异。共享标签的一致性同样不是独立配准证据。论文正文/表格/后续 release 的实例计数不完全一致，以本地文件清单为准，不能混用。 |

VEDAI NIR 来源已直接提取本地原论文 PDF p.6（期刊 p.192）核实；十折见 p.8（期刊 p.194）。本地原文位置：`01_文献/RGB-IR_20260905新增/VEDAI__2016_JVCIR__Vehicle_Detection_Aerial_Imagery_Small_Target_Benchmark.pdf`。HAL 在线原稿受访问限制，不以其他论文的转述替代此项。

J-STARS 的官方范围仍要求与应用遥感/Earth observations 的问题联系，以及新的、有意义的技术内容和充分实验描述。[官方作者说明](https://www.grss-ieee.org/publications/jstars-information-for-authors/) 因而建议把 DroneVehicle 的 UAV 单模态部署作为主问题，LLVIP/FLIR/M3FD 作为机制补充；若主表完全转成地面安防行人，投稿范围的叙事需要重新评估。这是选题适配判断，不是录用保证。

## 2. 这些已有方法排除了哪些“空白”叙事

| 方法与已核对出处 | 已有技术覆盖 | 我们不能直接当新颖点的内容 |
|---|---|---|
| CMDistill，J-STARS 18，1395–1409，2025；DOI 含 2024 online 年份。[出版记录](https://ieeexplore.ieee.org/document/10715640)、[原文全文展示](https://www.researchgate.net/publication/384915446_CMDistill_Cross-modal_Distillation_Framework_for_UAV_Image_Object_Detection) | PCC 特征、语义关系、IoU/BCE response 蒸馏；训练双模态、部署单模态。IEEE 页面本工具访问受限，机制依据原文全文展示和项目已审计实现交叉核对。 | “用 IR 教师帮助 RGB”“减小特征差异负作用”“同时转移定位与类别知识”已有直接先例。G 若仅 IoU+BCE，不是首次几何蒸馏。 |
| CCLKD，Geo-spatial Information Science，2026。[出版原文](https://www.tandfonline.com/doi/full/10.1080/10095020.2026.2633014) | 类别分区、预测熵自适应温度、多层 KD、类约束对比。**原 PDF pp.7–9 明确对候选框 spatial localization distributions 做 KL 与对比**；由 xywh 回归输出经 sigmoid/softmax 构造，并非逐边 DFL bins。 | 本地旧笔记称“只有 logit/feature/relation，没有定位蒸馏”过粗。可以严格区分分布参数化与对齐方式，但不能声称跨模态定位分布蒸馏无人做。论文三数据集是 OGSOD/DroneVehicle/VEDAI，不能误写 LLVIP。 |
| Localization Distillation (LD)，CVPR 2022。[原论文](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html) | 边界定位分布蒸馏；valuable localization region；区分分类与定位的蒸馏区域。 | “只蒸馏框的分布”“只在有价值的区域蒸馏”都是已有检测 KD 思路。迁到 RGB–IR 可作为必要 baseline，不能只改数据集当机制创新。 |
| CMKD-net，TCSVT 2026；DOI 10.1109/TCSVT.2026.3670458。[出版 DOI](https://doi.org/10.1109/TCSVT.2026.3670458) | 已读本地原 PDF 首页及方法：融合教师到缺失模态学生；channel/spatial 特征分布与表示一致性、实例关系、rotation-adaptive RoI pooling；VEDAI/DroneVehicle 的有向检测。 | 融合教师、对象 RoI、实例关系、同数据集上的模态缺失叙事均已有。HBB 三种子与四臂归因增强证据质量，但单独不自动构成方法新颖性。 |
| Infrared-Privileged UAV Detection via Cross-Modal Vector-Quantization，AAAI 2026，正式发表 2026-03-14。[AAAI 原文页](https://ojs.aaai.org/index.php/AAAI/article/view/37692) | IR 作训练期 privileged information；从 RGB 层级预测多尺度 IR codebook indices，使用幻觉 IR 知识支持推理；DroneVehicle/VisDrone。 | 它是当前非常直接的相关方法。“避免直接复制 IR 特征、迁移 RGB 可预测的 IR 语义、UAV RGB-only”本身不能再当空白。其推理辅助结构与我们零额外学生结构可以区分，但仍需机制比较。 |

补充近期边界：**InfraNet**（2026-07 预印本，非本表已发表论文）已提出 QualGate，以 task-oriented quality 控制 RGB 引导，并包含 IR-only 部署版本，覆盖 LLVIP/FLIR/M3FD/DroneVehicle。[作者预印本](https://arxiv.org/abs/2607.03795) 另 DroneVehicle 原作者的 UA-CMDet 就使用跨模态 IoU 与 RGB 亮度量化不确定性（双模态融合场景）。因此“可靠性/亮度/几何门控”需落实为具体尚待核查的技术差异，不能泛称首创。

以上没有复现其论文主表，不引述其数字为已确认增益；也没有判断哪个方法“必定可靠”。最有用的比较是固定本项目 recipe 下，逐一核对它们的数学损失与实际实现后再复现。

## 3. 本轮 probe 指标如何定义，如何避免误读

### 3.1 配准：分开文件、图像、标签三个层次

1. **文件配对覆盖**：同 ID 两侧文件存在比例、尺寸/方向一致性、重复/缺失率。100% 只说明文件级成对。
2. **图像几何代理**：边缘距离、gradient correlation、phase-correlation shift/response。跨模态材料响应不同，低相关不等于错配；单个平移模型对尺度、旋转、局部视差无能为力；低 response 的 shift 不应当成可信标定值。固定样本图应同时展示原图与跨模态轮廓叠图。
3. **标签几何代理**：只有两侧独立标注时才可作额外证据。先同类一对一匹配，保留所有候选，包括低 IoU/未匹配的质量分布；若先筛 IoU>0.5 再报告 IoU 高，是循环筛选。共享/复制标签应明确 `label_iou_not_independent_registration_evidence=true`。

可比较的尺度归一位移：

`d_center = sqrt((cx_s-cx_t)^2 / w_ref^2 + (cy_s-cy_t)^2 / h_ref^2)`，其中 `w_ref,h_ref` 在协议中固定为两框宽高几何平均，或统一学生框宽高；不能跨脚本混用。再报宽高比的 `abs(log(w_s/w_t))`、`abs(log(h_s/h_t))`，区分平移与轮廓尺寸差。图内密集同类对象会产生错误匹配；必须报告匹配覆盖率和边界/截断情况。

HBB 的 IoU 不能表征 OBB 方向偏差。在航拍车上将有向框转 HBB 会同时扩大背景、缓和小偏移、遮掩角度问题；本项目结论应限定 HBB。

### 3.2 centered spatial linear CKA

对每图、每层，把空间位置作为样本：`X∈R^(n×Cs), Y∈R^(n×Ct)`；沿位置轴分别去均值，`Xc=X-mean_rows(X), Yc=Y-mean_rows(Y)`，计算：

`CKA = ||XcᵀYc||F² / (||XcᵀXc||F * ||YcᵀYc||F)`。

这正是需要的中心化线性形式；其通道旋转/排列不变性适合独立模型整体表征比较。[CKA 原论文](https://proceedings.mlr.press/v97/kornblith19a.html) 但它回答的是空间表征结构是否相关，不是学习该结构会提升 AP。

- P3/P4/P5 使用相同坐标采样原则；插值/下采样方案、token 数必须记录。层间通道数与 token 数不同，不能仅按 CKA 大小给蒸馏层排序。
- FG/BG 各自提取 token 后各自中心化；mask 来源要记录。P5 极小对象常只有 1–3 token，样本太少或中心化方差近零应记 N/A；两个样本的 centered CKA 可退化地接近 1。汇总需给有效样本数。
- 去掉 letterbox/padding 与白边；这些共享低频区域会人为提升相关。
- 汇总逐图分布，不把同图几千像素当独立重复；区间按图像或场景组 bootstrap。相邻视频帧相关时按场景更合适。
- 报 `ΔCKA_pair = CKA(paired)-mean_k CKA(independent donor_k)`，而不只报裸 CKA。随机 donor 可能同时改变光照/类别/目标位置；若只想问实例配对贡献，后续预注册同类/同场景条件 donor。另一种 spatial permutation null 回答空间对应作用，不能与图像 donor null 混为一谈。
- `paired>null` 说明配对带来表征对应；CKA 很高可能冗余也可能任务有用，CKA 很低可能是互补也可能噪声。两端都不能推出“空间大/空间小”。

### 3.3 通道能量热图

建议明示 `E(h,w)=mean_c F(c,h,w)^2` 或 `mean_c abs(F)`，选一种贯穿全报告。热图每图 min-max 后“更亮”不代表该模型原始激活更大；原始数值还受 BN/权重尺度影响，不能跨独立模型直接相减解释知识量。

更有区分力的两个描述量是：`FG_energy_fraction=sum_FG E/sum_valid E` 与 `FG_enrichment=(mean_FG E)/(mean_BG E)`；前者受目标占图面积影响，二者应连同 FG 面积率一起给。每层在对应 GT/预测框边界显示能量质心偏移，以观察模型关注的是目标、热尾迹、车道或背景。能量并非 gradient attribution，“热点在目标上”也不证明该热点对预测有正向因果贡献。

独立训练模型的第 c 通道没有天然同义性；不使用“同编号通道差图”给 shared/private 语义贴标签。

### 3.4 共同目标命中矩阵应当是诊断主体

先建立共同目标集合 `J={(g_sj,g_tj)}`。两侧都存在同类对应 GT，或确认共享标签；分别在各自本模态坐标系，对预测与 GT 做一对一匹配。冻结 conf=0.25/IoU=0.5 作为本轮描述阈值，绝不把该矩阵称 AP。

| 状态 | 可以说什么 | 不能说什么 |
|---|---|---|
| T 命中、S 命中 | 双方当前 detector 都找到目标；再比较定位误差，判断是否有精定位机会。 | S 已命中就没有 KD 空间。 |
| T 命中、S 未命中 | 教师拥有额外检测能力；值得查看 S 是否有弱候选/局部可见证据。 | 这些目标全能被 S 通过 KD 学会，或它们就是学生可达上限。 |
| T 未命中、S 命中 | 当前教师可能给出错误监督，直接强制复制存在风险。 | 所有表示知识都无用，或该对象所有 KD 梯度必定有害。 |
| 两者未命中 | 当前固定模型难例；可能是微小、遮挡、标签噪声、输入缺观测等。 | 数据没有信息或换教师无意义。 |

至少同时报告：`n_Tonly/|J|`、`n_Sonly/|J|`、`n_both/|J|`、`n_neither/|J|`、J 对两模态 GT 的覆盖率。在 both-hit 上给 `δIoU=IoU(pred_t,g_t)-IoU(pred_s,g_s)` 的分位数与正向比例；两 GT 形状差过大时这个差也受标签口径影响，需按标签一致性分层解释。

分类与定位要解耦：另做 class-agnostic GT 匹配，先问“有没有正确位置候选”，再看其类别是否正确；这样可以分清 T-only 是漏检、错类、定位未达 IoU0.5，还是只是置信度略低。标注外 FP 与重复检测也需保留，否则仅 GT 命中矩阵会漏掉负迁移的重要来源。

固定 baseline 可见性 proxy 不等于物理可见性。尤其 LLVIP IR 复制标注的目标，即使 RGB 未命中，也不能仅凭黑图缩略图判它从 RGB 不可识别。需将其写成“本模型当前未学会/弱响应”，而非不可约信息量结论。

## 4. 一个最小、可证伪的干预假设

**假设 H_loc：** 在 DroneVehicle 的 RGB-only 部署任务中，跨模态密集特征模仿把错误的像素/实例对应与教师特异响应一并传给学生；若只在跨模态可对应的共同目标上，转移教师确实更准的、相对于该目标的定位分布，而保留 RGB 的直接分类与检测监督，则能改善学生精定位，同时减少教师更差目标上的损伤。

这是假设，当前 probe 只能判断是否存在值得检验的机会。它不保证收益、不保证减少所有负迁移，也未证明新颖。

**最小起点：** 先保留原 detector、初始化、增强、更新次数与所有监督 loss，仅加真正逐边 DFL 分布的 KL；去掉全图 PCC/关系/分类 KD，选择一个经 token 覆盖与共同目标误差证据支持的层集合。不要再根据“P3 CKA 高”直接丢 P3；航拍小车可能主要由 P3 承载。

首轮用最简单共同 GT 前景匹配，不同时加入 learned gate、对抗分解、频域组件。若 probe 明确存在模态框偏移，可以**另立下一版本**比较原坐标与目标相对坐标的传递，不把坐标修正和 selection 一次全开后只跑一个成败结果。

目标相对坐标版本的可实现含义：教师框边界的离散概率先还原为图像坐标，以对应 GT 的中心/宽高定义单个目标的局部仿射映射，再变回学生 anchor 的边界距离 bins、做概率质量重采样。它迁移的是教师预测相对本模态 GT 的残差与不确定性，不直接把 IR 物体边缘强行当 RGB 边缘。每个尺度 stride、anchor origin、resize/flip 的变换必须明确。仅 resize 两张 feature map 或截断 DFL logits 不等价于坐标对齐。

若试教师质量选择，必须直接乘在 unreduced KL 上，并 `stop_gradient` 于权重，按固定归一化规则控制剂量。先做教师/学生本模态 GT 定位质量差的明确规则，或稳定的冻结参考学生；不能把在线学生置信度高简单定义为可学习性。训练集 baseline 可能过拟合，固定参考学生的选择也需登记，避免把训练集完美拟合当可迁移证据。

**最小 null/对照与其问题：**

| 对照 | 需要排除的解释 |
|---|---|
| 同流水线 native/weight0 | 训练预算、初始化、数据管道或普通 regularization 造成 gain。 |
| same-modal 同剂量、匹配 teacher 容量/训练预算 | 收益只是一般 KD，而非跨模态信息。 |
| paired 与固定 shuffled donor | 配对信息是否必要；shuffled 需 donor 规则与覆盖率匹配，不能仅制造完全错误框使其崩溃。 |
| 常量同平均剂量 / 在同类与尺度桶内打乱 gate | 选择机制是否比简单减少 KD 总量有用；正标量乘 teacher feature 不算这个干预。 |
| GT-only distribution，固定宽度或匹配熵 | 如果使用 GT 对齐/重定中心，增益是否只是软标签/额外 GT 监督。共享标签映射本身带来的收益不能都归教师知识。 |
| 同类 donor 的定位残差/分布交换 | 若 global shuffled 过于恶意，该对照问“配对实例的定位知识”是否超越按类形状先验或分布平滑。 |

不是要求第一轮同时跑所有复杂矩阵：先验证最小 DFL-only 的 matched native/same-modal/paired/shuffled，机制正信号存在后才检验坐标或质量选择，且各自 null 必须跟随。最终增益结论仍须 seeds 0/42/123、mean±SD 与逐 seed，使用 accepted analyzer 和封存 test。

**可证伪预测：** 若 H_loc 是主要机制，最小候选应在共同目标上提高精定位（如 AP75/定位误差及标准 mAP50–95），并在教师质量差的预定义子集减少相对全量 KD 的损伤；paired 应胜 same-modal 和关键 null。若仅 beat 全量 CMDistill 而仍输 native，结论只是修复/减轻既有负迁移。若 beat native 但不胜 GT-only/same-modal，不能叫跨模态实例知识成功转移。若 probe 中 teacher 优势主要是 RGB 无弱候选的 T-only 命中，而 both-hit 精定位差很小，则应降低这一定位候选的优先级。

## 5. 对现有 W1 结论的补充边界

W1 的三种子 matched N/L 负差若身份链完整，可支持“当前全量 KD 在该开发协议下净负”。但“亮/暗两个桶都负”只能说明**按这两个桶对任务整体 gain 的简单解释不成立**。它不能证明目标级/错误类型级质量选择无效，更不能推出“所有 When 都错、只剩 What”。两个桶中的误差异质性可以很大，且当前分桶只是亮度 proxy。应将原 README 的这部分解释降为对简单亮度门控的反证，不扩大成所有门控的否定。

本项目更有效的下一步决策链应为：模型/划分身份 → 共同目标与坐标策略 → 模态特异错误 → 最小知识种类 → 对应 null → 多种子净收益。热图与 CKA 是这条链上的诊断证据，而不是方法收益的替代终点。
