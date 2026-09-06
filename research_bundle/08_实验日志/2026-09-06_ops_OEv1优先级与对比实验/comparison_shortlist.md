# OEv1 优先阶段的最小对比清单（2026-09-06）

> **建议优先完成 OEv1 同代码 P/N 三 seed，再并行补 CCLKD 现有权重的独立评估与一个同剂量选择对照；新增外部方法首选 FGD。CMDistill、CCLKD、FGD 构成最小候选外部清单，是否将 CCLKD 列为完整方法取决于实际训练源码审计。MGD 为备用，CrossKD 后置。** 本文件是代码/文献和成本评估，没有启动 GPU、修改队列或宣称任何新收益。

## 1. 推荐投入顺序

| 优先级 | 项目 | 最小新增工作 | 能回答什么 | 成本与边界 |
|---|---|---|---|---|
| P0 | OEv1 paired 与同代码 weight0，0/42/123 | 完成正在进行的六个端点、统一独立 last 评估 | 整套干预是否有净收益 | 现有主线；不改参数，不拿历史 native 替换未完成同代码对照 |
| P0 | CCLKD 已完成三 seed 的评估与身份审计 | 核实实际 trainer/配置/权重，补三个独立 last eval | 已有外部适配是否可用于比较 | 评估通常远低于重新 E200；先核实端点，不依据 loss≈0.69 判输赢 |
| P1 | OEv1 同数量、同分母随机选择 | 冻结抽样空间/随机种子，canary 后先启一个 seed | 选择规则相对等剂量干预有无作用 | 使用现有冻结教师、参考和 paired loader；完整增益需要补足三个 seed |
| P1 | OEv1 same-modal | 在同预算下用 RGB 教师产生对象判别目标 | IR 是否超过普通 RGB 教师/自模态辅助 | 独立协议，不能仅把一个张量别名改为 RGB 就认为控制成立 |
| P1 | CMDistill-adapted/corrected | 复用三 seed 独立端点，核对初始化/teacher/数据与评估；必要时统一 trainer 重跑 | 全图多层/关系/响应知识迁移能否竞争 | 已有工程与结果，低增量成本；历史结果是主要 recipe 一致的参照，不自动等同 OEv1 精确运行路径 |
| P2 | FGD-YOLO11 cross-modal adapted | 对照官方完整四项 loss，接入当前 paired trainer；CPU 梯度/数值及 GPU canary；冻结权重后 E200×3 | 对象区域与注意力引导的特征 KD 能否解释 OEv1 收益 | 中等工程；共享现有单 IR 教师，不需新训融合教师；仍需实测显存 |
| P3 | MGD adapted（备用） | 官方随机 mask + generation block，接入相同特征层 | 通用表征重建正则是否已经足够 | 相对 FGD 模块简单，但与 OEv1 的判别选择问题关联较间接；不建议同时铺开 |
| P3 | CrossKD adapted（后置） | YOLO11 分类/DFL 分支的 cross-head 路由与梯度审核 | 任务空间蒸馏/监督冲突处理能否竞争 | 头路由与优化路径改动较大；不能把普通预测 KD 改名 CrossKD |

上述成本为结构性相对判断，没有对新方法做吞吐、显存测量，不是 GPU 时数承诺。对比不应先全部单 seed 筛掉不利结果再只给有利者补 seeds；正式比较的配置、展开条件和报告范围应先冻结。

## 2. 三个外部方法为什么合适

### CMDistill：任务领域最直接，已有结果应充分复用

本地 `yolo_osssl/rgbt_cmdistill_kd.py` 与 `tools/train_rgbt_cmdistill.py` 可复用。现有方法包含全图 Pearson 特征相关、深层空间关系、框 IoU 与分类 BCE，冻结 IR 教师、部署 RGB 学生。当前有效身份为 corrected/adapted：原文印刷形式中的相关/IoU最小化方向经过修正，一些 YOLO11 选择和损失权重由项目显式规定。不能写成作者代码精确复现。

已核实 Drone 三 seed 独立 last：native **53.9535±0.3558 mAP**，CMDistill **53.6048±0.3324**，同 seed 差 **−0.3488±0.2918**。负结果不应删去，也不构成刻意选择弱对手的理由：接入主表前仍需确认公平设置，并保留额外有竞争力的通用方法。

论文入口：[CMDistill，DOI 10.1109/JSTARS.2024.3479717](https://doi.org/10.1109/JSTARS.2024.3479717)。本次 DOI 网页抓取失败，方法细节来自已存本地论文阅读和代码，不以网页访问失败推断论文或代码不存在。当前工作区未取得经核验的作者官方实现。

### CCLKD：领域相近、较新，先把已跑实验审清楚

论文采用易检测模态帮助难检测模态，包含自适应温度及分类约束对比，涉及 logit/feature/relation 多层知识。来源：[出版社原文](https://www.tandfonline.com/doi/full/10.1080/10095020.2026.2633014)。因此值得保留为外部比较。

本地有 `rgbt_cclkd_kd.py`、`train_rgbt_cclkd.py`、对应测试与训练记录，但有两个不能忽略的版本问题：

1. 本地旧 `configs/research/rgbt_cclkd_protocol_drone.yaml` 是 512/batch16/mosaic1；**并行审计已确认实际已跑三 seed 是 640/batch32/nbs64/E200、SGD，与当前主线主要增强一致**。旧配置不能代表已跑设置，最终引用以 94 实际 run receipt 为准。
2. 本地现行 trainer 的头注释与调用显示 v1 只接入 **LLD+CCL**，FLD/RLD 尚未接入；loss 库有函数不等于已训练调用。实际 94 已跑 trainer 是否相同，必须通过回执源码确认。若是局部版本，主表应标 `CCLKD-adapted (LLD+CCL subset)`，不能冒充完整 CCLKD，也不能用它的负结果否定全文方法。

因此，先做三次独立评估是合理低成本工作；是否补完整 CCLKD 的统一协议训练，取决于源码身份审计和资源预算，不应直接复制历史名字开新长训。

### FGD：最值得新增的通用对比

FGD 将前景、背景分开，利用空间/通道注意力并加入全局关系。它直接针对“哪些特征区域更值得蒸馏”，比再增加一个无选择全图 MSE 更能检验 OEv1 的差异。来源：[官方代码](https://github.com/yzd-v/FGD)、[官方 loss 实现](https://raw.githubusercontent.com/yzd-v/FGD/master/mmdet/distillation/losses/fgd.py)。

移植必须保留四项结构：前景特征、背景特征、注意力差、关系项。官方源码默认 temp=0.5，四项权重依次 0.001/0.0005/0.001/0.000005；这些是**源码默认值**，不等于所有官方检测器推荐配置，更不保证在 YOLO11 的 batch reduction 下剂量合适。最终选择应核对相应官方 config、确定 sum/mean/按 batch缩放语义，预注册合理的固定配置或相同调参预算，不能盲抄数值后宣布 FGD 不行。

当前本地定向搜索只找到旧 SAR 的 FGD 结果 CSV及论文资料，没有找到已经审计可直接启动的 RGBIR FGD trainer。推荐接入已有 OEv1 paired loader和统一检测训练路径，但新建方法身份和 canary；不能宣称此刻已可直接跑正式比较。

## 3. MGD、CrossKD 与融合教师为什么暂不优先

**MGD**：随机遮盖学生特征，经 generation block 重建教师完整特征，官方公开实现。它适合补充通用生成式 KD 基线，结构容易理解，但当前不需要与 FGD 同时扩大战线。来源：[ECCV 论文页面](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/140_ECCV_2022_paper.php)、[作者代码](https://github.com/yzd-v/MGD)。

**CrossKD**：把学生中间特征送入教师的部分检测头，匹配 cross-head prediction 与教师 prediction；其动机涉及缓解检测监督与蒸馏监督的冲突。必须正确处理冻结教师参数、仍允许梯度经过教师头回到学生、BN状态、分类和 DFL 分支。没有这条路由就不是 CrossKD。对 YOLO11 的移植与验证投入大于直接特征模块，目前后置。来源：[CVPR 2024 论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf)、[作者代码](https://github.com/jbwang1997/CrossKD)。注意这是 cross-head，名称本身不表示跨传感器模态方法。

**HNEWA-inspired**：已有 RGB+IR 六通道融合教师和三 seed KD 端点，可以作为另表参照；教师信息量和结构与 OEv1 单 IR 教师不同。不要为增加主表行数重训一个新融合模型，也不要把融合教师学生的成绩与单 IR 教师学生混写成相同教师能力下的 loss 比较。

## 4. 公平协议应明确写成什么

| 方法/臂 | 教师输入 | 训练 KD 时是否另外访问 IR GT | 额外训练资源 | 推理输入 |
|---|---|---|---|---|
| OEv1 | IR | **是**，对象对应、IR区域与教师正确性检查 | 冻结 IR 教师 + 冻结 RGB 参考 | RGB |
| OEv1 weight0 | IR辅助路径执行但loss系数0 | 相同诊断路径 | 同上 | RGB |
| CMDistill 当前适配 | IR | 当前核心 loss 不额外读取 IR GT | IR 教师 | RGB |
| CCLKD 当前本地局部适配 | IR | 当前核心 LLD+CCL不额外读取 IR GT；实际回执待确认 | IR 教师与温度等训练模块 | RGB |
| FGD 建议主适配 | IR | 前景mask用 RGB GT；若改用双侧GT须另列变体 | IR 教师、训练期关系模块 | RGB |
| MGD 建议适配 | IR | 核心 mask/reconstruction不需要 | IR 教师、训练期生成模块 | RGB |
| CrossKD 建议适配 | IR | 依分类/定位监督协议说明，不能默认需要双侧GT | IR 教师头的可微辅助路径 | RGB |
| HNEWA-inspired | **RGB+IR融合** | 融合教师训练标注需单独登记 | 另训六通道教师 | RGB |

“KD 时不读取 IR GT”不表示整套方法完全不用 IR 标注：这些 IR 教师通常都先用 IR 标注训练。表中区分的是**学生训练阶段新增的对象级标注访问**。

主比较建议固定 Drone同一 train/dev/封存test、RGB YOLO11n学生、完全相同 seed 对应初始化张量、同一个冻结 IR teacher checkpoint hash、输入640、E200、batch32/nbs64、同SGD及增强、统一last-EMA独立评估。固定0/42/123，报告逐seed和mean±sample SD，mAP50–95与AP50分列，所有失败/偏差保留。

外部方法可以有其本质所需的额外模块与损失层，但要记录训练 FLOPs/耗时/峰值显存与推理参数；不能强行删去原方法关键项来“统一”。同样，不能只给 OEv1 大规模调参，而让基线使用未校准的默认剂量。主表若只按方法原生推荐层配置运行，P3/P4与P3/P4/P5的差异要说明；“只因知识类型而获益”的机制判断另用共享支持集/层数的对照。

## 5. 机制对照如何避免再制造混杂

- **随机选择**：必须写清从哪个集合抽 K 个。当前 loss 中 `paired_random` 从基础 E 抽与P相同数量，可同时检验教师正确性筛选与优势排序的整体作用；若想单独检验排序，需要另设从eligible中随机抽K，不能将两者混称。分母仍为相同基础 E，固定抽样随机流，记录每batch K和总KD剂量。
- **same-modal**：保持对象证据算子、检测初始化和预算。若把教师直接设为同一冻结 RGB 参考，原 q=(reference误差−teacher误差)+ 恒为0，会把 same-modal 变成零损失；必须预先规定适当的共同选择mask或独立同模态教师，报告新增差异。
- **shuffled**：不能打乱整张IR后导致跨模态GT无法匹配、K几乎为0，再用P胜出证明配对内容有效。实例级同类/尺度 donor 需保持ROI与目标剂量，冻结规则并记录其同时改变的因素。
- **GT-only**：使用与OEv1相同的对象集合与mask，但教师软证据换成预先固定的标签目标/对应监督，可以帮助判断增益是否主要来自额外IR标注可得性或难例加权。实现前冻结具体目标和剂量，不能看AP后设计有利版本。
- **全量对象证据**：可检查是否真需要选择，但若all-E与top-half分母不变，总剂量自然变化；需记录并设置剂量匹配版本，不能把剂量差全部解释成选择效果。

## 6. 原创性表述边界与当前范围

OEv1 是我们提出并实现的**候选方法**，应优先验证；这与“已证明文献新颖且有稳定收益”是不同命题。前景/背景区分已有FGD，任务输出空间与监督冲突已有CrossKD，实例筛选和教师质量选择也存在相关研究。对象局部logit证据、冻结RGB参考的可学性代理、质量筛选及剂量控制这一具体组合，仍需针对最近邻方法做正式查新和机制对照。本文件不是完整 novelty audit，不宣称首次。

本阶段将 VEDAI 作为已完成诊断的归档即可，不新增其训练、反向教师或跨折实验。以 Drone主结果和LLVIP后续验证构建连续的 RGB–热红外问题，比现阶段同时展开 RGB–NIR 方向更集中。这个范围决策来自当前投稿目标与资源优先级，不代表 VEDAI 无研究价值。

OS-SSL 可降为已完成/已启动工作的外部参照，不必继续把其完整九臂与初始化修复当作OEv1论文的前置条件。是否让正在接近完成的一次微调自然结束，由实时队列/保存状态决定；没有必要因为已投入成本继续填满整个矩阵。本子任务未操作队列。

## 7. 本地复核入口

- [前一轮全景报告](../2026-09-06_audit_实验全景与设置对照/README.md)、[逐seed历史与数据](../2026-09-06_audit_实验全景与设置对照/dataset_and_history.md)。
- [CMDistill loss](../../03_现行工程/SpaceNet6_OTD_official_reproduction/yolo_osssl/rgbt_cmdistill_kd.py)。
- [CCLKD loss库](../../03_现行工程/SpaceNet6_OTD_official_reproduction/yolo_osssl/rgbt_cclkd_kd.py)、[本地局部trainer](../../03_现行工程/SpaceNet6_OTD_official_reproduction/tools/train_rgbt_cclkd.py)、[旧协议配置](../../03_现行工程/SpaceNet6_OTD_official_reproduction/configs/research/rgbt_cclkd_protocol_drone.yaml)。
- [OEv1对象证据实现](../2026-09-06_train_RGBIR对象判别蒸馏首轮/code/object_evidence_loss.py)、[正式训练配置](../2026-09-06_train_RGBIR对象判别蒸馏首轮/code/config_drone.yaml)。

本报告基于2026-09-06工作区文件与当日官方网页/作者代码只读核验；实际实验运行设置以同期94回执为准。
