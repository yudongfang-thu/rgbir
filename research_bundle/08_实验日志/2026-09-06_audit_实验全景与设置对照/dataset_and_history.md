# 数据诊断与历史对照：实验全景说明素材（2026-09-06）

> **当前方向有数据依据：DroneVehicle 主要检验 IR→RGB 的对象判别证据迁移；LLVIP 保留条件性定位机会；VEDAI 应按 RGB→NIR 诊断。既有全量 CMDistill 未产生净收益，HNEWA-inspired 对强同模态对照的三 seed 优势很小且不稳定。** 本文件只复核已有证据，不代表新训练结果，也不重新启动 GPU。

## 目的与证据口径

为整体进度说明整理已完成的数据诊断、旧 RGBIR 方法设置和可比较结果。正式训练最新状态由同目录主报告独立记录；本文件不把 9 月 5–6 日历史快照冒充实时状态。

AP 数值均转换为百分数，差值用百分点（pp）。主指标为 mAP50–95，AP50 单独注明。所有 SD 为 sample SD（ddof=1）；比较先按相同 seed 作差，再计算均值和 SD。以下结果属于开发集描述性证据，不能替代 accepted analyzer、封存测试或完整四臂归因。

## 已完成的数据和六个 baseline 诊断

六个 baseline 均为 YOLO11n、seed42、E200、last.pt，输入以原生模态加载并 letterbox 到 640。不是为 probe 重新训练六个模型，而是核对并复用已有有效权重。对共同对象逐一核对，输出完整预测、P3/P4/P5 激活图、正确中心化 CKA、5 个独立 donor 对照，以及局部配准图。

| 数据集 | 数据和训练/开发划分 | 六baseline中的两模态成绩：RGB / IR或NIR | 本轮诊断量 | 合适的研究角色 |
|---|---|---|---:|---|
| DroneVehicle | hbb_v1；17,990 train / 1,469 val；5类 car/freight car/truck/bus/van；840×712 去边后为640×512，OBB转HBB | 单seed CSV mAP **53.798 / 59.659**；AP50 **75.994 / 80.860** | 200对 | 主任务：IR训练期辅助，部署RGB检测器；优先前景/类别判别 |
| LLVIP | grouped_v1；9,619 fit / 2,406 dev；1类person；按文件前缀整组切分；官方test3,463对未用于本probe | 单seed CSV mAP **32.867 / 48.833**；AP50 **71.629 / 92.611** | 200对 | 机制补充：IR→RGB，判别与条件性定位 |
| VEDAI | 512、paper8类、official_fold01；1,089 train / 121 val；test为同一val别名，没有独立封存含义 | 单seed CSV mAP **37.018 / 34.874**；AP50 **63.110 / 61.362** | 全部121对 | 遥感小目标光谱补充；实际为RGB–近红外，应先按RGB→NIR看 |

Drone/LLVIP baseline 使用batch32、nbs64；VEDAI这对模型使用batch64。三组均为SGD、imgsz640、200轮。上表是同一数据集中两模态资源/能力描述；跨模态GT不一定相同，差值不是可蒸馏收益上限。尤其不能将表内 baseline CSV 与后续独立评估端点直接混用。

VEDAI 的 paper80 教师已经从诊断排除：paper80 train 与 official_fold01 val 有 **95/121** 张交集，混用将造成严重污染。现在使用同一 official_fold01 的 RGB/NIR 模型。

正式图像抽样 RNG 固定为20260906；特征示例按RGB亮度10%/35%/65%/90%分位固定选择，共12组双模态图，加3组局部配准图。热图显示的是通道 L2 激活强度，各图独立归一化，不能用红色多少推断知识更好。

## 对象层面比总体 AP 更有价值的发现

命中定义：conf≥0.25、同类IoU≥0.5、一对一匹配。两侧GT先按同类IoU≥0.1对应，再在各自模态GT上判断预测；不是新的AP评估。

| 诊断方向 | 共同对象 | 双方命中 | 教师独有命中 | 学生独有命中 | 双方未命中 | 双方命中时教师−学生IoU均值 / 教师更准比例 |
|---|---:|---:|---:|---:|---:|---:|
| Drone IR→RGB | 3,083 | 2,505 | **360（11.68%）** | **78（2.53%）** | 140 | **+0.00790 / 49.46%** |
| LLVIP IR→RGB | 672 | 484 | **107（15.92%）** | **22（3.27%）** | 59 | **+0.04197 / 60.95%** |
| VEDAI RGB→NIR | 364 | 222 | **41（11.26%）** | **31（8.52%）** | 70 | **−0.00254 / 45.50%** |

这张表支持的设计判断是：

- **Drone：优先教“这是目标、这是哪个类别”，而不是普遍复制IR定位。** 360个IR独有命中中，RGB已有118个低置信但正确类别/位置候选（0.05≤conf<0.25）；另有错类等候选，错误标签会重叠，不能相加。双方已命中的目标上，IR定位优势并不普遍。最暗的50张图含320/360个IR独有命中，说明机会与条件相关，但未控制场景/目标数量，也没有证明整图亮度门控有效。
- **LLVIP：定位确有更明确的候选。** IR在双方命中的目标上平均IoU高0.04197；107个IR独有命中中，严格排除匹配竞争后25个属于RGB低置信正确框。person单类下，“分类知识”主要就是前景判别。IR清楚的对象未必在RGB中有足够可恢复线索。
- **VEDAI：不要默认IR为教师。** RGB总体更强，但精定位没有优势；41个RGB独有目标中，NIR已有20个低置信正确位置候选，15个存在高置信错类框，两者重叠5个。更适合检查类别、评分和小目标的光谱互补。

“教师独有命中”不是学生可达上限；“学生低置信正确候选”不是物理可见性证明；单纯抬高所有分数也未必改变AP排序。新方法需要同时监测漏检改善、背景误检与学生原有优势是否被损伤。

## 配准与特征相似性说明了什么

| 指标 | DroneVehicle | LLVIP | VEDAI |
|---|---:|---:|---:|
| RGB / IR或NIR GT数 | 3,116 / 3,276 | 672 / 672 | 364 / 364 |
| 对应框IoU均值 | 0.904 | 1.000 | 1.000 |
| 对应框IoU<0.5 | 83/3,083（2.69%） | 0 | 0 |
| 两侧标签字节相同 | 38/200 | 200/200 | 121/121 |
| 梯度相关均值 | 0.501 | 0.268 | 0.750 |
| phase全局位移/图像边长，中位数 | 0.00471 | 0.00368 | 0.00038 |
| 前景P3 paired−donor centered CKA | 0.582（200图） | 0.306（200图） | 0.466（93图） |
| 前景P4 paired−donor centered CKA | 0.575（198图） | 0.428（200图） | 0.496（33图） |

整体具有对象对应基础，Drone存在局部标注/几何差异；LLVIP/VEDAI共享标签IoU=1不构成独立配准证明。LLVIP低梯度相关和低phase响应可能来自成像机制，不能直接宣布严重错配。VEDAI全局边缘对应最好，但小目标在P4/P5有效前景token很少；P5仅1图可算，不能据此判定哪层最值得蒸馏。

配对前景CKA大于随机donor说明对象表征具有对应性；不能推出让特征相等就会提升检测。donor同时改变类别、位置和场景，真正的实例配对归因需要更严格的同类/同尺度对照。

## 已有监督KD：设置及结果

### CMDistill-adapted/corrected：多类知识一起迁移

学生为RGB YOLO11n，冻结IR seed42 YOLO11n作为教师，训练时共享几何增强，部署仅保留RGB学生。当前有效实现为协议适配版，不称作原论文逐项复现：

1. P3/P4/P5全图特征 Pearson 相关损失，正确方向为 `1−r`；
2. 最深层各空间位置的cosine关系矩阵，L1匹配；
3. 解码框的 `1−IoU` 与教师概率BCE。框损失仅选教师最大类别概率≥0.5的anchor，分类BCE覆盖全anchor×class；
4. 三大项权重都为1.0，按batch对齐Ultralytics损失剂量。

Drone N/L共有主要recipe：E200、batch32、nbs64、SGD、lr0=lrf=0.01、momentum0.937、weight_decay0.0005、warmup3、imgsz640、deterministic；mosaic/mixup/cutmix/HSV关闭，translate0.1、scale0.5、fliplr0.5。三seed为0/42/123，同一初始化文件路径与学生数据YAML；L记录56,722 optimizer updates。N为标准trainer、L为自定义paired trainer，精确逐batch和初始化tensor/更新数等价没有全部动态闭合；不能只凭目录名b32a2误判budget不同。

| Drone独立last评估，mAP50–95 | seed0 | seed42 | seed123 | mean±SD |
|---|---:|---:|---:|---:|
| N：新native | 54.3576 | 53.8156 | 53.6874 | **53.9535±0.3558** |
| L：CMDistill-corrected | 53.9003 | 53.2448 | 53.6692 | **53.6048±0.3324** |
| L−N（pp） | −0.4574 | −0.5707 | −0.0183 | **−0.3488±0.2918** |

AP50差值为−0.5133/−0.8367/+0.5197，mean **−0.2768±0.7085 pp**，因此只有mAP三seed全负，不能扩大成所有指标全负。seed42按RGB平均亮度中位数划分后，L−N在亮/暗两桶分别−0.623/−0.779 mAP；只能否定该KD在两粗桶获益，不能否定所有条件选择机制。

LLVIP同方法已有三seed独立开发评估 **34.0603 / 33.4440 / 33.3168**，mean **33.6070±0.3976**；本文件没有匹配的三seed新native可用于完整配对差值。不要用Drone的N或HNEWA不完整native来补空。

### HNEWA-CMKD-MSE-inspired：融合教师做全图特征KD

**这里教师不是单IR检测器。** 先用配对RGB+IR按通道拼接，训练六通道YOLO11n fusion teacher（T0，E200，seed42，last.pt）；再冻结教师，把P3/P4/P5（tap16/19/22）的全图MSE按三层均值、α=0.5加到普通RGB学生检测损失中。主recipe与上述协议一致：E200、batch32、nbs64、SGD、imgsz640、相同增强配置。模型文件和代码回执明确标注inspired port，不是exact reproduction。

| 臂 | 输入/目标如何构造 | 这个对照问什么 |
|---|---|---|
| b0 native | RGB直接检测，无KD | KD总体是否有用 |
| h1 paired | 教师输入当前RGB+正确配对IR | 成对跨模态教师能否帮助 |
| h2 shuffled | 保持当前RGB，只把教师IR半边换为固定donor | 配对信息是否有贡献；也可能因错误融合使该臂受损 |
| h3 same-modal | 冻结三通道RGB native seed42教师，只读RGB，使用相同特征MSE | 跨模态是否超过普通自模态KD |

三学生KD臂均有0/42/123独立last评估。b0只有seed0/123来源明确，seed42仅找到`_best`及`/tmp/best_probe/last.pt`不清端点，已排除；不能拿别campaign的N42补成四臂三seed完成。

| 数据集 / 臂，mAP50–95 | seed0 | seed42 | seed123 | mean±SD（完整三seed） |
|---|---:|---:|---:|---:|
| Drone paired | 54.3757 | 53.9132 | 54.1188 | **54.1359±0.2317** |
| Drone shuffled | 53.8095 | 53.9021 | 53.6880 | **53.7999±0.1074** |
| Drone same-modal | 53.8426 | 53.6472 | 54.2976 | **53.9291±0.3337** |
| Drone native | 54.3631 | 端点未闭合 | 53.6801 | 不给三seed均值 |
| LLVIP paired | 35.2280 | 33.7603 | 35.0475 | **34.6786±0.8004** |
| LLVIP shuffled | 34.9325 | 34.6128 | 33.4408 | **34.3287±0.7854** |
| LLVIP same-modal | 34.9785 | 34.3246 | 34.4601 | **34.5877±0.3451** |
| LLVIP native | 34.7709 | 端点未闭合 | 34.9442 | 不给三seed均值 |

| 同seed对比（mAP pp） | seed0 | seed42 | seed123 | mean±sample SD | 正向 |
|---|---:|---:|---:|---:|---:|
| Drone paired−shuffled | +0.5662 | +0.0111 | +0.4307 | **+0.3360±0.2894** | 3/3 |
| Drone paired−same-modal | +0.5331 | +0.2660 | −0.1788 | **+0.2068±0.3597** | 2/3 |
| LLVIP paired−shuffled | +0.2955 | −0.8525 | +1.6067 | **+0.3499±1.2305** | 2/3 |
| LLVIP paired−same-modal | +0.2495 | −0.5643 | +0.5874 | **+0.0909±0.5920** | 2/3 |

仅共同两seed 0/123时，paired−自身native：Drone +0.0126/+0.4386，**+0.2256±0.3012**；LLVIP +0.4571/+0.1033，**+0.2802±0.2502**。两seed描述不能满足本项目稳定增益最低要求，也不能与上面三seed臂平均直接相减。

这些结果说明普通全图模仿没有稳定获得强同模态教师以外的独特收益。paired>shuffled的差值可以来自有用配对，也可以部分来自shuffled伤害；必须同时看native和same-modal。

## 其他已有尝试及不能继续沿用的判断

| 项目 | 已做范围 | 当前可以说什么 |
|---|---|---|
| P3 causal v1 | Drone/LLVIP seed42，native、paired、same-modal、shuffled、random-dose，E200历史CSV | Drone分别53.798/53.509/54.095/53.936/54.367；LLVIP分别32.867/33.532/35.209/35.398/33.308。仅历史单seed、不同方法协议；LLVIP超过native仍输same-modal/shuffled，不能作跨模态独特收益证据 |
| CCLKD-adapted | Drone三seed训练完成，E200，每臂56,722更新 | 当前已核实快照未找到对应独立final检测JSON；loss约0.693或训练完成不能判成有效或无效 |
| 旧P2特征probe | 已做过图和CKA | native实际混入VEDAI模型、CKA未中心化；“蒸馏拉力+0.32”“P3饱和”等结论撤回，由六baseline新probe替代 |
| CGA-KD | 有预注册、canary及W1 native；W2冻结 | G并非所称DFL；C门控缩放被归一化抵消；I有通道错误；这些是实现偏差，不能当选择性蒸馏思想已被证伪 |

SAR历史只作为转向的背景：早期LADD的部分提升可由reload/继续训练解释；后期强同模态锚与配对对照显示监督KD净效应常小或不稳定，但不能证明所有SAR数据都无蒸馏空间。SpaceNet6 OS-SSL v2历史仍有paired−native **+3.328±0.909 AP50**、3/3正的独立正面证据，应与监督检测KD分开。

**旧“+9”与“+12.7”都不再作为路线判断依据：** FreqMix“SiXiang+9”实际发生错数据集比较，复核到的是OGSOD E400，gray/SAR donor相对现存主要recipe匹配native为−2.621/−2.628 mAP（两seed）；HNEWA“+12.7”来自precision误作mAP，真实paired−shuffled仅+0.336±0.289 mAP。前者有数据集身份错误，后者有指标列错误，不能以它们证明成功，更不能只把收益解释成增强而保留旧幅度。

## 当前数据覆盖和下一步边界

FLIR-aligned已解压，4类原版与去dog的3类版本需先冻结，暂未核实可用双模态baseline与独立dev。M3FD有4200对检测标签，但附带meta的train/val/pred均相同42个ID，不能直接拿来做检测划分。KAIST采集时仍为下载.part。该状态来自此前快照，当前准备状态如有更新应另查；这里没有把它们写成已完成的特征诊断数据集。

当前较合理的投入顺序是：先检验Drone上的单一对象判别知识是否超过同代码weight0，再补same-modal、合适shuffled与同总剂量随机选择来检验归因；其后以LLVIP检验定位机会和条件差异，以VEDAI检验方向变化与光谱小目标。现有观察支持这个实验顺序，不保证结果，也没有证明新颖性。

## 原始证据与复核入口

- [模型/划分身份清单](../2026-09-06_probe_RGBIR数据特性与可迁移知识/dataset_model_inventory.md)；同目录JSON保留远端args、YAML、数据治理记录。
- [521对完整诊断报告](../../07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md)、[特征图册](../2026-09-06_probe_RGBIR数据特性与可迁移知识/特征图册.md)、[probe_analysis.json](../2026-09-06_probe_RGBIR数据特性与可迁移知识/probe_analysis.json)。
- [当前结果复核](../2026-09-06_probe_RGBIR数据特性与可迁移知识/current_results_notes.md)、[N/L原始来源快照](../2026-09-06_probe_RGBIR数据特性与可迁移知识/current_results_sources.json)。
- [HNEWA/CMD逐seed独立评估表](../2026-09-05_audit_跨模态蒸馏全项目复盘/rgbt_eval_rows.csv)、[原始JSON与代码快照](../2026-09-05_audit_跨模态蒸馏全项目复盘/rgbt_eval_and_code_snapshot.json)、[原始args与训练回执快照](../2026-09-05_audit_跨模态蒸馏全项目复盘/rgbt_readonly_snapshot_v2.json)。
- [SAR方法史和证据限制](../2026-09-05_audit_跨模态蒸馏全项目复盘/history_notes.md)、[全项目复盘](../../07_研究分析/全项目复盘与研究诊断_20260905.md)。

本次只读原件；新增此说明，不改原始results、receipt、checkpoint。本文表格均从上述原始小产物和已复核诊断提取；没有访问test、引入新阈值、补造缺失端点或作跨campaign因果排名。
