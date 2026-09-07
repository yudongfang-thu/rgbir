# 外部对比准备：BCKD、FGD、LD

**结论：三类作者资产足以继续做可解释的 YOLO11n 协议迁移，当前仅完成文献与源码准备，没有实现或启动外部训练。优先比较 BCKD 的分类分量；FGD 和完整 LD 保留后续队列身份。已有 CMDistill / CCLKD partial 三 seed 只能保留历史参照，不能替代当前 workers4 的正式比较分母。**

日期：2026-09-07。执行者 `/root/review_matrix_spec`。范围仅公开文献、CPU 读取、现有结果审计，不含 SSH、GPU、新 AP、权重下载或原文数字复现。本文是外部准备线，不扩大当前 C1/L1 核心矩阵，不触发自动训练。遵循子工程 AGENTS §12–21；缺独立补充文件、作者回复或原条件资产不锁住本项目。

## 1. 检索覆盖与资产状态

先读本地 `01_文献` 的 FGD / LD PDF，以及 `2026-09-07_audit_native与蒸馏载体/{logit_literature,feature_literature}.md`，再补官方原文与作者实现。FGD 使用已有 10 页 PDF 提取方法与实验设置；没有新下载 arXiv PDF。所有 22 位作者分别用“方法 + 姓名 + GitHub”检索；完整查询及返回地址在 `external_baseline_search_log.json`。同名账户不冒充作者身份，未确认的个人账户不写成“已逐仓排除”。

| 方法 | 原文 / 补充阅读范围 | 作者实现与备用入口 | 可得性结论 |
|---|---|---|---|
| BCKD，ICCV 2023 | 正文方法、附录 A 算法 / B 实现 / C 补充实验 / D 可视化；CVF 独立补充可读 | TinyTigerPan/BCKD 的 main；配置、LDHead、NovelKDLoss | 代码、配置公开；权重链接存在但未下载核验 |
| FGD，CVPR 2022 | 已有正文 Eq.1–13、§4.2；单独补充未核实 | yzd-v/FGD 的 master 与作者 yolox 分支；FeatureLoss、distiller、YOLOX 配置 | 核心实现完整；不能把“单独补充未定位”写成“无补充” |
| LD，CVPR 2022 | arXiv 2102.12252v4 正文 §3 与附录 A1–A3 | HikariTJU/LD 的 main；Zzh-tju 的 LD / Rotated-LD 备用线；OpenMMLab 收录 | 区分 CVPR 2022 与 TPAMI 扩展；不把 OpenMMLab 精简版自动等同完整 Main+VLR |

来源：[BCKD 正文](https://arxiv.org/html/2308.14286v2)、[BCKD 补充](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Yang_Bridging_Cross-task_Protocol_Inconsistency_for_Distillation_in_Dense_Object_Detection_ICCV_2023_supplemental.pdf)、[FGD 正文](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.pdf)、[LD 正文与附录](https://arxiv.org/html/2102.12252v4)。

公开 API 检索了三仓的 repository、forks、releases、issues、源码树，各列表限制首 100 条；结果原始小记录和脚本为 `external_baseline_availability.json` / `external_asset_probe.py`。本次返回 BCKD 2 个 fork / 4 issue、FGD 50 个 fork / 99 issue（首页剔除 PR）、LD 53 个 fork / 81 issue；均无 release。数量是读取时快照，不是永久状态，也不意味着每个 fork 都经过实现等价审计。

FGD 的 [Ultralytics issue #95](https://github.com/yzd-v/FGD/issues/95)、LD 的 [YOLOv8 issue #59](https://github.com/HikariTJU/LD/issues/59) 提供第三方迁移线索，未建立作者认可的 YOLO11n 完整复现。检索还覆盖 [MMDetection LD](https://github.com/open-mmlab/mmdetection/tree/v2.28.2/configs/ld)、[MMRazor 方法合集](https://github.com/open-mmlab/mmrazor/tree/main/configs/distill) 和 [作者 Rotated-LD](https://github.com/Zzh-tju/Rotated-LD)。第三方 issue 中的失败或调参描述只作为工程线索，不作为方法有效性证据。

FGD 的 arXiv HTML 和尝试的 CVF supplement 地址未成功取得；已回退本地正文、官方 raw 代码、作者 yolox 分支及公开检索。未取得独立补充的事实已保留，不为此停止准备。三仓当前可读，不需要依赖不明镜像执行代码。运行前以日期、路径和源码原字节副本冻结实际用到的作者文件，不能只留下可变 main/master 链接。

## 2. BCKD：最接近 C1 的分类响应近邻

BCKD 把每个位置的类别视为独立二分类，保留 sigmoid 绝对分数。其分类损失是教师 soft target 的 BCE，乘 `abs(sigmoid(zT)-sigmoid(zS))^beta`，默认 beta=1。完整方法另有教师 / 学生解码框的 IoU 类定位项；因此只移植分类项时必须标 `BCKD-BCDL (classification-only, partial)`，不能写“完整 BCKD”。[作者仓库](https://github.com/TinyTigerPan/BCKD)

**源码差异必须保留。** `novel_kd_loss` 中教师 target detach，但差值权重的学生侧不 detach。加权 BCE 与加权 Bernoulli KL 此时不具有相同梯度：两者相差的教师熵项仍乘依赖学生的权重。不能替换为 KL 后声称作者原算子。类中保存的 T/threshold 不进入这个分类 forward，实际分类温度是 1；不能因构造参数有 T=10 就误写温度 10。[分类源码](https://raw.githubusercontent.com/TinyTigerPan/BCKD/main/mmdet/models/losses/kd_loss.py)

作者选定 GFL 配置：分类系数 1；定位是 **GIoULoss 系数 4**，不同于正文算法文字的 `1-IoU`；DFL-KD、VLR-KD、feature imitation 系数均为 0。定位的类别差权重通过 detach 得到。分类 / 定位在有效 dense 位置计算，以正样本计数归一；源码还有该层无正样本时蒸馏归零的行为。移植不能用我们的 E/K/质量门替换后仍只写 BCKD。[配置](https://raw.githubusercontent.com/TinyTigerPan/BCKD/main/configs/bckd/bckd_r50_gflv1_r101_fpn_coco_1x.py)、[head](https://raw.githubusercontent.com/TinyTigerPan/BCKD/main/mmdet/models/dense_heads/ld_head.py)

对当前研究的含义：独立 sigmoid 类别响应本身不是新贡献。C1 可争取的差异在于冻结参考下的对象选择、目标 / 非目标类别内容、对象分母及其跨模态净收益。BCKD 的差值加权也不自动判定“教师正确且对学生有用”，所以它适合检验我们的条件选择是否比响应差异权重更稳。

## 3. FGD：高容量特征迁移的真实对照

FGD 同时包含前景特征、背景特征、空间 / 通道注意力及全局关系四部分。GT 框形成区域和面积归一权重，教师注意力对特征项加权；全局项通过可学习上下文变换比较特征。它不是仅在一个局部块上做 MSE；只实现前景块就应标 partial。[FGD 正文](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.pdf)

优先参照作者 **yolox 分支**：三层 neck、温度 0.5、`alpha=.002, beta=.001, gamma=.001, lambda=.00001`、`init_student=False`。这比随手取 master 的默认 anchor-based 系数更贴近当前三层检测头；仍不等于 YOLO11n 作者复现。[作者 YOLOX 配置](https://raw.githubusercontent.com/yzd-v/FGD/yolox/configs/distillers/fgd/fgd_yoloxl_distill_yoloxm_coco.py)

实现重点：按增强后 canvas GT 投影到每一层，保留框面积缩放、重叠取最大权重、背景归一及 sum/batch reduction。不同通道用 1×1 对齐，同通道不凭空增添适配层。冻结的是教师检测器；FGD loss 内 teacher-side context transform 仍是可训练辅助参数，须与 student-side transform 一并进入 optimizer。它们不进入部署检测器 / EMA 检测头。漏掉全局辅助参数会造成“有 loss 无有效 FGD”。[FeatureLoss 源码](https://raw.githubusercontent.com/yzd-v/FGD/yolox/mmdet/distillation/losses/fgd.py)

对用户“选好对象后传特征是否更好”的判断：已有文献支持特征包含可迁移信息，但 FGD 不能直接证明我们的局部对象载体会更好。特征同时包含类别、位置、背景、纹理，容量增加也可能放大模态不匹配。先完成 C1 / C1_y 与定位的内容归因；FGD 用于比较完整特征迁移家族。若以后另测同 E/K/门的局部特征，只能作为新的受控载体实验，不能混在这次 FGD 名下。

## 4. LD：定位 logits 早有先例，区域和剂量是完整实现的一部分

LD 将四边距离分布做温度 softmax 后传递；Main 为分配正样本，VLR 由与 GT 的 DIoU 区间得到，常用下界比例 .25。作者完整 head 的有效配置还包含 Main 分类 KD，VLR 分类 KD 为 0。故我们 L1 的同 anchor DFL KL、随机定位选择或 GT 两 bin 控制都不是“完整 LD”。[LD 方法与附录](https://arxiv.org/html/2102.12252v4)

当前作者源码：Main / VLR 定位温度均 10、系数均 .25；Main 类别 softmax KD 温度 2、系数 10；不启用 feature imitation。KL 实现是 **bin/class 维 mean**，不是 sum，随后乘温度平方；Main 定位四边以 4 归一、VLR 以 16 归一；Main 使用 detach 的学生最大 sigmoid 分数权重，VLR 使用区域权重。源码末端没有再用 bbox 的质量分母归一这两个 LD 项。以上 reduction 必须单独核对，不能直接复用 L1 对象分母。[head](https://raw.githubusercontent.com/HikariTJU/LD/main/mmdet/models/dense_heads/ld_head.py)、[KD loss](https://raw.githubusercontent.com/HikariTJU/LD/main/mmdet/models/losses/kd_loss.py)、[配置](https://raw.githubusercontent.com/HikariTJU/LD/main/configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py)

GFL 的 `reg_max=16` 表示 **17 bins**；本项目 pinned YOLO 为 **16 bins**。保持本项目 T/S 原生 16 bins，不把 GFL 权重或分布硬塞入 YOLO，也不把 mean/sum 差异视为无影响。原作者仓库 [issue #74](https://github.com/HikariTJU/LD/issues/74) 也指出这一 reduction 差别；我们的解释以实际源码为准，不把 issue 结论替代验证。

YOLO 的 TAL 没有可直接套用的 ATSS IoU 阈值。确定的迁移方案是：**native TAL 完全保留；KD 旁路使用作者 ATSS topk=9、方形尺度 8×stride，在 P3/P4/P5 原生中心生成辅助框，计算 Main / VLR。** 旁路只决定蒸馏位置，不改变 native 标签分配、框回归或分类 loss。保留作者 VLR DIoU 与原始 IoU 阈值计算的细节，不事后换成 L1 质量门。[作者 assigner](https://raw.githubusercontent.com/HikariTJU/LD/main/mmdet/core/bbox/assigners/atss_assigner.py)

LLVIP 单类下跨类别 softmax 分类 KD 恒为零；不能宣称完整 LD 的分类内容得到验证。本次外部准备默认 DroneVehicle，若未来迁移 LLVIP，须显式记录退化而非偷偷替换成 Bernoulli KD。原文附录的 self-LD 也说明同模态蒸馏可能有收益，不能用 paired−N 自动归因于 IR。

## 5. 固定待实现配置与顺序

以下为 `PREPARED / NOT_IMPLEMENTED / NOT_QUEUED` 的配置约定，配套 `external_baseline_proposals.json`；不是当前 trainer 可直接接收的运行配置。保持 `PROTOCOL-ADAPTED` 身份，不使用 `AUTHOR-EXACT`。

公共协议：DroneVehicle、YOLO11n RGB student、与现有 N 相同通用初始化、同一 IR teacher seed42；原 split / 配对映射、E200、640、batch32、nbs64、workers4、SGD、学习率及增强均从当前 N 的有效配置继承；seeds 42→0→123，全程 E200，不据 AP 早停。固定 last/EMA，完整 dev=1469，主指标 mAP50–95，配对 pp 与样本 SD；教师无梯度，不进 optimizer / EMA / 部署模型。双模态几何增强同步，辅助模块初始化不能推进学生样本 RNG。

| 优先级 / ID | 固定 payload / 范围 | 初始系数与规则 | 必须保留的身份 |
|---|---|---|---|
| 1：BCKD_BCDL | P3/P4/P5 dense 分类响应；native TAL 正样本计数用于协议归一，无 C1 E/K 门 | 作者 beta=1、有效 T=1、分类系数1；差值权重学生侧有梯度 | classification-only partial；完整 BCKD 定位项不在此配置 |
| 2：FGD | P3/P4/P5 检测头前特征；RGB GT 区域，四项全开 | 作者 yolox 系数 .002/.001/.001/.00001，T=.5；sum/batch | full FGD mechanism，YOLO11 / 跨模态协议适配 |
| 3：LD_Main_VLR | 三层辅助 ATSS Main+VLR；定位+Main分类，native TAL 不变 | LD .25/.25、T=10；Main类别10、T=2；VLR类别0、feature0 | full selected-author LD mechanism；16 bins 与旁路 assignment 为适配 |

定位 / dense feature 对齐直接使用当前配对同步增强的坐标，不继承我们的教师正确性筛选，不把 L1 几何覆盖证据冒充所有 dense 区域的配准真值。须报告未经质量选择的朴素跨模态迁移身份与空间错配影响；其负结果不等于对作者原同模态检测压缩方法的否定。

**损失接入约定先于 AP：**外部算子显式输出平均每图的 KD 量，再按 `native_total.sum() + actual_B * KD_mean_per_image` 接入。本计划冻结作者内部各项相对系数，不安排按 AP 搜索权重。BCKD 保留正样本计数分母，FGD 已是 sum/B；LD 将作者批量求和型定位项改为图平均以适配 batch32，分类保留作者正样本 reduction。必须用固定教师/学生小张量分别给出作者原 reduction 与适配 reduction 的值及 B 倍关系，不能对已经 batch-normalized 的项二次除 B。这个批量约定是明确的协议改动，不声称数值剂量与作者 8 卡每卡2图完全相同。

未来启动前的必要工作限于：作者算子固定输入与梯度对照、实际 raw shape/坐标、全项非零、teacher 生命周期与辅助 optimizer、weight0 native 等价、各新路径至少24次成功 optimizer update、峰值 VRAM/RSS、旧/扩展 evaluator 固定 AP 等价。记录 native/KD 梯度量，**不根据新 AP 放大系数或放宽选择**。数值有限且有梯度的负 AP 不是停止理由；数据/指标/无效梯度/资源问题只停止对应 attempt，留存后修复。

LD / FGD 如有尚未支持的实现接口，先完成 CPU 适配；不等待作者回复，也不占用核心 C1/L1 资源。外部训练分配发生在类别赢家明确以后，通过同一 guard / lease 调度，GPU 动态选择，不假定能单卡三开。单独外部实现模块，不向当前独立方法的 N/C0/C1/C1_y/L1/L_GT 枚举塞入 joint 或伪装成原创实验臂。

## 6. 旧 CMDistill / CCLKD 的复用结论

数字直接复用 `2026-09-07_audit_RGBIR实施起点/comparator_analysis.json`，本轮未重新推理。均为 mAP50–95 百分数，seed 顺序 0/42/123，SD 为样本 SD。

| 历史臂 | seed0 | seed42 | seed123 | mean±SD | 相对其历史 native 的配对平均差 |
|---|---:|---:|---:|---:|---:|
| 历史 native，workers8 | 54.357648 | 53.815556 | 53.687433 | 53.953546±0.355778 | — |
| CMDistill corrected | 53.900286 | 53.244840 | 53.669168 | 53.604765±0.332435 | −0.348781±0.291793 pp；三 seed 均负 |
| CCLKD partial | 54.066006 | 54.493295 | 54.332596 | 54.297299±0.215820 | +0.343753±0.550510 pp；−/+/+ |

两者均为 `PROTOCOL-ADAPTED`。CCLKD 只含 LLD+CCL，缺 FLD/RLD，不称完整 CCLKD。当前 N/C0 是 workers4；历史方法 workers8，此外直接使用 IR GT 的权限与 OEv1 也不同。三 seed 不会消除 recipe 差异。

CCLKD 新评估 receipt 的 dev 1469 名单与 OEv1 对齐，但历史训练源码绑定较弱；CMD 历史 metric 单位为旧 fraction 语义，缺明确 bound roster。新 evaluator 重评旧 checkpoint 可以补评估证据，不能修复训练 workers 差异或未实现分量。

因此：可复用作历史描述、错误案例入口和复现边界；**不能进入当前 C1−N 的正式分母，不能直接写“打赢作者 CMDistill / CCLKD”。** 若将其纳入正式同协议比较，需要新的 workers4 适配训练与同端点评估；CCLKD 继续标 partial，除非真的补齐并另立方法身份。本轮不自动重跑这两项。

## 7. 可写与不可写的主张

可写的准备结论：分类 sigmoid 响应、定位 DFL 分布、前景/背景/全局特征已有明确先例；这些家族正好覆盖用户提出的载体问题。当前方法的价值须落在“哪些可靠对象传哪部分内容、相对匹配 N 和同掩码控制是否稳定获益”。

目前不可写：logit / DFL KL / 高容量 feature 本身原创；旧 comparator 负结果证明作者方法无效；新方法已优于这些外部基线；paired−N 已排除同模态正则化。任何保留方法的跨模态增益仍须自身三 seed、paired/shuffled/same-modal/N 四臂及 accepted analyzer。外部方法只有 paired 数字时只作协议下比较，不给它或本方法额外跨模态归因主张。

下一步只实现上述三个固定配置的接口和必要测试；待核心赢家确定后，由根调度器分配实际训练预算。本文没有把未来最多9个 paired端点或外部归因扩展混入当前12/15个核心训练承诺。
