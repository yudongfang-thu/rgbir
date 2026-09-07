# 条件筛选后的局部特征蒸馏：文献核验与实验建议

结论：**可以把“在哪些对象上蒸馏”与“传递什么内容”解耦，并在同一选择集合上比较输出分布和局部特征；但特征维度更多不保证可迁移信息更多。该组合存在充分先例，原创空间必须落在可验证的任务条件、信息增量及负迁移边界。**

日期：2026-09-07。范围：只读研究调研，不改训练代码，不启动 GPU；研究判断是下一版假设，不改当前已冻结实验。

## 1. 已核验的主要文献

| 文献、发表身份 | 原文实际做法 | 对本项目的含义和限制 | 实现身份 |
|---|---|---|---|
| Wang et al., **FGFI**, CVPR 2019 | 估计对象附近值得模仿的 anchor 位置，在这些区域蒸馏高层特征响应。 | “挑对象附近的局部区域再传特征”已有明确先例；属于同模态检测压缩，不能据此保证 RGB–IR 可迁移性。 | [作者仓库](https://github.com/twangnh/Distilling-Object-Detectors) 可访问；[论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.html)。 |
| Guo et al., **DeFeat**, CVPR 2021 | 将 neck 的对象/背景特征与分类头 proposal 分开加权，指出背景也包含检测知识。 | 支持保留有限对象上下文；不能从“前景好”推导“背景都应删掉”，也不能把普通前背景解耦当新方法。 | [论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Distilling_Object_Detectors_via_Decoupled_Features_CVPR_2021_paper.pdf) 明确给出 Huawei Noah DeFeat 路径；本次网页未成功加载代码，不称已经复现。 |
| Dai et al., **GID**, CVPR 2021 | 用师生差异选择 general instances，再传 feature、instance relation、response 三类知识，不依赖 GT 正负划分。 | **最直接的“选择机制与载体可分开”先例**。我们的潜在差别须是跨模态对象身份、特定任务可靠性和学生收益证据，而不是仅把损失改为 feature。 | [论文](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_General_Instance_Distillation_for_Object_Detection_CVPR_2021_paper.html)；找到 [Detectron2 实现](https://github.com/daixinghome/Distill_GID_detectron2)，未完成作者身份/逐项实现审计，暂不标严格官方复现。 |
| Yao et al., **G-DetKD**, ICCV 2021 | 区域特征在 FPN 多层之间做语义引导软匹配，并迁移区域关系的对比知识。 | 同分辨率/同层数并不保证语义对应；可以借鉴区域/关系表示。其软匹配主要处理金字塔语义层级，**不能说它解决 RGB–IR 像素几何误差**。 | [论文](https://openaccess.thecvf.com/content/ICCV2021/html/Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive_ICCV_2021_paper.html)；本次未核定作者完整实现。 |
| Yang et al., **FGD**, CVPR 2022 | 前背景分别处理，教师空间/通道注意力突出重要位置，并补充全局像素关系。 | Table 1 中前景、背景不区分地一起蒸馏低于分别处理；“传得更多”已有负面例子。适合作为通用特征 KD 外部基线，不能直接当跨模态配准模块。 | [作者代码](https://github.com/yzd-v/FGD) 可访问；[论文](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html)。 |
| Wang et al., **CrossKD**, CVPR 2024 | 学生检测头中间特征输入冻结教师的后半检测头，让 cross-head predictions 匹配教师输出，缓解 GT/教师输出监督冲突。 | 一个折中：由教师任务头约束特征中的任务相关方向，避免逐元素复制全部特征。原文输出模仿可胜过此前特征法，因此不能预设 feature > logit。跨模态迁移仍需验证。 | [作者代码](https://github.com/jbwang1997/CrossKD) 可访问；[论文](https://arxiv.org/abs/2306.11369)。 |
| Tong et al., **CMDistill**, JSTARS 18, 2025（2024 在线） | PCC 归一化特征、深层 affinity 关系、IoU 引导输出蒸馏；含学生特征适配层。 | **我们在跑的对比本身已覆盖特征/关系/输出三载体。** 原文 IV-C 报告使用全部特征层不如选择部分层，且直接跨模态特征匹配存在困难。不能把加入局部 feature 描述为从未做过的方向。 | [DOI](https://doi.org/10.1109/JSTARS.2024.3479717)；本地原论文/文本已读取。本项目实现保持 PROTOCOL-ADAPTED，不冒充作者代码。 |
| Ma et al., **CCLKD**, Geo-spatial Information Science, 2026 | 类别组织的自适应温度蒸馏包含 logit/feature/relation；另有类别约束对比。 | 同样已涉及多个载体。需要区分原文完整方法与本项目 partial 实现；当前对比结果不能归纳为“特征 KD 都无效”。 | [出版社全文](https://www.tandfonline.com/doi/full/10.1080/10095020.2026.2633014) 与本地 PDF 已核验；未查到可确认的作者完整代码。 |
| Xi et al., **CMKD-net**, TCSVT 36(7):9363–9377, 2026 | 融合教师到单模态有向检测；通道/空间特征 MSE+KL，实例 Gram 关系，旋转自适应 RoI pooling。 | “RGB–IR + 实例区域 + 特征/关系”直接近邻。RA-RoI 的目的之一是降低 pooling 量化误差，**不能把它当成原图配准保证**。 | [DOI](https://doi.org/10.1109/TCSVT.2026.3670458)；本地出版社 PDF 第1、4–6页核验。作者代码未核定；不是我们当前 HBB 协议。 |
| Kim & An, **CGDet / CGCMKD**, arXiv:2511.01435, 2025 预印本 | GT RoIAlign→GAP→投影做类别监督对比，另用 RGB 教师指导热图 FPN 特征。 | 直接支持对象级 pooled 表示与类别判别约束；使用 FLIR-aligned 三类。论文中输入拼接/推理移除及 CMG 为全图还是 RoI 的描述不够一致，不能据论文标题直接搭成可靠基线。 | [原文](https://arxiv.org/html/2511.01435v1)；本次未找到可靠作者代码。 |
| Thaker et al., **FreqKD**, arXiv:2606.11572, 2026 预印本 | DINOv2 RGB→IR；低频 MSE、高频弱 log-MSE，先 LoRA 表征蒸馏再检测微调。 | 作者 KAIST mAP50：无KD61.7、全特征MSE61.1、cosine62.1、response58.8、FreqKD64.1。是“全特征与输出都可能负迁移，需选择可传信息”的直接案例；未经本项目复现，不能移植这些增益数字。 | [原文](https://arxiv.org/html/2606.11572v1) 给出[匿名代码地址](https://anonymous.4open.science/r/freq_decoupled_kd-5E5A)，本次未能访问/审计，身份为“作者提供链接，代码未核验”。 |

两篇补充边界文献：

- **C²KD，Huo et al., CVPR 2024**：分析跨模态 soft-label mismatch，以在线选择、非目标类知识、双向和 proxy 处理模态差异；涉及分类与 RGB–Depth 分割，包含三 seed 和 Self-KD 比较。它不是 RGB–IR 检测，但不能再写“跨模态可靠性选择/自模态对照/重复实验无人做”。[论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.pdf)，[作者代码](https://github.com/huofushuo/C2KD/blob/main/README.md)。
- **UniDistill，Zhou et al., CVPR 2023**：先统一到 BEV，再在对象关键点迁移特征/关系，并在局部区域迁移输出；稀疏前景策略减少不对齐背景影响。它有真实 BEV 公共坐标表示，不能把“在二维两模态上裁相同坐标”视为同等保证。[论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_UniDistill_A_Universal_Cross-Modality_Knowledge_Distillation_Framework_for_3D_Object_CVPR_2023_paper.pdf)。

## 2. 为什么局部特征可尝试，却不一定更好

以下为本项目研究推断，而非某篇文献已经证明的项目结论。

完整特征含类别、形状、背景上下文，也含颜色、热辐射响应、纹理和教师内部编码方式。“教师在该对象分类对了”只证明任务输出较可靠，不保证对象区域里每个通道、每个像素都值得强制迁移。更大的张量维度还会增加噪声、不同模态不可观测细节、尺度/幅值差异和优化约束。

特征也不自动是“类别知识”：共享 FPN 局部特征通常同时影响分类与回归。若称任务条件蒸馏，应说明条件控制的是教师资格，还是实际迁移内容、梯度路径也按任务划分。可分别考虑靠近分类头的表示、低维投影，或 CrossKD 式用冻结任务头表达特征目标；每一种都是需要验证的新实现。

建议把载体比较定义为固定对象集合上的实验，而不是直接多加一个损失再看 AP：

1. 保留现有 C 作为参照，冻结其选择、对象分母和剂量定义。
2. 同集合比较当前证据、完整类别分布、一个 P3/P4 局部特征表示。先用小型 RoI 网格和训练期通道投影/归一化，不同时加入多层、关系、频率和注意力。
3. 如研究定位，单独保留当前 DFL 输出分布；局部 feature 不替代 DFL 的位置语义与几何检查。
4. 同集合不等于同优化强度。各载体用固定训练数据、盲于新 AP 的梯度测量定系数；直接共用数值 0.1 不是公平剂量。
5. 同时记录有效对象、像素/通道数、梯度范数/方向和损伤-修复；如局部特征额外产生可用性筛选，报告公共集合与原集合差异。

## 3. 配准不足时，局部区域怎么用

**同一对象身份、对应区域、像素对应是三个不同层次。** 同物体可以在两模态中采用各自 GT 框提取 RoI；这样无需假装同一绝对坐标一定对应同一像素，但细网格单元仍未必对应同一个物理部位。

可以按几何证据选择不同载体：

| 已确认的对应层次 | 合理候选 | 仍需验证的风险 |
|---|---|---|
| 对象身份可靠，像素/边界精度不足 | 各自 GT RoI 的 pooled embedding、通道统计或对象间关系 | pooling 降低位移敏感性，却损失位置细节；遮挡和背景污染仍存在。 |
| 近似区域对应，只有有限位移 | 小网格池化、有限局部匹配；定义相同对象内的搜索边界 | 搜索可能错配同类邻居；使用教师预测选匹配会产生确认偏差。 |
| 原生 anchor 坐标/stride/bin 的几何语义成立 | 同 anchor DFL 分布、细粒度局部图 | 几何误差可能直接转成错误定位监督；不能只靠模型相似度证明对应。 |

**AR-CNN，Zhang et al., ICCV 2019** 是 RGB–thermal 弱对齐的关键先例：独立的模态框与配对标注、区域位移预测、可信度加权融合、RoI jitter。它是双模态融合模型，不是单模态部署蒸馏，但说明“区域池化本身”不能替代对偏移的处理。可借鉴它的已知位移压力测试，对候选载体测量不同位移下的损失/梯度变化。[论文](https://arxiv.org/abs/1901.02645)，[作者代码及 paired 标注](https://github.com/luzhang16/AR-CNN)。

以上区域方法若要采用，应单列为下一版假设，不在当前同-anchor L 失败时静默切换，避免把两种定位机制混为一个实验。

## 4. 本地旧笔记需要纠正的地方

1. `01_文献/RGB-IR_20260905新增/00_方法调研综述.md` 中 CGDet “FLIR 单类”不准确；原文是 person/bicycle/car 三类。
2. `文献精读笔记_20260905.md` 将 FreqKD cutoff/权重完全写成“由测量决定，非手调”过强；原文明确 cutoff empirically selected，并报告 cutoff/merge-scale 扫描。其权重有频谱动机不等于所有超参已被数据唯一确定。
3. “全领域唯一重复统计”“四臂归因是空白”等领域级排他主张未通过完整查新。C²KD 已有重复 seed 和 Self-KD；严格四臂设计可作为本项目证据规范，但不据此自动主张算法原创。
4. CMKD-net 的旋转 RoI 积分解决采样量化问题，不能解释为完成 RGB–IR 注册；CGDet 的 FLIR-aligned 是改善后的配对版本，不代表可不经检查移用我们的数据。
5. CCLKD 本项目 partial/适配身份与原论文完整方法必须分开；不能用当前 partial 结果判定原论文特征/关系路径已被公平否定。

本文件不覆盖旧笔记，保留原始记录；这些纠正作为后续综述/稿件入口。

## 5. 复核入口和检索限制

使用 research-lit 技能，先读取本地笔记及 PDF，再查 CVF/arXiv/出版社/作者代码。arXiv helper 在技能约定默认路径未发现，使用 arXiv 官方页面与全文回退。未下载新 PDF，未把未核验代码标为已复现。

本地核验：

- `01_文献/FGD__2022_CVPR__Focal_and_Global_KD_for_Detectors.pdf`：前3页，含 Table 1。
- `01_文献/CrossKD__2024_CVPR__Cross_Head_KD_for_Object_Detection.pdf`：前3页，冲突动机、方法与作者仓库。
- `01_文献/G-DetKD__2021_ICCV__General_Detection_KD_Contrastive_Semantic_Feature_Imitation.pdf`：前3页。
- `01_文献/CCLKD__2026_GIS__Cross_Modal_Contrastive_Learning_Incomplete_Modalities.pdf`：前3页，并对照出版社正文。
- `01_文献/RGB-IR_20260905新增/CMKD-Net__2026_TCSVT__Cross-Modal_KD_Oriented_Detection_Modality-Missing_Visible-Infrared.pdf`：前3页及第4–6页方法。
- `06_历史工程_只读/LADD_public/tmp/cmdistill_pdf_text/cmdistill.txt`：摘要、III-B/C、IV-C；对应原 PDF 在 `comparison/cmdistill/paper/`。

这是目标明确的文献核验，不是穷尽式 novelty 证明。文献报告的增益只作为动机证据，不升级为本项目已支持结论。
