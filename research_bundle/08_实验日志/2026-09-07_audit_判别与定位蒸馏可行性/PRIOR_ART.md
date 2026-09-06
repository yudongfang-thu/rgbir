# 判别与定位互补蒸馏：有界文献核查

**结论：定位知识蒸馏、分类与定位分开选择、依据两任务质量不一致来蒸馏，均已有明确先例；“反过来再蒸馏定位”本身不足以主张第二个原创点。可研究的具体问题是：弱模态存在可学习对象时，如何同时确认跨模态对象身份、坐标可传递性及教师在目标模态坐标系中的定位优势，按任务决定传递或拒绝哪些证据。**

日期：2026-09-07；性质：可行性与相关工作核查，不是穷尽查新，不构成新颖性认证。本条目未启动训练。

## 核查范围与来源

先检索本地 `01_文献/` 的旧综述和精读笔记，并读取 LD、CrossKD、DKD、CoLD、GaLD 本地 PDF 前三页。外部仅采用原始论文、出版社/会议官方页面和作者仓库。按 research-lit 技能依次检查 ARIS_REPO、工作区 tools、Codex arxiv skill 的 `arxiv_fetch.py`，均未解析到可调用 helper，故降级至 web 检索 arXiv 元数据/正文。没有下载新 PDF，没有使用二手综述作为结论依据。

## 六篇最直接的原始工作

| 工作 | 已经完成的贡献 | 对当前提议的限制与差异 |
|---|---|---|
| **Zheng et al., LD, CVPR 2022** — [Localization Distillation for Dense Object Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html) | 将框表示为边界位置分布，独立传递语义与定位知识，并用 valuable localization region 判断不同区域适合传递哪类知识。 | 是“分任务、分区域蒸馏”的直接先例；没有据此解决 RGB-IR 实例错配、模态专属边界或目标模态坐标映射。不能把增加一个 DFL/KL 定位损失算新机制。 |
| **Tang et al., TBD, Pattern Recognition 137, 109320, 2023**（arXiv 初稿 2022）— [正式论文](https://www.sciencedirect.com/science/article/pii/S0031320323000213)、[作者预印本全文](https://arxiv.org/html/2208.03006v1) | 明确分析“分类分数高但定位质量差”及其反向不一致，用 Harmony Distillation 与 Task-decoupled Feature Distillation 联合处理分类、定位质量，并动态平衡任务掩码。 | 用户提出的任务对称性已有非常接近的动机先例；该工作研究模型压缩，没有验证跨模态同一实例的身份与坐标一致性，也没有证明本项目高置信低 IoU 对象有充分教师优势。 |
| **Wang et al., CoLD, IEEE TGRS 61, 2023** — [Category-Oriented Localization Distillation for SAR Object Detection and a Unified Benchmark](https://doi.org/10.1109/TGRS.2023.3291356) | optical→SAR 蒸馏中，将候选框按教师类别先验划分 target/non-target，并用教师 IoU 对候选框自适应加权。 | 已经覆盖“类别引导＋定位蒸馏＋教师定位质量加权”的跨模态组合；未提供本项目需要的 RGB-IR 原生双标注坐标映射和独立身份不确定性拒绝机制。不能只用任务名称或换数据集区别。 |
| **Wang, Luo, Fang & Yang, GaLD, ICASSP 2025** — [作者项目页](https://github.com/wchao0601/GaLD)；原文 DOI `10.1109/ICASSP49660.2025.10889285` | 将旋转框变为二维高斯分布来传递 optical→SAR 角度几何，并设计自适应权重，减少低质量角度信息的影响。 | “跨模态几何蒸馏＋拒绝劣质几何”也有先例；其旋转框表示与本文 RGB-IR HBB 坐标不一致问题不同。本次依据本地原始 PDF 的摘要及方法部分判断；作者仓库可见内容主要为数据说明，不能把仓库链接当成实现已充分公开或已复现。 |
| **Wang et al., CrossKD, CVPR 2024** — [Cross-Head Knowledge Distillation for Object Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html) | 将学生检测头中间特征送入教师检测头生成 cross-head 预测，以减轻学生 GT 监督与教师预测之间的目标冲突。 | 支持“教师输出与监督目标可能冲突”这一一般风险，不能直接证明 RGB-IR 的冲突来自配准，也不能直接解决教师 IR 框与 RGB GT 坐标/边界差异。 |
| **Zhang et al., AR-CNN, ICCV 2019** — [Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Weakly_Aligned_Cross-Modal_Learning_for_Multispectral_Pedestrian_Detection_ICCV_2019_paper.html) | 对可见光/热图对象位移设计区域特征对齐、可靠性重加权与 RoI jitter，并为 KAIST 构建每模态独立框及其对象关系。 | 直接说明“同一个对象”不等于“同一组像素坐标”，且身份对应与框标注需要分别建立；这是双模态融合方法，并未证明单 RGB 部署的定位蒸馏一定有效。 |

上述六篇结果均只作为文献先例；没有把其自报 AP 当成本项目可复现结果，也没有跨数据集比较数字。

本地另核对了 **DKD, CVPR 2022** 的前三页：其中 decoupling 指 target-class 与 non-target-class 蒸馏的分离，不是分类任务与框回归任务的分离。不能仅凭“Decoupled”标题把它当作本提议的完全相同方法，也不能将这个词本身作为差异化依据。

## 与当前构想的实质关系

从文献推导的判断如下，尚不是本项目的实验结论。

1. **不是纯粹对称问题。** 判别信息可以在两个各自对准对象的区域中比较；绝对框位置则属于某个传感器的像素坐标。IR 教师对 IR 标注很准，不等于其原始框适合监督 RGB 学生。定位支路必须多一道 geometry-transfer 条件。
2. **类别正确不等于同一实例。** 同一图内多辆同类车的置信度都可能很高，按类别一致或最近预测框硬匹配会把邻车当作教师。“不对准”恰好会使依赖预测框重叠建立身份的规则更加脆弱。
3. **未通过后处理检出不等于没有可蒸馏预测。** 稠密检测器在阈值过滤/NMS 前仍有位置预测。可以由可靠对象标注、共享坐标或一致的 assignment 建立监督，而不要求先产生最终 TP；但存在梯度不等于可学性成立。本轮若仅研究定位细化，先限定有可归属粗候选的对象，能减少额外身份与召回假设。
4. **定位优势应在 RGB 目标坐标系判断。** 至少需要 teacher 映射后预测相对 RGB GT 比学生/冻结 RGB 参考更好，并审计这一条件是否依赖真实标注形变。只比较 `IoU(teacher_IR, GT_IR)` 与 `IoU(student_RGB, GT_RGB)`，只能说明各模态各自预测质量，不能说明 teacher_IR 的坐标可直接复制。

因此合理的候选机制应写成对象和任务条件化的选择：身份可靠 ∧ 坐标可传递 ∧ 教师该任务可靠且有优势 ∧ 学生该任务仍有可学缺口。分类与定位可以有各自门控，也可以同时关闭；无须为了形式对称强行对每个对象选择一个分支。

## 差异化必须能够被证伪

| 拟验证主张 | 最小必要反证检查 |
|---|---|
| 定位支路确实利用了 IR 提供的几何信息 | 在相同 RGB/IR GT 使用权限、样本和剂量下，比较 GT-only、same-modal、有效 shuffled；若 GT-only 或同模态等效，不应归因为跨模态几何知识。 |
| 身份与坐标条件比普通高置信/IoU 门控有效 | 在同一合格集合、相同定位损失和剂量下，对比只有教师质量、随机选择、加身份/对齐检查；并分层统计有无配准偏差时定位误差的变化。 |
| “学生能分类但定位不佳”是定位可学子集 | 冻结阈值后报告该子集数量、教师映射后优于 RGB 参考的比例、改善/恶化边界与 AP75/定位错误变化；单靠全局 AP 增长不能解释具体机制。 |
| 改善不是复制新增 IR GT 的结果 | 若用逐对象 RGB GT 与 IR GT 的映射，应标明它是训练特权信息，并让对照获得同样映射；避免用 GT 对齐后本质直接得到 RGB GT，再将收益归于 teacher。 |

可主张的差异不是“我们也有两个损失”，而是能用数据证明：**哪种对象在完成身份和坐标检查后值得传递哪种证据，以及拒绝错误传递是否减少了负迁移。** 是否达到方法创新要求，还取决于下一步具体实现和对照结果；本次有界检索不能证明这一更窄构想无人做过。

## 对旧笔记的纠正

`01_文献/RGB-IR_20260905新增/00_方法调研综述.md` 第24、45行附近曾将“RGB-IR 定位分布级蒸馏未见”直接延伸为“方法创新点二”；其证据只够表示当时检索未见，不能证明新颖性。与此同时，CoLD/GaLD/LD 已存在，TBD 又为任务不一致提供直接先例。旧文件保留作历史记录，本说明覆盖其过强的新颖性判断。

本次没有确认“有身份且对齐可靠的 RGB-IR 定位劣势对象”在 DroneVehicle 的规模和可学上限；该证据必须来自本项目对象级分析，不能由上述文献代替。
