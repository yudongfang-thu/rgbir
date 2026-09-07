# 类别、定位 logits 与任务条件蒸馏：原文核验（2026-09-07）

结论：当前 C 是高度压缩的 logit 证据；完整类别响应有增加信息的可能，但 YOLO 应保留独立 sigmoid 语义，跨模态类间关系也可能相互冲突。当前 L 已是 DFL logits 分布 KL。按任务选择输出或特征存在多项先例，不能将简单 C/L 对称组合或加特征当作已验证原创点。

这是有目标的文献调研，不是穷尽查新；未新增训练、未变更冻结实验。论文结果不是本项目复现结果。使用 research-lit 技能，本地先读取 LD、DKD、CrossKD 前3页，再查询 CVF、arXiv、出版社、作者仓库；约定 arXiv helper 不可用，回退官方页面和全文。未下载新 PDF。

| 文献 / 作者 / 发表身份 | 已核验的相关结论 | 对本项目的含义与边界 |
|---|---|---|
| [DKD](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.pdf)，Zhao et al., CVPR 2022 | 将目标类与非目标类知识解耦，展示非目标类关系的重要性。 | 支持检查当前 GT 单通道压缩是否损失类间结构，但其 softmax 多类分类机制不可直接当作 YOLO sigmoid 的无修改替换；LLVIP单类没有非目标类别关系。 |
| [BCKD](https://arxiv.org/html/2308.14286v2)，Yang et al., ICCV 2023；[作者代码](https://github.com/TinyTigerPan/BCKD) | 检测分类以多个独立 binary maps 建模，采用 sigmoid/BCE 蒸馏以避免与普通softmax KD的协议差异；另有IoU定位项。 | 原始 anchor 全类别输出的优先基线；BCKD定位项与本项目DFL不同。应用到对象池化差分证据不是论文直接实现。 |
| [LD](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.pdf)，Zheng et al., CVPR 2022；作者仓库另列TPAMI 2023扩展，[代码](https://github.com/HikariTJU/LD) | 四边概率分布迁移；valuable localization region 区分类别/定位适用区域。 | 同anchor DFL KL与任务区域选择都有先例；我们的增量必须来自跨模态坐标、对象身份、相对优势、实际内容归因，不能把KL算子称原创。 |
| [CrossKD](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf)，Wang et al., CVPR 2024；[代码](https://github.com/jbwang1997/CrossKD) | 学生中间检测头特征经教师后半检测头产生cross-head输出，缓解GT与教师目标冲突。 | 可借鉴“让任务头筛出特征中哪些方向需要匹配”的折中。教师参数冻结不等于学生经过教师路径时关闭梯度；跨模态同样需要适配/对应验证。 |
| [Task-Balanced Distillation](https://www.sciencedirect.com/science/article/pii/S0031320323000213)，Tang et al., Pattern Recognition 2023；[预印本全文](https://arxiv.org/html/2208.03006v1) | Harmony Score衡量分类与定位协调，Task-decoupled Feature Distillation用两种任务掩码与任务权重指导特征匹配。 | 与“用类别/定位质量判断何处传局部特征”很接近。原文为通用检测压缩，不已证明RGB–IR误配下有效。出版社与预印本数字存在版本差异，本文不混用其AP。 |
| [Task Adaptive Regularization](https://arxiv.org/html/2006.13108v1)，Sun et al., arXiv 2020（本次仅确认预印本身份） | 共享学生proposal，在特征、分类、回归层迁移；教师回归后相对原proposal的GT IoU改善时启用定位监督。 | “教师在该任务更合理才教”亦有先例；注意其比较对象是原proposal，不可直接说与我们冻结R的教师优势门完全相同。 |
| [C²KD](https://openaccess.thecvf.com/content/CVPR2024/papers/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.pdf)，Huo et al., CVPR 2024；[代码](https://github.com/huofushuo/C2KD) | 跨模态soft-label关系可能冲突；用类别排序相关性进行在线选择，结合非目标类知识、双向代理。 | **反驳“完整类别logits必然更好”的直接跨模态证据。** 其实验主要分类/分割，不能照搬成Drone检测增益。全类别载体应检查跨模态类别排序/校准，而非只验证GT类别正确。 |
| [Logit Standardization](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html)，Shangquan Sun et al., CVPR 2024 | 以Z-score缓解logit尺度强匹配，保留关系信息。 | 对模态幅值差异有启示，但在ImageNet/CIFAR分类验证；不能未经评估抹去YOLO绝对置信度，也不能用于单类向量标准化。 |
| [Refined Logit Distillation](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Knowledge_Distillation_with_Refined_Logits_ICCV_2025_paper.pdf)，Wujie Sun et al., ICCV 2025；[代码](https://github.com/zju-SWJ/RLD) | 利用标签修正误导性的教师logits并保留类间相关结构。 | 支持从“教师是否整体强”转为“具体信息是否可靠”；原实验是图像分类，不是跨模态检测。 |
| [Local Dense Logit Relations](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Local_Dense_Logit_Relations_for_Enhanced_Knowledge_Distillation_ICCV_2025_paper.html)，Xu et al., ICCV 2025 | 对类别对logit关系递归解耦/重组并赋权，验证CIFAR/ImageNet/Tiny-ImageNet。 | 标题中的“local”是类别关系的局部性，**不是图像局部ROI特征图**，不可混为用户提出的空间局部载体。 |

## 对本项目的具体解释

1. 当前C的每对象一个连续标量是GT类别的前景相对背景证据。它是有意压缩过的logit信息，不是硬标签；压缩降低对逐像素配准依赖，也丢失非GT类别、绝对置信度、P3/P4差异和对象内部空间结构。
2. 最小载体变化是保留同一对象E/K、各模态自己的ROI/背景，将GT单通道改成5通道证据向量。仍是相对证据，不是原始anchor置信度；可以用同类向量损失作第一项对照，先不要同时换几何规则。
3. 原始anchor分类响应则用每类别Bernoulli分布，例如 p=σ(z/T)，KL(Bern(pT)||Bern(pS))；其对学生的梯度与软BCE相同。softmax跨C类仅保留相对关系，丢失独立前景置信度，尤其LLVIP C=1时恒为1。DKD类间关系不能直接解决单类LLVIP。
4. 当前L使用4×16 logits经温度softmax定义的四个边距离分布，KL本就保留各边的分布形状。直接raw-logit MSE还会限制每边任意共同偏移，不能视为无代价增加有效信息。
5. 特征载体可选择原生分类/回归头内局部表示、ROI归一化表示、对象关系或CrossKD式任务头输出。它们需要不同几何证据。共享FPN特征同时服务两任务，“类别门选中”不自动意味着整个特征都是可迁移类别知识。

## 下一版候选，非现行实验变更

- 先同选择比较：原C标量、全类别池化证据、单一小ROI网格特征。固定E/K与数据流，损失按通道/位置归一化并按冻结训练batch梯度校准；不直接把相同λ=0.1当作相同剂量。
- 全类别响应扩展需检查教师GT类以外的误导信息，不能仅凭GT类更强就认为全部类间关系都可靠。
- L独立重新审视绝对R IoU上限。离线cap敏感性支持有机会被挡掉，但不能由此选出AP最优阈值；新协议应冻结相对优势、对应风险与剂量规则。
- 若对象身份已知、像素精度不足，可研究各模态ROI pooled语义/关系；它不是绝对定位L的替身。细网格、边界和同anchor DFL仍须相称的几何对应证据。

具体数据与代码证据见 [payload_gate_audit.md](payload_gate_audit.md)；特征与跨模态近邻详见 [feature_literature.md](feature_literature.md)。
