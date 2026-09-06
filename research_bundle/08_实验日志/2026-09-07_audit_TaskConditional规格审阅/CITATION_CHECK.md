# Task-Conditional 规格：四项文献独立核验

**结论：R1、R2、R8、R9 均有可核验的一手来源；文档对其存在性和主要创新边界的判断基本成立。需要补充实际选择机制，避免把相关先例误写成与拟议跨模态方法完全等价。**

核验日期：2026-09-07。范围仅为用户规格第 2、20 节的 R1/R2/R8/R9；未训练复现，未验证论文报告的性能。本记录不修改用户原稿。下列“独立核验深度”指此次审阅，不是原规格作者当时的阅读深度。

| 条目 | 身份结论 | 原规格自述 | 此次独立核验深度 |
|---|---|---|---|
| R1 Task Adaptive Regularization | 已确认作者、标题、2020 arXiv | 摘要 | 作者预印本正文 §3.3–3.4、Eq. (5) |
| R2 TDKD | 已确认 ACM MM 2020、作者和 DOI | ACM 目录及作者说明 | ACM SIGMM 官方摘要＋作者代码 README |
| R8 TID | 已确认 Neurocomputing 624 (2025), 129386、作者及 DOI | 出版摘要/引言 | 出版摘要/引言＋相关 2024 作者预印本方法；出版全文仍未取得 |
| R9 Sigmoid-τ | 已确认 IVC 172 (2026), 106045、作者及 DOI | 出版摘要/引言、verification release | Elsevier 卷目录及摘要/引言＋论文直接链接的作者仓库 README；出版全文未取得 |

## R1：任务相对优势的先例确实存在，但参照量必须写准确

Sun 等的作者预印本包括共享学生 proposal、前景分类蒸馏与选择性回归。关键区别在 Eq. (5)：只有教师回归框对 GT 的 IoU **高于学生 RPN 的回归前 proposal**，才加入回归 KD。比较项不是学生最终回归框，也不是冻结的独立 RGB 参考。这支持“按可靠性选择定位知识已有先例”，不能直接推出其已解决双模态坐标差异。[作者 PDF，§3.3–3.4，p.7–8](https://arxiv.org/pdf/2006.13108)

建议在规格 R1 行补充这一比较对象；保留 arXiv 年份，不增添未经核验的会议归属。

## R2：分任务选样已有直接先例，但主要作用于特征

ACM SIGMM 官方目录确认 Liang 等、标题与 ACM MM 2020。摘要明确分类/回归特征解耦，任务特定卷积及适配卷积，并对不同子任务选择不同样本；还包含概率蒸馏。因此“两个 loss 或两个 mask 本身不足以主张原创”的提醒有依据。但不能把它简化成已经实现当前对象级 C/L 输出路由。[ACM 官方条目](https://sigmm.org/opentoc/MM2020-TOC-1)、[DOI](https://doi.org/10.1145/3394171.3414069)

作者实验室代码 README 与该概括一致；本轮未逐函数审计或复现。[作者代码](https://github.com/CASIA-LMC-Lab/TDKD)

## R8：TID 的“联合评估学习状态”不是独立双任务路由

出版社页面确认题目、卷号、文章号和 DOI，作者为 Hai Su、Zhenwen Jian、Yanghui Wei、Songsen Yu。摘要/引言支持：综合分类与回归输出，映射到特征区域并识别关键与薄弱区域。这与规格概括一致；它是特征蒸馏选择思想的先例，不能仅凭摘要说它已覆盖跨模态实例匹配或分别拒绝 C/L。[出版摘要/引言](https://www.sciencedirect.com/science/article/abs/pii/S092523122500058X)

另找到作者相关预印本 *Task Integration Distillation for Object Detectors*（2024，三位作者）。其中 Eq. (3) 将分类、回归离散评分相乘；Eq. (5) 按教师与学生综合评分差选薄弱区域。这个机制明确不同于分别计算两个任务优势。预印本作者列表和题目与出版版本不同，尚未逐式比较最终出版正文，**只能标作相关作者预印本证据，不能称为已核验出版最终公式**。[预印本身份](https://arxiv.org/abs/2404.01699)、[方法正文 §3.1–3.2](https://arxiv.org/pdf/2404.01699)

## R9：2026 年论文确实存在，公开仓库的复现范围有限

Elsevier 官方卷 172 目录确认 Xiangqun Shi、Xun Zhang、Xian Zhang、Yifan Su，文章号 106045、DOI 10.1016/j.imavis.2026.106045。出版摘要/引言明确 Sigmoid/BCE 一致的分类蒸馏，以及分别运行师生原生 task-aligned assigner 后构造分区，对框和 DFL 施加不同监督。规格的“现代 YOLO 分类 KD＋分区定位 KD 已有先例”成立。它不是跨模态配准方法；不能由这些文字推定任意 RGB/IR 同索引 anchor 可对应。[官方卷目录](https://www.sciencedirect.com/journal/image-and-vision-computing/vol/172/suppl/C)、[出版摘要/引言](https://www.sciencedirect.com/science/article/abs/pii/S0262885626001526)

论文直接链接的作者 README 进一步说明分区为 teacher-only、student-only、shared positives。仓库明确将 release 定义为 verification checkpoints，不承诺完整论文生产权重；README 的权重发布流程仍有待补充式文字。因此原规格保留这一限制是必要的，不能仅见仓库存在就称“原论文可完整复现”。本轮未下载权重、未运行其代码，也未逐式核验 Sigmoid-τ 的具体变换。[作者 verification release](https://github.com/zzxx-zx-ccc/Verification-Checkpoints-for-SigmoidTau-Distillation)

## 审阅建议与访问边界

1. 保留规格关于“任务解耦、双 mask、定位分布 KD 不是单独原创点”的判断。
2. 将差异比较细化为：参照模型是什么、选择依据作用于哪一任务、是否联合评分、监督的是特征/框/分布、如何建立跨模态坐标与实例对应。严格控制实验属于证据要求，不自动构成算法原创性。
3. 不必因近期论文年份而删除 R9；它已得到官方出版目录与摘要支持。
4. R8/R9 的直接 `open` 首次均返回 HTTP 403；限定标题的搜索随后取得同一出版社页面的可访问摘要/引言文本。403 是抓取限制，不是论文不存在的证据。
5. 四项核验不足以证明最终方法新颖；CoLD、LD、CrossKD 等仍应结合最终方法逐式比较，本子任务未扩展到这些文献。
