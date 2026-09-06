# 可直接交给其他模型的独立审计提示词

请独立审计这个 GitHub 仓库及当前分支中的 RGBIR 跨模态知识蒸馏项目。我们的目标是在合理实验成本内形成可投 J-STARS 的方法；请优先判断问题、证据和实现是否成立，不要迎合我们现有方法或为旧结论辩护。你可以得出“方法不值得继续”或“现有数据不能支持结论”。

请先读取根目录 [MODEL_REVIEW_GUIDE.md](MODEL_REVIEW_GUIDE.md)。新材料在 `research_bundle/`，原 `00_project_context/` 等目录是较早审计包，包含后来修正的概括。请沿勘误链追溯，不能只读旧 README 后作答。运行状态以具体快照时间为准，不把仓库中的“运行中”视为实时状态。

## 研究问题与已知边界

训练时有配对模态，推理时只用一个模态。当前 Object Evidence v1 主实验为 **DroneVehicle IR→RGB**；LLVIP 是 IR→RGB 机制补充候选；VEDAI 是 **RGB→近红外 NIR** 的方向反转候选，不能写成热红外。历史 RGB→SAR 工作是背景，不是当前方法的结果。

我们已用三个数据集、六个 seed42 baseline 对 521 对开发图像做实际预测、特征和配准代理诊断，观察到教师额外命中、学生独有命中、低置信正确候选，以及数据集之间不同的定位优势。请独立复算这些证据，判断它们是否支持“选择可迁移的对象判别信息”，不要把教师 AP 差、CKA 或特征热图当作 KD 可达收益。

新方法从各自真实训练 GT 匹配对象，提取 P3/P4 正确类别 logit 的前景−局部背景证据，按冻结 RGB 参考候选和 IR 教师质量代理选择对象，再对 RGB 学生施加 SmoothL1。IR 独立标注是新增训练期辅助信息。IR 教师和 RGB 参考固定 seed42，当前学生实验设计是 paired/weight0 × seeds0/42/123；只测试整体干预相对 weight0 的净结果。更多算子存在于源码，不等于完整归因实验已实施。

## 必须读取的材料

1. [最新综合诊断](research_bundle/07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md)，尤其现有结果、配准、对象互补、建议方法和证据限制。
2. [当前结果复核](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/current_results_notes.md)及同目录 `current_results_sources.json`，结合原仓库 `02_raw_results_dronevehicle/`、`04_hnewa_eval_records/` 重算主要对照。
3. [完整 probe 目录](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识)中的模型身份、正式 `*_full/` 预测/对象/特征结果、`probe_analysis.json`、源代码和特征图册。
4. [冻结协议](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏首轮/EXPERIMENT_PLAN.md)、[实现源码](research_bundle/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_object_evidence_v1)、首轮 `framework_notes.md` 和 canary 证据。
5. [三 seed 扩展记录](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/README.md)、同目录 `cross_seed_realization.md`、`ENDPOINT_ANALYZER_NOTES.md` 及可用最新端点快照。
6. 若评价路线选择，再读[历史复盘](research_bundle/07_研究分析/全项目复盘与研究诊断_20260905.md)、[文献笔记](research_bundle/01_文献/RGB-IR_20260905新增/文献精读笔记_20260905.md)、[OS-SSL-IR 实验](research_bundle/08_实验日志/2026-09-06_train_OS-SSL-IR迁移/README.md)。判断新颖性时回到作者原文；说明你独立核实的论文和仅依据本地笔记的内容。

如果工具无法读取某些文件或数据量超出上下文，请列出缺口并优先读源结果和实际代码。不要声称看完未读取的目录。原始数据和权重不一定在仓库中；路径清单不等于你已访问服务器或完成重训。

## 请完成的审计

### A. 重建可信结果表

检查方法身份、split、教师/学生、seed、checkpoint 端点和 AP 字段后，自行计算逐 seed 差值及 mean±sample SD。AP 从 0–1 转为百分点时明确标注；SD 用 ddof=1。禁止将 precision 当 mAP、不同数据集/recipe 拼表，或以 best 代 last。

至少复核 DroneVehicle CMDistill−匹配 native，以及 DroneVehicle/LLVIP HNEWA-inspired 的 paired−shuffled、paired−same-modal。对缺失 native seed 明确标缺失。检查旧总结中的“监督 KD 从未有效”“唯一强正例”“亮暗都负所以门控无效”是否超过证据。不要把 adapted/inspired 版本当成外部论文精确复现。

新 Object Evidence 只读取完整 E200、固定 last/EMA、独立开发评估和完整 receipt 对应的结果。若少于三对完整端点，不能形成完整三 seed 汇总；若没有终点，明确尚无性能判断。训练 CSV 的 AP=0 可能是关闭验证的占位。训练正常、KD 非零或 canary 通过不能替代最终 AP。

### B. 检验数据与“蒸馏什么”的联系

从保存的共同对象和预测重算：教师独有/学生独有/双方命中/双方未命中、低置信正确候选、错类候选、双方命中定位差、亮度和目标大小条件。处理候选重叠与一对一匹配竞争，避免重复计数。

评估对象判别、定位和局部结构三种知识各自的证据。请检查：LLVIP 复制标签对定位和配准判断的影响；DroneVehicle 独立标注和 HBB 转换的影响；VEDAI official_fold01 身份、paper80 教师排除及小样本边界。CKA 的有效 token/图像数和 donor 混杂、phase 低响应、独立图像归一化都应纳入解释。

### C. 审查实际 loss、数据流和归因设计

逐一对照源码与冻结协议：独立 IR 标签是否经相同几何变换；对象匹配是否最大有效匹配数优先；背景是否排除全部本模态 GT；空集合和极小目标怎样处理；教师/reference 是否冻结并排除在 optimizer/EMA/导出模型外；KD 标量是否只加一次；weight0 是否保持原生学生损失/梯度和同 seed 输入路径。

特别回答：

- 当前知识仅是 GT 正确类别的标量相对证据，它包含什么教师特有信息？GT-only、常数 margin 或额外前景/背景监督能否解释可能的收益？
- 基础集合 E 已依赖 IR GT 对应和区域有效性。怎样拆分 IR 标签辅助、教师内容、选择规则三个贡献？
- `q` 是代理质量差，是否存在分数尺度、训练集过拟合或冻结参考能力限制？它是否排除了真正有改善机会的难例？
- random 对照如何保持同 E、同 K、同分母和合理可比的有效剂量？相同 λ 是否足够？必要时如何记录 KD/检测梯度比例？
- shuffled 应使用什么 donor 单位和约束，才能检验实例配对而不是人为破坏几何？same-modal 对照如何公平定义？
- pinned loss 的标量广播导致历史 KD 有效系数放大三倍这一事实，如何影响旧负结果解释和新方法归因？请区分实现事实、待验证的原因和已证明的因果关系。
- 教师/reference 固定 seed42、学生初始化变化，而所测 DataLoader 顺序/首批增强一致时，三 seed SD 代表什么、不代表什么？

### D. 给出最少且能改变决策的后续实验

请最多列三个优先实验，而不是大范围超参搜索。每个写清：待证伪的主张、只改变的因素、对照、冻结知识/选择/剂量、数据和 seed、评估端点、主要指标、同时观察的收益与损伤，以及何种结果应继续或停止。

如果 P−N 为正，要说明仍不能证明什么；如果为负，也不要未经消融就断言 RGBIR 没有蒸馏空间。请优先讨论 same-modal、合理 shuffled、同 K random 和同 mask GT-only 的信息价值，不要机械要求堆完所有组合。明确哪些可以复用现有已匹配结果，哪些必须重新运行。

最后判断 J-STARS 主线：DroneVehicle 的航拍遥感部署问题是否清楚，LLVIP 作为地面机制补充是否合理，VEDAI 的 NIR 与方向反转带来什么信息；是否有充分理由增加尚未建立冻结 baseline 的 FLIR/M3FD/KAIST。新颖性必须与已有任务判别、区域选择、质量门控和跨模态 KD 工作比较，不能只靠换名字或增加损失项。

## 输出格式

请用中文输出，允许保留必要英文术语，并按以下顺序组织：

1. 你最重要的独立判断，区分已证实、合理假设、证据不足和被反驳。
2. 复算结果表，含逐 seed 数字、单位、端点、来源文件/字段及缺口。
3. 实现与实验协议问题，按是否影响结果有效性排序，提供准确代码路径与行号。
4. 当前方法的替代解释，以及各解释需要哪个最小对照才能区分。
5. 至多三个优先后续实验、继续/停止条件和可形成的最窄论文主张。
6. 你实际读取/运行的内容，未获得的证据，以及需要重新核实的文献。

不要输出编造的实验结果、无法复核的增益预估或投稿保证；不要为了契合维护者叙事而抹掉负 seed、失败 attempt、标注辅助或协议局限。我们希望你指出最有可能浪费下一轮训练的错误假设。
