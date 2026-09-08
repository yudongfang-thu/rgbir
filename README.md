> **2026-09-08 单批真实DFL信息读出完成**：固定32图80GT，保存真实四边DFL分布、同次解码及逐对象GT读出，区分额外信息和可迁移性。见[完整结果与边界](research_bundle/08_实验日志/2026-09-08_probe_DFL真实信息读出/FINAL_REPORT.md)。

> **2026-09-08 定位学习位置与目标诊断完成**：复用首32图与8批L2记录，核对实际学习anchor、教师坐标目标和GT控制，给出本版L2去留决定。见[完整结果与边界](research_bundle/08_实验日志/2026-09-08_probe_定位学习位置与目标/FINAL_REPORT.md)。

> **2026-09-08 双数据集完整推理分析完成**：Drone教师1469图28.69秒，完整队列3分33秒；CPU分析19.67秒。结合LLVIP训练侧覆盖，明确定位/置信度/特征的证据与反证。见[方向判断及复核入口](research_bundle/08_实验日志/2026-09-08_probe_Drone教师完整推理/FINAL_REPORT.md)。

> **2026-09-08 训练侧定位门覆盖核对完成**：复用首32图缓存与历史同批L2记录，11个双方粗检出而仅IR达IoU0.75的对象中，L2实际选4个，另7个因参考IoU≥0.70被前置门排除。全程CPU、无新增前向；门的覆盖与学习效果分别判断。见[原始记录及独立验收](research_bundle/08_实验日志/2026-09-08_probe_训练侧定位覆盖/FINAL_REPORT.md)。

> **完整dev检出机会复核完成（2026-09-08）**：[9.455秒CPU分析及方向决定](research_bundle/08_实验日志/2026-09-08_probe_开发集真实检出机会/FINAL_REPORT.md)。LLVIP 2406图/7879GT中，1304个对象双方检出但仅IR过IoU0.75；另1635个仅IR在0.25/0.5检出。先核定位训练覆盖，响应方向区分771个阈值敏感与864个低阈值仍未匹配；不当KD增益，不新增E200。

> **2026-09-08 同帧原生检测核对完成**：LLVIP固定32张训练图，旧assigned定义的教师机会9个，原生NMS后一对一匹配仅1个且已选中；selected中31/32双方已检出。旧低置信候选不能直接解释为实际漏检，完整dev AP不受此定义修正影响。见[结果、源码及原始小回执](research_bundle/08_实验日志/2026-09-08_probe_同帧选择覆盖/FINAL_REPORT.md)。下一项优先CPU复用完整dev预测，不新增E200。

> **2026-09-08 双数据集快速方向筛选已完成**：九次短训与九次完整dev评价已收口。分类、定位、置信度和局部特征关系的原值、去留范围、实际耗时、代理局限与协议修订见[综合结果](research_bundle/08_实验日志/2026-09-08_probe_快速方向筛选/FINAL_REPORT.md)。全部是单seed探索，不自动扩大E200；原F-rel阻塞与独立F-rel-GM结果分别保留。

> **2026-09-08 分钟级微调筛选已完成**：固定 2048 图、成熟同协议 N42 初始化、3 轮的 N-ft/C0-ft/C1-ft，含三次短验收、三次训练和完整 dev 评价的队列实测 **15 分 56.6 秒**。单 seed mAP50–95 为 **53.714083 / 53.662570 / 53.822939**；C1-ft 相对 N-ft +0.109 pp，但三臂均低于未微调 N42 的 54.513608，按事前规则不自动扩展。请读[最终报告](research_bundle/08_实验日志/2026-09-08_ops_小时级筛选重构/FINAL_REPORT.md)、[accepted 分析与原值](research_bundle/08_实验日志/2026-09-08_ops_小时级筛选重构/results_1022/COMPARISON_TABLES.md)及[本次发布范围](publication_checks/update_20260908_hourly_screen/README.md)。本轮验证短反馈可执行，不替代 E200 或三 seed 方法增益。

> **2026-09-08 E8 三臂结果已完成**：N/C0/C1 的 seed42 固定 E8 与完整 dev 评价全部完成，mAP50–95 原值为 **43.376542 / 43.920584 / 43.408107**。C1 未显示早期优势；单 seed 短日程不替代 E200，旧三 seed 及归因臂继续。完整六阶段队列耗时 **2.668001 小时**，共享负载下的预算兑现不称严格代码加速。请读[三臂阶段判断与原始复核](research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/three_arm_summary_20260908/README.md)和[实际吞吐](research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/three_arm_summary_20260908/THROUGHPUT.md)。

> **2026-09-08 C0 E8 端点**：Drone seed42 的 C0 已完成固定 E8 last/EMA 与完整 dev 独立评价，mAP50–95 原值为 **43.920584**；18 项限定回执复核通过。05:37 时 C1 为第 5/8 轮，保持三臂齐备后统一比较；本次不计算 C0−N 增益、不作 E200 等价或正式方法收益结论。请读[C0 原始端点与复核](research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/endpoint_C0_20260908_053913/README.md)。

> **2026-09-08 首个 E8 端点**：Drone seed42 的 N 已完成固定 E8 last/EMA 与完整 dev 独立评价，mAP50–95 为 **43.37654**；17 项限定回执复核通过。这是单独的短程 N 原值，C0/C1 尚未完成匹配比较，不是蒸馏增益或 E200 等价结论。请读[原始端点与复核](research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/endpoint_N_20260908_043812/README.md)。训练 CSV 的占位/末行短列不作为 AP 来源。

> **2026-09-08 性能修复与 E8 快速反馈**：原 block16 的真实训练轨迹失败保留；selected-only 在各 24 次成功更新上实现学习损失、梯度、模型/optimizer/EMA 状态逐位一致。正常日志频率仍需约 2.48 秒/C1批，已按先验预算启动共同 E8 的 N/C0/C1 单 seed 队列，尚无新 AP。请读[性能与失败证据](research_bundle/08_实验日志/2026-09-08_ops_训练吞吐诊断/README.md)、[E8 设置与启动](research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/README.md)、[既有数据分析综合报告](research_bundle/07_研究分析/RGBIR数据分析综合报告_20260908.md)。旧 E200 继续，本次不声称 E200 等价或方法增益。

> **2026-09-07 双数据集证据推进**：LLVIP完整2406dev旧RGB/IR baseline mAP32.8784/48.8529，定位优先；Drone全1469dev六端点表明少数类混淆贡献macro分类oracle的94.26%，不能将对象计数概括为全AP瓶颈。请先读[新阶段判断](research_bundle/08_实验日志/2026-09-07_probe_双数据集证据优先推进/STAGE_REPORT.md)与[原始证据/独立审阅入口](research_bundle/08_实验日志/2026-09-07_probe_双数据集证据优先推进/README.md)。已有C1继续，新定位方法尚未准入。

> **2026-09-07 baseline数据重诊断已完成**：两数据集各1024 train＋200 dev，当前Drone N42/IR42/N0与LLVIP baseline。Drone机会以低置信为主、LLVIP定位证据更强；局部IR特征未稳定超过独立RGB，固定ridge欠拟合已单列。请先读[阶段判断](research_bundle/08_实验日志/2026-09-07_probe_Baseline蒸馏机会重诊断/STAGE_REPORT.md)及[完整原始复核入口](research_bundle/08_实验日志/2026-09-07_probe_Baseline蒸馏机会重诊断/README.md)。这是推理/读出诊断，没有新增KD AP或定位准入。

# RGB–IR 独立分类与定位蒸馏：证据与复核入口

> **2026-09-07 17:06 follow-up:** The explicit post-hoc class adapter is independently accepted (17 tests and six actual endpoints). Original result files remain unchanged. C0 harm review remains REVIEW_REQUIRED with no missing items. C1 seeds were at epoch 5 as of 16:53; no complete C1 AP is available. [Evidence and review](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/heartbeat_20260907_1653/README.md).


**16:35 补充阶段：C1三seed进入第4轮；旧N/C0六次补评均与历史五指标逐项一致，实际观察桥接已接受。旧C0净正确对象+4/−3/+101、背景误检+3/+5/+46，后者触发专项复核。C0后续自动扩展暂停，已有训练继续；L仍缺几何证据。** [最新阶段报告](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/STAGE_REPORT_1635.md) · [真实对象诊断](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/old_c0_object_diagnostics_v1/README.md)。

**最新执行已切换为 C1 分类三 seed 优先、L1 定位条件准入。请先读 [实施状态与实际证据](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/README.md)，再读 [本轮复核请求](INDEPENDENT_KD_REVIEW_PROMPT.md)。**

旧 N/C0/random 九份 E200 last/EMA 独立端点已齐。C0−N 为 **+0.266655±0.144373 pp**，C0−random 为 **+0.174939±0.038853 pp**，均三 seed 同向。旧 C0 shuffled42/same-modal42 继续；四臂尚未齐，不能把这两个配对差当成已完成跨模态归因。

**C1 seed42/0/123已正式启动，λ=0.09227393550836771**；六条真实兼容、固定64批校准、两路径24-update canary及单卡双开实测均通过。新独立模块保留原生检测损失，C1用原C0选择与区域，传递P3/P4全类别相对raw logits。当前尚无新C1 AP，旧CL/CGT排程已被替代。

- [旧三 seed 原值、配对差及限制](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/OLD_RESULTS_1407.md)
- [冻结执行计划](research_bundle/refine-logs/EXPERIMENT_PLAN.md) · [追踪表](research_bundle/refine-logs/EXPERIMENT_TRACKER.md)
- [独立模块：源码、测试、训练与评价入口](research_bundle/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/)
- [首批24对几何诊断及 UNKNOWN 原因](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/PRIMARY_GEOMETRY_VISUAL_24.md)
- [实际失败、最小修复与资源证据](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/GPU_ADMISSION_FIRST_ATTEMPTS.md)
- [对象修复/损伤规则](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/object_error_analyzer_rect_v2/OBJECT_ERROR_ANALYSIS_RULES.md) · [外部基线准备](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/EXTERNAL_BASELINE_PREPARATION.md)
- [所有实验日志](research_bundle/08_实验日志/README.md) · [03:22 历史入口](ARCHIVE_README_20260907_0322.md)

L1 当前缺合格物理对应点证据，未进入新长训；0个接受点集不等于0个潜在定位机会或数据集未配准。两个数据集没有换64批、过采样或放宽门控。封存 test 不参与方法选择。归因、内容控制和 LLVIP 增量均分阶段触发，不一次性提交全部预算。

仓库保留正负结果、失败attempt和勘误。小体积原始数值与源码保持字节一致；Markdown链接适配GitHub。导出清单为根目录按时间命名的 `INDEPENDENT_KD_BUNDLE_MANIFEST_*.json`。不含凭据、大权重或数据集原图全集。
