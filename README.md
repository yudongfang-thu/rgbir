# RGB–IR 独立分类与定位蒸馏：证据与复核入口

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
