> **2026-09-09 13:06 最新：**[LLVIP作者RGB/IR完整复评与94恢复检查](ORIGINAL_REPRO_CONTINUATION_20260909.md)。新增mAP52.664239/67.014908%，接近修订版52.7/67.0，44项数值/覆盖检查通过。CFT三卡0更新OOM，LLVIP原训练10更新后OOM，均未开E200。94通过实际GPU/数据检查，旧训练未恢复；当前其他用户已占用全部8卡。以下为历史阶段记录。

> **2026-09-09 最新原文复现：**[CFT／LLVIP 作者权重复评与训练准入结果](AUTHOR_PROTOCOL_STATUS_20260909.md)。完整3463对test的AP50/AP75/mAP为97.376907/72.891348/63.537478%，距论文均≤0.13pp；覆盖/指标复核完成。原B32训练三次资源检查首批OOM，E200未启动。新增迁移版暂停。下方均为历史阶段记录。

> **2026-09-09 最新：**[90 实际复现接入与范围](REPRODUCTION_STATUS_20260909.md)：CFT 真实权重前向、BCDL 算子、CMD 22 tests 已执行；无新 AP。以下时间点为历史状态。

# 最新结果与执行状态

> **2026-09-07 17:06 follow-up:** The explicit post-hoc class adapter is independently accepted (17 tests and six actual endpoints). Original result files remain unchanged. C0 harm review remains REVIEW_REQUIRED with no missing items. C1 seeds were at epoch 5 as of 16:53; no complete C1 AP is available. [Evidence and review](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/heartbeat_20260907_1653/README.md).


[完整16:35阶段报告](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/STAGE_REPORT_1635.md)。C1三个seed正在E200第4轮，λ=.09227393550836771，尚无新C1完整AP。

旧N/C0/random九端点齐：C0−N +.266655±.144373pp，C0−random +.174939±.038853pp；四臂归因未齐。旧N/C0六次完整dev补评已全部成功，五汇总指标逐项exact；[观察桥接独立接受](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/legacy_observed_bridge_root_review_v1/REVIEW.md)。

新对象诊断：修复473/472/517，损伤469/475/416，净+4/−3/+101；背景误检+3/+5/+46。后者触发C0专项损伤复核，暂停其后续自动扩展，已有训练继续。[逐类、尺度和部分亮度结果](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/old_c0_object_diagnostics_v1/README.md)。

L仍因几何证据不足阻塞。正式分析器对posthoc逐类证据的显式接入仍待完成；不向旧JSON注入新字段。新三seed结果、内容消融和完整四臂尚未完成。
