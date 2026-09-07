# 最新结果与执行状态

> **2026-09-07 17:06 follow-up:** The explicit post-hoc class adapter is independently accepted (17 tests and six actual endpoints). Original result files remain unchanged. C0 harm review remains REVIEW_REQUIRED with no missing items. C1 seeds were at epoch 5 as of 16:53; no complete C1 AP is available. [Evidence and review](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/heartbeat_20260907_1653/README.md).


[完整16:35阶段报告](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/STAGE_REPORT_1635.md)。C1三个seed正在E200第4轮，λ=.09227393550836771，尚无新C1完整AP。

旧N/C0/random九端点齐：C0−N +.266655±.144373pp，C0−random +.174939±.038853pp；四臂归因未齐。旧N/C0六次完整dev补评已全部成功，五汇总指标逐项exact；[观察桥接独立接受](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/legacy_observed_bridge_root_review_v1/REVIEW.md)。

新对象诊断：修复473/472/517，损伤469/475/416，净+4/−3/+101；背景误检+3/+5/+46。后者触发C0专项损伤复核，暂停其后续自动扩展，已有训练继续。[逐类、尺度和部分亮度结果](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/old_c0_object_diagnostics_v1/README.md)。

L仍因几何证据不足阻塞。正式分析器对posthoc逐类证据的显式接入仍待完成；不向旧JSON注入新字段。新三seed结果、内容消融和完整四臂尚未完成。
