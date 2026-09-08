# 独立审阅初始提示原文

发送方：`/root`。接收方：`/root/loc_target_review`。模型身份不可观测，记unavailable。

> 作为独立审阅者，核本次 LLVIP cache-only 诊断的源码/数据/结论，不运行GPU/新模型/训练、不计算hash、不修改旧结果。工作区 E:/SHARE/光sar，读AGENTS。审阅root新条目 08_实验日志/2026-09-08_probe_定位学习位置与目标：ap_error将写anchor_join/，baseline_feature_analysis将写target_audit/。输入：08_实验日志/2026-09-08_probe_训练侧定位覆盖/{output_attempt1,bridge/output_attempt1}；08_实验日志/2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe/；08_实验日志/2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/llvip/及实际loss源码。任务是检验native匹配框与真实冻结参考学习anchor的身份链接、selected和基数、教师/GT目标差和导数/参数梯度的语义边界。现在可以先读原输入和源，不依赖root偏好；作者稍后通知产物齐备。实际输出独立CPU复算和小样例合适，不能为了审计扩实验。结果写新entry independent_review/ 下 EXPERIMENT_AUDIT.md/json，字段date,auditor,overall_verdict,integrity_status,checks(gt_provenance,score_normalization,result_existence,dead_code,scope,eval_type),claims；no-hash是本轮用户授权范围约束，hash字段标not computed，不许捏造。标真实agent名，模型未知写unavailable，不声称跨模型。提供prompt/response trace。项目主要模型使用已有train单批/8批，不读取test。首个5-10分钟审源码，收到实际结果再有界复核。

最终产物通知：anchor_join/output_attempt3、target_audit/output_attempt1；实际94捕获源码在source_identity/。独立初算完成后，root补充拟“停止本版L2仅放门/加剂量扩展，不否定所有定位”，要求独立限定行为判断支持范围。初始提示没有给出希望审计通过或希望否定定位的偏好。
