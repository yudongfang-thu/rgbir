# 2026-09-07 Independent KD 实施目录新增回执
新增独立实验模块与同日实验日志目录，仅增加文件，不移动或重命名原始结果/失败attempt/receipt/checkpoint。94阶段目录将随部署记录于实验日志。

2026-09-07跟进增量：同一实施日志内新增 `posthoc_class_adapter_v1` 与 `heartbeat_20260907_1653`，用于显式逐类接入、CPU检查和只读状态；独立审阅目录按实际回执登记。94新增 `RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/posthoc_class_adapter_cpu_v1`，仅两个小源码和CPU测试输出。无移动、覆盖或删除原始实验产物，冻结训练 release 不变。

17:54例行记录：本地同日志新增 `heartbeat_20260907_1754`，保存只读运行核对、小快照索引及脚本副本；服务器未修改目录或原实验产物。

18:54同类例行记录新增 `heartbeat_20260907_1854`；采用小快照和通用记录脚本，仅本地新增记录与更新索引，不改变服务器实验产物或目录。

18:58用户查询新增 `status_20260907_1858`，同样仅保存本地只读快照分析和脚本副本。

用户询问定位补救后，新增本地 `08_实验日志/2026-09-07_audit_定位补救与准入修订/`，保存候选修订分析、旧D2字段清点脚本与小体积JSON。仅增加分析产物并更新索引；没有移动、覆盖、删除原始证据，也没有修改冻结L1合同、训练release或服务器任务。L2仍是待冻结候选。

用户要求从baseline重新比较知识载体后，新增 `08_实验日志/2026-09-07_probe_Baseline蒸馏机会重诊断/` 与94数据盘 `RGBT_campaign/artifacts/rgbir_baseline_information_20260907/`。首次启动在导入阶段失败、未申请CUDA，原文件与日志留存；修复服务器release路径并独立审阅后使用新的 `attempt2/`，不覆盖原attempt。全部新诊断共用既有lease。C1及旧内容控制、原始权重/结果/失败记录均不修改。

补充完成：attempt2因逐对象GPU同步低吞吐按技术原因保留并停止；最终attempt3统一每图CPU后处理，两个数据集2448图已完成，回执与source原样保留。新增本地remote_exports、CPU分类/DFL及定位feature读出、独立复核和GitHub增量；原工程结构与任何历史大产物不变。

20:09例行增量：本地实施日志新增 `heartbeat_20260907_2008_runtime/`（只读快照比较、脚本及小回执）和 `heartbeat_20260907_2008/CONTINUITY_REVIEW.md`（文档连续性复核）。更新状态索引及已完成事项的过期描述；不修改服务器任务、原始产物或冻结科学矩阵，不为无新端点快照重复发布GitHub。

用户授权双数据集继续后，新增 `08_实验日志/2026-09-07_probe_双数据集证据优先推进/`，分为CPU定位压力诊断、AP错误分解、LLVIP配置准备和完整dev评估导出。旧资料只读、训练不改；94仅在实际部署后登记新 `RGBT_campaign/artifacts/rgbir_evidence_priority_20260907/llvip_full_eval_attempt1/`。执行前冻结协议，失败attempt及原始回执保留。

双数据集阶段完成：94新增有效 llvip_full_eval_attempt2 与 natural_flow_attempt2，均在原rgbir_evidence_priority_20260907根。两个旧attempt1分别因标签路径/配置导入失败，原始产物保留；本地保存完整有效预测、自然流/选择记录、CPU原始统计与独立审阅。仅补充导航/解释，不移动、覆盖、删除旧结果或权重。自然流初次SCP长路径失败通过tar流补齐缺失文件，已有文件逐字节一致；新增governance_group_sources为原TSV只读副本。

22:08心跳：本地新增实施日志heartbeat_20260907_2208_runtime，保存只读状态、原文before副本和文档连续性修正回执；无服务器目录或原实验产物变更。跟踪表B02–B04恢复原ID语义，新增B05；不改结果、门槛或训练。
