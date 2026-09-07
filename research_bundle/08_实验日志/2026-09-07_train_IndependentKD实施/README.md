# Independent KD v2 实施

2026-09-08 05:38：旧C1按42/0/123已完成45/44/45轮、进行中46/45/46；旧shuffled/same-modal已完成90/86轮、进行中91/87。五任务持续更新、无新E200端点/评估/失败。[只读增量](heartbeat_20260908_0536/README.md)。独立[E8队列](../2026-09-08_train_分类快速反馈E8/README.md)N/C0原值齐备，C1尚在第5轮，不能混入本E200判断。

2026-09-08 04:36：旧五个E200任务持续成功更新，C1按42/0/123为43/41/43轮，旧shuffled/same-modal为87/83轮；无新完整端点、评估或失败，last KD有限。[只读增量快照](heartbeat_20260908_0435/README.md)。独立[E8队列](../2026-09-08_train_分类快速反馈E8/README.md)的N已完成并接受短程原值，不能混入本E200比较。

2026-09-08 03:37：旧五个E200任务均持续成功更新，C1按42/0/123为40/38/40轮，旧shuffled/same-modal为84/80轮；无新完成、评估或失败回执，末次KD有限。[只读快照](heartbeat_20260908_0335/README.md)。另有独立[E8快速反馈队列](../2026-09-08_train_分类快速反馈E8/README.md)于03:31启动，不改变这里的E200身份或训练轨迹。

2026-09-08 01:12：C1按42/0/123为32/31/32轮，旧控制76/71轮；五任务持续成功更新，六项资源/lease检查通过，项目RSS140.64GiB，无新E200端点或失败。新增AMP skip：c_same_modal=1。末次KD有限，冻结科学条件保持。 [运行快照]（服务器/本地保留，未包含于本阶段发布：heartbeat_20260908_0112_runtime/README.md）。

9月8日00:12例行核对：C1按42/0/123为29/28/29轮，旧控制72/67轮。五任务持续成功更新，六项资源/lease检查通过，RSS140.74GiB，无新E200端点或失败。shuffled新增1次AMP skip已记录，当前KD有限；不据一次AMP跳步更改训练。 [本次快照]（服务器/本地保留，未包含于本阶段发布：heartbeat_20260908_0012_runtime/README.md）。

23:21用户进度查询：C1按42/0/123为26/25/26轮，旧shuffled/same-modal为70/64轮。3卡5训练持续更新，资源合规，RSS140.83GiB，无新E200端点或故障。[最新汇总]（服务器/本地保留，未包含于本阶段发布：status_20260907_2321_runtime/USER_STATUS.md）。

23:09例行核对：C1三seed第25轮，旧shuffled/same-modal为70/63轮。五任务成功更新增量均为正、无新增AMP跳步，六项资源/lease检查通过；3卡5训练、项目RSS141.51GiB，无新完整端点或故障。 [本次只读记录]（服务器/本地保留，未包含于本阶段发布：heartbeat_20260907_2309_runtime/README.md）。

最新运行核对（2026-09-07 22:08）：[五任务持续更新、资源合规](heartbeat_20260907_2208_runtime/README.md)。C1三seed第22轮，旧shuffled/same-modal为69/60轮，无新E200端点或故障。下方早期时间戳是历史快照。

新的[双数据集证据阶段](../2026-09-07_probe_双数据集证据优先推进/STAGE_REPORT.md)已完成并独立接受：完整dev AP、定位压力与两组真实64批；LLVIP定位候选207、Drone602，均未几何认证。当前证据优先LLVIP定位目标核验与Drone少数类混淆；旧“低置信占多数”仅为200dev对象计数，不能推广为完整macro AP瓶颈。已发布证据提交00bfb59及回执a819fe0。

历史查询（2026-09-07 20:09）：[五任务持续更新、资源合规]（服务器/本地保留，未包含于本阶段发布：heartbeat_20260907_2008_runtime/README.md）。C1按42/0/123的progress.epoch为16/15/16，CSV均完成15轮；进度与CSV分别刷新。旧shuffled42/same-modal42为63/53轮，项目RSS141.58GiB；无新增端点或失败回执。下面17:06为最新实施验收阶段。

新的[baseline中间结果诊断](../2026-09-07_probe_Baseline蒸馏机会重诊断/STAGE_REPORT.md)已完成2448图并发布：Drone优先核查前景置信度/排序，LLVIP定位目标质量更有支持；这些是信息机会诊断，未改变C1、L1准入或冻结新方法。[文档连续性复核]（服务器/本地保留，未包含于本阶段发布：heartbeat_20260907_2008/CONTINUITY_REVIEW.md）。

**状态（2026-09-07 17:06）：补评逐类接口已获独立接受，六端点完整接入，C0 harm仍为REVIEW_REQUIRED且无缺失项；16:53 C1三seed均第5轮，旧shuffled42/same-modal42第58/41轮。L1仍缺几何证据。** [最新跟进及接口验收](heartbeat_20260907_1653/README.md) · [16:35阶段分析](STAGE_REPORT_1635.md) · [完整准入与启动证据](FORMAL_LAUNCH_ACCEPTANCE_1535.md)。下面保留此前阶段明细，最新判断以跟进记录为准。

## 目的与固定范围
C1 三 seed 优先，C1/C1_y 与 L1/L_GT 独立，旧 C0 对照继续。Drone 定位不准入而 LLVIP 准入时转 LLVIP 并增加 N 三 seed；核心预算12/15，完整两线归因24/27，条件触发，不一次提交。

## 设置
继承用户2026-09-07执行计划及正式 OEv1 recipe；保留严格0.70定位门；固定64批校准/16非零批、每新路径24成功update。资源共用94既有lease，最多常规3卡/有证据例外4卡，实测双开，整卡留2GiB，项目VRAM<70%、RSS≤300GB及240GiB预约阈值。

## 已完成结果与证据

|工作|实际状态|证据|
|---|---|---|
|旧 N/C0/random|九份 E200 last/EMA、完整 dev 独立评估均完成|[三 seed 原值及配对统计](OLD_RESULTS_1407.md)|
|C0−N|+0.266655±0.144373 pp，三 seed 全正|[分析器原始输出](old_endpoint_analysis_1407.json)|
|C0−random|+0.174939±0.038853 pp，三 seed 全正；尚缺四臂内容归因|同上|
|C0 shuffled42/same-modal42|旧队列继续，15:35进入52/36轮；不是新 C1 对照|[只读快照](snapshots/2026-09-07T153517.173314_0800)|
|C1/C1_y|独立 trainer、原选择适配、逐类 Bernoulli KL、梯度观察实现并经交叉代码审阅|[分类梯度独立复核](C1_GRADIENT_INDEPENDENT_CODE_REVIEW.md)|
|L1/L_GT|严格门控、DFL、GT目标及支持域实现；几何未准入，不运行正式 canary 或 E200|[24 对审核](PRIMARY_GEOMETRY_VISUAL_24.md) · [5 对独立复核](INDEPENDENT_GEOMETRY_VISUAL_5.md)|
|C1 校准与canary|64/64批有限非零；C1/C1_y各24成功update；同卡双开实測通过|[实际验收](FORMAL_LAUNCH_ACCEPTANCE_1535.md)|
|N/C0 实际兼容|0/42/123×N/C0六轨迹全部通过；旧六端点补评、实际观察桥接及逐类/逐对象接入已完成并按各自范围接受|[实际回执汇总](stage_status_20260907_152201.json) · [首次失败与修复](COMPATIBILITY_FRESH_PROCESS_FIX.md) · [补评接入验收](heartbeat_20260907_1653/README.md)|
|C1 E200三seed|42/0在GPU2双开，123与旧same-modal共用GPU5；启动顺序42→0→123|[实际启动与成功更新](formal_live_20260907_153211.json)|
|对象修复/损伤分析器|15项已知真值测试及独立复核通过；尚无新 C1 结果|[规则](OBJECT_ERROR_ANALYSIS_RULES.md) · [独立回执](object_analyzer_independent_review_v1/review_receipt.json)|
|外部对比|BCKD/FGD/LD 公开实现与协议适配准备；CMDistill/CCLKD partial 身份审计完成，均未占新训练资源|[准备表](EXTERNAL_BASELINE_PREPARATION.md)|

当前旧 C0 相对 N 的汇总 recall 三 seed 同降，均值 −1.172702 pp。这不等于固定置信度的漏检率，但需要新对象诊断解释；不声称负迁移被消除。

## 定位阻塞范围

固定64自然批各2048张图。原小审核名单在 Drone 的命中上界为25/64批，LLVIP为11/64批，后者不能仅靠原名单满足16非零批。

本次实际看了24对固定候选（两数据集各12对），独立复核5对。合格对应点集为0、测得配准失败为0、UNKNOWN为24。原因是目前无法可靠给出覆盖三个象限、目标及anchor边界邻域的六点与严格误差证据；这不是数据集配准失败或定位蒸馏无效的结论。没有换自然训练流、过采样少数图片或放宽0.70门。

## 准入与失败记录

每个部署 release 和失败 attempt 均保留。CPU参考包18项与工程测试在94 pinned环境执行；这些不能代替六条30-batch/24-update真实兼容、64批梯度校准及新路径24-update canary。

最终release_gpu5的173项工程测试和18项参考包测试在94通过；六条真实兼容、64批校准、两路径canary、双开及完整dev评价等价均通过。正式readiness及配置已生成。早期两批临时λ未用于正式训练，最终λ来自完整64批。

首个formal campaign因队列误解析虚拟环境Python链接而在训练前失败。仅修LOG调度脚本，新建attempt2；NEW及科学验收不变，原失败记录保留。[修复说明](FORMAL_WORKER_VENV_FIX.md)。

## 产物与路径
本地代码：[独立实验模块](../../03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2)。冻结计划：[当前执行计划](../../refine-logs/EXPERIMENT_PLAN.md)，条件矩阵与LLVIP增量预算均已同步。

94部署及验收：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/`。实际正式输出在其 `formal_C1_gpu5_attempt2/runs/C1_seed{42,0,123}/`，位于授权数据盘artifacts下；最初拟定的独立runs根尚未使用。大权重及完整轨迹tensor留服务器，小产物按[最新采集清单](remote_admission_1532/source_manifest.json)保存在本目录。

## 局限与下一步
C1三个worker各自承担E200后独立评价，不按首seed AP停止其余seed。当前没有新C1完整AP。六次旧N/C0补评及实际观察桥接已完成，rect-v2对象诊断已通过；posthoc逐类接口17:06已独立接受，未来C1真实端点仍需核对其实际评价源码及回执。C0背景误检三seed同增，后续自动扩展暂停待专项复核，已有训练继续。三seed和四臂不足时不升级论文主张。阶段主包已推送到 `research/full-evidence-20260906`，提交 `1218adf`；见[实际发布回执](PUBLISH_RECEIPT_1218adf.md)，补充阶段发布另留回执。最新baseline诊断发布为 `6ed99b2`，见[发布回执]（服务器/本地保留，未包含于本阶段发布：../2026-09-07_probe_Baseline蒸馏机会重诊断/publication_receipt.json）。20:09例行无变化快照仅本地留档。
