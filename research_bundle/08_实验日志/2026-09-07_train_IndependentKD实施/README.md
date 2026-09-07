# Independent KD v2 实施

**状态（2026-09-07 15:35）：C1 三 seed 已正式启动，均已达到至少24次成功更新；λC1=0.09227393550836771。L1 因缺合格物理对应点证据暂不准入。** [完整准入与实际启动证据](FORMAL_LAUNCH_ACCEPTANCE_1535.md)。

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
|C0 shuffled42/same-modal42|旧队列继续，15:35进入52/36轮；不是新 C1 对照|[只读快照]（未导出的工作区路径：snapshots/2026-09-07T153517.173314_0800/）|
|C1/C1_y|独立 trainer、原选择适配、逐类 Bernoulli KL、梯度观察实现并经交叉代码审阅|[分类梯度独立复核](C1_GRADIENT_INDEPENDENT_CODE_REVIEW.md)|
|L1/L_GT|严格门控、DFL、GT目标及支持域实现；几何未准入，不运行正式 canary 或 E200|[24 对审核](PRIMARY_GEOMETRY_VISUAL_24.md) · [5 对独立复核](INDEPENDENT_GEOMETRY_VISUAL_5.md)|
|C1 校准与canary|64/64批有限非零；C1/C1_y各24成功update；同卡双开实測通过|[实际验收](FORMAL_LAUNCH_ACCEPTANCE_1535.md)|
|N/C0 实际兼容|0/42/123×N/C0六轨迹全部通过；旧六端点评价复用仍待桥接与逐类/逐对象补采|[实际回执汇总](stage_status_20260907_152201.json) · [首次失败与修复](COMPATIBILITY_FRESH_PROCESS_FIX.md)|
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
C1三个worker各自承担E200后独立评价，不按首seed AP停止其余seed。当前没有新C1 AP。L阻塞不影响分类。旧/新评价桥接候选仍为DRAFT，旧N/C0六last的逐类与objects须补采，才能完成升级前损伤检查；[最小补采方案](EVALUATION_BRIDGE_PREPARATION.md)。三seed和四臂不足时不升级论文主张。阶段证据发布到 `research/full-evidence-20260906`；发布提交以实际回执为准。
