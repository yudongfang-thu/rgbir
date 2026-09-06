# RGBIR Object Evidence v1 三 student-seed 扩展独立审查

> 2026-09-06：训练、评估、seed传递和扩展launcher通过独立静态审查；8项控制流CPU检查通过。新卡各两臂canary通过后可按冻结计划正式扩展。该判断不代表方法有检测收益。

## 审查范围与计划

原 GPU4 上 paired seed42/E200 → weight0 seed42/E200 队列保持；新增两条串行队列覆盖 weight0 seed0 → paired seed0、paired seed123 → weight0 seed123。使用已经执行过双臂 canary 的 release_v2 trainer、loss、loader、config，除了显式 `--seed 0/123` 不变更方法与训练超参。IR 教师与 RGB 参考均固定既有 seed42 baseline。

审查依据为工作区 README、实验日志索引、最近三个实验条目、首轮 EXPERIMENT_PLAN.md，以及工程 AGENTS.md 的资源、回执和研究约束。审查只读代码与已有证据，不运行 GPU，不修改训练代码，不计算摘要校验值。

## 1. Seed 与回执一致性：通过，有一项记录提醒

- `train_object_evidence.py::run` 先将 CLI seed 覆盖到内存 cfg，再创建 trainer；`build_trainer` 将该值传入原生 Ultralytics overrides，数据顺序、增强和学生训练使用新 seed。
- KD 调用的私有 seed 同步使用覆盖后的 cfg seed。paired 选择本身是确定性的，不通过该参数额外改变方法。
- launch_manifest、completion_receipt、`emit_bound_run_receipt` 使用覆盖后的实际 seed；原生 `args.yaml` 应同时核验为 0/123。
- `evaluate_object_evidence.py` 读取该 run 的 completion_receipt，再用其 arm/seed 标识最终指标和 eval 回执，不会把新增 run 标成 seed42。固定 last/EMA 与完整 val 1469 图不随 seed 改变。
- `protocol_config.yaml` 仍是基础配置副本，内部 seed 为42。这是“基础配置 + CLI override”的记录结构，不是实际训练 seed 错误；不能单独以该文件判断 run 身份。建议扩展 launcher 另存 seed override/effective config 说明，并用实际 args、launch manifest、completion/eval receipt 闭合身份，不改正在运行的 release_v2。

## 2. 科学解释与 outcome-blind：允许扩展，限定证据范围

完整 endpoint AP 尚未产生时，按已有 0/42/123 计划增加重复并冻结六格，不是依据检测结果修改方法；中间 loss、对象数量、吞吐只支持工程状态判断。本目录 EXPERIMENT_PLAN.md 已于新正式 run 启动前冻结全六格、主指标、配对差值、样本SD和次指标，不按结果改方法或提前停可解释的对照臂，满足本次扩展的 outcome-blind 要求。

三次重复估计的是**固定教师、固定参考、固定数据划分条件下，学生训练随机性的 paired 干预净变化**。不是三套独立教师重训，不估计教师训练随机性，也不支持跨数据集泛化。建议主统计固定为各 seed 的 `100 × (paired mAP50–95 − weight0 mAP50–95)`，报告三项差值、均值、样本标准差（ddof=1）与正负方向；各臂 mean±SD 同时列出。

N/P 六格能检验总干预是否改善此开发协议的结果，不能分离真实跨模态内容、IR 独立标签、对象选择与额外训练正则。shuffled/same-modal 等归因对照尚未完成，不能升级为“已证明跨模态独特贡献”“已避免负迁移”或 paper-ready claim；没有 accepted analyzer 也不升级证据等级。即使三 seed 结果全正，也保留这些边界。

## 3. 资源与新 launcher：静态审查通过，实际 canary 由主 agent 验收

每张卡必须先实测 canary 再开启新正式训练，guard lease 覆盖全程。新任务不能仅根据 lease 数判断全项目 GPU 使用：任何既有但未出现在 guard 注册表中的项目 CUDA PID 仍须计入最多三卡、每卡最多两个项目 CUDA PID、项目显存严格小于70%、全卡实际空闲至少2GiB、全项目 aggregate RSS 小于300GiB与240GiB接纳阈值。

主 agent 的最新资源快照说明新分配候选 GPU5/6 当前空闲，原 GPU4 只有既有 OEv1 训练；本次新卡上每次最多一个 OEv1 CUDA 任务。10000MiB VRAM 与49152MiB RSS 是预约，仍应由新卡各两臂24实际更新 canary的峰值和当时实占证明可接纳。资源暂时不足只排队；技术失败保留 attempt，不能将训练退出码误认成可无限重试的资源排队。

已审本目录 `expansion_worker.py` 与8项控制流测试源码：

- 每 seed/stage/attempt 建独立 worker目录，每臂独立日志/结果/seed override；写覆盖后 effective_config，同时继续调用原始 release_v2 与 CLI seed，保留源码不变。
- 单卡 flock + 每次唯一 candidate GPU + 单 GPU/单 CUDA PID guard预约；资源排队固定原卡，不回退第四张卡。
- 只有 `QUEUED`、无 `LAUNCHED` 且退出2才重试，已启动训练退出2不重跑；原 attempt 存在即拒绝，失败产物保留。
- 某一臂训练或eval技术失败，不阻止另一独立臂执行；总状态记 partial_failed，不误记成功。
- full 要求相同 seed、GPU、source 路径下的 accepted canary；canary逐张量验证 P/N 初始化及首批输入相同，检查实际更新、非空选择、非零KD梯度、weight0精确等价、教师无梯度。

初审发现初版 canary 比较记录峰值后直接通过，而既有 `project_resource_guard.py::inspect` 只统计资源，不会因超过某run预约值自动失败。实现agent已修复：`validate_canary_resources` 对每臂检查 NVML峰值≤10000MiB、RSS峰值≤49152MiB、实际GPU等于本worker分配、CUDA PID≤1；比较通过保存 `resource_reservation_checks_passed=true`，full准入必须具备该标记。已复读修订代码，缺口关闭。

已读取94 pinned CPU的 `launcher_cpu_tests_attempt2.log`：8项全部通过，覆盖资源正负边界、真正资源排队可重试、已启动子进程退出2不重试、训练/eval失败继续另一臂、已有attempt不覆盖、seed-specific canary准入及命令seed/stage传递。测试在CPU运行，不替代真实canary显存与训练核验。

## 最终状态

- 科学协议与 seed/eval 标识：通过；固定教师的条件性与基础 seed42配置覆盖需在扩展日志说明。
- launcher 静态审查：通过，资源预约验收缺口已修复；8项控制流CPU测试通过。
- 新卡 canary：已复核本地两份比较回执。seed0/GPU5、seed123/GPU6均双臂24实际更新，初始化与首批tensor相等；4个canary的NVML峰值均6304MiB，RSS峰值最高28766MiB，全部低于10000/49152MiB预约；单卡单CUDA PID与资源验收标记通过。正式运行持续状态以主agent快照为准。

## 4. 固定端点汇总器追加审查：通过

已只读审 `analyze_three_seed_endpoints.py`、对应fixture测试和 `endpoint_snapshot1/summary.json`，并读取更新后的 `endpoint_cpu_tests_attempt2.log`：10项CPU检查通过。未改训练、未运行GPU。

- 汇总器完全不读取 results.csv。缺终态completion/eval或训练/评估COMPLETED回执时保持pending、metrics=null；snapshot1实际为0/6完整端点、0/3完整pair、three_seed_summary=null，没有把训练CSV的val占位0纳入结果。
- 实际seed来自launch CLI/manifest、原生args.yaml、completion及train/eval receipt交叉核对，不被基础protocol_config的seed42覆盖；检查run对应arm与固定seed42教师/reference路径。
- 接受的端点必须为完整E200、该run实际存在的last.pt、fixed_budget_last_ema、val、0–1单位、未访问official test；指标快照与外部endpoint JSON逐字典一致。缺证据、错误arm/seed、非200轮、best端点与不匹配metric snapshot均不接纳。
- 同seed差值计算为 `100 × (paired − weight0)`，主指标mAP50–95；只有六格全部完成才生成三seed汇总。`statistics.stdev` 使用样本SD（ddof=1）；fixture差值1/2/3pp正确输出2±1pp。各seed方向保留，不用已有baseline或缺失格补位。

无阻止本次描述性汇总使用的问题。本工具检查回执/source snapshot的存在和指标一致性，结合本轮已审、冻结的原始实现使用；它不另行宣称成为canonical accepted analyzer，也不将三seed P/N升级为四臂归因或confirmatory test证据。snapshot1早于新full启动，里面的pending_not_started是当时快照，不是当前队列状态。
