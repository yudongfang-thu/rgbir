# selected-only 两批 GPU 回放与新 24-update 入口审阅

**结论：两批固定 raw bundle 的实际 GPU 数值回放通过；新 `compare_24_selected_only.py` 可进入原 lease 下的有界 24 次成功更新诊断。** 尚无该新入口的实际训练轨迹结果，不构成正式训练切换、E200 等价或训练提速结论。旧 v1 / block16 的轨迹失败继续保留，不能说已被两批 raw 回放修复。

2026-09-08，loc_stress。读取 `remote_selected_probe_attempt1` 小回执/源码与新 24-update 入口，并在 CPU 独立复跑比较器真值测试。未启动 GPU/SSH、未加载真实权重/大型轨迹，未计算新 hash、未改原 harness 或候选源码。

## 实际两批 GPU 回放

| batch | base / selected | old 中位数 s | selected-only 中位数 s | old / thin |
|---|---:|---:|---:|---:|
| 0 | 435 / 124 | 1.7456882321 | 0.9269087720 | 1.883344× |
| 1 | 423 / 93 | 1.6196302010 | 0.8445863910 | 1.917661× |

依据：[batch 0 receipt](../remote_selected_probe_attempt1/replay_00_attempt1/receipt.json)、[batch 1 receipt](../remote_selected_probe_attempt1/replay_01_attempt1/receipt.json)。两个均为 `PASS_NUMERIC_EQUIVALENCE_ONLY`，各做普通/完整统计两次检查。全部被比较的浮点张量（含 selected S/T delta、quality、C0/KD loss、student score gradient）`exact=true`；选择/映射/region/background mask、C0 stats/计数和所检查 Torch RNG 状态也相同。S/T/R raw scores 均为 FP16，确认实际走薄路径。未计算的 R/base delta 没有被伪装成测量零，metadata 明确缺失。

每侧原定两次 warmup、七个保留计时，old/new 顺序交替。表中是**原完整 selection＋stats＋raw-score backward** 对 **薄 selection＋minimal stats＋raw-score backward**，薄路径计时不做 full diagnostics；不是单一 pool 算子比较，也不含模型、DataLoader、native loss、模型参数 backward、optimizer/EMA。不能用表中约 1.9× 表述训练/epoch 提速。

本地当前 `selected_only_v1.py` 和 `benchmark_selected_only.py` 与此次收集远端版本直接字节相同（14,574 / 10,911 字节）。两阶段资源 profile 均 COMPLETED、exit 0、monitor_errors 空，GPU 4 的整卡已记录最小空闲分别 11,894 / 11,916 MiB；本任务 NVML 峰值 1,336 / 1,318 MiB，项目已记录 RSS 最大 145,339 / 145,334 MiB。这里是已有采样回执审阅，不是另做全时段资源监控。

## 新 24-update 入口的差异审查

对照了先前已独立审阅的 `performance_candidate/update24/compare_24_updates.py` 与新的 [compare_24_selected_only.py](../performance_candidate/update24_selected_only/compare_24_selected_only.py)。新入口独立复制，核心 compare-tree、optimizer/AMP/EMA/worker 流观测逻辑保留；新增内容为 selected-only factory、selected S/T 比较范围、薄学习覆盖计数、独立统计计时和逐批 loss 文件。

- old worker 用原 `build_classification_selection` 和原 criterion/loss；new worker 必须通过 `make_api(selection_module, classification_module)` 与 `make_criterion_type` 共同绑定 selector＋loss。两边都用私有 `build_trainer` globals 副本、`historical=False`、原 yolo11n 初始化、C1 seed42、B32/workers4、E200 学习率配方和冻结 coefficient `0.09227393550836771`，不是 E20 截短配方。helper/runtime/project 模块实际来源逐项核对。
- 两个 worker 是 fresh sequential Python 子进程，单既有全局 lease；协调器不得初始化 CUDA。上限为 **24 个真实 optimizer.step、最多 96 次 update attempt、最多 384 批**，AMP skip 不计成功 update。旧/新训练输出位于新的数据盘目录；失败不恢复、不覆盖，不调用正式 train/verify run 或 hash/evidence emit。
- 初始学生/EMA/optimizer/scaler/RNG、T/R 状态、每批 GT/元数据/worker RNG、首次图像、逐 attempt scaled/applied gradients、student、optimizer、EMA 和控制状态仍保存比较。每批增加 `loss_####.pt`，记录本批 criterion 的 native/KD/weighted/total/target/off-target loss 及实际 B/coefficient。完整 inventory 必须与 receipt 批数/attempt 数闭合。
- candidate `sanity=True` 时每批 full statistics 仍读取同次 raw、在 no_grad 中执行旧完整统计；**学习 loss 始终 thin，组件/shared-gradient observer 仍读取 learning payload**。完成 receipt 和 CPU compare 入口都硬要求 `thin_learning_batches==batches`、`full_diagnostics_batches==batches`、`fallback_batches==0`。因此不能用 dtype fallback 或全统计替代薄学习图后冒充验证。
- `selection_snapshot` 对 old/new 都只存 selected S/T delta；quality、C0 scalar/stats 与完整 region/base/matched 身份仍存。明确标 `delta_collection_scope=selected_S_T_only` 及三类 uncollected learning delta，不比较伪造零。sanity 日志中的完整统计仍是独立真值，但它不是 learning placeholder。
- source snapshot 从实际入口、实际 candidate source 和实际 reference release 的 `.py` 复制，逐文件 byte identity，并保存 config/模型 stat。比较 old/new snapshot 字节、config 字节及模型 stat；未新增 hash。它不等价于对整个安装环境逐文件做新 byte attestation，固定环境验证和既有 pinned source 证据仍需保留。

## 时间段与接受边界

batch_start 记录总 audit/diagnostics 累计，batch_end 先同步测得 span 和两类增量，再单独保存该批 loss；这次保存不挤入该批 span，仍计入全训练 audit。selection audit callback 在 `api.build` 的完整诊断计时结束后执行；`api.loss` 的完整统计计时与 audit/save 分开，未发现同一区间被同时作为 audit 和 detached diagnostics 扣两次。

保留完整 observed wall、仅扣 audit wall、再扣额外完整统计 wall 三种数值。新臂 sanity 多做一次完整统计，第三种是插桩环境下的辅助估计；同步、缓存、worker prefetch/重叠与落盘都仍改变吞吐，不能解释为生产速度。逐批 span 不含 loader fetch；首次六批为预定 warmup。原 fixed-epoch shared-gradient 检查仍包含在训练路径，没有随性能计时关闭。

`selection_floating` 与最终 trajectory precision 分开：q/C0/selected delta 若在新动态轨迹上产生误差，需要读取该类别原始结果；即使最终 trajectory 指标通过，也不能隐藏它。`numeric_agreement` 对相同 nonfinite bit 模式单独标记，不代表这些梯度可用；实际 applied gradient finite 和 AMP lifecycle 检查仍保留。

## 独立 CPU 验证

[update24_cpu_attempt1.json](update24_cpu_attempt1.json)：19 项全部 PASS，Torch 1.8.0+cu111，CUDA 未初始化。包含固定容差方向、NaN payload / signed-zero 与 finite 状态分开、clone closure/global binding、完整 inventory、缺少 attempt/loss 文件拒绝、选择身份变动拒绝、参数或本批 KD loss 差异不能通过 trajectory，以及 fallback 不能冒充 thin learning。未在本地假装运行真实 trainer。

**PASS_FOR_BOUNDED_24_UPDATE_DIAGNOSTIC** 仅授权根继续检查这一个新候选的实际 24 次成功更新；实际 pinned 环境/显存/轨迹/覆盖计数必须由新 attempt 回执证实。旧失败产物与结论不变。
