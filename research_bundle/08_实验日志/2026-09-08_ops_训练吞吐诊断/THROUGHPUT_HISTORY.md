# 既有训练吞吐历史核对（2026-09-08）

**旧 C0 的真实 E200 每轮中位数为 200.8–216.9 秒，完整训练约 11.16–12.74 小时；C1 在 01:12 历史快照的近期中位数为 1112.4–1160.0 秒/轮，约慢 5.25–5.54 倍。这个时间差已经可追溯，但不能仅凭跨时段记录归因于 C1 算法或同 GPU 并发。**

本核对只读本地原 CSV、completion、canary、lease 和 monitoring 快照，没有运行服务器/GPU/新训练、没有改当前计划或解释方法 AP、没有计算新 hash。根任务另做当前服务器 profiling；下文 01:12 数值是历史观测，不是实时 ETA。

## 完整旧 N/C0/random 的真实耗时

取 2026-09-07 16:53 完整小产物快照的 `results.csv`，只抽取 epoch/time。time 为训练累计秒；每轮耗时为相邻 time 差，主中位数排除第 1 轮，另列后 20 轮。completion 总耗时比 CSV 末行多约 10–15 秒，分别保留；均不把独立终点评估或排队加进训练时间。

这些九个端点均已完成 200 轮、112600 batch，即每轮 563 batch，optimizer successful updates 为 56694–56697。recipe 为 Drone、B32、640、workers4、AMP、SGD；表中 GPU 来自实际 completion 的 physical gpu_ids，而非目录名。

| 方法 | seed | GPU | ep2–200中位秒 | 最后20轮中位秒 | CSV累计小时 | completion累计小时 |
|---|---:|---:|---:|---:|---:|---:|
| N/weight0 | 0 | 5 | 217.20 | 214.70 | 12.1146 | 12.1181 |
| N/weight0 | 42 | 4 | 204.64 | 201.05 | 11.5041 | 11.5078 |
| N/weight0 | 123 | 6 | 212.00 | 212.00 | 11.9629 | 11.9663 |
| C0/paired | 0 | 5 | 216.90 | 216.85 | 12.3452 | 12.3494 |
| C0/paired | 42 | 4 | 200.80 | 201.00 | 11.1562 | 11.1590 |
| C0/paired | 123 | 6 | 214.80 | 220.00 | 12.7322 | 12.7358 |
| random | 0 | 2 | 251.16 | 250.45 | 13.8845 | 13.8884 |
| random | 42 | 2 | 251.85 | 259.90 | 13.9696 | 13.9729 |
| random | 123 | 4 | 229.70 | 231.80 | 12.5785 | 12.5823 |

原快照的 `C0` 目录表示 C 的 seed0，`C42/C123` 表示另两个 seed；本表统一写方法 C0，避免把 `C0` 目录误当全部方法。

N/C0 同 seed 的 physical GPU 恰好相同，但运行时段与外部负载不同，不能据此称严格同负载性能 benchmark。random42 与 random0 在 GPU2 的正式双开有 9月6日23:59 快照/准入实证；9月7日03:20 lease 又明确 GPU4 同时为 random123 与旧 shuffled42。random 与 C0 的差值同样混有共享资源影响，不能全部归因选择随机化。

直接来源：

- `../2026-09-07_train_IndependentKD实施/snapshots/2026-09-07T165357.033246_0800/raw/{N0,N42,N123,C0,C42,C123,R0,R42,R123}/{results.csv,completion_receipt.json}`。
- `../2026-09-06_ops_单卡并发与计划澄清/parallel_evidence/parallel_live_snapshot.json`、同目录实验 README。
- `../2026-09-07_audit_RGBIR实施起点/snapshots/2026-09-07T032011.072078_0800/resource_leases_raw.json`。

## C1 和当前内容对照的历史速度、同卡关系

C1 最初第 2 轮已分别用 1074.27/1108.51/1081.38 秒（seed42/0/123）；当时等速外推 E200 为 59.68/61.58/60.08 小时。这个慢速不是到最后一个快照才出现。

01:12:33 原状态如下；“已完成”取 throughput.completed_epochs，不能用 progress 正在进入的 epoch 冒充完成轮数。

| 任务 | 正进入/已完成epoch | 近期秒/轮 | 近期分钟/轮 | 该快照等速剩余训练小时 | 本项目同GPU任务 |
|---|---:|---:|---:|---:|---|
| C1 seed42 | 32 / 31 | 1112.4 | 18.54 | 52.221 | GPU2：C1 seed0 |
| C1 seed0 | 31 / 30 | 1160.0 | 19.33 | 54.778 | GPU2：C1 seed42 |
| C1 seed123 | 32 / 31 | 1126.7 | 18.78 | 52.892 | GPU5：旧 same-modal42 |
| 旧 shuffled42 | 76 / 75 | 1084.0 | 18.07 | 37.639 | GPU4：当时无第二个本项目训练 |
| 旧 same-modal42 | 71 / 70 | 998.9 | 16.65 | 36.071 | GPU5：C1 seed123 |

目录 `formal_C1_gpu5_attempt2` 不是实际 GPU 位置：三 seed 实际为 GPU2/2/5。当前调度器动态择卡，必须看 lease。

GPU4 当时只有一个**本项目** shuffled 任务，但整卡仍有他人占用，不能当“空卡独占”基准。五个本项目训练同时共享主机 CPU/内存/存储，workers4 每任务、数据增强/几何跟踪/选择实现也和早期训练入口不完全相同。旧内容对照本身也达到约 17–18 分钟/轮，说明应同时检查执行路径与主机共享资源；本地历史证据不能拆分各原因的因果份额。

同一 16:53 快照内旧 shuffled/same-modal 尚未完成，分别只到 CSV 57/40 轮；其最后20轮中位数 560.85/628.20 秒。之后 01:12 近期值升到 1084.0/998.9 秒，进一步说明时段负载/运行条件可变，不能给这些 partial 运行套用固定全程速度。

直接来源：`../2026-09-07_train_IndependentKD实施/initial_throughput_20260907_162119.json`、`running_state_20260908_011232.json`、`heartbeat_20260908_0112_runtime/summary.json`。历史快照仅提示当时趋势，剩余小时未包含终点评估、排队和未来负载。

根任务 01:57 的只读实时采样另见本目录 `host_profile_20260908_015752.json`：112逻辑CPU，load约35；GPU2/5整卡平均util约96.2%/94.9%、功耗109.7/120.6W，两张卡上的各项目进程SM利用率约46–48%。GPU4有约3.8GiB他人占用，shuffled进程SM均值33.65%、整卡54.1%。这支持“共享GPU确实分摊执行机会”，却不能把全部约5倍历史减速归给并发；整卡高util也不等同单个任务充分利用硬件。八卡当时均有占用，不能援引剩余两张空卡的四卡例外。具体内核/逐对象同步热点与block16全类别pool候选由根任务/另一执行者单独验证，本报告未将候选实现当作已实现的加速。

## Canary 能说明什么

以下 canary 都是 30 batch、24 successful optimizer updates、6次 AMP skip。总秒包括短测初始化/首批等开销，不是纯 GPU kernel 时间，不能乘以 E200 当可靠 ETA。

| 短测 | physical GPU | 总秒 | 秒/batch |
|---|---:|---:|---:|
| 旧 C0 seed42 | 4 | 51.148 | 1.705 |
| 旧 C0 seed0 | 5 | 30.881 | 1.029 |
| 旧 C0 seed123 | 6 | 32.861 | 1.095 |
| 旧 N seed42 | 4 | 29.622 | .987 |
| 旧 random seed42 | 2 | 30.615 | 1.020 |
| C1 canary | 2 | 71.466 | 2.382 |
| C1 companion canary | 5 | 75.626 | 2.521 |
| C1_y canary | 2 | 73.300 | 2.443 |

这些短测证明真实更新/显存准入等技术条件；不同 seed、时段、GPU及首批开销让比值不能当稳定吞吐增量，更不是学习有效性验证。`c1_profile2_attempt{1,2}` 为 PROFILED、0 optimizer updates，分别24.395/23.919秒；本报告保留其原值但不把它们当训练 canary。

来源：旧短测在 `../2026-09-06_ops_单卡并发与计划澄清/concurrency/raw/RGBT_campaign/runs/`；新短测在 `../2026-09-07_train_IndependentKD实施/remote_admission_1532/`；每条完整路径见 `history_evidence_v2/canary_times.csv`。

## 对验证延迟与短 recipe 的判断

三 seed 从一开始进入 E200，优点是完成后能直接获得稳定性证据；代价是三项长任务同时占用资源，而第一条正式结论仍须等待某一 seed 完整200轮及固定last/EMA独立评估。当前协调器明确无中途AP判断、训练后才评估，所以正确性 canary 通过并没有缩短学习假设反馈周期。约18.5–19.3分钟/轮的运行成本已在最初两轮出现，E200级反馈自然接近60小时；01:12只完成30–31轮且无独立端点。这是验证延迟的直接流程原因，不能把等待仅解释为“多seed统计要求”。三seed并发不等于三倍完成时间，也不保证三倍总体吞吐。

若用户另行授权短 recipe，必须单独冻结匹配比较；本报告没有据此改变已有训练：

1. 判断 C1 相对不蒸馏的净效应，至少需要同短 recipe 的 N/weight0 与 C1；判断内容是否比旧 C0 更好，还需同短 recipe C0。不能拿短 C1 对旧完整 E200 N/C0。
2. 新短 schedule 与 E200 中途 checkpoint 是两种干预。epochs改变会改变LR衰减的时间尺度，warmup占比也不同；需明确究竟保留 E200 前缀、还是从头定义短 schedule，并在所有比较臂一致。
3. 对齐初始参数、head/class remap、train/dev、B32/nbs/梯度累计、样本顺序与增强、workers、优化器/LR/warmup/AMP、教师/R身份、成功更新或样本预算、固定endpoint/EMA和同一独立评估器。教师错误归因还需同短 recipe 的 shuffled/same-modal；定位需同mask/同预算的GT控制与既定几何合同。已有 E200对照只能作历史背景。
4. 单seed短程只能快速判断工程和条件效应方向，不能替代三seed增益及四臂因果归因；短程排序也可能不代表E200排序。预先冻结验证与扩展规则，不能看一次dev结果再挑epoch/阈值。

即使缩短为10轮，在上述历史速度下单个C1仍约3.1–3.2小时；这只是线性成本说明，不是新计划或承诺。先厘清约5倍减速的具体执行原因，是根任务当前profiling要解决的问题。

## 本次小产物与局限

`history_evidence_v2/summary.json` 保留原输入 path/size/mtime、completion身份、GPU和统计；`completed_and_partial.csv`、`epoch_times.csv`、`canary_times.csv`、`running_throughput_snapshots.csv` 提供复核小表；入口 `analyze_throughput_history.py` 与实际源码副本 `history_evidence_v2/runner_source.py`。统计没有改旧CSV或原始结果。

首次读取 CSV 时错误地对一个未使用指标空格调用 strip，尚未产生耗时统计；失败源码与说明保留在 `history_evidence/`。修订只读取要求的 epoch/time 两列，并写独立 v2 目录；没有填补或排除任何已记录epoch/time行。没有新 hash，没有方法AP结论，没有修改公共索引或现行排程。
