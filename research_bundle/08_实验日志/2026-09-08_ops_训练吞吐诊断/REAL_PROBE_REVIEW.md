# 真实 capture / block16 回放独立审阅

结论：真实两批 capture 在限定诊断范围内通过；`replay_00_attempt1` 的冻结数值等价检查失败，`replay_01_attempt1` 未执行。约 **1.81×** 是 batch 0 的 selector＋KD/statistics＋raw-score 梯度回放耗时比，不能作为正式训练提速或 block16 替换准入结论。

审阅日期：2026-09-08。审阅者：独立子代理 loc_stress。仅阅读本地源码、JSON、日志，进行小文件直接字节比较、JSON trace 相等检查及已有中位数的除法；未运行模型、GPU、SSH或新科学实验，未加载权重或 raw tensor bundle，未计算新 hash，未修改候选代码、冻结容差、旧结果或运行任务。既有审阅 `performance_candidate/candidate_review.md` 保留；本报告补充真实执行证据。

## 1. 实际执行与归属

入口为 [run_real_probe.py](remote_real_probe_attempt1/run_real_probe.py)，原始结果根为 [remote_real_probe_attempt1](remote_real_probe_attempt1)，对应远端 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_throughput_20260908/real_probe_attempt1`。

| 实际阶段 | 回执状态 / 进程退出 | 独立审阅范围 |
|---|---|---|
| capture，两批各 32 图 | `COMPLETED_DIAGNOSTIC_CAPTURE_NOT_TRAINING_ADMISSION` / 0 | PASS：有限真实输入、原 selector、独立 KD score 梯度与合成损失反向诊断 |
| batch 0 old / block16 回放 | `FAIL_EQUIVALENCE` / 2 | FAIL：冻结 `atol=1e-6, rtol=1e-5` 未全部通过；可保留观察耗时 |
| batch 1 old / block16 回放 | 无该阶段 receipt | NOT_RUN：队列在 batch 0 退出后停止 |
| 正式训练实现切换 | `optimizer_updates=0`，`formal_training_admitted=false` | 未准入，也未执行 |

`screen.log` 的异常来自 dispatcher 对 `c1_real_bundle_0` 的非零退出；resource profile 的 `monitor_errors=[]`。这不是资源超限或 capture 失败。队列计划三阶段不能写作三阶段已完成。

本地当前候选、收集到的远端候选和 capture 开始时的源码快照，三个版本的 `capture_real_batch.py`、`benchmark_candidate.py`、`pool_block16.py` 均直接字节相同，分别为 14,434 / 15,123 / 2,289 字节；本地与收集入口 `run_real_probe.py` 亦字节相同。此为小文件比较，不是新 hash。收集清单声明只收集小文件、`raw_tensor_bundles_downloaded=false`；本次审阅不声称独立重算了远端 raw bundle。

## 2. Capture 的有效性与限制

依据：[capture receipt](remote_real_probe_attempt1/capture_attempt1/receipt.json)、[执行源码快照](remote_real_probe_attempt1/capture_attempt1/sources/00_capture_real_batch.py)、[input config](remote_real_probe_attempt1/capture_attempt1/sources/input_config.yaml)。

| 批 | common / base / selected | 独立 KD raw-score 梯度 L2 | 学生有限梯度张量数 | 串行诊断管线秒数（排除独立 KD probe） |
|---|---:|---:|---:|---:|
| 0 | 451 / 435 / 124 | 0.00764485635 | 255 | 10.9708926382 |
| 1 | 429 / 423 / 93 | 0.00755168265 | 255 | 2.2475234880 |

两批各 32 图，`trace_00.json` / `trace_01.json` 与 `../2026-09-07_train_IndependentKD实施/remote_cpu1/coverage_drone_attempt1/natural_batches.jsonl` 前两条完整 JSON 相同，包含输入路径、配对/增强元数据、GT 类/框和形状。此检查不是图像像素字节比较。数据流为固定 calibration/probe seed 20260907 的普通 DataLoader，不是正式 trainer 的 sampler/worker 流。

学生使用已完成 `full_weight0_s42_attempt1/weights/last.pt`，教师和 reference 分别为既有 DroneVehicle IR42 / RGB42 native endpoint；不是在训 C1 学生快照。学生为 private train mode，T/R 冻结且 eval，模型前向使用配置 AMP。输入权重前后 stat 相同支持未改写输入文件；不把 stat 相同当作内容 hash 校验。

修复后的代码单独计算 `autograd.grad(kd, student_scores, retain_graph=True)` 并要求有限、非零；上表为实际通过值。因此可以确认本次 KD 诊断在学生 raw-score 上有有效梯度，而不仅是 native loss 使总反向非零。随后执行 `native.sum()+B*lambda*KD` 的 unscaled backward，学生参数梯度存在且有限，T/R 参数未产生梯度。

这些事实不证明 KD 单独到所有共享参数的梯度、optimizer/scaler/EMA 更新有效性、连续 24 update 等价或完整训练等价。该 capture 没有 optimizer/scaler/EMA update，也没有运行 block16。学生 train mode 的私有 BN 状态可随两个前向改变；不能把两批描述为同一个完全不变模型状态上的重复采样。

阶段计时使用 CUDA synchronize，破坏常规流水重叠；第一批含明显初始化开销，两批样本不足以估计稳定训练吞吐。独立 KD probe 已从管线时间中扣除；另做的 pool replay 不属于该管线，不能再次加到训练分母。嵌套 pool 计时使用 `track_peak=False`，不再重置外层 peak 计数；各 stage 的 reserved/allocated 仍有缓存与已有存活张量，不能解释成该 stage 的独占增量显存。

## 3. 冻结数值检查：保留失败，不外推方法失效

依据：[replay 00 receipt](remote_real_probe_attempt1/replay_00_attempt1/receipt.json)、[benchmark 快照](remote_real_probe_attempt1/capture_attempt1/sources/01_benchmark_candidate.py)。

| 比较项 | exact | 固定容差 | 最大绝对差 |
|---|---|---|---:|
| student delta，4,350 元素 | 否 | PASS | 1.9073486328125e-6 |
| teacher delta，4,350 元素 | 否 | **FAIL** | 1.9073486328125e-6 |
| reference delta，4,350 元素 | 否 | PASS | 1.9073486328125e-6 |
| valid levels / selected / labels / eligible / 索引映射 | 是 | PASS | — |
| quality，435 元素 | 是 | PASS | 0 |
| C0 loss / C1 KD loss | 是 | PASS | 0 |
| student score gradient，1,344,000 元素 | 是 | PASS | 0 |
| student box、T/R score/box gradient | 两边均 None | PASS | — |
| matched/base object IDs、C0 stats、selected count | 是 | PASS | — |

不能写作“除了 teacher delta，其他所有浮点量都 exact”；S/R delta 也不同。`allclose` 逐元素使用绝对与相对容差，三个张量最大绝对差相同但 pass 不同并不矛盾。旧 receipt 未提供失败元素数、具体对象/类/层、selected 子集归属或局部相对误差，所以本报告不把失败进一步归因为某个对象、未选对象或已证实的舍入机制。

约 1.9e-6 的差异与重新分组浮点归约相容，但该解释需要实现者正在做的差异定位支持。失败本身不足以证明科学方法无效、选择门已经变化、KD 有害或增益不存在。应保留固定检查的 FAIL；也不能因数值很小而事后放宽容差、改批或直接写“等价通过”。

候选只替换原 selector 函数 globals 中的 pooling 实现，沿用原 GT 匹配、reference/teacher 条件、q、rho、稳定排序和 base 分母。代码未改变门或分母；实际 batch 0 的对象/选择/质量也 exact。该证据仍不足以担保全部后续批次。

回放读取相同 raw scores/DFL，使用相同克隆与梯度输入；T/R 也设为可求梯度叶子来检测泄漏，两边都没有通路。shape-only feature placeholder 不影响本 C1 所读的布局信息，但不是特征蒸馏数据。Torch CPU 与当前 CUDA RNG 在比较前后相同；这里没有 Python/NumPy/worker RNG 的全状态等价断言。capture 与回放传入的 selection_seed 不同，但当前 paired `_choose` 不使用随机分支，不构成本次 old/new 比较的不公平来源。

## 4. 性能结论只能到回放层

该批 old/block16 均预热 2 次、计时 7 次，顺序交替，同一设备、同一 raw 输入、同一 selector/KD/statistics/score 梯度路径；有同步计时，比较口径合理。原始计时保留在失败 receipt，因为脚本先采样，随后按数值检查结果写 FAIL 并退出 2。

- old 中位数：**1.6607235752 s**；block16：**0.9176034031 s**。
- 两个中位数之比：**1.8098489713×**；对应该回放耗时降低 **44.7468%**。
- 这条路径不含 DataLoader、S/T/R model forward、native loss、模型参数 backward、optimizer 或验证/checkpoint，且并非正式 trainer AMP 全路径。
- 既有 synthetic 的约 7.47× / 11.66× 仅为两个预定形状的 pool forward＋raw-score backward；不能替代本次真实回放，更不能写成“训练加速 7–12 倍”。

只可表述“block16 在这一真实 raw batch 的失败回放中表现出缩短局部耗时的潜力”。不计算或报告正式训练倍数、每 epoch 预计时间、GPU 迁移后收益或科学 AP/KD 收益。capture 原实现的串行管线时间也不能与回放 block16 中位数直接相除。

## 5. 资源与清理

两阶段通过同一 dispatcher / lease 路径串行调度，使用 GPU 4，各只有一个本任务 CUDA PID；资源 admission 的项目活动卡为 `[2,4,5]`，未援引四卡放宽条款。8192 / 4096 MiB 是初始诊断显存预约，不是假称已经实测的峰值。

| 阶段 | 本任务 NVML 显存峰值 MiB | 本任务 RSS 峰值 MiB | 已记录整卡最小空闲 MiB | 已记录项目总 RSS 最大 MiB |
|---|---:|---:|---:|---:|
| capture | 5,546 | 7,577 | 8,246 | 151,340 |
| replay 00 | 1,322 | 1,344 | 12,472 | 145,270 |

依据为两个 `*_resource_profile.json`（各 8 个样本）、`*_status.json`、`*_admission.json` 和阶段 receipt。已有样本符合显存余量与项目内存限制，`monitor_errors=[]`；不能把有限采样表述为审阅者另做了全时段监控。capture 的 `finally` 对实际普通 DataLoader iterator 调用 worker shutdown；dispatcher 等待子启动器退出并记录 0/2，失败终止剩余队列。当前小产物没有独立的“退出后全部 PID/lease 均不存在”快照，所以本报告不另声称独立完成了事后全局进程清场验证。

## 最终范围

**PASS：真实两批原路径 capture 的来源、非零独立 KD score 梯度和限定反向诊断；FAIL：block16 的第一批冻结数值等价；NOT_RUN：第二批比较；未准入正式训练切换。** 实现者可继续解释已存在差异，同时保留此次全部失败产物及固定协议。本报告不提出重开科学方法、改变门/分母或因 tiny rounding 重新选择批次。
