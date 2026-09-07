# 独立 KD 共用资源调度器实现

后续修订：首次 bootstrap 允许在显存余量足够时与其他用户共享 GPU；当前规则和 25 项测试见 `RESOURCE_DISPATCH_SHARED_GPU_REVISION.md`。下文保留初版“完全空卡”限制的历史记录，该额外限制已取消；第四卡例外仍须另外两张真实空卡。

结论：新 `resource_dispatch.py` 已实现并通过 24 项 CPU 检查，可由统一执行者部署。没有启动任何 GPU 工作。新校准/兼容/AMP canary 的首次短测仅能进入无任何 GPU 进程、无项目预约的空卡；资源不足每 30 秒重新检查。

## 文件与真实依赖

- 工程文件：`experiments/rgbir_independent_kd_v2/resource_dispatch.py`。
- CPU 测试：`test_resource_dispatch.py`。
- 未修改原 `task_conditional_reference` 或项目 `tools/project_resource_guard.py`。
- 2026-09-07 只读 SSH 查看了 94 上真实 guard 的常量、`acquire`、`bind` 与共用锁代码。服务器实际 `MAX_ACTIVE_GPUS=4`，单 GPU 项目 CUDA 进程上限为 2，显存预约严格小于 70%，RSS 预约阈值为 240 GiB。原 acquire 没有核查第四卡之后的两张空卡条件。
- 新调度器运行时从指定项目根加载真正的 guard，并使用其 `DEFAULT_LEASE_FILE`。每次新调度目录保存这份真实 guard 源码副本，不生成另一套租约文件或 schema。

## 原子准入

先进入原 guard 的 `_locked_state`，在同一锁内刷新所有项目 lease、采集 GPU 显存和全部 GPU 进程、计算动态候选，再通过 `_AlreadyLockedView` 调用原 guard 的 `acquire`。视图仅提供已持有的同一 state；实际租约仍由原 acquire 创建。并发的旧/新项目启动器无法在选卡与预约之间插入另一份 lease。

全进程采样来自一次 `nvidia-smi -q -x` 的 processes 节点，同时包括 compute 和 graphics；不把小显存图形进程当成空卡。空卡还要求没有项目 active/pending lease，且显存已用少于 100 MiB。第四张项目 GPU 只有在这次锁内采样显示占用后仍剩至少两张完全空卡时才允许，并记录前后列表与规则依据。若安装的 guard 本身最多支持三卡，则保留三卡上限。

GPU 准入仍检查项目实际占用、未兑现预约、最多两个项目任务、项目总显存严格小于 70%、整卡留出 2 GiB 余量。主机准入沿用 240 GiB 的有效预约阈值；实际 RSS 硬界使用 300,000,000,000 字节，满足工作区要求的 300 GB。

CPU 并发测试用了真实项目 guard 源码，仅替换 OS 文件锁和进程/GPU采样夹具，确认两个同时申请首次短测的线程不会取得同一空卡。测试锁没有操作服务器或 CUDA。

## 首次短测与 profile 身份

- `bootstrap_profile=true` 仅允许 `calibration`、`canary`、`compatibility` 短测；不允许正式训练。
- 首次短测必须独占一张当前无进程、无预约的卡，并预约和限制为 16,000 MiB；RSS 由实际 manifest 指定（本轮 32,768 MiB）。
- `calibration_profile_only=true` 要求实际回执是 `PROFILED`、两批、零 optimizer update。它只生成资源测量，不能当作 64 批校准完成或 lambda 接受。
- 正式 64 批校准要求实际回执 `CALIBRATED` 且 `total_batches=64`。canary 要求 `canary_completed` 和至少 24 次 optimizer update。兼容轨迹要求 `ACCEPTED`、`trajectory_exact`、至少 24 次 update 和 30 批 loader 暴露。
- profile 复用同时匹配 `profile_key`、计算路径和真实 execution binding，包括有效配置、实际加载源码和小输入内容。配置支持 YAML 和 JSON；允许计划中的未定系数 null。
- 校准峰值与 AMP 训练峰值分开。C1 的两批校准 profile 可支持同路径的后续 64 批资源预约，不能借给新 C1 AMP canary。新 canary 首次仍须 bootstrap；成功 canary 的同路径峰值可用于正式训练。
- 本实现没有为了省一次短测而给新 wrapper 自动借用旧 OEv1 峰值。N/C0 新 wrapper 也可直接用受限空卡短测，保留旧路径和新路径身份。

## 执行、监控与证据

先把 lease 绑定到活着的 dispatcher，再启动子进程，使子进程从出生开始就在父进程树的租约内。只通过 `CUDA_VISIBLE_DEVICES` 和原 guard 环境变量传入动态 GPU，不在实验配置中硬编码卡号。

运行时约每秒检查共享项目 RSS、GPU 项目占用、全卡空闲显存、任务数和本任务峰值；超过预约或硬界时，只终止本次 Popen 子进程及其后代。不会终止其他 lease/用户的进程。若结束后仍有该 lease 下 CUDA 后代，保留 lease 并记录，避免把它隐去。

日志和失败 artifact 不覆盖。每个新队列目录保存 manifest、真实 guard 副本；每个 job 保存追加式事件、日志、原子准入依据和资源 profile，失败另存 failure JSON。原始训练/校准/评估回执保持在命令对应的输出目录。

轮询监控不等于对其他用户瞬时分配的全时刻硬保证；准入采样与项目预约是原子的，但不控制非项目用户的后续行为。共享负载变化由运行监控发现后终止本任务并保留证据。

## 最小 manifest 示例

以下路径为待执行者替换的明确占位示例，不是已执行结果：

```json
{
  "jobs": [{
    "id": "drone_c1_profile2_attempt1",
    "kind": "other",
    "stage": "calibration",
    "calibration_profile_only": true,
    "bootstrap_profile": true,
    "formal": false,
    "profile_key": "drone-C1-calibration-b32-w4",
    "vram_mib": 16000,
    "rss_mib": 32768,
    "config": "/PROJECT/CONFIG/drone_C1.yaml",
    "result_receipt": "/PROJECT/RUNS/profile2_attempt1/calibration_receipt.json",
    "command": [
      "/PROJECT/ENV/bin/python",
      "/PROJECT/V2/calibrate_independent.py",
      "--config", "/PROJECT/CONFIG/drone_C1.yaml",
      "--output", "/PROJECT/RUNS/profile2_attempt1",
      "--profile-only"
    ]
  }]
}
```

运行入口：

```text
python /PROJECT/V2/resource_dispatch.py --manifest /PROJECT/manifest.json --output /PROJECT/新队列attempt目录
```

真实长命令由主执行者放在 screen/tmux，输出只位于项目 runs/artifacts 或数据盘。后续 64 批校准保持同 `profile_key`，去掉 `calibration_profile_only` 与 bootstrap，增加 `requires_profile` 指向上述 dispatcher 实测资源 JSON，并将预约设置为不少于已测峰值。C1 canary 使用独立的 training profile_key 和 bootstrap。

manifest 中独立评价优先于新训练，顺序稳定。外部队列责任的 job 若标注 `queue_owner` 为其他队列会拒绝；本轮不生成/重复旧 random seed42 评价任务。根执行者已报告旧 random 三 seed 评价完成并按预期退役，本文件不另行认领这些任务。

## 验证范围

2026-09-07 在本地 Python 3.8 CPU 运行 `python -m unittest test_resource_dispatch -v`：**24/24 通过**。包括实际 guard schema/API 复用、并发空卡选择、第四卡及第五卡限制、pending lease、图形进程、双开/三开、70% 显存、2 GiB 余量、240 GiB 预约、300 GB RSS、profile路径隔离、YAML/null配置、两批PROFILED不冒充校准、正式训练实际完成状态等。

另由非作者 `/root/review_matrix_spec` 有界审查原子视图、bootstrap和停止边界，未发现绕过 guard/误杀问题；指出的 `training_completed` 状态与 YAML 读取问题已修复并加入测试。新 evaluator 的完成状态由 evaluator 作者补齐。本文件不替代实际 GPU canary、峰值测量或正式 readiness 接受。
