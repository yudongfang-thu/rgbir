# 单卡并发只读核查（2026-09-06 23:46，94）

> 用户规则允许每卡最多3任务；当前单卡单任务来自旧队列和较保守的资源预约，不是用户禁止多开。对现配方，双任务显存有余量，三任务按已经观测到的正式训练峰值将不足2GiB余量，不能直接三开。

## 已核查的当前状态

| 任务 | GPU | CUDA PID | guard PID | CSV已完成 / 当前epoch | guard记录峰值VRAM MiB | guard记录峰值RSS MiB | 近5轮秒/轮 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random42 | 2 | 975949 | 975933 | 6 / 7 | 7630 | 28825 | 203.752 |
| N42 | 4 | 1564206 | 1564095 | 170 / 171 | 7632 | 28898 | 201.280 |
| P0 | 5 | 3743565 | 3743513 | 60 / 61 | 7632 | 28809 | 217.260 |
| N123 | 6 | 3966512 | 3966414 | 53 / 54 | 7632 | 28800 | 212.080 |

GPU1/3/7完全空闲；GPU0由其他进程1070544占用约19.5GiB，不应加入。当前已占4卡，仍3空卡，符合用户最新§2.1放宽条件。GPU2/4/5/6物理空闲显存约16.43GiB，各1个CUDA PID。

项目lease覆盖进程树RSS约115179MiB（112.48GiB，按RSS计提共享页）；预约196608MiB（192GiB）；主机available约948300MiB。不要将主机used53GiB与项目树RSS112GiB当成矛盾，RSS可重复计提共享页。

## 为什么现在没有双开

真实guard源`SpaceNet6_OTD_official_reproduction/tools/project_resource_guard.py`仍有：

- `MAX_CUDA_PROCESSES_PER_GPU=2`，低于用户允许的3，但不阻碍双开。
- `PROJECT_VRAM_FRACTION=0.70`：后续完整query核实`memory.total=24564MiB`，实际guard准入线17194.8MiB；此前used+free=24084MiB不包括480MiB保留区，不能将其当成total。
- 每个现有长训预约10,000MiB；两个预约合计20,000MiB，因此被70%线拒绝。
- `HOST_RSS_ADMISSION_MIB=240*1024`：4个任务各预约48GiB，第5个相同任务将恰好240GiB，被`>=`条件拒绝。实际每任务峰值约28.2GiB。
- 正式同卡第二训练需要显式`--profiled-second-train`；现worker没有传这个参数。

guard存在acquire/bind/release/inspect，没有安全的既有lease资源预约修订API。`inspect`会写lease刷新峰值，本次为严格只读，直接读原JSON后调用无写入的`_usage`，未调用inspect或acquire。

随机对照队列源`rgbir_oev1_random_20260906/random_worker.py`硬编码GPU2与seed42→0→123。驻留worker PID975898。没有命令行seed选择、停排flag、跳过已接管seed机制；在磁盘修改原worker不会改变其已载入的代码。停止worker还可能破坏其管道中现行42日志读取，因此不应直接杀worker去接管在跑训练。

## 显存判断

短canary仅6304MiB，正式训练已升至7632MiB。必须采用正式训练峰值作为资源判断基准，不能压低到canary值。

- 双任务：约2×7632=15264MiB，含少量驱动开销后仍剩约8.8GiB，具有可测试余量。
- 三任务：约3×7632=22896MiB，24,084MiB卡只剩约1.16GiB，低于2GiB规则。当前配方不建议3开。
- 单次利用率14%或47%不能证明并行吞吐收益；应先同卡双任务canary，比较稳定进度、显存、主机RSS和两个任务合计更新速度。

## 最小安全加速建议（本分工未执行）

优先只在已经使用的GPU2/5/6上补第二个训练，不触碰P/N训练、不增加物理卡数量，不在GPU4即将完成的N42上抢资源。random0/123优先于新方法变体。

两条可审计实施路线：

1. **保留旧guard阈值，新增锁内资源预约修订API。** 核验lease_id、owner/job PID/cmdline、仍在训练、正式峰值证据；仅调整期望预约，不释放/重绑lease，不改历史admission；记录原值、新值、证据来源和修订理由。例如正式峰值7632MiB、RSS≤28898MiB已知时，VRAM每任务8300MiB、RSS32768MiB是可评估的预约候选，2×8300=16600低于旧70%线，6×32GiB=192GiB。预约不是强制上限，仍须监控物理余量，短canary若超过候选就不放行。禁止裸手改lease JSON或为腾预约先release仍在训练的lease。
2. **将guard政策显式更新至用户允许范围内，同时保留保守空间。** 例如90%显存线、2GiB物理余量、双训练profile、主机准入280GiB且硬上限300GiB；每个新任务仍10,000MiB VRAM、32GiB RSS。两个10000MiB预约低于90%线且留约4GiB。必须版本化旧源码、更新policy元数据、验证边界和跨launcher共享同lease文件；不能启动绕过原guard的旁路。是否采用由root决定，本只读分工不修改。

队列接管也需显式解决：新建支持`--seed/--gpu`、共享原独占输出和同一guard的worker；先确认random0/123任何canary/full均未开始。旧驻留worker在seed42训练和评估完成后会进入seed0；它在canary输出存在时`assert not out.exists()`立即失败，从而不会重复GPU训练。若选择利用这一既有fail-closed行为作为接管保障，必须提前写seed迁移回执和预期旧队列终止原因，后续明确记录为调度接管，不记为seed42训练失败；新worker须先完整创建并锁定同一canonical输出。更整洁的方案是在旧42完成后再自然退役队列，但无法立刻把0/123并行提前。不能仅修改旧脚本文件并声称驻留队列已支持停排。

## 证据

- `remote_snapshot.json`（23:45）和`remote_snapshot_v2.json`（23:46）；旧快照保留。
- `raw/`：精确当前guard、lease JSON、三个OEv1 campaign worker/queue/canary/进度/receipt小文件副本，来源与SHA256在snapshot的files字段。
- `read_remote.py`、`fetch_remote.py`、`summarize_snapshot.py`：只读采集与本地解析脚本。

本分工未启动GPU任务、未发信号、未修改94文件、未申请/释放/更改租约。
