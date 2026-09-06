# 独立并发启动核验（2026-09-06 23:59:40，94）

> 已独立只读确认：R0与R123的canonical同卡canary均通过；R0已在GPU2与R42共同进行正式训练，已真实更新64次；R123在GPU4完成canary后等待N42训练及独立评估结束，正式训练尚未开始。原P/N与R42训练未中断，当前资源符合用户规则。

## 核验范围

读取94原始status、ownership、canary比较与completion、lease JSON、`/proc`进程身份、`nvidia-smi`及各run进度。未启动任务、未发信号、未修改远端文件；没有调用会刷新lease文件的guard.inspect，仅使用纯内存`_usage`计算当前计提。

原始快照保存在`independent_launch_raw.json`与`independent_launch_raw_v2.json`，第二次时间为UTC15:59:40（北京时间23:59:40）。采集/下载脚本为`independent_read_launch.py`、`fetch_independent_launch.py`。

## Canary与正式训练身份

| 项目 | R0 | R123 |
|---|---|---|
| 物理卡 | GPU2 | GPU4 |
| canonical canary比较 | passed | passed |
| 真实optimizer update | 24 | 24 |
| 与对应P的init/首batch逐张量相等 | 是 | 是 |
| 共同检查batch / 选择对象不同的batch | 30 / 30 | 30 / 30 |
| 两CUDA PID同时观测次数 | 31 | 33 |
| canary期间卡总显存峰值 | 13975MiB | 13975MiB |
| canary期间最少剩余显存 | 10109MiB | 10109MiB |
| 后续状态 | **正式训练epoch1，64真实更新** | **等待N42独立评估** |
| worker PID | 1149860 | 1149862 |

R0正式训练guard PID1154629、CUDA PID1154725，job_id=`oev1_random_parallel_full_s0`，有独立绑定lease；canonical full目录`runs/rgbir_oev1_random_20260906/full_paired_random_s0_attempt1`。因此这里的“已开始”有进程、租约和真实optimizer更新三重证据，非仅status标记。

R123状态为`waiting_primary_N42_evaluation`，canonical full目录不存在；当前无R123 CUDA PID。没有把它记为正式训练在跑。新worker实际GPU4，与早期拟议GPU5不同，以实际代码、ownership与状态为准。

## 原任务继续推进

| 任务 | GPU / CUDA PID | 当前epoch | 真实更新数 |
|---|---|---:|---:|
| R42 | 2 / 975949 | 11 | 3273 |
| N42 | 4 / 1564206 | 175 | 49435 |
| P0 | 5 / 3743565 | 65 | 18539 |
| N123 | 6 / 3966512 | 58 | 16539 |

两次快照之间R42的真实更新从3233增加到3273，其余P/N同样增加。原串行R42 worker PID975898和guard PID975933继续存活，日志读取管道未被本次核验或调度迁移中断。

旧串行队列之后在canonical seed0 canary已经存在时会fail-closed，这属于提前记录的接管安排；当前尚未发生，不应提前宣称已验证旧队列退役。其最终退役原因仍需按原traceback核对。

## 资源合规

- **5个正式训练，4张物理卡**：GPU2两任务，GPU4/5/6各一任务；GPU1/3/7仍完全空闲，满足第4卡使用后至少2空卡条款。
- GPU2当前总used13975MiB、free10109MiB，另外三张我们的卡free≥16429MiB，全部明显大于2048MiB。GPU0是他人进程1070544，未加入。
- 项目实际RSS计提143694MiB（约140.33GiB），预约196608MiB（192GiB），小于用户300GiB和现guard240GiB准入线。数据进程计入树RSS。
- GPU2现有R42和新R0各预约8300MiB，总16600MiB，仍走原guard；全局guard源码未因本次预约修订而改变。
- `nvidia-smi memory.total`为24564MiB，used+free为24084MiB，差额480MiB为保留部分；guard按total算70%线为17194.8MiB。实际headroom应直接使用free，不能用total减used替代。

## 局限

本次证明同卡canary与正式启动可行，并未完成双长训全程峰值或最终吞吐基准。R0当前首轮显存6304MiB，按照既有全程训练经验后续可升至7632MiB，因此预约仍按正式峰值保守计提，不能按首轮占用再降。没有新AP结果，也没有改变OEv1方法、训练预算或选择阈值。
