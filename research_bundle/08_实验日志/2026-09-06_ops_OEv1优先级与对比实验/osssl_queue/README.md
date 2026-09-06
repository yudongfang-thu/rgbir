# OS-SSL 队列只读核查与安全停止边界（2026-09-06 23:09–23:10）

> 用户调整研究优先级时，OS-SSL 已完成 4/9 个微调，IR-only seed0 刚自动启动；实际 worker 没有 stop-after-current 标志。建议先停止两个 OS-SSL 调度 shell，再保留原 checkpoint 并停止新启动的 IR-only0，不能将研究收缩记为方法失败。本子任务未在 94 写文件、发信号或修改队列；实际操作回执由主 agent 单独记录。

## 当前状态

| 初始化 | seed0 | seed42 | seed123 |
|---|---|---|---|
| paired | 未启动 | 未启动，GPU0 worker 重试资源申请 | 200 epochs 完成 |
| shuffled | 200 epochs 完成 | 未启动 | 200 epochs 完成 |
| IR-only（代码 sar_only） | 23:04 自动启动，23:09 CSV 已完成3轮 | 200 epochs 完成 | 未启动 |

当前未启动的是4个微调：paired0、paired42、shuffled42、IR-only123。此前22:26的“3/9完成、shuffled0在跑、5个排队”已过时。

23:09 GPU2=OS-SSL IR-only0；GPU4/5/6=OEv1 N42/P0/N123。其他GPU空闲。OS-SSL当前任务加入时占4卡，属于已存在的空闲服务器放宽条款范围，不是本次新启动。

## 已核查的进程边界

| 作用 | PID | PPID | starttime ticks |
|---|---:|---:|---:|
| GPU0 OS-SSL worker | 3894177 | 3894173 | 183404002 |
| GPU2 OS-SSL worker | 3894186 | 3894183 | 183404002 |
| IR-only0 resource guard | 878944 | 3894186 | 188878062 |
| IR-only0 trainer | 878988 | 878944 | 188878157 |

精确命令：

```text
bash /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/osssl_ir_20260906/worker3.sh 0 /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/osssl_ir_20260906/jobs2_gpu0.txt
bash /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/osssl_ir_20260906/worker3.sh 2 /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/osssl_ir_20260906/jobs2_gpu2.txt
```

GPU1 的 screen 3894174 仍存活，但其下只有空闲 bash 3894176，没有执行中的 worker、guard 或训练。旧 GPU3/4/5/6 等 OS-SSL screen 多为已结束后 `exec bash` 的空壳，不等于仍有任务。

IR-only0 当前租约：`osssl-ir-sar_only-s0-f1fb4773cc3d`。租约文件位于：

```text
/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/runs/.project_resource_leases.json
```

## 可安全执行的停止流程建议

1. 实时重读两个 worker 的 `/proc/<pid>/cmdline`、`stat` starttime，与上述精确内容一致才行动。不能使用 `pkill python`、`pkill -f osssl` 或对 screen/session 整体发信号。
2. 仅对两个 worker shell 发 `SIGSTOP`，再核验状态为 T。先阻止其下一次资源申请；不要暂停或杀掉 OEv1 的 worker。
3. 重新枚举两个 worker 的所有后代，并核对 guard job_id、训练 output 路径、GPU UUID；应只有新开的 IR-only0 训练及其 dataloader。GPU0 正处于60秒重试，暂停时仍可能存在一次刚已发出的 guard，必须重新检查，不能只信旧 PID 表。
4. 保留 IR-only0 原始 `weights/last.pt` 和 `best.pt`；停止前确认 checkpoint ZIP/CRC 完整，停止后记录大小、mtime、SHA、CSV 完成轮数。无需复制大权重。若正在写 checkpoint，应等一次有效文件出现后再停。
5. 单独向 IR-only0 trainer 及已核实的 dataloader 后代发 SIGTERM；保持 guard 878944 活着。资源 guard 第773–778行轮询子进程，第779–800行 finally 清理：子训练退出且无存活CUDA后代时，自动释放该租约；若存在CUDA后代，保留租约而非错误释放。
6. 不要先 SIGTERM guard：其无 SIGTERM handler，直接 SIGTERM Python 不保证执行 finally。也不要在 CUDA 子树仍存活时手动删除租约。
7. 待训练子树与租约退出后，结束已暂停的两个 shell（针对精确 PID，TERM 后 CONT，使默认终止信号得以处理），或保留为明确暂停状态并登记。屏幕空壳没有GPU成本；不需要清除历史 run 或队列文本。
8. 复查 nvidia-smi、精确 OS-SSL 进程与租约，确认只移除了OS-SSL GPU2任务；OEv1三个训练 PID应保持不变。记录停止原因为 `USER_REPRIORITIZATION_OEV1`，保留已完成和中断产物，不把部分训练当终态成绩。

## 为什么不能仅改 worker 文件或创建 stop 标志

`worker3.sh` 用外层 `while read` 读取 jobs，内层循环遇到资源退出码2则 sleep60 后重试；没有任何 stop/pause 文件检查，也没有 signal trap。正在运行的 bash 可能已解析完整 while 块，修改磁盘源文件不能可靠影响当前循环。修改 jobs 文件也可能受到已打开 FD/缓冲的影响，不能作为唯一阻断。

本次只读扫描未发现已有 stop/pause/hold 文件或控制逻辑。多个 agent/线程共享队列，动作前必须重复核对进程身份并在项目日志登记新的研究优先级；不要修改全项目 resource guard 策略，以免影响 OEv1 或其他工作。

## 证据产物

- `queue_snapshot.json`：23:09:19 UTC+8 的完整小文件、进程、worker日志尾部、九run进度、screen和GPU快照。
- `stop_safety_snapshot.json`：23:10:05 UTC+8 的进程身份、租约与checkpoint只读检验。
- `remote_source/`：实际 worker/job 列表、pipeline 和 guard 源码原副本，SHA保存在queue快照。
- `read_remote_queue.py` / `fetch_queue.py`、`read_stop_safety.py` / `fetch_stop_safety.py`：仅CPU只读采集与本地保存脚本。

23:10 checkpoint检查：best.pt与last.pt均为10,748,051字节；mtime为23:09:51；SHA256=`74edfc32deaf7910c45346a8e500cc7007ff46b858fdddc22cc15ad481c47179`；ZIP CRC正常。训练仍在推进，实际停止时应重新记录，不能沿用这个hash宣称最终状态。

## 结论边界

暂停后，原计划九个微调将不完整；不能再承诺三seed OS-SSL结论，也不能因为中止说它被证实无效。保留已完成结果作参考，后续只有明确需要作为OEv1的相关比较时才恢复，且先处理同模板初始化与RGB-only对照缺口。
