# C1 持久协调器独立审阅

**结论：CODE_SCOPE_ACCEPTED，无必要代码修复。可由根在实际正式配置、profile 与准入齐备后部署；本审阅未执行 prepare、coordinate、worker、screen 或任何 GPU 任务。**

日期：2026-09-07。审阅者 `/root/review_c1_spec` 未编写或修改 `formal_campaign.py`；作者 `/root/review_matrix_spec` 确认稳定后，审阅其完整脚本、文档、15 项 CPU fixture，并核对冻结 `resource_dispatch.run_job / atomic_acquire / measured_reservation` 的实际接口。没有修改 NEW。

## 关键合同

- **实际启动顺序成立。** 42→0→123 依赖前置训练 job 的真实 dispatcher LAUNCHED 事件；SCREEN_REQUESTED / SCREEN_STARTED / QUEUED 均不能代替。队列门在每次真实资源申请时重查，提前存在的等待 worker 也不能越过前置 seed。
- **阶段顺序成立。** worker 单独调用真实 `run_job(train)`，完整 E200 completion 加对应成功训练资源 profile 通过后，再调用 `run_job(evaluation)`。没有将两段交给会提前排序 eval 的 ordered_jobs；不会因仅有学生 completion 而忽略失败的资源监测结果。
- **评估责任持久。** 每 seed 的独立 screen worker 负责 train→eval，manifest 保留两项 job、路径与状态，coordinator 退出不会丢掉已有 worker。已完成训练且资源回执有效时，明确恢复该 worker 只补 eval，不重训。失败/缺失 eval 回执保持待核对身份，不伪装 DONE。
- **技术暂停范围正确。** paused 只阻止新的训练资源准入；已在运行的其他 E200 没有被 coordinator kill，完成后仍可请求 eval。优先级与暂停的判定点为 campaign admission 锁内的资源准入；已经获准的任务不被事后撤销。资源自身越界仍由原 dispatcher 处理它拥有的违规子树。
- **全局资源路径保持。** campaign 只在内存中包住 `atomic_acquire` 增加顺序/暂停/评估优先限制，再原样调用当前 release 的真实原函数；它没有复制或放宽资源规则，没有新建 lease 池、没有硬编码物理 GPU。实际 source/profile/config 验证和共享全局 lease 锁仍是原 dispatcher。
- **无 AP 早停。** 调度仅读取训练/评估完成与技术资源回执，未读取或判断 AP 数值。训练命令无 max-steps，recipe 要求 E200；E200 后固定 last/EMA 独立评估。三个 seed 已运行的结果不会改变后续 method/lambda。
- **原证据保留。** 仅原子替换派生 state.json；manifest、frozen inputs、事件、attempt日志和结果不覆盖。不完整训练不自动恢复成连续 E200；已发布 eval 而资源回执缺失要求技术复核，不再次推理覆盖。

## 锁与恢复核对

coordinator 与每 seed worker 各有非阻塞 flock 单例；campaign 状态变更和资源优先级同用 admission.lock。该锁内调用原 global guard，没有发现反向获取 campaign/global 锁的环路。创建 detached screen 时持有 bookkeeping 锁，worker 等待它，不把 screen 创建当训练准入。

JSON 完成文件处于写入窗口时，coordinator 保守阻止新训练，不将暂时不可解析的评估判为完成。worker screen 意外消失会保存暂停/待eval身份；不擅自接管同名其它 screen 或重启失败训练。

## 验证范围

本版补充核对资源预约：正式 VRAM 为 `ceil(实际测量峰值) + 256 MiB`；主机 RSS 为 `max(8192, ceil(实际进程树峰值) + 4096) MiB`。这些固定余量写入 campaign manifest。原 bootstrap reservation 仍用于验证当时测量确实在预约内，但不会被直接当作正式任务需求。原 profile 文件和字节副本不变，真实 `dispatch.measured_reservation` 仍核验该 profile 的源码、config、数据及计算路径绑定后才使用新的预约数。

新增反例 CPU fixture 中，bootstrap 上限 16000 MiB、实测显存 1370 MiB 的评估只预约 1626 MiB；实测 RSS 6260 MiB 对应 10356 MiB，含小数峰值也向上取整。资源共享仍需满足原全局 lease 的总量、任务数、70% 和全卡 2 GiB 余量规则。本版没有改变真实 canary 测量记录，也没有重测或启动 GPU。旧 v1 审阅产物保留。

`D:/Anaconda/envs/KGJ_proj/python.exe -m unittest test_formal_campaign -v`：**15/15 通过**，输出在 `cpu_tests.log`。

这些测试使用明确合成配置/profile/dispatcher 和替代锁，只验证控制流和状态合同；不证明 Linux screen/flock/真实 GPU 的实际运行结果。真实共享资源、formal readiness 和 endpoint/analyzer 验收仍由原链路执行，当前审阅不能替代它们。

所审原字节副本与本文件同目录。后续 coordinator 源码若变化需重新核对；本代码接受不是自动启动许可，也不是性能结论。
