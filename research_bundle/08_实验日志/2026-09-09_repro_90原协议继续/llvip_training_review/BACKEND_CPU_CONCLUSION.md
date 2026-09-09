# LLVIP baseline 后端 CPU 收口

**仅设置 `cudnn.benchmark=False` 的下一次 GPU 尝试不应启动：90 当前原训练路径本来就将它设为 False。现有24更新目标未完成，不能通过无差别重跑升级资源准入。**

2026-09-09 13:04:14，在90实际 `environments/cft90/bin/python`、Torch `2.10.0+cu128`、实际 `external_reproductions/llvip_author_baseline/author_source/yolov5` 中，仅导入原train并调用原seed函数。检查进程隐藏CUDA、关闭可选wandb，不构造模型、不读图片、不训练，也未在seed之后改后端。结果见 [backend_cpu_check.json](backend_cpu_check.json)：

| 检查项 | 实测 |
|---|---|
| train.RANK / LOCAL_RANK / WORLD_SIZE | -1 / -1 / 1 |
| 原调用 | `init_seeds(1 + RANK)` |
| seed 实参 | 0 |
| cudnn.benchmark / deterministic | False / True |
| CUDA initialized，调用前 / 后 | False / False |

这次CPU检查证明当前源和环境的初始化结果；旧训练回执没有逐批记录此开关，因此不把今天的CPU打印伪装成旧GPU进程的直接观测。旧v2包装/启动命令没有RANK或后端覆盖，且原非负RANK分支会进入DDP；现有证据不支持“LLVIP此次因为seed1开启benchmark”的前提。CFT原训练用 `init_seeds(2+rank)`，单进程rank=-1时是seed1，两者不能混同。

## 两次训练尝试事实

同一次只读连接读取90原failure.json，结构化摘要已放上述JSON；主任务另行归档完整产物。

| attempt | 实际结果 | 更新/AP |
|---|---|---|
| llvip_train_canary_attempt1 | 原opt.yaml序列化遇PosixPath类型错误；5.5998秒，未进入训练批 | 0更新，无AP |
| llvip_train_canary_attempt2 | 10批完成；第11批申请100MiB触及0.68 allocator上限；39.1566秒 | 10次真实SGD，10次scaler尝试，0跳过，无AP |

v2的Torch峰分配/保留为16282.46/16570 MiB。前10批批末allocated约697.24–697.70 MiB，reserved从3924逐步到16514 MiB；这与批末持续保存大量训练图的典型表现不符，但不能只据这两个计数区分临时分配、工作区、缓存分块或碎片，更不能据此认定benchmark搜索。PyTorch区分张量占用allocated和缓存分配器管理的reserved。[内存管理说明](https://docs.pytorch.org/docs/2.10/notes/cuda.html#memory-management)

排除前4批后的6批，平均训练加数据等待为0.9606655秒，E200纯训练早期外推80.2689小时。它来自共享GPU6、原warmup阶段，仅6个样本且未含每轮验证/保存，不能称作者方法稳态成本或完整训练实测；当前更不能据此声称≤10小时。原回执明确另一进程占约1.03 GiB，但显存数也不能量化共享计算争用。

结论：保留失败与10步执行证据，暂停同设置重跑。`benchmark=False` 与 `deterministic=False` 是不同改变，不能把后者藏在“关闭benchmark”名下；若以后排查，须先提出与现有设置确实不同的单一后端/分配器假设并实测，继续保留模型、B8/1280、优化器、数据及资源上限。本次没有启动该诊断。[PyTorch后端设置区别](https://docs.pytorch.org/docs/2.10/notes/randomness.html#cuda-convolution-benchmarking)

本次是父任务明确授权的90仅CPU检查；没有GPU操作、后端额外覆盖、新hash、框架修改或AP计算。审阅者为复用subagent。
