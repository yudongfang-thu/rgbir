# CFT 原训练显存与三卡技术检查

**三卡原 DataParallel 技术检查也在首批前向触及 0.68 显存分配上限，成功更新为 0。实际分片 11/11/10、各卡 autocast 开启且卷积输出 FP16，已排除这两项工程疑点。连同三次旧失败，原 B32/1024/E200 在当前 90 资源约束下没有训练准入；停止同存储方式盲试，不提高 GPU 上限或取消显存额度。**

## 既有证据

原作者源码来自 `../2026-09-09_repro_RGBIR对比方法优先接入/cft_forward90/cft_author_source.tar.gz`，已记录作者 commit `fb591c9b163177c0e950db08e213e24ddc912d41`；本次不新计算 hash。三份原失败回执位于 `../2026-09-09_repro_CFT原文协议/server_training/`，原文件保留。

| attempt | GPU 数 | allocator 比例 | 失败时成功更新 | Torch 峰值 allocated / reserved MiB |
|---|---:|---:|---:|---|
| train_canary32_attempt1 | 1 | 0.65 | 0 | 15478.72 / 15728 |
| train_canary32_attempt2 | 1 | 0.68 | 0 | 16374.77 / 16498 |
| train_canary32_dp_attempt1 | 2 | 0.68 | 0 | device0 16247.91 / 16570；device1 16232.13 / 16572 |
| cft_dp3_canary_attempt1 | 3 | 0.68 | 0 | device0 16338.30 / 16560；device1 16388.71 / 16516；device2 16424.53 / 16562 |

双卡报错明确进入 DataParallel replica 0，申请 128 MiB 时失败；报错仍显示约 7.19 GiB 物理空闲，但当前进程允许约 16.19 GiB。不能把这些空闲显存视为可以绕过项目额度的空间。guard 的轮询峰值不覆盖所有瞬时峰值，本表优先使用每设备 Torch 记录。

三卡回执已逐项读取：[failure.json](cft_dp3_execution/cft_dp3_canary_attempt1/failure.json)、[events.jsonl](cft_dp3_execution/cft_dp3_canary_attempt1/events.jsonl)。实际物理卡为 0/1/3；总耗时 23.236591 秒，首批前向在 SiLU 申请 44 MiB 时拒绝，未执行到 loss 或 backward。完整 train loader 为 12025 图、376 批；首批来自 official train 010001–010032。每卡输入分别为 `[11,3,1024,1024]`、`[11,3,1024,1024]`、`[10,3,1024,1024]`，RGB/IR 一致；autocast=true、target=float16、grad=true，三卡第一卷积输出均为 float16。不能把 23 秒失败检查时间当训练吞吐。

## 源码诊断

- `train.py:615` 的原分支在 `rank=-1` 且可见 GPU 多于一张时使用 `torch.nn.DataParallel(model)`。输入 RGB/IR 都是 batch 维在第 0 维的张量；两卡应各 16，旧日志的主线程 `[32,6,1024,1024]` 只证明全局 batch。新三卡记录已验证实际 worker 分片。
- `train.py:755` 将模型前向及原 ComputeLoss 放在 CUDA autocast 内；GradScaler 也启用。PyTorch 2.10 的 DataParallel worker 显式传播 autocast 状态。旧日志中的 deprecated 警告不代表 AMP 关闭。新入口只观察 worker 的 autocast 和第一卷积输出 dtype，不强制改 dtype。[PyTorch parallel_apply](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/torch/nn/parallel/parallel_apply.py)、[DataParallel](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/torch/nn/parallel/data_parallel.py)
- `train.py:577` 的 FP32 EMA 是作者原行为；206,247,222 参数的单份 FP32 参数约 787 MiB。`train.py:605` 在训练前删除初始化 checkpoint/state dict 引用。`train.py:715` 先把全局 6 通道输入转 GPU0 float32（B32、1024 约 768 MiB），再交给 DP。模型副本、梯度归并、EMA 与输入使主卡额外承载开销；未发现足以解释此次失败的异常图保留或多余持久 checkpoint 副本。不能为通过检查而去掉 EMA。
- 原 DDP 函数分支存在于 `train.py:656`，但 CLI 在 `train.py:1005–1010` 仅令 rank -1/0 调用 `train_rgb_ir`。直接 torchrun 原 CLI 不能保证所有 rank 进入训练，存在等待非零 rank 的风险。本次不将 DDP 当作无需修改即可运行的修复。

## 已执行入口与协议边界

`train_canary_dp3_diagnostic.py` 只允许 3 个可见 GPU，沿用现有 guard/lease，不创建资源池。默认 allocator 比例 0.68，与旧双卡失败相同；物理 GPU 由主执行方动态核实和分配。全局 B32 预计被原 DP 分成 11/11/10。`utils/torch_utils.py:77` 的作者 CLI 要求 batch 整除 GPU 数；直接调用原 `train_rgb_ir(hyp,opt,device)` 绕过该 CLI 检查，属于**并行执行条件重建**。模型、损失公式、全局 batch、1024 分辨率、E200、初始化、SGD、原 warmup/AMP/数据增强和完整 official train12025 均保留。每副本 BatchNorm 的 batch 与单双卡不同，因此不宣称历史训练轨迹精确相同。

入口默认 `--successful-steps 2 --max-batches 16`，仅在原 SGD.step 真正返回后计数；GradScaler 跳过的更新不计。完成目标步后立即退出，最后一步的 scaler.update/zero_grad/EMA 尚未执行，不生成可恢复 checkpoint，不执行 AP。可以显式增加目标更新数和 batch 上限，仍不能把首次 warmup 两步当稳定吞吐。新目录拒绝已有输出；包装及兼容文件复制进该 attempt。

首次真实训练前向按卡记录 RGB/IR shape/dtype、autocast 状态/目标 dtype、grad 状态与第一卷积输出 dtype，并断言 11/11/10 与 FP16 卷积输出。只保存元数据，不持有激活张量。无关闭 AMP、变更分辨率、减小全局 batch 或训练公式替换。

## CPU 已验证的最小兼容

作者 `utils/loss.py:167` 将 gain 建为 float，`:178` 令 `gain[2:6] = shape[[3,2,3,2]]`，`:211` 却对 long 索引用 float 上界执行 inplace clamp。此前三次 OOM 未触达该代码。新模块 `cft_loss_index_compat.py` 只对该原函数做两处 AST 边界替换，不修改磁盘原源：

```diff
- gj.clamp_(0, gain[3] - 1)
+ gj.clamp_(0, p[i].shape[2] - 1)
- gi.clamp_(0, gain[2] - 1)
+ gi.clamp_(0, p[i].shape[3] - 1)
```

网格维度本为整数；使用 shape 可避免每个 head 对 CUDA gain 做 int() 同步。静态检查严格验证原 gain 赋值与两处 clamp 结构，结构不符即失败。CPU preflight 从原函数构造两个版本，逐张量核对全部 tcls/tbox/indices/anchors：运行版本使用 shape 整数边界，参考版本保留 gain 但把边界转 long。空目标及含边缘、内部、非方形网格的目标均精确相等。

90 原环境 CPU PASS 已读取 [loss_cpu_preflight.json](cft_dp3_execution/loss_cpu_preflight.json)：空目标各层匹配 0/0/0，内部与边缘目标 24/33/27；原写法两组均出现 Float→Long 错误，CUDA 未初始化。新主入口先 import author_train、执行兼容安装与 CPU 检查，之后才调用 CUDA 可用性检查和 allocator 设置。本地 Python 缺 Torch；本地只做 AST、接口字段及执行顺序检查，结果在 `CFT_TRAIN_STATIC_CHECKS.json`，不冒称本机跑过 Torch。

## 时间与证据范围

12025/B32 按原 loader 约 376 批/epoch，E200 约 75,200 批。10 小时留给训练及每轮评价、保存等全部工作，平均每批总预算约 0.479 秒；这只是预算反推，尚无实测支持。首两步用于证明前向、反向和真实更新可行；若通过，再看适当更长范围的吞吐与显存。所有尝试需保留资源失败、兼容差异及原始回执，不能把技术检查升级成训练完成、AP 或全文复现结论。

唯一保留的工程候选是把原 Model.forward 放在每个 DP worker 内的 `torch.autograd.graph.save_on_cpu(pin_memory=True)` 上下文中。官方实现把 backward 所需保存张量移至 CPU，反向时搬回原设备；它不重算前向或 BatchNorm，因此有希望保持原算子/梯度语义，但会增加 host RAM 与 PCIe 传输。不能据此保证显存通过或 10 小时内完成，也不能只在 DP 主线程外层套上下文就认定所有 worker 已覆盖。此候选本次未实现、未运行、未取得准入；当前优先继续已能执行的作者 baseline 权重复评。[PyTorch 2.10 save_on_cpu 实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/torch/autograd/graph.py)

本次审阅者是已有复用 subagent，非 fresh 或跨模型独立审阅。未 SSH、未运行 GPU、未更改旧源、旧 attempt 或共享索引。
