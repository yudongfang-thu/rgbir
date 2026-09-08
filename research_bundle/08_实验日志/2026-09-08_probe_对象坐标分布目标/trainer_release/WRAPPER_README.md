# OBJECT_DFL 独立短筛 wrapper

**源码已冻结，可进入 pinned CPU 与有界实际校准/canary；尚无本方法 GPU 梯度、训练或 AP 结果。** 本目录只适配已执行 direction 短筛骨架，不改旧 release，也不接入另一路 C1 加速部署。

固定协议见父目录 [SHORT_SCREEN_PROTOCOL.md](../SHORT_SCREEN_PROTOCOL.md)，来源接口核对见 [TRAINER_INTERFACE_REVIEW.md](../TRAINER_INTERFACE_REVIEW.md)。本地 [WRAPPER_CPU_attempt1.json](WRAPPER_CPU_attempt1.json) 为 15/15 合成 CPU 合同通过，CUDA 未初始化；[CLI_CPU_attempt1.json](CLI_CPU_attempt1.json) 验证三个命令入口。实际 L3 算子及其独立小真值见 [L3_LOSS_INTERFACE.md](L3_LOSS_INTERFACE.md)。这些不替代 pinned 实际运行。

入口与旧文件名保持兼容：

- `calibrate_object_dfl.py --reference-dir <release_gpu5> --config <N模板> --output <新校准目录>`：固定首 8 批逐批恢复完整初始状态，P3/P4 源模块共享参数梯度；仅 DFL 的至少 4 批有限非零比值决定 `min(1,median)`，GT 共享系数，另保留其真实单位梯度与 cosine。
- `train_object_dfl.py --reference-dir <release_gpu5> --config <有效冻结配置> --output <新目录> --canary`：至少 24 次成功 optimizer 更新；正式运行把 `--canary` 换为 `--canary-receipt <对应canary.json>`，完整 3 epoch / 192 batch。
- `evaluate_object_dfl.py --reference-dir <release_gpu5> --config <同配置> --checkpoint <run/weights/last.pt> --output <新评价目录> --native-config <configs/llvip_native_evaluation.yaml>`：完整 2406 图/7879 GT；不用 Drone-only binding。

三臂为 N/L3-DFL/L3-GT，scope `OBJECT_DFL_FT3_BNFROZEN`，endpoint `OBJECT_DFL_FT3_BNFROZEN_LAST_EMA`；完成 status 为 `OBJECT_DFL_CANARY_COMPLETED` / `OBJECT_DFL_TRAINING_COMPLETED` / `OBJECT_DFL_EVALUATION_COMPLETED`。保留文件名 `canary.json`、`completion_receipt.json`、`direction_evaluation_receipt.json`、`direction_config.yaml`。`method_identity` 和 `object_dfl` 列出新方法定义，不能拿旧 scope 的结果替代新匹配控制。

`object_dfl_criterion` 只调用 `l3_distribution_loss.compute` 返回的未加权 KD；后者已含温度平方与 base 归一，外层只加一次 `B_actual*lambda`。N 走同 DFL 辅助/选择，但 λ=0 并检验 native 总值；GT/DFL 同 raw 计算时硬核 base/normalizer/selected/anchor/对象身份相同。T/R eval、no_grad，GT/教师目标 detached，EMA/optimizer 保持原隔离。

原自然 2048 图、seed42、B32、nbs64、AMP、workers4、SGD1e−4 恒定、warmup0、BN running statistics 冻结而 affine 可训练、fresh optimizer/EMA 保留。64 是每轮 batch 数，不能叫 64 次成功更新；实际 receipt 报 update attempts / AMP skips / successful updates / EMA updates。首30批跨臂及正式训练对 canary 的流由 queue 复核。

新协议避免 setup 下载/推理无关 AMP 检查模型。`validated_amp_prior.json` 和 `validated_amp_prior_config.yaml` 是前次完成 DFL probe 回执与实际配置的逐字节副本；运行时验证原 actual_amp、三模型当前 path/stat 及 pinned 版本。只在 setup 期间局部替换 native trainer 的 `check_amp`，`finally` 恢复；每臂及校准相同，实际 autocast 仍为 True，不称与旧 setup 完全等价。部署不能漏这两个输入文件。

校准每批末 CUDA 同步，调用既有 lease helper 刷新本任务 NVML/进程树资源并读取框架峰值。按既有预定 margin 取整后硬限显存 8192MiB / RSS32768MiB，首批另写 `first_batch_resource_receipt.json`；任意超额立即技术失败，不能继续凑够 8 批。全部原始检查留 `calibration_resource_batches.jsonl`，外部 guard 仍独立生效。新训练资源预约须用本路径实际 canary，不能用旧 L2 峰代替。

目录的 `WRAPPER_SOURCE_ADAPTATION.json` 记录初始来源适配，不是最终部署字节回执；最终本地源码清单见 `FINAL_WRAPPER_INPUTS.json`。父目录 `prepare_trainer_wrappers.py` 仅是最初生成步骤，后续已追加 AMP/资源门与合同检查，**不要用生成器重建或覆盖冻结源码**。GPU 部署与唯一 global queue 由 root 负责。本 wrapper 工作没有 GPU、SSH、AP 读取或新 hash。
