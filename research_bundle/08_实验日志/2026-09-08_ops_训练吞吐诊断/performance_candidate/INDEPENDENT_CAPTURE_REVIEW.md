# 独立静态审阅：真实两批 capture

**PASS_FOR_BOUND_1_OR_2_BATCH_DIAGNOSTIC_CAPTURE：本次只读静态审阅没有发现阻塞真实 1–2 batch 诊断采集的技术问题。它不构成 24 次更新 canary、正式训练切换或实际加速的验收。**

审阅者：独立子任务 ap_error；日期：2026-09-08。已读 `capture_real_batch.py`、`benchmark_candidate.py`、`pool_block16.py` 及其原 runtime/selection/classification/raw_prediction 调用关系。本次不运行 GPU/SSH，不修改作者代码，不计算新 hash。root 在此回执落盘前已收到 READY；此文只是记录已完成的审阅。

- 学生是独立加载 checkpoint 的内存副本，设为 train 并启用梯度；T/R 保持 FP32 模型参数、eval、冻结，并在 no_grad 中前向。三者前向遵守 cfg 的 AMP，学生先从 checkpoint 对应 `args.yaml` 恢复 args，再初始化原生 criterion。总 backward 为原生 loss 加 B×系数×KD，无 optimizer、scaler、EMA 更新及权重写回。
- 输入固定为旧自然流 seed20260907 的前 1–2 批；验证原 64 批回执和完整批号，并逐批 exact 比较旧 source/augmentation/双标签 JSON；B32、workers4、640 有显式约束。这是诊断自然流，不是当前正式训练 sampler 或恢复点。
- bundle 保存真实 raw scores/DFL、标签与原 feature 逻辑形状/dtype。feature 值由零 stride 占位，适用范围为本次 C1 layout/selector；不能用来复核 feature KD 或模型前向。模型不为 bundle 重复推理；独立 replay 与原 pipeline 分开报告。
- 原生加 KD backward 后、zero_grad 前要求存在学生梯度且全部有限；辅助模型不得泄漏梯度。输入 checkpoint 只检查前后 size/mtime，不能把该项描述为 checkpoint 的逐字节证明。运行源有字节复制比较，cfg/学生 args 也留副本。
- 刚补的独立 KD 检查使用 `autograd.grad(kd, s['scores'], retain_graph=True)`，不写参数 `.grad`，在原总 backward 前要求该梯度存在、全部 finite，且 float L2 有限并严格大于 0。该批若无有效梯度立即失败，未提供换批补救。探针全段前后 CUDA 同步，单独计时并从 pipeline wall span 扣除；独立探针结果不能代替训练更新验收。
- `timed` 的普通 stage 在同步后开始计时并独立记录 peak；pool replay 内层明确传 `track_peak=False`，不会重置外层 replay 的峰值区间。单独 pool replay 带逐调用同步，其耗时既不与原 pipeline 相加，也不是正常重叠执行下的生产吞吐。外层总 span 仍包含逐批核验与有限性检查等诊断开销，代码已明确串行仪器化口径。

本次接受只覆盖代码接口与上述诊断边界；真实 source/flow、数值、资源和 bundle 等价性仍以 root 随后收集的实际运行回执为准。synthetic pool 倍数不能直接外推完整训练提速。
