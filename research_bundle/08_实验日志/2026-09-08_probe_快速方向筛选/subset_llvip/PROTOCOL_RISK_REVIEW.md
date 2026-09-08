# 新方向快筛协议短评

**统一 BN running statistics 冻结、SGD lr0=1e-4/lrf=1、warmup=0、FT3/192 batch、fresh optimizer/EMA 是一个新的受限微调协议，必须重跑各数据集匹配 N。不能复用上一轮 Drone N-ft 53.714083，也不能与旧 E8/E200直接做方法增益归因。** 本文件仅静态科学边界意见，未启动训练、GPU 或新增 AP。

该设置可减少小子集更新 BN 统计及较大学习率对成熟模型的共同扰动，适合测试“候选在保守微调下是否出现方向信号”。它尚未被证明更能预测完整训练收益；上一轮三臂低于起点也未单独证明是 BN/LR 导致。即使这轮恢复 baseline，仍是协议整体差异，不是 BN 单因果验证。

冻结的是 BN running_mean/running_var/num_batches_tracked，须明确 affine γ/β 是否继续训练，并在每次模型进入 train 模式后保持同一规则；teacher/reference 仍为冻结 eval。fresh EMA 的状态初始化/更新及评价 last/EMA 口径应在各臂一致。这里 1e-4 是旧 args 中 .01×.01 的配置终段目标，不凭 args 声称旧最后一个实际 optimizer step 的 LR 恰为该值。无 warmup/恒定 LR 与 fresh momentum 仍是重新定义的短程动力学。

LLVIP 原 visible42/IR42 是单类 person baseline，当前 2048 子集是已治理 fit 内自然 source×密度×面积样本；共享 GT 或完整配对只提供实例对应，不认证物理几何。新的 L2-box 若使用 IR-GT→RGB-GT 的目标相对映射，应明确是 GT 条件下的框迁移候选，并用 L2-GT 控制共享目标/额外监督解释，不能解除原 L1 bin 对齐与物理几何阻塞。

所有臂应使用同一固定子集、起点、增强流和预算。小样本、单 seed、192 batch 可提供低成本方向反馈，不能把阴性结果判为所有训练阶段无效，也不能把阳性结果升级为正式多 seed 增益。不要按新 dev AP 补采、换子集、调阈值或自动延长；新增候选遵循本轮事前分流规则。对象/图像覆盖统计不等于梯度贡献或定位收益。
