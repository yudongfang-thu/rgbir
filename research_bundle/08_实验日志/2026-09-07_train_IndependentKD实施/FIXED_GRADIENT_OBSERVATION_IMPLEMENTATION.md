# 固定时刻共享参数梯度观测实现

**结论：已为 C1/C1_y/L1/L_GT 接入预先固定 epoch 0/10/50/100/199 首个训练 batch 的只读梯度观测；10 个新数值/无副作用测试与 27 个分类回归测试共 37/37 通过。没有运行 GPU；新的 criterion 字节需要重新绑定到真实验证回执。**

日期：2026-09-07。此项按根明确授权修改 `independent_criterion.py`，新增 `gradient_observation.py` 与 `test_gradient_observation.py`；不修改冻结 legacy、分类纯核、trainer、calibrator、optimizer 或资源调度。原任务所称 §8.4 的“正式训练只观察”在当前规格实际为 **§8.5**；§8.4 是 L 的校准公式。本实现采用 §8.5 明列的五个 epoch。

## 冻结观测合同

- 仅 C1/C1_y/L1/L_GT，在每个指定 epoch 的第一个真实训练 criterion 调用记录一次。没有 signal 条件，零 KD 也占用该 epoch 的唯一观测槽，不会在后续挑一个有信号的 batch 替换。
- Θ 是 `model.model[-1].f` 的前两个实际输入特征模块中全部可训练参数；与当前校准器一致，并记录实际 module indices / parameter names。解析失败不回退整个模型。
- 原生梯度为 `grad(native_total.sum(), Θ)`；KD 梯度为 `grad(actual_B * λ * KD_unit, Θ)`。实际 B 只乘 KD 一次，native 已包含真实 B。
- C1 额外记录 `actual_B*λ*target_loss` 与 `actual_B*λ*0.25*off_target_loss_unit` 的范数/夹角；C1_y 对应 off-target 实际系数为 0。没有把“删项”解释成总范数必然下降。
- 全部使用 `autograd.grad(... retain_graph=True, create_graph=False, allow_unused=True)`，不调用 backward、optimizer、scaler，不写 Parameter.grad，不清理已有累积梯度，也不再执行模型前向。下一次实际训练 backward 仍可使用原图。
- 局部梯度立即 detach，仅保留 Python 标量/名称等 JSON 记录；helper 实例只保存已观测 epoch 的整数集合。图引用在当前调用返回后释放。
- 梯度保留真实训练精度，范数/内积用 float64 归约以避免有限 FP32 平方溢出。梯度非有限时记录 `NONFINITE_OBSERVATION`、对应 metric=null 和非有限 tensor 数，不改变原本的 AMP skip/更新逻辑。有限零梯度无 epsilon 放大。

## 产物与字段

每次运行写 `shared_gradient_observations.jsonl`，包含：

- epoch、全局 batch index、固定 epoch 内 index=0、固定 schedule；
- Θ 实际名称、实际 B、冻结 λ、实际梯度 dtype；
- native / weighted KD 的范数、比值、余弦，存在/缺失/非有限参数 tensor 数；
- C1 的 target/off-target 范数、夹角及 target/native 与 target/full-KD 剂量比；
- 观测时的 optimizer successful updates / attempts / AMP skips / EMA updates、GradScaler scale、AMP 启用状态；
- 当前 base / selected 数及学生文件路径，便于复核预定样本。

该指标是当前 batch 未缩放、未裁剪的梯度观测，不冒充累积梯度、momentum 或实际 optimizer 更新的方向。调度/系数都不根据观测结果变化。

## N/C0 与源码绑定

N/C0 在 `IndependentCriterion.__call__` 开头仍直接 `return super().__call__`，不构建或执行 observer，不改变其损失、样本流或 RNG。新增模块本身没有随机调用。

但 `independent_criterion.py` 源文件确实变化，execution binding 按字节核验，因此**旧字节生成的 compatibility/canary/calibration binding 不能自动借用给当前 release**。根已暂停部署等待本项完成，应统一部署新 release 并按最终绑定规则运行对应真实验证；不能仅因 N/C0 数学未改就手工重标旧 binding。

## CPU 验证

本地 Python：`D:/Anaconda/envs/KGJ_proj/python.exe`，torch 1.8.0+cu111，纯 CPU。

`python -m unittest test_gradient_observation test_classification_logit -v`：37/37 通过，真实输出保存在 `fixed_gradient_cpu_tests_attempt1.log`。

新 10 个测试覆盖解析可算的范数/余弦、C1 分量抵消导致 C1_y 范数反增、部分 batch 的一次缩放、已有 Parameter.grad/模型 buffers/三类 RNG 不变、观测后正常 backward 仍可累积得到预期梯度、固定 schedule 且零信号不换样、L-only/缺失参数梯度、有限 loss 非有限梯度的 JSON 安全记录、零 native 无 epsilon、实际参数名称与解析失败、记录无 tensor、拒绝历史臂与错误 eta。

三个变更文件额外语法编译通过。真实 pinned GPU 的新路径显存峰值/梯度/API 兼容仍需接下来的受管 canary 证据；本文件不声称这些已经完成。
