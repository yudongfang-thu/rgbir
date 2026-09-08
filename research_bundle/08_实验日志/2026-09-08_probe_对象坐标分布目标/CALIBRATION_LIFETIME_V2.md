# 校准 attempt1 资源失败与计算图生命周期修复

**attempt1 已停止且没有训练；发现并修复校准器跨批保留上一批学生图的明确代码缺陷。预算、样本流、选择门、温度、损失和校准系数规则不变，资源能否回到上限内仍需同固定 8 批的新 attempt 实测。**

原证据保留在 `training_evidence_1703/calibration/llvip/`。首批 NVML/框架 allocated/reserved 峰为 5454 / 4389.493164 / 4954 MiB，含预定余量预算为 5888 MiB，通过。第二批对应 8506 / 7376.645508 / 8006 MiB，含余量预算 8960 MiB，超过固定 8192 MiB，资源门立即抛出 `OBJECT_DFL_CALIBRATION_FAILED`。这不是 AP 或梯度无效导致的停止；整卡仍有余量、外部 guard 未报错，不豁免本次候选预算。

v1 校准先对 native、DFL、GT 三个标量使用 `autograd.grad(..., retain_graph=True)`。`for arm,(loss,stats) in losses.items()` 的 Python 循环变量 `loss` 留在函数作用域；最后删除 `losses` 字典时没有删除最后的 GT scalar，因而仍有路径到共享学生前向图及 saved tensors。下一批前向发生时，上一批图还活着，直到新循环覆盖 `loss`。独立审阅者确认了这条引用链；该缺陷与观察到的第二批峰值增长相符，但本次不声称它已解释全部显存组成。

修复前的原样源码保存在 [failure_source_v1/](failure_source_v1/)，有逐字节复制回执。v2 仅改 `trainer_release/calibrate_object_dfl.py`：

- 将 native 返回的 items 显式命名为 `native_items`，不依赖它是否 detach。
- 每批读数与原资源门完成后，连同原引用显式删除最后的 `loss`、`stats`、`native_items`。
- 在每批开始、清理前、清理后记录框架 allocated，另记清理后 reserved，写 `calibration_allocation_lifecycle.jsonl`；同步读数，不调用 `empty_cache()`，不靠清空缓存掩盖问题。
- 完成回执添加 `calibration_graph_lifetime=v2_explicit_loop_loss_and_native_items_cleanup`。

原计算图内的梯度读取、native+KD 数学、T/R 冻结、参数恢复、8 批 roster、4/8 非零要求、共享 λ、温度、base 分母和全部质量门没有修改。若重新执行，必须新目录从第 1 批开始，不能 resume 失败的第二批或放宽资源上限。

受影响 wrapper 的 [15 项 CPU 真值](trainer_release/WRAPPER_CPU_LIFETIME_V2.json)重新通过，calibrator CLI 成功；CUDA 未初始化。独立的 saved-tensors 弱引用测试见 [CALIBRATION_LIFETIME_INDEPENDENT_CPU_attempt2.json](independent_review/CALIBRATION_LIFETIME_INDEPENDENT_CPU_attempt2.json)，验证旧引用保留、新清理释放，两个合成批次梯度 norm 相同。此 CPU 依据证明生命周期缺陷和修复机制，不伪称已测得新 GPU 峰或新 λ。
