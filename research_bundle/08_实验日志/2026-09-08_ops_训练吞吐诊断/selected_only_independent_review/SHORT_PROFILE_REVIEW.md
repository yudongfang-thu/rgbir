# short_profile 快审

**READY：可在根安排的既有 lease 下执行三臂各 24 次成功更新的性能探针。** 仅审查 `short_screen_draft/short_profile.py`、协议、builder 与固定 criterion；未运行 GPU，没有新 hash。此结论不是 E20 训练或效果准入。

2026-09-08，loc_stress。N/C0 保持原 builder/criterion；C1 通过配套 selector＋loss 的 criterion factory。外层 wrapper 只让第一次调用 `sanity=True`，其后 False，原 first3/every100 日志与 shared-epoch observer 仍在原函数内执行。它改变的是性能探针诊断频率，不是宣称重做了既有状态等价验证。

第六批结束和第 24 个成功 optimizer update 所在批结束 CUDA 同步，二者之间持续 wall 包含中间 loader fetch/正常日志和 callback，没有逐步 tensor/optimizer 大复制；中间 CPU timestamp 不能当单批 GPU latency。成功更新与实际 optimizer.step 调用闭合，最多 96 attempts/384 批。peak reset 在 trainer.train 之前，含 setup；最初 native canary 保存发生在热身界限前。失败及 loader 清理路径保留。

初审唯一要求补齐的是 C1 实际 thin/fallback 归属。最新版本已在逐批 row 和 final receipt 记录 `thin_learning_batches`、`full_diagnostics_batches`、`fallback_batches`，并要求 thin==实际批数、fallback==0，full diagnostics 数量有效。此项解除；N/C0 不伪造专用 thin 计数。

独立复跑 [8 项 CPU 真值](short_profile_cpu_review.json) 通过，未导入 torch/runtime。包括 first-sanity-only、连续窗口含批间等待、非固定 AMP skip 计数、缺失终点/乱序/非有限时钟/超限拒绝和 fallback 负例。

实际产物仍须验证第一原组合梯度检查、真实 C1 覆盖和资源。约 30 批的窗口可能看不到每 100 批完整统计和后续 epoch observer；不将其外推为长期零开销或完整 epoch 实测。
