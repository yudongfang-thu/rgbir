# selected-only：24 次真实更新诊断

这是新的独立旧/新比较入口，**不修改原 block16 入口或其失败结果，不准入正式长训**。保留原已接受 `update24/compare_24_updates.py` 的观察、比较和 fresh worker 主体；运行时仍复用 pinned release 的 `verify_compatibility.py` CPU/RNG/梯度/state/loader helpers，不调用其 run 或证据 hash 入口。

调用：

```text
PINNED_PYTHON compare_24_selected_only.py --reference-dir EXISTING_RELEASE_GPU5 --config ORIGINAL_FORMAL_C1_S42_YAML --candidate-source performance_candidate/selected_only_v1.py --output NEW_DATA_DISK_ATTEMPT
```

新增部署依赖只有本目录 `compare_24_selected_only.py` 与冻结 `selected_only_v1.py`；原完整 release 和 cfg 路径必须已存在。由 root 在原 global lease/screen 下启动，作者不执行 SSH/GPU。输出必须在 `/mnt/dataset/yudongfang/` 新目录；旧/新 worker 都新建独立解释器，顺序运行 `old`、`selected_only`。

两臂保留原 C1 seed42 的 E200 recipe、yolo11n 初始权重、B32/nbs64/workers4/AMP，实际 λ=`0.09227393550836771`，不改 native、GT、选择、优化器、EMA 或学习率日程。每臂恰好 24 次真实 optimizer.step，最多 96 次尝试，AMP 跳步另计。原 `max_steps=24` 令 sanity=True，因此保留每批 sanity 检查和完整统计。

新臂必须调用 candidate 的 `make_criterion_type`，学习 loss 始终来自 selected-only 图；原完整 selector/loss 只在同 raw 的 no_grad 中采集诊断，不新增模型 forward。实际 `thin_learning_batches` 与 `full_diagnostics_batches` 都必须等于总 batch，`fallback_batches=0`；否则本次诊断失败，不能用原图的回退伪装成薄路径验收。原 target/off-target/shared-gradient observer 仍读取真实 learning payload。

每批 selection 快照比较完整身份、所有 common-valid scales、base/selected/eligible、quality/C0 和所有 region/mask；浮点 delta 仅采集 selected S/T。`unselected_student_delta`、`unselected_teacher_delta`、`reference_delta` 明确不属于学习快照的采集范围，不当作零测量或数值失败。完整诊断日志仍来自真实 detached 原统计。

新增每批 `loss_####.pt` 保存 native、KD unit、加权 KD、总 loss、target/off-target 与实际 B/λ；清单按 batch 数闭合，缺失即拒绝。初始/首批真实像素、逐批 source/双标签/增强元数据/worker 与父 RNG、原全部 scaled/applied gradients、参数/optimizer/EMA/scaler/AMP 计数按原比较规则保留。后续每批像素不额外存，不能把元数据 exact 写成所有像素已重比。

原 atol=1e-6、rtol=1e-5 不变，float64 比较 `abs(new-old)<=atol+rtol*abs(old)`。选择与控制先要求 exact，再报告中间数值和全训练轨迹的 numeric/bitwise 两层；matched AMP 非有限模式单列，不当作可用梯度。任何失败保留原值。

计时同时保留完整同步 batch wall、audit copy/save/check、扣审计后的 wall、新臂额外 no_grad 完整诊断时间及再扣这段后的时间。前 6 批固定热身，loader fetch 不在该 batch callback 区间。完整统计会改变缓存与重叠，因此这些分层不能当作无插桩正式 epoch 吞吐。额外诊断时间累计在候选 factory，原学习/反传/sanity 不从图上删除。

CPU 入口 `test_compare_cpu.py` 基于原已接受 16 项，另测 thin fallback 拒绝、每批 loss 清单缺失拒绝、KD 差异导致 trajectory numeric fail。这里只准备与审阅，不执行真实训练；实际 GPU 资源由 root 的已有 guard 实测。
