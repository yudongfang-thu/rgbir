# 新方向训练与定标入口限定快审

**READY：修订后的训练、criterion、8-batch 定标源码未见确定运行或学习剂量阻断；5/5 独立 CPU 合成检查通过。** 这是 bounded source/API 审阅，不是新 GPU 运行或正式端点验收。

审阅 `train_direction.py`、`direction_criterion.py`、`calibrate_direction.py`，并对照 pinned builder、selected-only API、shared_parameter_set 与原 trainer。回执和源 stat 见 [cpu_receipt.json](cpu_receipt.json)，复核入口见 [check_cpu.py](check_cpu.py)。无新 hash/GPU/SSH/AP 输入。

- 原 builder 被私有克隆；辅助模型从原 full data identity 加载，保持 eval/frozen，仍由原 setup 排除 optimizer/EMA。学生全 state（含 head/BN）与输入 checkpoint 精确对照，新的 optimizer/EMA 不继承旧运行状态。
- criterion 保留 `native.sum() + B * lambda * KD`，selected-only build/loss 与 L2 variant 的实参匹配；shared_parameter_set 解包为 indices/named。CPU 直接执行当前 make_type 函数的小型替身：非零剂量的总 loss/gradient 组合及 λ=0 原生 loss/gradient exact；LLVIP N 使用空 L2 路径也可微、无 selected 阻断。
- BN wrapper 每次 `_model_train` 后只将 BN 设 eval；其 affine 参数保持可训练。CPU 直接执行当前 wrapper，确认 running buffers 不变且 affine 梯度存在。真实运行还会逐 BN buffer torch.equal 校验；此处不冒充实跑已通过。
- 定标不进行 optimizer step，每批恢复全部初始化参数/buffer，teacher/ref 为 frozen 模型。补充后在 setup 后与输入 checkpoint EMA/model.float 的所有状态逐项 torch.equal 对照；方向 lambda 规则保留原预定口径。

审阅发现并由根任务即时修复一项确定读出错误：cosine 原先仅在双方非 None 参数交集计算两边 norm，会漏算单侧梯度。现 dot 用交集、norm 各自计入全部非 None，CPU 反例 `[3,4]` 对 `[3,0]` 正确返回 0.6。此修复不改变定标 norm/系数。另补齐上述定标输入 checkpoint 对照，避免仅还原 setup state 却声称已验证输入初始化。
