本入口只准备一个新 F-rel-GM 的 canary、FT3 训练与完整 Drone dev 评价；只有显式匹配投影通过，才能把已完成 main N/C1 用作本次单 seed 对照。

固定入口为 `run_feature_gm_queue.py --release-dir <feature_gm_release_v1> --output <feature_gm_attempt1>`；可显式提供 `--reference-dir`、`--main-screen`、`--wait-for`。默认等待 `confidence_attempt1/queue/completion.json` 的 `CONFIDENCE_MATRIX_COMPLETED`，每 30 秒只检查 CPU 文件，无 lease 空等；失败则退出保留证据。

固定顺序：F canary（至少 24 次成功 update）→ 与 main N/C1 的共同配置、实际初始化 checkpoint stat、各 canary 和完整训练前 30 个 B32 批逐项相等 → F 训练 3 epoch/192 batch → FP32 full native dev 1469 图/22462 GT。只跑一个新训练臂，输出布局为 `canaries/F-rel-GM`、`runs/F-rel-GM`、`evaluations/F-rel-GM`。输入 checkpoint 只做 stat，队列不加载权重、不计算新 hash。

`queue/matched_control_projection.json` 的状态为 `FEATURE_GM_MATCHED_CONTROL_VERIFIED`，`controls` 恰好包含 N/C1；每臂保存共同配置、first30、完整训练 first30 和初始化 stat 的实际检查结果及原文件 stat。候选配置/canary 另有独立绑定。显式跨 scope 投影不会把新方法伪装成原 `DIRECTION_FT3_BNFROZEN` 的 F-rel。

新方法 scope 为 `FEATURE_RELATION_GM_FT3`，endpoint 为 `FEATURE_RELATION_GM_FT3_LAST_EMA`，λ 固定 14.438521129817886，不重新校准、不扫参数、不按 AP 改队列。旧 F-rel 的数值上限 BLOCKED 记录保留；这是校准后、首次 F AP 前固定的协议修订。该系数来自异种 loss 的梯度剂量匹配，不能声称跨 loss 的 λ 本身可比；实际剂量分布由独立固定 8 批读出报告。

仅使用既有 `resource_dispatch.run_job` 与全局 lease。首次 canary 预算 8192 MiB VRAM /32768 MiB 进程树 RSS；训练按实际 NVML、PyTorch 与 RSS 峰值加原固定余量计算，超过相同上限即停。评价预算 2048/8192 MiB。Python 保留虚拟环境入口原路径，不 resolve 到底层解释器。

CPU 小检查覆盖固定剂量/身份、共同配置负例、成功前置门、资源上限、CLI、两对照 projection、first30 内容变更及初始 stat 变更。实际 GPU 执行与资源准入由 root 调度；本目录 CPU PASS 只说明入口可提交受限诊断，不代表 F 已训练、评价或获得增益。
