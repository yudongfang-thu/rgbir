# 固定子集筛选器：计划审阅

**PLAN_ACCEPTED_SOURCE_AND_PINNED_PENDING。** 新任务是已知 C0 方向的廉价筛选工具校验；计划范围自洽。此状态不是 GPU 准入，尚无新执行结果。

审阅输入：本 entry README、上一 entry `NEXT_FAST_SCREEN_TASK.md`、既有子集 `2026-09-08_ops_小时级筛选重构/subset/README.md`。沿用 experiment-audit；按本轮明确要求使用路径、stat 与逐字节身份，不计算新 hash。

- 固定身份 `DRONE_SUBSET2048_PRETRAIN_E8_CHECK`；既有自然分层 2048 图/33196 GT、完整 dev 1469 图/22462 GT、seed42、N/C0 λ=0/.1。两臂各 8×64=512 batch，原通用 yolo11n、正常 BN、原 E8 优化与增强、fresh optimizer/EMA、fixed last EMA。既有子集合同中的成熟微调用途不自动绑定本任务；新配置和回执必须明确通用预训练初始化。
- 原 C0 算式、门、base 分母及 pinned historical=False 路径保留。比较只回答这个子集/seed/日程能否保留已知方向，不是独立方法增益、一般排名或 E200 预测性；非正向停止这个筛选器版本，不推出 C0 无效。不重抽子集、扫 λ 或改变轮数。
- 五类检测头按原生方式从通用 80 类 checkpoint 建立；要求两臂实际初始 state 全张量逐项相等，且训练重置到共同初始状态。不得要求五类头与原 80 类 checkpoint 全张量相等。
- 服务器数据盘保存 N canary 前 30 批的双模态 uint8 张量，C0 canary 与两臂正式训练用实际逐批 `torch.equal` 比对，并绑定 shape/dtype、batch 编号和双标签。若保存位置是增强后、归一化前 loader batch，应明确该取样阶段；它不是未增强源图文件。约 2.4 GB 参考只留服务器，不上传。30 批是流前缀，不能表述为全部 512 批像素一致。
- 两个新解释器 canary 各需至少 24 次成功 optimizer 更新，最多 48 次尝试，AMP skip 和 EMA 调用单列。实际有限 loss/梯度、C0 非零 KD、资源及计时为后续准入证据；不凭既有 C1 吞吐代替本路径测量。
- 45 分钟是本 attempt 执行硬上限，排队、本地实现/审阅时间另记。预算必须计入两 canary、初始化、服务器像素保存/比对 I/O、两次完整 dev 评价和启动交接；按正常 cadence 实测外推训练并至少加 20% 训练余量。执行中检查累计实际计时，超时保留 incomplete，不缩轮数、改 batch/workers 凑预算。全局 lease 和原资源纪律不变。

下一步只审新增源码差异、少量受影响 pinned 检查、部署逐字节身份和实际预算门；不重复旧 C0 大型测试矩阵。源码与实际 pinned 证据齐前保持 pending，任何旧 READY 不适用于这个新身份。

审阅者：`/root/object_transport_review`，独立于实现与执行；模型身份未暴露，不声称跨模型审阅。未运行新测试、forward、GPU 或 AP 重算。
