**固定状态交换中，替换 FT 参数产生的下降大于仅替换 BN 统计；BN 统计漂移不足以单独解释旧 FT 的下降，但这些结果不是参数与 BN 的可加因果分解。**

独立回执级复核通过，限 Drone、seed42、原 N42 与旧 2048 图/3轮 N-ft 的两份固定 EMA。没有重跑 GPU、加载权重或读取新 direction AP。小复算见 [STATE_SWAP_RESULT_receipt.json](STATE_SWAP_RESULT_receipt.json)，脚本见 [review_state_swap.py](review_state_swap.py)。

|内存模型组成|mAP50–95 %|相对初始化 pp|AP50 %|AP75 %|
|---|---:|---:|---:|---:|
|初始参数 + 初始 buffers|54.513608|0.000000|77.179654|64.166566|
|FT 参数 + 初始 buffers（parameter_only）|53.245420|−1.268189|75.805512|62.122758|
|初始参数 + FT buffers（buffer_only）|54.251864|−0.261744|77.015863|63.510963|
|FT 参数 + FT buffers（完整旧 N-ft）|53.714083|−0.799526|76.442313|62.776442|

两种交换均保留同类同形状同 dtype 的全部注册状态：256 parameter tensors、243 buffer tensors。两份源 checkpoint 间 239 个参数 tensor、162 个 buffer tensor 有变化；发生变化的 buffers 全部是 BN running mean/variance，未发现变化的非 BN buffer。这里的“参数”包含卷积、检测头以及 **BN affine 参数**，不能解读为仅卷积权重。

parameter_only 使用 FT 参数和初始 buffers；buffer_only 使用初始参数和 FT buffers。composition 明确 raw 复制与统一 FP32 投影 exact、无 optimizer/梯度/checkpoint 写入。实际 evaluator 接收该内存模型，forward identity 回执为 true，两臂各 48 次 forward；不存在被原 checkpoint 重新加载覆盖后的假交换。来源 stat 与旧 N-ft 独立评价 checkpoint 闭合。两臂实际 native kwargs、完整 1469 图 roster/loader 序列与旧 native profile 一致；推理前及捕获后均为 22462 GT，五类 AP 宏均值和 fraction[0,1] 单位均复算通过。上表仅将原值乘 100，差值单位为 pp。

完整 FT 比 parameter_only 高 **0.468663 pp mAP**，说明 FT buffers 与 FT 参数组合后，表现优于把 FT 参数直接放回初始 buffers。两项单独替换的下降相加为 −1.529933 pp，实际完整 FT 为 −0.799526 pp；二阶数值残差为 **+0.730407 pp**。该残差描述非加性，不是可解释的独立收益、因果贡献比例或 BN 修复量。交换模型还可能打破共同适应关系；不能据此声称“参数导致了某百分比损伤”或“冻结 BN 一定恢复 AP”。

两次交换评价分别为 12.925 / 12.395 秒，单 GPU 4、1 CUDA 进程，回执峰值显存均 1370 MiB，RSS 4074 / 3963 MiB。资源和组成取自实际小回执；本审阅没有再次装载完整 tensor 核对。交换结果来源见 [parameter_only](state_swap_evidence/parameter_only/swap_evaluation_receipt.json) 与 [buffer_only](state_swap_evidence/buffer_only/swap_evaluation_receipt.json)；初始化原值来自 [已接受 native profile](../2026-09-07_train_IndependentKD实施/remote_admission_1532/evaluator_profile_attempt2/native_metrics.json)，完整旧 FT 原值来自 [N-ft 独立评价](../2026-09-08_ops_小时级筛选重构/snapshot_1016/evaluations/N/hourly_evaluation_receipt.json)。

新方向筛选的 `lr0=1e-4 / lrf=1 / warmup=0 / BN running statistics frozen / 3 epochs` 已在读取本次交换 AP 前固定。继续执行该冻结协议；这些结果只补充旧 FT 的诊断，不据此再修改学习率、门、剂量或预算。单 seed 内存交换不替代多 seed、匹配训练对照，也不升级为正式增益结论。
