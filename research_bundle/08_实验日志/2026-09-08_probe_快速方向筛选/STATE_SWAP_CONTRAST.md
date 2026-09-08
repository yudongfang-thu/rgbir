# 事前固定：参数与全部 registered buffers 交换

本诊断仅执行两次完整 dev 推理，不训练。共同来源为同协议完整 N42 last/EMA 与已完成 FT3 的 N-ft last/EMA，原始完整模型 AP 复用已接受结果：N42 mAP54.513608、N-ft53.714083（百分制），不重复评价。

原值来源：[FT3最终报告](../2026-09-08_ops_小时级筛选重构/FINAL_REPORT.md)、[正式N42原生评价](../2026-09-07_train_IndependentKD实施/snapshots/2026-09-07T165357.033246_0800/raw/N42/evaluation_val.json)。本文件不重新计算它们。

|变体|parameters|所有 registered buffers|其余模型结构/属性|
|---|---|---|---|
|PARAM_ONLY|N-ft|初始 N42|初始 N42|
|BUFFER_ONLY|初始 N42|N-ft|初始 N42|

在读取两个新混合模型 AP 前冻结。比较原始 checkpoint 参数/全部buffer的键、dtype、shape、layout；包括 nonpersistent registered buffers，不能仅用 state_dict 漏掉它们。混合后逐项验证来源与统一FP32转换。改动buffer逐键分类为BN running_mean/running_var/num_batches_tracked或其他；如果存在非BN改动，只能解释为全部buffer组，不能称纯BN效应。BN的可学习weight/bias属于parameters。

保持原pinned native full dev1469图/22462 GT、640/B32/workers4、FP32、conf.001/iou.7/max_det300等。原native profile只约束评价源码/数据/kwargs，混合模型另标BN_PARAMETER_SWAP_DIAGNOSTIC，不冒充训练完成端点。混合在CPU内存完成；YOLO.val调用与Autobackend均要求实际使用同一内存module，禁止静默重载checkpoint。交换在原生eval fuse之前完成，之后允许与原生评价相同的融合和缓存重建。禁梯度、optimizer和权重写出；仅保存小回执、源码及评价预测。

两次运行分别报告AP50/AP75/mAP和逐类原值。以 F(initial_params,initial_buffers)、F(FT_params,FT_buffers)、F(FT_params,initial_buffers)、F(initial_params,FT_buffers) 组成固定2×2状态对照。混合模型可诊断“替换哪组状态改变了当前推理结果”；参数与buffer存在相互作用，不能把两项单独差值相加当独立贡献，也不能仅凭恢复量断言训练退化唯一由BN造成。若buffer改动仅BN统计，仍需区分BN统计置换与完整训练因果机制。

root使用现有globallease各评估预约2048MiB显存/8192MiB RSS，基于相同结构FP32评价的既有实测。不新增初始/FT完整模型评价，不扫BN层或重训，不访问test，不计算新hash。脚本CLI与新scope见evaluate_state_swap.py；仅准备和CPU真值检查，GPU由root统一执行。
