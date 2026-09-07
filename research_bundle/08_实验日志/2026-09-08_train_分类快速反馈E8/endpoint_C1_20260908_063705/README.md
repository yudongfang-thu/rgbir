# C1／seed42 E8 完整端点

**C1 完成固定 8 轮及完整 dev 独立评价，mAP50–95=43.408107；26 项限定回执核对通过。三臂汇总见 [阶段报告](../three_arm_summary_20260908/README.md)。**

## 设置、完成与评估

Drone train17990/dev1469；YOLO11n 原通用初始化、seed42、640、batch32/nbs64/workers4、SGD/AMP、原增强与数据流；独立 8 轮学习率日程、warmup3。实际分类系数为 0.09227393550836771，定位系数为 0；使用已通过真实更新一致性验证的 selected-only criterion_factory。

训练 4504 批，2667 次成功 optimizer 更新、7 次 AMP skip，2674 次 attempt/EMA；实际训练入口耗时 4391.556851 秒（73.192614 分钟）。固定 E8 last/EMA 在完整 dev 1469 图、22462 个 GT 上独立评价，耗时 14.738947 秒。未访问封存 test。

|指标|百分制原值|
|---|---:|
|mAP50–95|43.408107|
|AP50|64.712517|
|AP75|50.145839|
|Precision|65.220361|
|Recall|64.067837|

AP 唯一来源是 `evaluations/C1/short_evaluation_receipt.json`；不用禁用训练内评价时的 CSV 零值或末行错位列作为 AP。原始 CSV 保留，前两列可用于已完成 epoch 的计时。

## 核对范围与局限

[独立核对](ENDPOINT_REVIEW.md)覆盖实际系数、48 条加权 KD 与 FP32 total 日志、训练计数、checkpoint stat、已批准 candidate、配置与实际 evaluator 合同、评估总体及逐类均值、源副本和资源。三臂 actual args 仅运行名称与保存目录差异，评估合同一致；共同源文件 stat 相同，预期 candidate 和分支配置差异单列。

训练实测 NVML 峰值 7726MiB、进程树 RSS 28837MiB；评价 1370/4036MiB。复用原 global lease，资源核对无越界。累计日志证明截至 4500 批 thin/full/fallback=4500/48/0，最后 4 批未另存该统计，不能将其写成已观测 4504/48/0。

这是单 seed 的独立 E8 端点，不是 E200 前缀、正式三 seed 结果或完整训练轨迹等价保证。它不能单独决定 C1 最终效用，也不能代替 C1_y 内容控制、固定阈值对象错误诊断或四臂跨模态归因。

原始服务器路径、已收小文件及保留远端的 checkpoint/预测见 `collection_receipt.json`；复核脚本与机器可读回执为 `review_receipts.py`、`endpoint_review_receipt.json`。本次未重新训练或评价、下载权重、新计算 hash。
