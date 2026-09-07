# C0 E8 端点限定回执交叉核对

**PASS：可报告 Drone、seed42、独立 E8 日程的 C0 原值。** 2026-09-08，loc_stress。复用 N 端点已采用的小回执核对方法，只调整臂身份、实际加权 KD 核对及十进制 300 GB 资源上限。没有重新评估、GPU、权重/state 加载、新 hash 或新的科学分析器，不计算 C0−N。

- 训练完成 8 epochs；runtime 为 17990 train 图、563 批/轮，回执 4504 批 = 8×563。2667 次成功 optimizer 更新 + 7 次 AMP 跳步 = 2674 次尝试，EMA 更新 2674。计数在实际回执与既有包装器语义下闭合；没有重新记录所有 optimizer 调用。
- 实际配置与启动冻结 C0 配置字典完全一致。原通用 `SpaceNet6_OTD_official_reproduction/artifacts/int8_cross_modal_stage2_v1/weights/yolo11n.pt` 初始化路径在配置、args、训练回执一致，`pretrained=true`、`resume=false`。实际分类系数 0.1、定位系数 0。48 条日志恰为批次 1/2/3 和每 100 批至 4500，均为 paired、B=32、kd_weight=0.1、weighted KD>0；`loss_unweighted × B × 0.1` 与记录的 weighted KD exact，按原 FP32 乘法/加法复算的 total loss 也 exact。这里只核对已记录日志，不外推逐批独立梯度审计。
- 训练/独立评估 checkpoint 的路径、10,753,555 bytes、mtime_ns=1788814557908049694 一致，指向本 C0 E8 run 的 `weights/last.pt`，身份为 `SHORT_SCREEN_E8_LAST_EMA`。本次未加载 checkpoint，结论是 stat 一致。
- canonical dev roster 与本地 roster、已接受 native contract/binding 一致；实际 loader roster 是同一无重复 1469 图集合。推理前及 capture 后均 22462 GT，五类顺序为 car / freight car / truck / bus / van。此处核对 population，未开展逐图 GT 字节审计。
- 实际 native kwargs 与已接受口径一致：imgsz640、batch32、workers4、conf0.001、IoU0.7、max_det300、rect=true；half/augment/single_cls/agnostic_nms=false、quantize=null。实际身份为 Torch2.10.0+cu128、Ultralytics8.4.115；训练/评估 source manifest 的来源字节一致字段均为 true，本次未重新获取远端源码。

独立评估回执单位为 [0,1] fraction；百分数原值为 **mAP50–95 43.92058、AP50 65.07173、AP75 51.11326**。五类三个 AP 指标的算术均值均与主值 exact。AP 来源仅为 `evaluations/C0/short_evaluation_receipt.json`，不是训练 CSV 或 TIDE oracle。

训练/评估 queue 均 COMPLETED、exit0、monitor_errors 为空。分别 484/8 条资源采样，整卡最低剩余 8391/14661 MiB；任务 VRAM 峰值 7630/1370 MiB、RSS 峰值 28884/4027 MiB，均低于预约。采样项目总 RSS 峰值 173001/147855 MiB，已明确以 `MiB × 2**20 ≤ 300000000000 bytes` 检查，低于十进制 300 GB。资源结论限现有回执和采样范围。

原始日志局限保留：CSV 表头及前 7 行各 15 列，最后一行 8 列，其末三个值是 LR；不能按表头将它们当作 AP。epoch/time 前两列完整。该问题不改变独立评估来源，不触发重新评估或停止后续臂。

18 项小交叉核对均通过，见 [endpoint_review_receipt.json](endpoint_review_receipt.json) 和复用脚本 [review_receipts.py](review_receipts.py)。本结论仅支持单 seed E8 C0 原值，不支持方法增益、E200 等价、最终收敛或完整三臂/三 seed 完成判断。
