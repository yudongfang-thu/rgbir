# N E8 端点限定回执交叉核对

**PASS：这份端点可作为 Drone、seed42、独立 E8 日程的 N 基线原值报告。** 2026-09-08，loc_stress。沿用已接受的 pinned native 评估口径，只读本目录实际小回执并做 CPU 算术核对；没有 GPU、重新评估、权重加载或新 hash，不新增科学验收。

- 训练回执完成 8 epochs，runtime 为 563 批/轮，合计 4504 批。2667 次成功 optimizer 更新 + 7 次 AMP 跳步 = 2674 次尝试，EMA 更新为 2674。计数与既有包装器语义一致；本次未重新记录全部 optimizer 调用。CSV 的 epoch/time 前两列覆盖 1–8 轮。
- 实际配置与启动冻结配置字典完全一致；原通用 `SpaceNet6_OTD_official_reproduction/artifacts/int8_cross_modal_stage2_v1/weights/yolo11n.pt` 初始化路径在配置、args、训练回执一致，`resume=false`。分类/定位实际系数均 0；48 条正常频率 KD 日志均 `kd_weight=0`、weighted KD=0、total=native，批次恰为 1/2/3 和每 100 批至 4500。配置中沿用的 `kd_weight: 0.1` 元数据不是本 N 臂实际学习系数。
- 训练/独立评估的 checkpoint 路径、10,753,555 bytes、mtime_ns=1788812556756139776 完全一致，均指向本 E8 run 的 `weights/last.pt`，端点身份 `SHORT_SCREEN_E8_LAST_EMA`。未加载 checkpoint，此处是 stat 交叉核对，不声称本地验证权重内容相同。
- 实际 contract 的 canonical roster 与本地 roster、已接受 native contract/binding 均一致，实际 loader roster 是同一无重复 1469 图集合；rect 可改变遍历顺序。推理前与 capture 后 GT 均 22462。类映射依次 car / freight car / truck / bus / van。此处核对 population，不替代逐图 GT 字节审计。
- 实际 kwargs 与已接受口径完全一致：imgsz640、batch32、workers4、conf0.001、IoU0.7、max_det300、rect=true，half/augment/single_cls/agnostic_nms=false，quantize=null；实际身份 Torch2.10.0+cu128、Ultralytics8.4.115。训练/评估 source manifest 的 70/18 条来源字节一致字段均为 true；没有在本次重取远端源码。

独立评估回执单位为 [0,1] fraction；换成百分数为 **mAP50–95 43.37654、AP50 64.98349、AP75 50.30199**。五类 AP50/AP75 的算术均值与主值 exact，mAP50–95 均值与主值仅差 5.55e-17。指标源是 `evaluations/N/short_evaluation_receipt.json`，不是 TIDE oracle。

训练/评估 queue 两阶段均 COMPLETED、exit0、monitor_errors 为空。资源采样分别 653/7 条，整卡最低剩余 3112/14661 MiB；任务峰值 VRAM 7630/1370 MiB、RSS 28957/4052 MiB，低于各自预约。采样项目总 RSS 峰值 173130/147951 MiB，未超过 300 GiB。以上为回执及采样范围，未重新监控。

**原始日志局限保留：** 训练 CSV 表头及前 7 轮各 15 列，末轮仅 8 列，最后三个值是 LR，不能按旧表头解释为 AP；训练内禁用评价的零指标也不是结果。末行短列不改变这里独立评估来源，不据此重评或停后续臂。

小复算见 [endpoint_review_receipt.json](endpoint_review_receipt.json)，源码 [review_receipts.py](review_receipts.py)。17 项限定交叉核对均通过。仅支持这一单 seed E8 N 原值；不支持方法增益、E200 等价、最终收敛或完整三臂/三 seed 完成判断。
