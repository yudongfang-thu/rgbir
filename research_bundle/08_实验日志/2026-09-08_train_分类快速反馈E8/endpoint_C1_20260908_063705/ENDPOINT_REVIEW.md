# C1 E8 端点与三臂共同口径限定核对

**PASS：可报告 Drone、seed42、独立 E8 日程的 C1 原值；三臂具备同一短程日程与 native 评估口径下的描述性比较条件。** 2026-09-08，loc_stress。复用 C0 小回执核对，补查 C1 factory/批准来源与三臂共同字段；未重新评估、GPU、权重加载或新 hash，未计算任何臂间差值。

- C1 完成 8 epochs、4504 批 = 8×563，train population 17990。2667 次成功 optimizer 更新 + 7 次 AMP 跳步 = 2674 次尝试，EMA 更新 2674。计数在实际回执与既有包装器语义下闭合，并非重新记录全部 optimizer 调用。
- 实际配置与启动冻结 C1 配置完全一致；原通用 `SpaceNet6_OTD_official_reproduction/artifacts/int8_cross_modal_stage2_v1/weights/yolo11n.pt` 初始化路径在配置、args、训练回执一致，`pretrained=true`、`resume=false`。分类系数 **0.09227393550836771**，定位系数 0。48 条日志的 `loss_unweighted × actual_B × coefficient` 与 weighted KD exact，按原 FP32 乘法/加法复算的 total loss 也 exact。
- candidate 为 `criterion_factory`，训练回执、实际 admission、启动 admission 中的 source/approved_source_copy 相同。实际 source manifest 记录 `selected_only_v1.py` 14,574 bytes、byte_identity=true；启动保存的该源码与本地已审 candidate 逐字节一致。已审入口会在训练前比较 candidate 与 approved copy，并共同绑定 selector/loss。此处沿用来源回执，没有重新获取远端源码。
- **薄路径计数范围：** 48 条累计日志均 thin_path_used=true、无 fallback、完整统计已采集。最后记录到第 4500 批，thin/full/fallback=4500/48/0；最终训练回执未存这些计数，不能写成 4504/48/0。该缺失不等同最后 4 批异常，也不补造字段。
- 训练/独立评估 checkpoint 的路径、10,753,555 bytes、mtime_ns=1788819030007123690 一致，均指向本 C1 的 `weights/last.pt`，身份 `SHORT_SCREEN_E8_LAST_EMA`。这是 stat 核对，未加载权重。
- dev canonical roster 与已接受 native contract/binding 一致，实际 loader 为同一无重复 1469 图集合；推理前及 capture 后 GT 均 22462。五类顺序 car / freight car / truck / bus / van。实际 kwargs 为 imgsz640、batch32、workers4、conf0.001、IoU0.7、max_det300、rect=true；half/augment/single_cls/agnostic_nms=false、quantize=null，Torch2.10.0+cu128、Ultralytics8.4.115。这里核对 population 与实际设置，不替代逐图 GT 字节审计。

独立评估单位为 [0,1] fraction；百分数原值为 **mAP50–95 43.40811、AP50 64.71252、AP75 50.14584**。五类 AP50/AP75 均值与主值 exact，mAP50–95 均值仅差 5.55e-17。AP 来源仅为独立评估 receipt。CSV 表头/前 7 轮 15 列、末轮 8 列的既有局限保留，不能从末行按表头读取 AP。

训练/评估 queue 均 COMPLETED、exit0、monitor_errors 为空。资源采样 1261/8 条，整卡最低剩余 8295/14661 MiB；任务 VRAM 峰值 7726/1370 MiB、RSS 峰值 28837/4036 MiB，均低于预约。采样项目 RSS 峰值 172881/147945 MiB，以 `MiB × 2**20 ≤ 300000000000 bytes` 检查，未超过十进制 300 GB。资源结论限已有回执和采样。

## 三臂共同口径

三臂配置仅存在预定 arm/method_id/分类系数和 C1 candidate/审阅元数据差异，数据、通用初始化、T/R、增强、SGD、B32/nbs64/workers4、seed42、8 epochs/8 LR horizon、warmup3 均相同。实际 args 仅 name/save_dir 不同。三份实际评估 contract **全字典一致**，端点、fraction 单位、native binding 与 population 一致。

来源回执共有 68 条训练源、17 条评估源，其 path/bytes/mtime_ns/byte_identity 一致。非共有项仅各臂 config/admission，及 C1 增加的 selected-only 源码；学习 branch 按原 N/C0 与 C1 factory 配置区分。此处是源 stat/复制回执范围，不表述为本次远端全源码逐字节复取，也不保证三臂所有训练像素或全轨迹一致。

小核对 26 项通过，见 [endpoint_review_receipt.json](endpoint_review_receipt.json) 和 [review_receipts.py](review_receipts.py)。三臂可以交由受限分析器计算同 seed E8 的原值与百分点差，不产生跨 seed SD、显著性或正式方法增益；不证明 E200 等价、最终收敛或全部研究冲刺完成。
