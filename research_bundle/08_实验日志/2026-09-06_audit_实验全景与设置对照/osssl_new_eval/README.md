# OS-SSL 两个新增独立 last 评估（2026-09-06 22:28）

> 首个同 seed 独立评估差为 paired123 − shuffled123：+0.708405 mAP50–95 / +0.533024 AP50 个百分点；正面观察从训练 CSV 得到独立 last 口径的支持，但仅一个微调 seed，不能判定三 seed 配对归因或迁移门通过。

## 目的

核对 22:22–22:23 新出现的两份 `metrics_record.json`，确定权重、数据角色、臂身份与历史 native 背景差，并保留可复算小证据。本次仅 SSH 读取文件，不运行推理、不使用 GPU、不修改训练和队列。

## 设置与来源核对

- 数据集 DroneVehicle，学生输入 RGB，5 类检测，开发 `val`。三份独立评估记录均指向同一 `rgb.data.yaml`，明确 `split=val`，权重均为各自 `weights/last.pt`，不是 stdout 中的 best。
- paired 与 shuffled 都是 detection finetune seed123。`args.yaml` 模型分别指向 `clean_paired.pt` / `clean_shuffled.pt`；训练 completion receipt 虽仍写 `native_weight0` / `CGA-KD-W1`，其中 config 分别为 `protocol_clean_paired.yaml` / `protocol_clean_shuffled.yaml`。真实身份由模型和 config 交叉确定，不能按复用的 receipt arm 名认成 native。
- 两臂微调共同设置：E200、640、batch32、workers8、SGD、lr0=0.01、lrf=0.01、momentum=0.937、weight_decay=0.0005、warmup=3、AMP 和 deterministic 开启；mosaic/mixup=0，translate=0.1、scale=0.5、水平翻转=0.5。三个比较对象 CSV 均有完整 epoch200。
- paired/shuffled 指标文件 mtime 分别是北京时间 22:22:42 / 22:23:29；本次采集 22:28:24。当前 val 图像目录中 1469 张、1469 个唯一文件路径；生成的文件清单明确是采集时目录枚举，并非评估执行时回执。
- 当前 `eval_rgbt_detector.py` 强制 existing last.pt，禁止 test，调用 `eval_yolo_detector.evaluate`，再调用 `YOLO(...).val(...)`。脚本默认 imgsz640/batch32/workers8。指标记录没有记录完整 CLI、实际 batch/device 或代码快照，因此不把这些默认值伪称为执行时完整验证；当前脚本仅作实现背景。

## 结果

指标为百分数，差值为百分点。全部为同 seed123 独立 last 记录：

| 臂 | mAP50–95 | AP50 | AP75 |
|---|---:|---:|---:|
| paired SSL | 53.932152 | 75.856702 | 63.083680 |
| shuffled SSL | 53.223747 | 75.323679 | 62.740825 |
| 历史 W1 native（初始化混杂） | 53.687433 | 75.730143 | 63.306324 |
| paired − shuffled | **+0.708405** | **+0.533024** | **+0.342855** |
| paired − 历史 native（仅背景） | +0.244719 | +0.126560 | −0.222644 |
| shuffled − 历史 native（仅背景） | −0.463686 | −0.406464 | −0.565499 |

与 21:46 训练 CSV 口径核对：

| 来源 | paired123 mAP / AP50 | shuffled123 mAP / AP50 | P−S mAP / AP50 |
|---|---:|---:|---:|
| E200 CSV | 53.942 / 75.849 | 53.273 / 75.354 | +0.669 / +0.495 |
| 独立 last | 53.932152 / 75.856702 | 53.223747 / 75.323679 | +0.708405 / +0.533024 |

两种口径方向相同，数值不同。CSV 中 +0.495 不能四舍五入成通过 +0.5 门；独立 last 的 +0.533024 超过该数值门槛，也不能替代三 seed 和全部对照条件。

截至 22:28，`sar_only_rgb_s42_e200`（实际 IR-only SSL→RGB 检测）仍没有独立 `metrics_record.json`；此前 CSV mAP54.623/AP5076.822 保留 CSV 标签，不填入本表。

## 结论与边界

1. 新结果强化了“本次预训练产物下，paired 在 seed123 微调优于 shuffled”的有限观察。仍只有一次各臂 SSL 预训练与一个完整同 seed 比较，不计算三 seed mean±SD，不宣布跨模态配对作用已确立。
2. 历史 native 检测头初始化不匹配的证据仍成立，参见[17:30 初始化审计](../../2026-09-06_audit_RGBIR晚间进度与新结果/osssl/README.md)。本次未重复读取大权重，也未重新测量该混杂。P−旧N 的 +0.245 mAP 不能称作 SSL 净收益。
3. RGB 目标检测仍缺 RGB-only 自模态 SSL 和同模板零 SSL native。IR-only 是辅助模态单模态预训练，不是目标 RGB 自模态对照。
4. 同次服务器 README 新记录运行偏差 D3：这两次短评估在 GPU2 绕过资源守卫执行（记录称此前四租约满载、手工检查显存余量15.5GB）。这是已有运行的审计事实，本次未执行该操作。需保留偏差，不能把本轮端点写成资源守卫全流程验证通过；此项本身不直接证明数值有误。
5. 两个指标文件具有明确 last/data/seed/arm，已由原字节采集与 CPU 复算核对；其回执完整度仍低于 OEv1 的运行时 roster/receipt 证据链，不升级为 accepted analyzer 结论。

## 产物与采集记录

- 原始文件：[raw/](raw)；完整复算：[summary.json](summary.json)；来源、mtime、字节数：[source_inventory.json](source_inventory.json)。所有文件采集期间 size/mtime 均未改变。权重只记录服务器路径、大小与时间，不下载。
- 采集器：[collect_remote.py](collect_remote.py)；SSH 封装和复算：[fetch_and_analyze.py](fetch_and_analyze.py)。服务器原始 run 位于 `/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/osssl_ir_20260906/`。
- 首次本地分析误以为 data val 字段是清单文件，实际是目录，因此在缺 `val_roster.txt` 处报错。第一次来源清单保留为 `source_inventory_attempt1.json`；已改为对当前目录只读枚举。该错误发生在审计脚本，不是训练或评估失败。
