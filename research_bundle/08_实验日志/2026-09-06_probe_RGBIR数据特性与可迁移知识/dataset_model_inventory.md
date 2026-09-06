# RGBIR 数据与 baseline 身份盘点（2026-09-06）

> 结论：DroneVehicle、LLVIP、VEDAI 均已有同数据划分、同架构的两模态 seed42 baseline 可用于本轮特征与误差诊断。VEDAI 必须使用 official_fold01 的 RGB/NIR 配对模型；paper80 教师训练见过 official 验证集中的 95/121 张，不能混用。FLIR-aligned、M3FD 暂无核实的项目 baseline 与冻结开发集；KAIST 尚未解压。M3FD 附带 meta 列表不是有效检测划分。

## 范围与证据等级

- 94 服务器只读 CPU 元数据核查：训练 args、数据 YAML、训练 CSV、checkpoint 存在性与大小、数据目录及治理 receipt。
- 未调用 torch.load、未使用 GPU、未训练、未更改服务器数据/配置/模型。下列模型结构身份由 args 中的 `yolo11n.pt` 支持，实际加载后的模块身份由主 probe 再确认。
- 本表数值均为单 seed 训练 CSV 的最后一行，**仅用于描述 baseline 和定位资源**；不是新方法增益结论，也不替代独立评估或 accepted analyzer。
- 检索 RGBT_campaign/runs 全部 87 份 args，没有 FLIR/M3FD/KAIST data 路径。结论范围限于现行项目可追溯训练记录，不能证明整台机器任何角落均无相关权重。
- 原始小产物文本、路径、SHA256 和逐模型记录见同目录 `dataset_model_inventory.json`。可复采脚本：`collect_dataset_model_inventory.py`。

## 可以直接使用的模型

表中两个模态均采用 last.pt、seed42、YOLO11n、imgsz640、SGD、200 epochs 已完成。AP 是原 CSV 的 val 口径，单位为百分点。

| 数据集/划分 | RGB/visible AP50 / mAP50–95 | IR AP50 / mAP50–95 | 描述性发现与限制 |
|---|---:|---:|---|
| DroneVehicle hbb_v1，val 1469 对 | 75.994 / 53.798 | 80.860 / 59.659 | IR 的单模态 AP 较高，但两模态 GT 数不同；不能把 5.861 点差直接当作可蒸馏空间 |
| LLVIP grouped_v1，dev 2406 对 | 71.629 / 32.867 | 92.611 / 48.833 | 此模型对 IR 优势明显；仍须检查 RGB 有观测且 teacher 正确的共同目标 |
| VEDAI512 paper8，official_fold01，val 121 对 | 63.110 / 37.018 | 61.362 / 34.874 | 当前 baseline 对是 RGB 更强，不能默认 IR 是强教师；样本很少，且 val/test 是同一列表 |

### DroneVehicle

RGB：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/rgb_seed42_native_b32a2/weights/last.pt`

IR：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt`

数据 YAML：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/rgb.data.yaml`
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/infrared.data.yaml`

输入与标签：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/data/processed/dronevehicle/yolo/hbb_v1/{rgb,infrared}/images/val/`
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/data/processed/dronevehicle/yolo/hbb_v1/{rgb,infrared}/labels/val/`

类别顺序：`car, freight car, truck, bus, van`。原始 840×712 图像已去除边框，裁为 640×512；polygon 经裁剪后取 HBB 包络，不是原始 OBB 检测协议。治理 receipt 中 val RGB 22462 个框、IR 24490 个框。因此图像配对、对象对应、标签完整性是三个不同问题。

用于与 adapted_v2/CGA-W1 比较的另一个 RGB native：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/cgkd_w1/native_rgb_s42_e200/weights/last.pt`

其 args 仍为 DroneVehicle RGB，最后 CSV 与 formal_native s42 对应指标一致。该模型也已记录，主 probe 若研究 native→KD 效应，应读取具体训练配置和独立 eval，不能只凭目录名判断匹配。

### LLVIP

RGB：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/llvip/visible_seed42_native_b32a2/weights/last.pt`

IR：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/llvip/infrared_seed42_native_b32a2/weights/last.pt`

数据 YAML：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbt_p3_causal_v1/prepared/llvip/{visible,infrared}.data.yaml`

输入与标签：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/data/processed/llvip/yolo/grouped_v1/{visible,infrared}/images/dev/`
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/data/processed/llvip/yolo/grouped_v1/{visible,infrared}/labels/dev/`

唯一类别 `person`。grouped_v1 以文件前两位为整组划分：fit 9619 对、dev 2406 对（前缀 01/04/07/12/25）；dev 每模态 7879 个框。官方 test 3463 对仅在既有治理 receipt 中记录，本盘点没有读取其图像或推理。不要把 official train 12025 对训练的权重混入当前 grouped fit→dev 诊断。

### VEDAI：只配 official_fold01

RGB：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_cmdistill_paper_reconstructed_v2/native_rgb_s42_b64_e200/weights/last.pt`

IR：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_cmdistill_paper_reconstructed_v2/teacher_ir_s42_b64_e200/weights/last.pt`

数据根：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/cmdistill_native/data/processed/VEDAI512_paper8_hbb_official_fold01/`

数据根下：`configs/vedai512_{rgb,ir}_hbb.yaml`；`images/{rgb,ir}/val/`；`labels/{rgb,ir}/val/`。类别顺序：`car, pickup, camper, truck, other, tractor, boat, van`。模型两侧都是 batch64、200ep、相同 fold。数据 YAML 将 `test` 指向 `val`，故该 121 张端点没有独立封存测试含义。

以下教师排除出本 probe：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/cmdistill_vedai_native/teacher_ir_y11_s42/weights/last.pt`

它实际使用 `VEDAI512_paper8_hbb_paper80_seed0`，CSV 260/计划300 epoch，末行验证指标为零；不应从文件名 `teacher_ir_y11_s42` 认定其可用性。

| 对照 | paper80 | official_fold01 |
|---|---:|---:|
| train 图像数 | 997 | 1089 |
| val 图像数 | 249 | 121 |

交集按图像 stem 实算（详见 JSON）：paper80 train ∩ official val = **95**，paper80 val ∩ official val = **26**。误配 paper80 教师会让 official 验证集大部分样本成为教师训练样本。

**模态身份：VEDAI 的 IR 是近红外 NIR，不能直接等同热红外。** [原数据官网](https://downloads.greyc.fr/vedai/)给出数据来源与基准协议；[Cross-modal contrastive learning-based object detection under incomplete modalities](https://www.tandfonline.com/doi/pdf/10.1080/10095020.2026.2633014)的 VEDAI 数据段明确描述三通道 RGB 加一通道近红外。原始数据论文的 HAL 全文端点此次被访问保护阻拦，官网本身没有详细波段参数。当前 NIR 说明不据此推导温度对比、夜间热显著性等机制。

## 尚不具备本轮 baseline 特征诊断条件的数据集

### FLIR-aligned

数据根：`94:/mnt/dataset/yudongfang/datasets/FLIR_aligned/x/`；已解压 `visible/`、`thermal/`、`coco_annotations/`。

- `train.json`：4129 图、32256 annotations，4 类 car/person/bicycle/dog。
- `test.json`：1013 图、8604 annotations，同 4 类。
- `train_new.json` / `test_new.json`：图像数相同，移除 dog 注释后分别为 32161 / 8591 个 annotations，3 类。
- 附带 README 明确解释 `_new` 的类别差异；这不是可互换的文件名。
- 未找到现行 RGBT 项目的两模态已训 baseline、冻结 train→dev 划分和对应治理 receipt。若后续纳入，需要先冻结3/4类口径，并从 train 设计不泄漏的开发划分，保留原 test。

### M3FD

根：`94:/mnt/dataset/yudongfang/datasets/M3FD/`。

- `labels/` 4200 个 txt，`Annotation/` 4200 个 XML，4200 个 stem 同时在 `vi/`、`ir/` 中存在。
- `vi/`、`ir/` 各 4242 图，其中42张没有该检测标签；另有 `Vis/`、`Ir/` 各300图。大小写和数据子集不能混淆。
- **`meta/train.txt`、`meta/val.txt`、`meta/pred.txt` 完全相同，均42行 `001.png` 至 `042.png`。它们不是可用于检测泛化评估的划分。** 当前元数据与多份已解压 zip 共存，不能从 `meta` 名字假设其属于4200对检测集。
- 无核实的本项目两模态 baseline 与冻结开发集。可以先做4200个带标签配对的 CPU 身份/配准抽查；训练前需另行冻结检测划分并排查相邻帧泄漏。

### KAIST

根：`94:/mnt/dataset/yudongfang/datasets/KAIST/`。

2026-09-06 01:33 UTC 快照只有 `download_and_verify.log` 与 `kaist-cvpr15.tarq2r8gmvi.part`（35,684,089,856 bytes）；没有解压目录。下载状态与早期“4.3GB”记录已不同，但不能把仍为 `.part` 的文件当作已完成数据。没有可核实的 baseline 或开发数据可用于本轮特征图。

## 给特征 probe 的直接约束

1. 模型、输入、类别、split 必须成套。VEDAI 官方 fold 的 RGB/IR 同对模型已经可用，不再使用历史 P2 错误 native。
2. 跨模态 AP 差只能作为错误诊断入口。DroneVehicle 两侧标签不同，需同时匹配共同 GT 与分析各侧未匹配 GT。
3. 相似度、热力图、标签偏移都只支持表示/标注层面的观察，不能单独证明可蒸馏增益。
4. VEDAI 当前模态强弱与 Drone/LLVIP 不同，是检验方向假设的价值所在；不能统一令 IR 当教师。
5. FLIR/M3FD/KAIST 当前缺口明确保留。不得用其他数据集模型生成图后称作这些数据集的已训练 baseline 证据。
