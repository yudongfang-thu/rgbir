# CCLKD 已有端点与独立评估准备（2026-09-06 23:11）

> DroneVehicle 三个学生 seed 的 200 轮 CCLKD 适配版已训练完成，均有标准 YOLO11n last 权重；补三次独立验证即可利用已有训练成本。但这些端点只有 LLD+CCL，不能作为完整论文 CCLKD 的复现结果。

> **23:21更新：三次独立评估已完成并通过证据检查。** mAP54.297±0.216；相对历史native描述性+0.344±0.551pp，2/3seed正。完整结果、回执和边界见 [RESULTS.md](RESULTS.md)，以下保留23:11启动前盘点。

## 目的与执行范围

核对是否能以最小计算代价把历史 CCLKD 作为 OEv1 的参考方法。此次子任务只执行 94 CPU 文件读取与 checkpoint ZIP/pickle 静态检查，没有模型反序列化、没有 GPU、没有远端写入。启动与最终评估由主任务负责。

## 已有实验设置

| 项目 | 实际训练证据 |
|---|---|
| 方法身份 | `CCLKD-adapted partial / PROTOCOL-ADAPTED`；receipt 原文 `CCLKD-adapted (GIS 2026); not an exact reproduction` |
| 臂 | `cclkd_literal`；只启用 `LLD` 和 `CCL`，各权重 1 |
| 禁止误称 | FLD / RLD 没有接入此轮正式训练，不能写成完整 CCLKD |
| 学生 | RGB YOLO11n，五类，通用 `yolo11n.pt` 初始化 |
| 教师 | 冻结 IR YOLO11n baseline，seed42 |
| 数据 | DroneVehicle RGB train17990 / val1469，HBB五类 |
| 跨模态输入 | 当前 RGB 与正确配对 IR，几何变换同步 |
| 标注 | 学生使用原生 RGB GT；蒸馏阶段未直接使用 IR GT；IR 教师自身来自有监督训练 |
| 训练 | seed0/42/123，各200轮、640、batch32、nbs64、workers8 |
| 优化 | SGD，lr0=.01，lrf=.01，momentum=.937，weight_decay=.0005，warmup3 |
| 增强 | translate=.1、scale=.5、fliplr=.5；mosaic/mixup/HSV等关闭 |
| 初始化证据 | v2日志：80类转5类，按类别名 remap3/5，加载451/499张量；与旧 native 的主要初始化路径一致 |
| 验证 | 训练设置 val=False，最终CSV仍有epoch200验证数值；统一比较应另做last评估 |
| 环境 | Ultralytics8.4.115、PyTorch2.10.0+cu128、Python3.10.20 |

准备回执中 `teacher_labels_used_by_kd=false` 与 loader、criterion 读取方式一致。注意：OEv1 在蒸馏中使用真实 IR GT 对应与正确性判断，因此 OEv1 对这一历史方法的比较存在辅助标注使用差异，需要在表格中明列，不能把全部差值归因为对象证据设计。

## 三个端点

公共根目录：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_cclkd_adapted_v1/`。

| seed | run目录 | last大小 | last修改时间 | CSV末行mAP / AP50（百分制，仅背景） |
|---|---|---:|---|---:|
| 0 | `cclkd_drone_seed0_b32_e200` | 5,487,258字节 | 09-03 16:50:47 | 54.061 / 76.488 |
| 42 | `cclkd_drone_seed42_b32_e200` | 5,487,258字节 | 09-03 14:33:41 | 54.491 / 77.023 |
| 123 | `cclkd_drone_seed123_b32_e200` | 5,487,258字节 | 09-03 16:26:02 | 54.323 / 76.298 |

每个目录有 `completion_receipt.json`，状态 `completed`，200行训练CSV及 `weights/last.pt`。三个 last 的 pickle globals 只含标准 Ultralytics/torch 模型模块，不含 `CCLKDCriterion`、`__main__`、教师或自定义训练 wrapper，可以用普通 `YOLO(...).val(...)` 读取。此次静态检查不等于已经执行过 GPU 评估。

截至 23:11，三个 run 内均没有独立 eval/metric 文件；RGBT_campaign 与 SpaceNet6 artifacts 中按 CCLKD 路径检索，也没有独立 eval/metrics/summary JSON。完整范围见 `inventory.json`。

## 当前配置与历史实现的限制

当前 `configs/research/rgbt_cclkd_protocol_drone.yaml` 已经变成 **512、batch16、mosaic1**，不能拿它给这三个旧端点填写实际训练设置。实际设置应来自各run的 `args.yaml` 与 v2训练日志，两者均显示640、batch32、mosaic0。

当前 `rgbt_cclkd_kd.py` 含 v3 的 logit-domain 修订，旧正式日志名为 v2。此次保存的是“当前可见源码”，没有找到与三个历史训练执行逐字绑定的源码快照，不能把当前v3源码自动追认成当时实现。历史receipt明确的 LLD+CCL身份、参数和训练完成状态仍可记录。

末期训练日志中 LLD 约2e-5，CCL约0.693。它们提示应复查损失梯度与历史实现，但仅凭损失标量不能断言没有梯度，也不能把端点评估成败外推至论文完整方法。

## 建议的最小独立评估

1. 不重新训练；固定三个已存在的 last。依次在同一 GPU 做完整1469张RGB验证集评估。
2. 使用原run `args.yaml` 中的数据路径、640输入；评估batch32、workers4可与现有OEv1独立评估保持一致。固定 split=val，不访问test。
3. 输出写入新 `runs/.../cclkd_partial_s{seed}_last_eval_attempt1/`，不在历史run中覆盖原始文件。
4. 输出 `evaluation_val.json`，记录权重路径、seed、`fixed_budget_last_ema`、单位0到1、`CCLKD-adapted-LLD-CCL-v2-historical`、辅助标注使用差异。
5. 通过 `emit_bound_run_receipt` 记录当次评估源码/配置/命令/版本/完整val roster/资源峰值/指标，方法身份 `PROTOCOL-ADAPTED`。明确当次receipt不补写成历史训练源码完整回执。

现有普通入口：

```text
/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/tools/eval_yolo_detector.py
  --checkpoint <run>/weights/last.pt
  --data /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/rgb.data.yaml
  --output <new-output>/evaluation_val.json
  --split val --imgsz 640 --batch 32 --workers 4 --device 0
```

该入口本身不写受lease绑定的完整回执。更合适的是复用 OEv1 `release_v2/evaluate_object_evidence.py` 的评估与receipt结构，改为接收独立输出目录、历史status=`completed`、从run args读取协议，并移除 OEv1 特定身份。

资源入口：

```text
<python> <repo>/tools/project_resource_guard.py run
  --job-id cclkd_partial_s42_last_eval_attempt1
  --kind eval --candidate-gpu <动态选择的物理GPU>
  --expected-vram-mib <canary实测后预约值>
  --expected-rss-mib <预约值> --free-safety-mib 2048
  -- <python> <具备receipt的评估脚本> ...
```

guard自动设置 `CUDA_VISIBLE_DEVICES`，评估脚本使用逻辑device0。不得用未实测的理论显存值加入他人正在使用的卡。可在释放后的空卡做小 canary，然后按实测峰值预约完整评估。主任务正在统筹OEv1与OS-SSL资源；本子任务没有预约或启动。

运行时 helper 所属项目规范为 `/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/AGENTS.md`；RGBT_campaign内未发现AGENTS。用户当前工作区AGENTS是服务器资源限制的最新明确授权，仍保留guard、receipt与原始证据保护。

## 本地证据

- `inventory.json`：三个端点、200行CSV、checkpoint静态模块清单和检索范围。
- `source_manifest.json`：26个小文件来源与修改时间；没有哈希。
- `raw/RGBT_campaign/runs/...`：三个run的args、receipt、CSV。
- `raw/log_excerpts/`：v2训练日志开头和末尾，不复制几十MB逐batch日志。
- `raw/SpaceNet6_OTD_official_reproduction/`：当前源码、当前配置、guard、receipt工具与AGENTS。
- `inspect_remote.py` / `fetch_inventory.py`：此次CPU只读检查与证据下载脚本。
