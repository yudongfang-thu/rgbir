# 可执行原模型交接：LLVIP 作者 YOLOv5l 基线

首选为 **LLVIP 数据集作者发布的 YOLOv5l 单模态检测权重复评**。作者模型包 Google Drive 端点已实际返回二进制，主任务随后在 90 完成下载、解包和完整作者仓库取得。AMFD 的 LLVIP 云盘没有现成 checkpoint，当前不把 KAIST checkpoint 误用为 LLVIP。此目录只有源码子集、小元信息和执行包装；本子任务没有 SSH、GPU、模型前向、安装、训练或新摘要计算。

## 可部署资产与证据

- [LLVIP 作者仓库](https://github.com/bupt-ai-cz/LLVIP)，运行源码子目录 `yolov5/`。主任务报告 90 完整 clone revision：`c1a655cce437ebfd990a97b04fc48fbb99f4c47b`；这是读取已有 Git 引用，不是本子任务新算摘要。
- [作者模型下载页](https://drive.google.com/file/d/1SPbr0PDiItape602-g-bstkX0P7NZo0q/view)。实际 [下载端点](https://drive.usercontent.google.com/download?id=1SPbr0PDiItape602-g-bstkX0P7NZo0q&export=download&authuser=0&confirm=t) 返回 HTTP 200、`application/octet-stream`、173,666,266 字节、文件名 `yolov5_trained_model.rar`。本地仅读取 128 字节并验证 RAR5 魔数，没有下载大权重；见 `LLVIP_author_weight_endpoint_check2.json`。
- 主任务报告已在 90 解出 `yolov5_visible.pt`、`yolov5_infrared.pt`，各 93,884,963 字节。文件名给出模态线索，仍先以 CPU identity 读取实际 model class、names、yaml、stride、参数数和 checkpoint 选择字段；此文不宣称前向已经通过。

90 路径（以下 `ROOT=/mnt/dataX/ydf/projects/RGBT_campaign_90`）：

| 资产 | 路径 |
|---|---|
| 完整作者源码 | `ROOT/external_reproductions/llvip_author_baseline/author_source/yolov5` |
| 两个权重 | `ROOT/external_reproductions/llvip_author_baseline/extracted_attempt1/yolov5_trained_model/yolov5_{visible,infrared}.pt` |
| 独立原协议数据视图 | `ROOT/data_author_protocol/llvip_baseline_attempt1/{visible,infrared}/data.yaml` |
| 官方清单 | 同独立视图下的 `test_roster.json`，包装要求 JSON 含 `stems` 数组 |
| 既有隔离 Python | `ROOT/environments/rgbir90/bin/python` |

本地 `LLVIP_author_source/` **不是完整仓库**，只有 6 个成功取得的模型/工具源码。raw 逐文件下载遇网络超时后已停止；官方 codeload 包超过 30 MB 的预设读取上限后终止且未存包。正式执行使用 90 完整 clone，不能使用此 6 文件子集拼造原实现。`AMFD_source/` 是 48 个选取的 Python/配置/文本文件；不包含 MMDetection 基座或作者媒体。

## 原论文目标与必须保留的版本区别

[LLVIP 原论文 v1 第 5.2 节和 Table 3](https://arxiv.org/html/2108.10831v1#S5.T3)给出的 YOLOv5l 结果如下，单位为百分数：

| 单独输入 | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|
| visible | 90.8 | 51.9 | 50.0 |
| infrared | 94.6 | 72.2 | 61.9 |

论文正文描述早期 16,836 对数据与 70/30 划分；[作者 README](https://github.com/bupt-ai-cz/LLVIP#yolov5)说明后来移除低质量图片，发布 15,488 对版本并重训/复评。现有完整数据为 official train 12,025、test 3,463，不能声称与早期论文历史划分逐字节相同。当前主评采用独立视图中的 **官方 previous annotations**，test 7,931 GT；2023-02-21 更新版为 8,302 GT，不能混用。

身份保持 **`PAPER-RECONSTRUCTED`**，另用描述字段说明“作者公开单模态基线权重”。上述论文数作为原论文参考端点；结果应同时报告数据发布版本差异，不能直接升级 AUTHOR-EXACT。作者 README 的后续更新结果位于嵌入图片，网页工具本轮未成功取到该图片，故本交接没有猜读更新表数值。

## 原评估入口与冻结值

入口是作者 [`yolov5/val.py:run`](https://github.com/bupt-ai-cz/LLVIP/blob/main/yolov5/val.py)，权重由作者 `models.experimental.attempt_load` 加载，单模态 3 通道 YOLOv5l；精确模型参数数由随后 CPU 身份回执给出。作者 README 指定 `--img 1280`。其原命令多写一个悬空 `--data`，需要删除这一语法错误；不改变评价参数。

| 项目 | 本候选原公开入口 | 与 CFT 的区别 |
|---|---|---|
| 输入 | 一个模态，1280，rect=True/pad=0.5 | CFT RGB+IR 两流，1024 |
| eval batch | CLI 默认 32 | CFT CLI 默认 64 |
| precision | CLI 未传 `--half`，FP32 | CFT 原入口 FP16 |
| conf/NMS | 0.001 / 0.6 | CFT NMS 0.5 |
| GT 匹配 | 原 `process_batch`，`IoU >= threshold`，按 IoU 去重 | CFT 原匹配为严格大于，不能替换 |
| AP | 原 `ap_per_class/compute_ap`；AP75 取列 5 | 当前本源 recall 末端补 1.0；CFT 补 recall[-1]+0.01，不能互换 |

`val.run` 函数的默认 `half=True` 与 CLI `--half` 默认 False 不同，包装显式传 `half=False`。其余冻结为无 TTA、`single_cls=False`、`save_hybrid=False`、不走 COCO JSON 分支。保存预测 TXT 和原 AP 输入，关闭绘图只减少输出。三个 AP 从作者原返回的完整 AP 数组捕获，不以 CFT 函数或新版 Ultralytics 重新打分。

作者要求 Python>=3.6、torch>=1.7、torchvision>=0.8.1，另有 NumPy/OpenCV/Pillow/PyYAML/SciPy/matplotlib/pandas/seaborn/tqdm 等依赖；这些是宽范围下界，不是原运行环境的精确锁定。先复用已存在的 `rgbir90` 隔离环境做 CPU 身份检查，不自动执行 pip 安装。新版 torch 旧 pickle 加载与 NumPy int/float 别名适配均留痕；没有沿用 CFT 的 24 类重绑补丁。

## 小执行包装

`llvip_author_baseline.py` 支持以下顺序，当前仅通过 AST、模板 JSON 和 CLI `--help` 检查，尚无该包装的模型运行结果：

1. `identity`：强制 CUDA 不可见，在 CPU 读取一个原 checkpoint，保存 `identity.json`；不前向、不算 AP。分别处理 visible/infrared，不凭结果反推模态。
2. 填写 `llvip_author_protocol.template.json` 为具体冻结配置：对应权重、模态、已完成 identity 回执、独立 data.yaml、含 3,463 个 stem 的名单。其模板状态表示资产尚需具体化，不是额外用户批准门。
3. `canary`：必须先绑定既有 project resource lease；仅逻辑 GPU 0，由 lease 的 CUDA_VISIBLE_DEVICES 映射，allocator 上限 0.68。使用原 `val.run` 完成 **2×B32** 的预处理、原前向、原 NMS 和逐图匹配，然后在 AP 汇总前正常停止。记录框架峰值、lease 资源记录和两批耗时；不计算 AP。
4. `evaluate`：要求同权重、模态、数据和冻结参数的 CANARY_PASS 回执，再由主任务根据实测峰值完成资源准入，跑完整原入口。包装拒绝 NMS 超时；验证名单集合、3,463 图、7,931 GT，并保存完整原指标输入和三 AP。

GPU 模式会先调用既有 `require_bound_lease_from_environment`，不会创建新资源池。allocator 限额不是 NVML 总显存峰值，实际进程峰值仍以 canary/现行统一 guard 为依据。输出目录必须为新目录，不覆盖失败/成功 attempt。

作者 `select_device('0')` 会改写 CUDA_VISIBLE_DEVICES；包装已先确认 lease 只暴露一张卡，调用作者的 CLI 默认 `select_device('')` 分支，仍返回逻辑 cuda:0，同时保留调度器给定的物理映射。这项设备选择说明写入回执。

**禁止摘要的处理**：本源 `utils.datasets.get_hash` 确实调用 `hashlib.md5` 对总大小和路径字符串生成缓存摘要。包装在任何 dataloader 创建前，将这一缓存键函数明确替换为普通 `(策略名, ((path,size),...))` 元组；模型、加载图片、标签转换、预处理、匹配和 AP 函数不变。数据视图必须独立于 CFT，以免覆盖旧 cache。回执记录这项缓存元信息适配；不会一边内部算 MD5 一边声称 `new_hash=False`。

CPU 身份示例（实际路径填到 90 后执行；本子任务没有执行）：

```bash
python -B llvip_author_baseline.py identity \
  --source /mnt/dataX/ydf/projects/RGBT_campaign_90/external_reproductions/llvip_author_baseline/author_source/yolov5 \
  --weights /mnt/dataX/ydf/projects/RGBT_campaign_90/external_reproductions/llvip_author_baseline/extracted_attempt1/yolov5_trained_model/yolov5_visible.pt \
  --output /mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/llvip_baseline_visible_identity_attempt1
```

在已有资源调度器包装内执行 `python -B llvip_author_baseline.py canary --source ... --protocol ... --output ...`。通过后给协议增加已完成的 `canary_receipt` 路径，再以同样 lease 路径调用 `evaluate`。固定输入尺寸、模型、精度、阈值；若技术资源失败，保留失败记录，不依据 AP 改协议。

## AMFD 的确切核查结果

[AMFD 作者仓库](https://github.com/bigD233/AMFD)公开 LLVIP R50 双流教师与 R18 单流学生配置，但“单流”仍含 RGB+IR，不能当 RGB-only 推理。官方 Google Drive 的 Models 目录本轮实际只有 KAIST 的两个模型；LLVIP 子目录实际只有 `anno`。已解析的小云盘目录原件与回执保留。

| 公开文件 | Google Drive ID | 元信息大小 |
|---|---|---:|
| KAIST student `singlel_fasterrcnn_7_23.pth` | `1zAUqX3C7POtt_lbttyfxP7X4u-VJb78p` | 362,375,129 B |
| KAIST teacher `Teacher_Fasterrcnn_7_66.pth` | `1NmAzrY9w-Jp0DOWyahmsYu1Fa5qWVkQV` | 524,887,069 B |
| LLVIP `test_annotations.json` | `1Tu1UjOrwVhAXO7JQYgePAnOCVqD2yFfn` | 3,189,213 B |
| LLVIP `train_annotations.json` | `1zDLk4T98RWj9We5W5K8PGxnIC9pAiYBJ` | 12,814,287 B |

AMFD 需 Python3.9.1/torch1.12.1+cu116/torchvision0.13.1/MMCV2.0.1/MMDetection3.1.0/MMEngine0.8.4。LLVIP 配置原评为(1333,800) keep-ratio、B4、CocoMetric；蒸馏配置为30,000 iterations，尚缺该数据集 teacher/student权重。若转原训练，还需独立闭合基座环境、作者标注、教师初始化和实测时间；当前不为此延迟已取得的 LLVIP 原模型资产，也没有再访问任何受限 Baidu 链接。

本交接不产生新 AP、训练成功、跨方法增益或三 seed 结论；后续可执行身份由主任务 CPU 读取与 canary 回执推进。
