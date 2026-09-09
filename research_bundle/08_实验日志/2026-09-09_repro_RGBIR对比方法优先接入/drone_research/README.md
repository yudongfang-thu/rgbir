# DroneVehicle 作者实现审计与 90 接入顺序

结论（2026-09-09）：本轮核验到的最完整、最容易先做作者权重评估的源实现是 **M²D-LIF（ICCV 2025）**；完整作者源包已下载到本目录。它在推理时使用 RGB+IR，并输出 OBB，必须单列。优先方向中的 CMDistill、CCLKD、CMKD-Net 暂未核验到可下载的作者训练代码及权重；现有本地 CMDistill/CCLKD 适配代码不能改称完整原论文复现。

本轮仅做本地文献读取、公开作者源审计和小体积下载；未 SSH、未训练、未装环境、未执行下载的源代码、未计算新 hash。以下接入步骤是交付给主任务的技术路径，尚未执行。

## 1. 五个候选：可用性与协议

| 候选 | 作者代码/权重实际状态 | 输入、框与 split | Backbone / recipe 与 90 判断 |
|---|---|---|---|
| **CMDistill，JSTARS 2025**，DOI 10.1109/JSTARS.2024.3479717 | 本地有论文与明确标注 non-official 的 CMDistill-style 工程；未找到可核验作者 repo/权重。不能将本地 YOLO11 移植当成 YOLOv5s 原实现。 | 训练 IR teacher→RGB student，推理 RGB-only；YOLOv5 四坐标 HBB 检测。论文评估 DroneVehicle，但本轮未找到可直接执行的精确 Drone split 清单/标签转换及完整 epoch 设置。 | YOLOv5s；640、SGD lr .01 / momentum .937 / wd 5e-4、B64；论文使用 4090。对当前 RGB-only 问题最直接，但现在只能做 protocol-adapted，不是最快作者复现。[论文 DOI](https://doi.org/10.1109/JSTARS.2024.3479717) |
| **CCLKD，GIS 2026**，DOI 10.1080/10095020.2026.2633014 | 论文及本地 LLD+CCL 部分适配已有；未核验到作者软件 repo/权重。出版社有补充 ZIP，文件名含 sources/figures，未检查内容，不能断言全球无代码。 | IR→RGB，推理 RGB-only；YOLO HBB 口径，原始 OBB→HBB 转换仍缺执行证据。Drone train 17,990 / val 1,469 / test 8,980。 | 主要 YOLOv5，另有 YOLOv11 系列；Drone 512²、200 ep、SGD、B16、lr .01、m .937、wd 5e-4、常规增强+MixUp，Tesla V100。完整方法含 LLD/FLD/RLD/CCL；本地只 LLD+CCL，不是完整作者方法。[出版社](https://www.tandfonline.com/doi/abs/10.1080/10095020.2026.2633014) |
| **CMKD-Net，TCSVT 2026**，DOI 10.1109/TCSVT.2026.3670458 | 本地完整论文，无文内作者代码 URL；本轮未找到对应作者可运行包及权重。与 Hnewa CMKD / 自动驾驶 3D CMKD 不是同一工作。 | 融合 teacher→单 RGB 或单 IR student；**OBB**，VOC2007 mAP50；Drone 17,990 / 1,469 / 8,980，原图 640×512。 | MMRotate，双分支 LSKNet teacher 12 ep（8/11 衰减）；student RetinaNet/FCOS/ATSS/LSKNet，36 ep、AdamW lr 1e-4、wd .05、B4（24/33 衰减）。含作者 RARoIPooling 等未获得源实现，不适合先在 10h 内重建全方法。[论文 DOI](https://doi.org/10.1109/TCSVT.2026.3670458) |
| **M²D-LIF，ICCV 2025**，**首接入** | 官方完整 repo 可访问；本地已下载完整分支 ZIP：**4,638,044 B、1,251 文件**，包内无 .pt/.pth。README 发布百度权重/标签链接；链接存在，下载内容与权重身份尚未核验。 | 训练及推理均 **RGB+IR（6 通道）**；Drone **OBB**；17,990 / 1,469 / 8,980。官方测试数据目录内部仍命名 `images/val`，不能按目录名推断 split。 | 作者自带 Ultralytics 8.1.44 fork；双分支 YOLOv8/C2f+LIF、OBB head，源码默认 scale=m。发布入口：100 ep、B8、640、AMP=false、lr .001、CWD .8；论文 SGD lr .01、m .937、wd 5e-4，二者差异需留档。先拿 checkpoint 做评估，比先训练两个 teacher 更直接。[作者库](https://github.com/Zhao-Tian-yi/M2D-LIF)，[官方论文](https://arxiv.org/html/2503.11780v2) |
| **C2Former，TGRS 2024**，备用 | 官方完整 MMRotate fork 与配置可访问；仓库含 `pretrain_weights/resnet50-2stream.pth` 的文件入口，只证明预训练骨干存在，未下载/加载。README 未核验到 Drone 最终权重（有 KAIST 结果链接）。 | 训练及推理 RGB+IR；**OBB le135**、旋转 IoU .5；定制 matched 数据集。图像 `id.jpg` + `id_tir.jpg`，标签 `*_tir.txt`，多边形8坐标+类别+difficulty。官方 train_total/val_total/test_total 路径没有提供可核对的成员清单。 | Two_Stream_S2ANet + C2Former 双 ResNet50 + FPN；resize(512,640)，B1、workers2；SGD .001/.9/1e-4；文件虽名 1x，继承 schedule_2x，实际 **24 ep**，16/22 衰减。mmcv 1.5.3–1.8.0、mmdet 2.25.1≤v<3.0.0，需要旧栈独立环境及编译算子；首接入成本高于 M²D。[作者库](https://github.com/yuanmaoxun/C2Former) |

本表没有混列论文 mAP：HBB/OBB、RGB-only/双模态、原始/对齐标签、val/test、模型规模均不同，直接数值排名无意义。

## 2. 已交付的真源包与证据

- `M2D-LIF_master_author.zip`：来自 `https://codeload.github.com/Zhao-Tian-yi/M2D-LIF/zip/refs/heads/master`，原 ZIP 保留；只读中央目录，未解压/执行。`AUTHOR_ARCHIVE_RECEIPT.json` 记录名称、字节数及文件清单，不生成 hash。分支快照按下载日期留档，**尚未确认 commit ID**（GitHub API 请求返回 403）。
- `public_sources/M2D-LIF/`：实际下载的 README、environment.yaml、Drone YAML、训练/验证入口、模型 YAML、OBB trainer/validator、dataset.py。此处是便于审查的选择性快照，单独不能替代完整 ZIP。
- `public_sources/C2Former/`：README、主模型配置和数据配置；另外三份 raw 下载失败，网页工具仍可读取。逐文件 URL、HTTP 状态、错误和字节数见 `RAW_FETCH_RECEIPT.json`。
- `local_papers/`：三篇用户已有 PDF 的文本提取及原路径回执。未把提取文本视为代码实现证据。

M²D 作者权重：[百度网盘](https://pan.baidu.com/s/1GKDkfhJrKeskrnDNRzmFXw?pwd=vmvr)；标签：[百度网盘](https://pan.baidu.com/s/1NBDTPIgdlvnARiO_Uy8c-w?pwd=ir9b)。本轮仅核验 README 发布这些 URL，网页工具未取得网盘文件列表/文件，不得记为权重已下载。

## 3. M²D-LIF 在 90 的最短实际接入步骤

目标 root 是 `/mnt/dataX/ydf/projects/RGBT_campaign_90`；以下路径是**建议的新隔离位置**，本轮没有建立远端目录。

1. 将现成 ZIP 复制到 `ROOT/reproductions/M2D-LIF/source/`，在独立目录解出作者 fork。保留原 ZIP，所有路径与运行适配另存副本。当前 `environments/rgbir90` 的 torch 2.10 / stock ultralytics 8.4.115 不能当作作者已匹配环境；先用隔离进程从作者目录 CPU 导入并输出 `ultralytics.__file__`。若兼容性失败，再按作者 `environment.yaml` 单独建环境到 dataX，不改现有环境。作者 pin 为 Python 3.10.6、torch 2.1.0、torchvision .16.0、CUDA 12.1、NumPy 1.24.4；环境文件中的用户 prefix 必须移除/改至 dataX。
2. 优先绑定 90 已有 `DroneVehicle_M2D` OBB 组织（主任务另行确认绝对路径和内容），而非拿正在准备的 HBB `rgb.data.yaml` 直接替代。接口要求 `images/{train,val}`、同名 `images_ir/{train,val}`、`labels/{train,val}`；类别顺序 **car, truck, feright_car, bus, van**。作者把 freight 拼成 feright，id 顺序仍须保持。检查每个标签类别+8多边形坐标、配对及 split 数量；不得凭文件夹名声称与作者网盘标签相同。
3. 先取得作者 `checkpoint/multimodal/DroneVehicle.pt`，记录取得日期、URL、文件大小、模型 names、OBB head、6通道和训练 args；无新 hash。没有 fusion 权重时，原样验证就绪条件尚未满足。90 的 generic `yolo11n.pt` 不是该模型的复现权重。
4. 作者 `val_obb.py` 只需模型、数据和 device 路径适配，核心为 `OBBValidator(args={model:..., data:..., imgsz:640, batch:1, rect:True})`；物理卡由已有唯一 lease 调度器绑定，进程使用逻辑 device=0，不能继承原 device=5。输出全部设为 ROOT/runs 下独立实验目录。**当前封存 test 的限制仍有效，作者测试集 8,980 对正式评估不在本轮授权执行范围，也不是入队下一步。**本轮先仅做 train/已治理 dev 的接入检查；若作者训练集包含当前 dev，作者 checkpoint 在该 dev 上仅做机械检查，不能作为泛化指标或方法裁决。仅记录未来协议接口：作者测试配置的 `path` 指向 `DroneVehicle_test`，其 `val` 字段仍为 `images/val`；现在不执行它。
5. **无新 hash 约束仍要处理**：作者 `ultralytics/data/dataset.py:get_labels` 无论 `cache=False` 都会调用 `get_hash` 校验/生成标签缓存。不能直接启动验证规避此约束，也不能伪造 hash。若主任务接入，需要保留原文件，在独立适配副本明确禁用标签缓存读写及 hash，改用当次内存标签解析，并记录此唯一数据管道改动。本轮未改、未运行。
6. 权重验证跑通后再决定是否排入原论文训练。开始训练前必须处理下述已证实的问题，不需要新增用户许可门：原 wrapper 同时把 RGB 和 IR teacher 指向 `dv_ir.pt`；正确 RGB teacher 的身份目前未知。短资源 canary 仅测显存/速度来检验 10h 预算，不将 FT3 或短步结果重新用于方法方向裁决，也不承诺当前 3090 上全训练必然 10h 内结束。

作者的训练原入口是 `python train_dist_obb.py`，但**原样不应作为当前生产命令**。作者验证原入口是 `python val_obb.py`；上面 1–5 的数据、权重、环境、GPU 和无 hash 适配完成后才形成可执行的复现调用。

### 已证实的代码细节

- `train_dist_obb.py` 第 15–19 行：两次 `attempt_load_one_weight('./checkpoint/dv_ir.pt')`；不能默认为正确双教师。
- `ultralytics/nn/tasks.py:parse_model` 未指定 scale 时明确选 `m`，不是 stock 默认 n；模型 YAML 的 `nc:3` 会被 Drone 数据 `nc:5` 覆盖，构造函数 `ch=3` 也会被 YAML `ch:6` 覆盖，这两处不构成 bug。[作者模型构建](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/ultralytics/nn/tasks.py)
- `ultralytics/data/base.py` 对路径做 `images`→`images_ir` 替换并按作者次序合并图像；保留作者通道处理，不自行改成 RGB+灰度4通道。[作者 loader](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/ultralytics/data/base.py)
- 作者训练入口 lr0=.001 与论文 .01 不同。先读最终 checkpoint args 和发行配置确定具体实验，再标记是作者 checkpoint 重评估、发布入口训练或改配训练，不能静默选一个值后宣称严格复现。[训练入口](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/train_dist_obb.py)

## 4. C2Former 可执行备用入口

作者 README：

```text
python tools/train.py configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py --work-dir work_dirs/C2Former
python tools/test.py configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py CHECKPOINT --out RESULT.pkl
```

接入时仅改独立 config 副本的 `data_root`、`img_prefix`、标注目录及 `load_from`；work-dir 指 ROOT/runs。需先补齐对应 Drone checkpoint 或按作者24ep训练，复用 matched 的 TIR 标签定义；不能把当前普通 HBB 数据/YOLO11 权重直接灌入。有效日程和标签格式见[作者日程](https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/configs/_base_/schedules/schedule_2x.py)、[作者数据类](https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/mmrotate/datasets/dronevehicle.py)、[依赖边界](https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/mmrotate/__init__.py)。

## 5. 未入选条目与本地证据边界

- **Infrared-Privileged UAV Detection via Cross-Modal Vector-Quantization（AAAI 2026）**：训练用 RGB+IR、测试 RGB-only，方向很贴近；[AAAI 正文入口](https://ojs.aaai.org/index.php/AAAI/article/view/37692) 给出 `github.com/chenbys/InfraredPrivilegedUAV`。本轮网页与本地 HTTP 实测均 404，因此暂不作为可跑代码来源；不是对论文方法的否定。
- **Hnewa/Rahimpour CMKD**：本地确有作者代码审计，Faster R-CNN R50-FPN、Seeing Through Fog、RGB+三张 gated slices teacher→RGB student；不是 Drone RGB-thermal，也不是本表 CMKD-Net。既有下载回执存在失效链接/缺训练清单，不能拿仓库同名冒充当前候选的官方实现。
- 本地 `06_历史工程_只读/LADD_public/comparison/cmdistill/README.md` 明确写 non-official CMDistill-style / paper-aligned；本地 `08_实验日志/2026-09-06_ops_OEv1优先级与对比实验/cclkd/README.md` 明确为 CCLKD-adapted partial，3seed 200ep 完成也不能补足尚未实现的 FLD/RLD。
- 审计按“本地 PDF/历史实验→出版社/作者论文→作者 repo/具体源码”顺序完成。搜索未发现是有范围的否定：未检查全部补充材料，也未联系作者；没有将搜索结果页或 ResearchGate 的后来论文摘录当作作者代码链接。

## 产物与局限

本目录保存报告、结构化候选、HTTP 回执、原源 ZIP 与选择性源文件。主任务负责实验总目录 README/索引登记；本子任务没有修改原工程、任何已有 README/index 或远端内容。

现在可以移交的真实进展是“作者完整 M²D-LIF 源包已取得且关键入口已读”，不是“作者权重已取得”或“90 已跑通”。最短未完成链路为：作者 fusion checkpoint → 数据类别/OBB/split 核对 → 作者 fork CPU 接入与无 hash 适配 → 受 lease 管理的短资源检查 → train/已治理 dev 机械接入检查。封存 test 不加入本轮队列；若作者训练覆盖当前 dev，不升级泛化结论。
