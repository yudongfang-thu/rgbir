# SpaceNet6-OTD 可审计复现报告（执行中）

> 截止：2026-08-21  
> 主机：gp94（`yudongfang`）；LADD90（后续矩阵）  
> 状态：R0 身份门控失败；teacher 两协议完成；R1 四格三种子全部终端验收（Baseline 46.2 / AKD 45.2 / OS-SSL 52.9 / Full 52.6，AKD 无正效应）；R4 因子诊断部分完成；R2 paper300 的当前终态/暂停状态待 receipt 审计；新优先级为三数据集 YOLO OS-SSL 四臂判别计划（`docs/YOLO_OS_SSL_MULTIDATASET_PLAN.md`）；2026-08-21 09:44 用户确认 gp94 有 4 张空卡

## 1. 复现范围与结论上限

本工作从公开论文、官方仓库、官方权重和 SpaceNet-6 原始数据出发，重建
Baseline、AKD、OS-SSL 与 OS-SSL+AKD。作者未公开 TIFF→PNG 脚本、OS-SSL
自定义 dataset、完整切块规则和随机种子，因此结论上限是：

> 在冻结数据与公开实现约束下的 paper-faithful reimplementation。

不得声称 bit-exact 作者复现。R0 已证明公开最终权重与当前能从公开证据恢复出的输入字节不完全匹配。

## 2. 权威代码、环境与服务器

- 官方 SN6-OTD commit：`4a8deadf7256180b565290156d72b33ce99bf03b`。
- OpenSelfSup commit：`1db69ec`。
- 远端规范根：
  `/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction`。
- Python 3.10.20、PyTorch 2.0.1+cu118、torchvision 0.15.2、MMCV-full 1.7.2、
  MMDetection 2.19、NumPy 1.23.5。
- RTX 4090 CUDA ops、真实 optimizer transition、checkpoint save/load/resume、COCO
  evaluator、双卡 NCCL/SyncBN、BYOL EMA 与梯度累积均已通过真实 CUDA mini-e2e。

兼容补丁仅处理依赖/API/加载兼容性，不改变模型、损失、优化器或指标定义。

## 3. 数据身份与存储

用户自有真实副本（不是指向其他用户目录的唯一软连接）：

| 数据 | 文件数 | 字节数 |
|---|---:|---:|
| SAR-Intensity TIFF | 3,401 | 44,096,706,206 |
| PS-RGB TIFF | 3,401 | 8,279,217,548 |

检测划分以官方 JSON 为准：622 train + 200 test。论文写 620+200，是公开文字与
JSON 的差异。OS-SSL 严格排除 200 test，共 3,201 对：622 detection-train + 2,579
no-tank。后者仅提供无标签背景/场景表征，不是监督检测中的“无油罐类别”。

主线 v1 SAR 规则：`[HH,VV,HV]`、`rint(x*255/92.88)`、clip 到 uint8；EO 为
RGB TIFF→PNG 像素不变。文件清单 SHA-256 为
`e58d946ea472f79d77fb0fc11cb7a7ae9d30b2512de92932f1beb17b5aa6b2c5`，OS-SSL pair manifest SHA-256 为
`98a8b981fe8f7efb8cde223fed6a5d14fa6c2e18d1b3ee2938db6551754d76b4`。

## 4. 官方权重审计

| 权重 | 大小 | SHA-256 |
|---|---:|---|
| OS-SSL BYOL epoch 500 | 507,367,660 B | `ccc99446505953cf26978027a5d490309289cb0d5d558ff958fca8231adceea1` |
| Full detector epoch 60 | 330,280,167 B | `531e0b94432f128ae69da39e22204bbe21cf6104cdaa439308d138d371fc1770` |

OS-SSL checkpoint metadata 表明实际发布训练为 500 epochs、有增强、每卡 48、
累积 32、LARS，而论文写 300 epochs、无增强。两者已拆成独立协议，禁止混写。

## 5. R0 官方最终权重门控

论文 Full 目标 AP50=56.2%，冻结通过区间为 `[55.7%,56.7%]`。官方 checkpoint
加载和评测链完整通过，但所有预注册输入身份均未通过：

| 输入身份 | AP50 |
|---|---:|
| 主线 `[HH,VV,HV]`、固定 92.88、rint | 37.6% |
| `[HV,VV,HH]` | 29.7% |
| orientation=1 单独旋转 | 0.0% |
| raw truncate | 9.4% |
| 固定 92.88 truncate | 37.7% |
| 逐图逐通道 min-max | 41.7% |
| `[HH,VV,HV]` p0.05/p99.95 | 50.6% |
| 逐图 joint min-max | 40.7% |
| SatShip 256/92 | 25.3% |
| 外部 strip statistics 最佳 | 41.9% |
| 作者项目图 `[HH,VV,VH]`、固定 92.88 | 38.2% |
| 作者项目图 `[HH,VV,VH]`、p0.05/p99.95 | **52.0%** |

作者项目页图像与 3,401 张原始 SAR 做唯一匹配后，RGB 与原始四波段的相关矩阵明确支持
`[HH,VV,VH]`，而论文图注写 `[HH,VV,HV]`。该新证据将最佳 AP50 提高到 52.0%，
但仍距目标 4.2 点。因此 R0 结论为：

> `FAIL_RELEASED_CHECKPOINT_INPUT_IDENTITY_UNRESOLVED`

公开资料仍缺少作者生成 PNG 的精确字节规则，或 checkpoint 与论文结果的对应证明。

## 6. OS-SSL 重建与预检

已恢复缺失的 paired SAR/EO datasource 和 dataset，保持上游 BYOL forward、loss、EMA
与 LARS 不变。冻结的非作者确认切块为 900×900 上 5×4 全覆盖：

- patch 256×256；
- x=`[0,161,322,483,644]`；
- y=`[0,215,429,644]`；
- 20 patch/source pair，共 64,020 对。

已通过：

- 1-pair、20-patch 单卡 epoch1 + resume epoch2；
- 双卡 DDP、NCCL、SyncBN、distributed sampler、梯度累积、EMA；
- 981 个 tensor 在恢复后继续更新且所有浮点 tensor 有限；
- 4090 单卡 micro-batch 16 的一次真实更新，reserved VRAM 约 4,390 MiB；
- 四卡×16=global micro-batch 64 的资源可行性。

完整 64,020 对 patch 已于 06:39:39 完成物化并通过独立终端验收：SAR/EO 共
128,040 个 PNG 的 SHA-256 与 PIL 完整性全部通过，全部几何、文件名和训练列表严格一致，
32 个确定性源 pair 的 640 对裁剪逐像素重算通过。terminal receipt SHA-256 为
`da0a5a496cf52f1db4f2c5bb152ff38ce6cb7aa461f2f0e51f8a3e70301305a8`。

## 7. R3 修正版诊断对照

R3 数据规则为 `[HH,VV,VH]`，逐图逐通道 positive p0.05 / all-pixel p99.95，
clip 后 truncate。它是依据已访问 R0c 结果保留的诊断分支，不是无偏确认性实验。

已物化并独立全量验收：

| 身份 | 数量 |
|---|---:|
| detection train SAR | 622 |
| immutable reference test | 200 |
| no-tank OS-SSL SAR | 2,579 |
| SAR manifest | 3,401 |
| OS-SSL pairs | 3,201 |

终端 receipt SHA-256：
`2a48429d381256d7d037b4f84615a1ce028f9f6b0f2aa3c114111894034440da`。

## 8. 当前执行矩阵

| 阶段 | 单元 | 当前状态 |
|---|---|---|
| Teacher | optical Faster R-CNN 24e / public 36e | 已完成冻结终态 |
| R1 | Baseline / AKD / released OS-SSL / Full，三种子 | 已完成并终端验收 |
| R4 | AKD 因子诊断 | 部分完成；未开始单元不再优先启动 |
| R2 | paper300 seed168120232 | 不再假定运行中；等待 process、exit、checkpoint 与 receipt 审计 |
| R2 | 其余 paper300/released500 五个预训练 | deferred，不启动 |
| YOLO OS-SSL | 三数据集 Native / SAR-only / shuffled / paired | 新核心矩阵，尚未授权启动 |
| Faster R-CNN OS-SSL | SiXiang、OGSOD E60 四臂 | YOLO 提前终态后的可选项 |

2026-08-21 09:44 用户确认 gp94 有 4 张空卡。允许一张卡运行多个任务，也允许在显存安全时与他人共享；不抢占或停止他人任务。该资源快照不代表任何旧任务已经 COMPLETE，也不替代启动前的进程与显存复核。

## 9. 当前边界与下一步

1. 冻结 YOLO OS-SSL 尚缺的 optimizer、LR、EMA、projector、归一化、增强和 backbone key map；
2. 将旧 R2/R3/R4 machine-readable queue 调整为 deferred/paused，避免旧调度器继续启动已降级实验；
3. 启动前复核 gp94 实时剩余显存，初始将四张空卡按三个 SSL arm 加一个 Native E60 装箱；显存允许时继续同卡加入任务，也允许与他人共享；
4. 完成三数据集真实 CUDA mini-E2E 后，按 `docs/YOLO_OS_SSL_MULTIDATASET_PLAN.md` 的优先级执行；
5. 所有科学结果仍需终态 receipt、哈希绑定和整组读取，不使用中间 AP 改协议。

最终报告将在获授权实验终端验收后补充 mAP、配对差值、GPUh、训练曲线、权重和日志哈希。
