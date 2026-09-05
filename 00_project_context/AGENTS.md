# AGENTS.md — 工作区协作规范与服务器使用原则

> 适用于所有在本工作区工作的 AI agent 与人类协作者。结构总入口见 [README.md](README.md)。

## 0. 新会话启动检查单（防进度丢失）

1. 读 [README.md](README.md)（目录地图）→ [08_实验日志/README.md](08_实验日志/README.md)（**实验索引，最新条目在最上**）→ 最近 2-3 个实验条目的结论；
2. 需要证据背景时读 [05_实验证据_按服务器/实验证据对照总表.md](05_实验证据_按服务器/实验证据对照总表.md)；
3. 继续任何实验前，确认该实验的条目已存在或新建（见 §5）；
4. 会话内产生的**每一个新实验结论，落盘前不算完成**。

## 1. 服务器角色与连接

| 服务器 | 角色 | 连接 |
|---|---|---|
| **94**（10.103.12.94，gpuserver94，8×RTX 4090 24G） | **现行主力**（训练/评估/复现） | `ssh 94`（yudongfang，免密已配） |
| 90（10.103.12.90，inspur-NF5468M5，8×3090） | 次要；历史工程与数据 | `ssh 90`（ydf，免密已配） |
| L20 | 远端结果仓（canonical 协议期证据） | 仅远端，本地只有 manifest |

- **禁止在任何文件中写入服务器密码**；一律使用 ssh 别名 + 密钥。
- 凭据类文件一律放 `C:\Users\MSI-PC\guangsar_credentials\`（E:\SHARE 是 SMB 共享，不得存放凭据）。
- **TUN/虚拟网卡模式注意**（2026-09-05）：本机开 TUN 后默认路由被 198.18.0.1（metric 0）抢占；当前代理对私网直连放行故 `ssh 94/90` 正常（~0.5s）。若 SSH 断连：用管理员权限执行 `route add 10.103.12.94 mask 255.255.255.255 10.81.0.1 metric 1`（90/92 同理换 IP），强制走物理网卡（ASIX USB，网关 10.81.0.1）。

## 2. 94 服务器使用原则（强制）

1. **同时最多占用 3 张 GPU**（与其他同学共享显卡是合理常态，不要求独占空闲卡）。
2. **每张卡最多 3 个任务**。
3. **显存纪律**：加入他人正在使用的卡前先 `nvidia-smi` 查当前占用；**我们的任务加入后，卡上必须始终剩余 ≥ 2GB 显存余量**（即任务峰值 ≤ 启动时空闲显存 − 2GB）。起任务前用 canary/短步数实测显存峰值，禁止按理论值拍脑袋。
4. **全部任务总内存占用 ≤ 300GB**（机器 1TB，须给其他用户留余量；多进程 dataloader num_workers 要计提）。
5. 长任务一律 `screen`/`tmux`，会话命名 `<campaign>_<arm>_<seed>`；训练输出只写项目 runs 目录或 `/mnt/dataset/yudongfang`，**不得写系统盘**（系统盘仅剩 ~100G）。
6. 大文件传输/解压用 `setsid nohup ... < /dev/null > log 2>&1 &`，避免占用会话。

## 3. 90 服务器使用原则

- `/mnt/dataY` 已 **95% 满**，禁止写入大数据集/权重；只做小体积整理与轻量评估。
- 90 上的 ogsod400-research-core 等目录属 canonical 链路，动前先对照 [05_实验证据_按服务器/实验证据对照总表.md](05_实验证据_按服务器/实验证据对照总表.md)。

## 4. 研究规范（复现甄别）

背景：遥感检测小圈子普遍单 seed、不公开代码、事后编故事的风险高。我们反着来：

1. **先复现，再信任**：引用任何文献数字前，优先在 94 上复现其核心表；复现失败是正常且可写的结论（如实记录到实验证据对照总表，标 `NOT_REPRODUCED`）。
2. **多 seed 强制**：任何增益结论 ≥ 3 seeds（0/42/123），报告 mean±SD 与逐 seed 方向。
3. **四臂归因强制**：paired / shuffled / same-modal / weight0（或 native）缺一不可——没有归因的增益不进论文。
4. **outcome-blind**：先冻结 estimand/协议/对照组，再跑；禁止看结果后修改方法或阈值补救。
5. **自模态增强基线强制**：任何输入层跨模态增益必须打赢 sar/ir-self 增强臂（FreqMix 教训：shuffled≈native≈self，+9 AP 是增强不是跨模态）。
6. 结果口径遵循 `03_现行工程/ogsod400_clean_protocol` 的证据等级；无 accepted analyzer 不升级 claim。

## 5. 实验记录纪律（强制，防进度丢失）

1. **每个实验一个目录**：`08_实验日志/YYYY-MM-DD_<类型>_<短名>/`，类型 ∈ {probe, train, repro, audit, ops}。
2. **结论必须用 md 落盘**：目录内 README.md 按模板写（目的/设置/结果/结论/产物路径/局限与下一步），一句话结论放最前。
3. 目录内存：脚本副本 + 小体积关键产物副本（summary.json 等）；大文件（权重/数据集）只记服务器路径。
4. 完成后在 `08_实验日志/README.md` 索引表加一行（最新在最上）。
5. 服务器端同步建议：在 94 的对应项目 `artifacts/` 下留同名脚本与产物，本地目录记录其路径；两边以路径+日期互指。

## 6. 数据集登记（94，/mnt/dataset/yudongfang）

| 数据集 | 位置 | 状态 |
|---|---|---|
| LLVIP | `projects/RGBT_campaign/data/raw/LLVIP`（visible/infrared/Annotations，train+test 15,488 对） | ✅ 已审计（100% 配对，grouped split 政策见 governance） |
| DroneVehicle | `projects/RGBT_campaign/data/raw/DroneVehicle{,_clean}` | ✅ HBB 转换 receipt 在 governance |
| VEDAI | `projects/RGBT_campaign/cmdistill_native/data/processed/VEDAI512_paper8_hbb_paper80_seed0/`（paper8 协议处理版；原始 tar 在本地 06） | ✅ |
| M4-SAR | `datasets/M4-SAR/`（optical 8.0G + sar 14G，images/labels 就位，split 见 default.txt） | ✅ 解压完成（2026-09-05） |
| FLIR-aligned | `datasets/FLIR_aligned/x/`（coco_annotations/thermal/visible） | ✅ 解压完成 |
| M3FD | `datasets/M3FD/`（Annotation/ir/vi/labels/meta 已解出；4 个原始 zip 保留可后清理） | ✅ 解压完成 |
| KAIST | `datasets/KAIST/`（gdown 断点续传中，~4.3GB/.part，完成后自动 MD5 校验） | ⏳ 下载中 |

## 7. 结构变更纪律

- 本地/服务器目录结构改动必须留回执（本地 `99_整理回执/`，服务器 `PROJECTS_INDEX_*.md`）。
- 含 junction/硬链接的目录不得用穿透式递归删除。
- 原始 results、失败 attempt、receipt、checkpoint：不移动、不覆盖、不重命名（去重政策见 `04_方法演化档案/hub索引文档/DEDUPLICATION_POLICY.md`）。
