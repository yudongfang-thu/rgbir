# 光sar · 跨模态知识蒸馏工作区

> 目标：跨模态蒸馏方法研究，投稿 **J-STARS**。
> 结构重整：2026-09-05（回执见 [99_整理回执](../99_整理回执/README.md)）。
> 历史 canonical 工程及证据规范见 [03_现行工程/ogsod400_clean_protocol](../03_现行工程/ogsod400_clean_protocol/EXPERIMENT_AUDIT.md)；当前 RGB–IR 执行进度见下方最新条目。

> **最新综合审计（2026-09-05）**：[全项目复盘与研究诊断](../07_研究分析/全项目复盘与研究诊断_20260905.md)。已核对历史终态及 94 原始小产物：FreqMix“SiXiang +9 AP”为错误比较；P2 对照模型/CKA 与 CGA 实现存在待纠正问题；SpaceNet6 OS-SSL 有独立正面证据。下文旧交接状态和证据总表须结合该审计读取。

> **最新RGBIR诊断（2026-09-06）**：[数据特性与蒸馏方向诊断](../07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md) · [baseline特征图册](../08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/特征图册.md)。已完成六baseline、521对开发图像及局部配准图：Drone优先对象判别迁移，LLVIP有更明确定位机会，VEDAI为RGB–NIR且须核验教师方向；新native与KD主要recipe一致，旧目录名推断及过强门控结论以新复核为准。

## 目录地图

> **当前执行：独立分类与定位蒸馏（2026-09-07 15:35）**：[实施、准入与原始证据](../08_实验日志/2026-09-07_train_IndependentKD实施/README.md)。C1三seed已按42→0→123启动，λ=.09227394；六条真实兼容、64批校准、两canary及双开通过。N/C0/random九份E200独立端点已齐：C0−N +0.266655±0.144373 pp，C0−random +0.174939±0.038853 pp；旧shuffled42/same-modal42继续。定位24对尚无合格对应点证据，不等于数据集未配准或定位无效。旧CL/CGT排程已替代，以下旧进度仅作历史追溯。

> **Task-Conditional执行（2026-09-07 03:22）**：[冻结计划与执行入口](../08_实验日志/2026-09-07_train_TaskConditional首轮/README.md)。独立定位/GT算子、D1/D2、几何合同、校准工具及统一lease调度已部署94；N/C与两种C内容对照各24次成功更新通过，新旧C/N真实等价通过。严格C42−N42仅+0.144554pp，AP75−0.327243pp。几何接受范围内D2为0，CL/CGT不准入；按预设分支启动C-shuffled42，C-same-modal42等待资源。以下旧状态须结合此入口读取。

> **Task-Conditional规格审阅（2026-09-07）**：[DRAFT-v1总评](../08_实验日志/2026-09-07_audit_TaskConditional规格审阅/SPEC_REVIEW.md)。认可D1对象机会/D2实际anchor机会分开及原C冻结；附录14项CPU检查通过。新长训前仍需补几何误差范围、R/native owner关系及GT-DFL温度定义，建议同mask GT控制前置。此轮为审阅，未启动新定位训练。

> **定位分支可行性复核（2026-09-07）**：[判别与定位蒸馏可行性](../08_实验日志/2026-09-07_audit_判别与定位蒸馏可行性/README.md)。新CPU重算支持条件定位机会：Drone/LLVIP细化候选241/3083、117/672；但Drone原样复制IR框会使双方命中对象的相对IoU均值变为−0.04564。实例身份、目标坐标质量和学生可学性须分别判断；未检出不等于不能蒸馏。定位分支尚未实现/训练，原OEv1保持不变。

> **当前方法与单卡并发（2026-09-06 23:59）**：[计划沿革与并发调度](../08_实验日志/2026-09-06_ops_单卡并发与计划澄清/README.md)。主方法仍为OEv1，random用于机制验证；规则允许每卡多任务。实测短测通过后GPU2已R42+R0正式双开，R123短测通过、等待N42完整评估后接GPU4。N42/P0/N123进入175/65/58轮；物理4卡、5正式训练、3张空卡。以下23:25串行描述为历史状态。

> **当前研究优先级（2026-09-06 23:25）**：[OEv1优先级与对比实验](../08_实验日志/2026-09-06_ops_OEv1优先级与对比实验/README.md)。按用户目标收敛到OEv1；OS-SSL队列已停、既有产物保留，VEDAI暂缓。CCLKD partial三seed独立评估补齐，mAP54.297±.216；OEv1同剂量随机选择对照完成canary并在GPU2启动seed42，0/123串行。以下22:28快照仍保留历史意义，旧“完成OS-SSL矩阵”计划已被本条取代。

> **RGBIR最新进度（2026-09-06 22:26–22:28）**：[实验全景与设置对照](../08_实验日志/2026-09-06_audit_实验全景与设置对照/README.md)。主线IR→RGB对象判别蒸馏OEv1，3/6独立端点、0/3同seed配对；N42/P0/N123完成146/38/31轮。OS-SSL 3/9微调完成，新增seed123独立last paired−shuffled +0.708mAP/+0.533AP50；仍缺三seed、同模板native与RGB-only控制。全景说明已本地落盘；GitHub上一版快照仍为21:46/c6073b9。

| 目录 | 内容 | 说明 |
|---|---|---|
| [01_文献](../01_文献) | ~60 篇论文 PDF + 文献分析 | KD 经典（FGD/LD/CrossKD/SFTN）、跨模态/SAR 蒸馏、文献综述 md；旧文件名以硬链接别名保留；**[RGB-IR 调研批次](../01_文献/RGB-IR_20260905新增/00_方法调研综述.md)（2026-09-05，34 篇 + 方法综述）**；**[文献精读笔记](../01_文献/RGB-IR_20260905新增/文献精读笔记_20260905.md)（5 篇深读 + 全库速读 + 三条设计启示）** |
| [02_汇报材料]（未导出的工作区路径：02_汇报材料/） | 5 个 PPT + draw 绘图脚本 | 组会汇报、示意图、LADD 演讲稿 |
| [03_现行工程](../03_现行工程) | **ogsod400_clean_protocol**（canonical）+ SpaceNet6 复现 | 70 个方法版本目录、实验审计、同事交接包、预注册协议；内部结构不要改动 |
| [04_方法演化档案](../04_方法演化档案) | 按时期归并的方法史与索引文档 | TSKD→LADD→CoRe-LADD→重审→clean protocol 的完整叙事；hub 索引文档（方法目录/归档目录表/去重政策） |
| [05_实验证据_按服务器](../05_实验证据_按服务器) | 大体积运行结果归档 | 90 服务器 formal 实验、3090 迁移快照、autodl checkpoint 备份、本地归档；**⭐ [实验证据对照总表](../05_实验证据_按服务器/实验证据对照总表.md)——每个实验的配置/服务器/数据集/结果/分析/路径** |
| [06_历史工程_只读](../06_历史工程_只读) | LADD / LADD_public / CoRe-LADD 完整旧工程 | 含 runs、日志、remote_snapshots、VEDAI 数据集与内部数据视图链接；只读追溯 |
| [07_研究分析](../07_研究分析) | 研究思考与方法评估 | ⭐ [配对质量与蒸馏潜力评估](../07_研究分析/配对质量与蒸馏潜力评估_20260905.md)——逐数据集分析 + probe 实验清单 |
| [08_实验日志](../08_实验日志/README.md) | **每个实验一个目录，结论 md 落盘** | 新会话先读这里防丢进度；索引表最新条目在最上（AGENTS.md §0/§5 有强制规范） |
| [AGENTS.md](WORKSPACE_AGENTS.md) | 服务器使用原则 + 研究规范 | 94 三卡/三任务/显存留 2G 余量/总内存 300G 上限；复现甄别与四臂归因纪律；实验记录纪律 |
| [99_整理回执](../99_整理回执) | 移动映射、链接重建日志、盘点数据 | 每次结构操作都有回执，可追溯 |

## 快速上手

- **继续研究**：进入 [03_现行工程/ogsod400_clean_protocol](../03_现行工程/ogsod400_clean_protocol)，先读 `EXPERIMENT_AUDIT.md` 与 [交接包](../03_现行工程/ogsod400_clean_protocol/handoff_packages/ogsod400_colleague_handoff_20260723/README.md)（一句话结论：已有强基线与完整审计体系，尚无 paper-ready 新方法；SX-APR / MM-ARCS / PSRMD 三条线的最新决策都在交接包里）。
- **查方法历史**：[04_方法演化档案/2026-04~06_LADD主线/LADD_EVOLUTION_AND_EXPERIMENTS.md](../04_方法演化档案/2026-04~06_LADD主线/LADD_EVOLUTION_AND_EXPERIMENTS.md) 是最完整的单篇叙事；51 个方法族的机器可读目录见 [hub索引文档/catalog]（未导出的工作区路径：04_方法演化档案/hub索引文档/catalog/）。
- **查某个服务器的结果**：直接进 [05_实验证据_按服务器](../05_实验证据_按服务器) 对应目录；L20 服务器结果只在远端 `/private/results`（本地无副本，清单见交接包 catalog）。

## 服务器 ↔ 本地证据对照

| 服务器 | 角色 | 本地证据位置 |
|---|---|---|
| **94**（10.103.12.94，gpuserver94，8×4090） | **现行主力**：RGB-T 蒸馏（RGBT_campaign）+ SpaceNet6 复现；旧 OGSOD/freqmix 实验在其 `projects/ydf/` | 服务器端为主；SpaceNet6 工程本地副本在 [03/SpaceNet6_OTD_official_reproduction](../03_现行工程/SpaceNet6_OTD_official_reproduction)；freqmix 等记录见 [04 方法演化档案]（未导出的工作区路径：04_方法演化档案/2026-08_服务器环境总结/） |
| 90（10.103.12.90，8×3090） | 前主力，OGSOD 400ep 协议 | [05/90+3090迁移包_20260704]（未导出的工作区路径：05_实验证据_按服务器/90+3090迁移包_20260704/） 内 `90/` 子树 |
| 3090 迁移机 | 2026-07-10 快照 | [05/3090快照_20260710]（未导出的工作区路径：05_实验证据_按服务器/3090快照_20260710/） |
| 117 / 5090d | 历史远程训练 | [06/LADD/remote_snapshots]（未导出的工作区路径：06_历史工程_只读/LADD/remote_snapshots/）、[06/LADD/docs]（未导出的工作区路径：06_历史工程_只读/LADD/docs/） |
| autodl（seetacloud） | 早期 checkpoint | [05/autodl备份_20260614]（未导出的工作区路径：05_实验证据_按服务器/autodl备份_20260614/） |
| L20 | 现行远程结果仓 | 仅远端，本地清单见 [03/.../catalog/remote_results_manifest.csv](../03_现行工程/ogsod400_clean_protocol/handoff_packages/ogsod400_colleague_handoff_20260723/catalog/remote_results_manifest.csv) |

## 数据集位置（本地）

- **VEDAI 512**：[06/LADD_public/comparison/cmdistill/.../data]（未导出的工作区路径：06_历史工程_只读/LADD_public/comparison/cmdistill/）（raw tar + interim + processed 视图，视图经硬链接零拷贝）
- **SpaceNet6 OTD**：[03/SpaceNet6_OTD_official_reproduction](../03_现行工程/SpaceNet6_OTD_official_reproduction)
- **OGSOD / SiXiang**：不在本地（在各训练服务器上）；协议与划分定义见 03 工程内文档

## 重要约定

1. **旧 hub 已退役**：原 `00_RESEARCH_HUB` 的方法档案视图由本 README + 04 目录取代；其导航文档全部保存在 [04/hub索引文档](../04_方法演化档案/hub索引文档)，去重政策仍有效（见 [DEDUPLICATION_POLICY.md](../04_方法演化档案/hub索引文档/DEDUPLICATION_POLICY.md)）。
2. **链接层**：目录链接为 Windows junction、文件链接为同卷硬链接。删除/移动含链接的目录时不要用会穿透链接的递归删除；`git` 已按决策全部删除（44.6GB，2026-09-05）。
3. **凭据**：不在本目录内；已隔离至 `C:\Users\MSI-PC\guangsar_credentials\`（LADD 工程内留有指针）。涉及服务器口令的两个 md：`04/2026-08_服务器环境总结/FINAL_SERVER_SUMMARY.md`、`06/LADD/SERVER117_README.md`，建议尽快改密并清除。
4. **大目录成因**：44.6GB git 历史已删；剩余大头是 .pt 权重（31.5GB）与训练日志（14.5GB），均为证据或可再生物，按需另行处理。
