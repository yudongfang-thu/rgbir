# 跨模态蒸馏全项目复盘（2026-09-05）

> 结论：项目存在可复核的跨模态正信号，但当前自研监督 KD 未获得稳定净增益；本轮发现 FreqMix 错数据集比较、HNEWA precision 误引、P2 对照身份/CKA 问题及 CGA 预注册与代码偏离。状态：审计完成，未启动或修改训练。

## 目的
梳理长时间线内各方法的实际证据，定位效果不理想的原因、证据冲突及后续验证优先级。

## 设置
- 范围：本地 03/04/05/07/08 目录内关键文档、代码与小型结果；94 服务器当前 RGBT_campaign 及相关结果只读核查。
- 方法：文档与原始 JSON/CSV/脚本交叉核对，区分原始复算、历史记录、假设；不把不同协议或不同指标合并比较。
- 不运行 GPU 实验，不读取 checkpoint tensor，不改变服务器训练状态。

## 结果
全部 AP 差值以百分点表示，具体证据边界见综合报告。

| 核查项 | 结果 | 来源/等级 |
|---|---|---|
| SiXiang full-CMD A8/A9 | paired−H_S +1.051±0.900 mAP，3/3 正；seed42仅 +0.035 | accepted 本地产物，exploratory pilot |
| SX-APR | 21格完整，六个门均失败；paired−GT gate −0.804 mAP | accepted 本地产物 |
| SpaceNet6 OS-SSL v2 | paired−native +3.328±0.909 AP50，3/3正 | 本地终态 analyzer，未重新读取全部远端输入 |
| SpaceNet6 监督 P3 | locked mAP +0.117，逐seed −0.748/+0.664/+0.435 | 本地汇总与报告核对 |
| FreqMix | 实为 OGSOD E400；gray/SAR donor 相对现存同主要recipe native −2.621/−2.628 mAP | 94 CSV/args/data YAML/源码，本轮两seed复核 |
| RGBT HNEWA DroneVehicle | paired−same-modal +0.207±0.360 mAP，2/3正 | 94 独立 final eval JSON 描述性重算 |
| RGBT HNEWA LLVIP | paired−same-modal +0.091±0.592 mAP，2/3正 | 同上 |
| P2 probe | native实际VEDAI；linear_cka未中心化；不能解释native→dist的纯KD效应 | 94 args + 本地脚本 |
| CGA-KD | G不是DFL；C特征缩放被损失归一化抵消；I canary通道错误，另有optimizer/目标风险 | 94源码、实际失败日志、trainer源码静态审查 |

## 结论
1. 研究的长期困难同时包括真实科学负结果、工程/协议问题与汇总污染，不能统称“所有跨模态方法无效”。
2. 先关闭证据与实现身份缺口，再用强匹配基准和最小归因矩阵检验新机制；降低复杂分解臂优先级。
3. 本次描述性重算不升级方法 claim，不替代 accepted analyzer、独立复现或尚未完成的训练。

## 产物路径
- [综合报告](E:/SHARE/光sar/07_研究分析/全项目复盘与研究诊断_20260905.md)。
- `history_notes.md`、`canonical_notes.md`、`replication_notes.md`、`cga_code_review.md`：四份定向审计，含精确来源。
- `collect_rgbt_readonly.py`、`collect_rgbt_records.py`：经 `ssh 94 python3 -` 执行的纯只读 CPU 采集脚本。
- `rgbt_readonly_snapshot_v2.json`：89个CSV及args/receipt，包含原始CSV、路径、SHA256、列宽异常。初版 `rgbt_readonly_snapshot.json` 保留解析错误，已由v2补齐，不能作完整结果表。
- `rgbt_eval_and_code_snapshot.json`：独立评估JSON、准备receipt、当前依赖源码和协议配置。
- `rgbt_eval_rows.csv`、`rgbt_descriptive_analysis.json`：逐seed指标与配对差值，`analyze_rgbt_snapshot.py` 可重算。排除身份未闭合的seed42 best临时路径。
- `replication_sources.json`：FreqMix与native/H_S远端小文件、来源与哈希。
- `train_cga_kd.remote_snapshot.py`、`ultralytics_trainer.remote_snapshot.py`、`canary2_invariant.remote_snapshot.log`：当时源码与真实失败证据。
- `audit_manifest.json`：本次审计文件清单与SHA256；由 `build_audit_manifest.py` 生成，附报告本地链接验证结果。
- 94原始根：`/mnt/dataset/yudongfang/projects/RGBT_campaign/`、`/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/`、`/mnt/dataset/yudongfang/projects/ydf/`。大权重/数据未下载。

## 局限与下一步
L20 结果部分仅有本地 manifest/审计转述；未访问的远端结果不可视为已复核。
本次未重跑P2、未修复CGA、未评估新的封存test；RGBT native仍在运行的状态仅对应采集时刻。后续代码/训练更新不由此快照自动涵盖。完整报告及主索引已落盘，历史原始实验资料保留。
