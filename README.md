# RGB-IR 跨模态蒸馏项目 · 外部审计材料包

> 打包日期：2026-09-06。用途：供外部高级模型审计员读取并独立诊断本项目。
> 审计对象：RGB→IR（及 RGB→SAR 历史）跨模态蒸馏目标检测研究。项目主要运行于局域网服务器（10.103.12.94，简称 94），本仓库收录其中**小体积文本证据**（评估 JSON / results.csv / 配置 / 代码 / 审计报告）。**不含**权重、数据集图像、任何凭据。

## 0. 审计员需要知道的五件事

1. **项目目标**：跨模态蒸馏目标检测，拟投 J-STARS。当前主战场 RGB-IR（DroneVehicle/LLVIP/VEDAI），历史主线为 RGB→SAR。
2. **最重要的背景文档**（先读）：`00_project_context/全项目复盘与研究诊断_20260905.md`——一次全项目审计，已确认：历史总结中的 FreqMix "+9 AP" 系**跨数据集错引**（实为 OGSOD 且为负结果）；HNEWA "+12.7pp" 系 **precision 误作 mAP**；监督 KD 在协议匹配下从未净胜 no-KD 对照。
3. **最新鲜的硬结果**（审计后产生）：`02_raw_results_dronevehicle/`——协议匹配 3 seeds 对照：**现有全量 KD（cmdistill_corrected，PCCFD/SLRD/IBCLD）比 no-KD 净负 −0.35 mAP50-95（N 53.954±0.356 vs L 53.605±0.332，逐 seed N≥L 为 3/3）**；昼夜分桶中 L 在 day 与 night 两桶都低于 N（night AP50 −1.75）。
4. **争议未决的方法方向**：`00_project_context/方法预注册_CGA-KD_20260905.md`（几何锚定/不变分解/条件门控三组件）及其偏差记录 D1——实现与预注册不符（G 未实现分布损失、C 门控因 PCC/cosine 尺度不变性为空操作、I 有优化器与目标冲突缺陷），W2 已冻结。
5. **已知唯一强正例**：SpaceNet6 OS-SSL 预训练（paired−native +3.3±0.9 AP50，3/3 胜全部对照），但迁移 OGSOD/SiXiang 失败（+0.20/−0.45）。尚未在 RGB-IR 上测试——这是下一步实验。

## 1. 目录地图

| 目录 | 内容 |
|---|---|
| `00_project_context/` | 项目入口文档：主 README、AGENTS（服务器使用与研究规范）、证据对照总表、方法预注册（含偏差 D1）、配对/蒸馏潜力评估（含勘误头）、全项目复盘审计 |
| `01_audit_20260905/` | 全项目复盘审计的完整工作产物：分线笔记（history/canonical/replication/cga_code_review）、RGBT 只读快照与复算脚本、FreqMix 来源 SHA 清单、CGA 代码审查与当时源码快照 |
| `02_raw_results_dronevehicle/` | 协议匹配 N/L 3-seed 评估 JSON（eval_rgbt_detector 原始输出）、昼夜分桶 summary、P2/P3 探针 summary |
| `03_freqmix_sources/` | FreqMix 原始 results.csv（6 臂）+ OGSOD native/H_S 对照 results.csv（4 文件）+ 训练 args.yaml（4 文件）+ launch/patch 脚本——用于独立验证"跨数据集错引"结论 |
| `04_hnewa_eval_records/` | HNEWA 五臂的独立评估 JSON（eval_records，含 DroneVehicle/LLVIP 各臂各 seed）——用于重算 paired/shuffled/same-modal 归因 |
| `05_code/` | 关键代码：cmdistill 训练器/评估器/启动器/资源守卫、KD 项实现、CGA-KD 训练器（含缺陷版本）、native 训练与评估脚本 |
| `06_experiment_log/` | 项目实验日志（每个实验一个条目，含勘误链） |

## 2. 建议阅读顺序

1. `00_project_context/全项目复盘与研究诊断_20260905.md`（§5/§6/§7 最关键）
2. `00_project_context/方法预注册_CGA-KD_20260905.md` + 偏差 D1
3. `01_audit_20260905/cga_code_review.md` 与 `05_code/train_cga_kd.py` 对照
4. `02_raw_results_dronevehicle/` 自行复算 N/L 对比与昼夜分桶
5. `03_freqmix_sources/` 自行复算 FreqMix 的数据集归属与数值
6. `07_questions_for_auditor.md`——我们希望你仲裁的问题

## 3. 完整性说明

- 所有 results.csv 为完整文件（非截取）；评估 JSON 为工具原始输出（schema: rgbt-detector-metrics-record-v1）。
- 服务器原始路径已在文件名或审计 manifest 中记录；SHA256 抽样见 `01_audit_20260905/replication_sources.json` 与 `audit_manifest.json`。
- 勘误纪律：分析文档的更正以勘误头形式追加，不无痕改写；实验日志见 `06_experiment_log/`。
