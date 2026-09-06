# 08_实验日志 · 索引与规范

> 目的：**每个实验（探针/训练/评估/复现/审计）一个目录，结论落在 md 里**，保证新开对话不丢进度。
> 新会话启动时先读本 README 与最近条目，再读 [AGENTS.md](../workspace_context/WORKSPACE_AGENTS.md)。

## 目录命名

```
YYYY-MM-DD_<类型>_<短名>/
```
类型：`probe`（探针/评估）、`train`（训练）、`repro`（复现）、`audit`（审计）、`ops`（数据/工程操作）。

## 每个目录必含

1. **README.md —— 结论记录**（模板见下），这是唯一必写件；
2. 实验脚本副本；
3. 关键产物副本（summary.json 等小文件；大文件只记服务器路径）。

## README.md 模板

```markdown
# <实验名>（<日期>）
> 一句话结论放最前面。
## 目的
## 设置（模型/数据/超参/服务器/GPU/条件与对照臂）
## 结果（关键数字表）
## 结论（可直接引用的判断，含证据强度）
## 产物路径（服务器路径 + 本目录副本清单）
## 局限与下一步
```

## 条目索引

> **勘误链（2026-09-05 晚）**：P2/P3 探针条目与《配对质量与蒸馏潜力评估》已加勘误头；CGA-KD 预注册登记偏差 D1（W2 全臂冻结，G/C 实现与预注册不符、I 暂缓）。引用下列早期条目结论前，先读其勘误头与 [audit 条目](2026-09-05_audit_跨模态蒸馏全项目复盘/README.md)。

| 日期 | 条目 | 类型 | 一句话结论 |
|---|---|---|---|
| 2026-09-06 | [audit_RGBIR夜间结果与GitHub更新](2026-09-06_audit_RGBIR夜间结果与GitHub更新/README.md) | audit | 21:46：OEv1 P123=54.637/N0=54.346新增，3/6端点但0/3配对；OS-SSL P123−shuffled123 CSV+0.669mAP/+0.495AP50；更新原GitHub分支 |
| 2026-09-06 | [audit_RGBIR晚间进度与新结果](2026-09-06_audit_RGBIR晚间进度与新结果/README.md) | audit | 17:30：OEv1首个独立端点P42 mAP54.658，历史参照+0.843但0/3完整配对；OS-SSL 2/9完成，证实旧native检测头初始化混杂 |
| 2026-09-06 | ops_GitHub完整审计包（未收录的来源路径：`2026-09-06_ops_GitHub完整审计包/README.md`） | ops | 已推送并回读核验：research/full-evidence-20260906，803文件/104.1MiB；完整诊断图册、代码/原始结果/94快照及模型审计指南 |
| 2026-09-06 | [train_RGBIR对象判别蒸馏三seed扩展](2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/README.md) | train | 17:30：P42独立E200 mAP54.658；N42完成61、N0完成162、P123完成151轮；1/6端点、0/3完整配对，详见晚间audit |
| 2026-09-06 | [train_RGBIR对象判别蒸馏首轮](2026-09-06_train_RGBIR对象判别蒸馏首轮/README.md) | train | 已启动：19项CPU检查+两臂24更新canary通过；94仅GPU4，DroneVehicle paired seed42/E200运行、weight0串行排队；暂无性能结论 |
| 2026-09-06 | [train_OS-SSL-IR迁移](2026-09-06_train_OS-SSL-IR迁移/README.md) | train | 17:29：2/9微调完成、paired123@103；IR-only42 CSV54.623、shuffled123 CSV53.273；新增native初始化混杂与RGB-only控制缺口，详见晚间audit |
| 2026-09-06 | [probe_RGBIR数据特性与可迁移知识](2026-09-06_probe_RGBIR数据特性与可迁移知识/README.md) | probe | 已完成：六baseline/521对图；Drone优先前景判别、LLVIP有定位机会、VEDAI为RGB–NIR；12组特征图及3组配准图已落盘 |
| 2026-09-05 | [audit_跨模态蒸馏全项目复盘](2026-09-05_audit_跨模态蒸馏全项目复盘/README.md) | audit | 完成：FreqMix +9 为错数据集比较（实为 OGSOD 且为负）；HNEWA +12.7pp 系 precision 误作 mAP（真实 paired−shuffled +0.336±0.289）；P2/CGA 依据与实现有误；OS-SSL 有正例，监督 KD 净增益仍弱 |
| 2026-09-05 | [train_CGKD-W1_native](2026-09-05_train_CGKD-W1_native/README.md) | train | 预注册 CGA-KD 后第一个训练臂：**已完成**：N 53.954±0.356 vs L 53.605±0.332（3 seeds 协议匹配）——现有全量 KD 净负 −0.35，昼夜两桶皆负；门 1 通过 |
| 2026-09-05 | [probe_DroneVehicle昼夜分桶](2026-09-05_probe_DroneVehicle昼夜分桶/README.md) | probe | ⚠️已勘误：条件差异观察保留（白天 RGB +2.2、夜间 IR +11.2 AP50），"必然负迁移"表述撤回——代理分桶/标签口径/recipe 混杂 |
| 2026-09-05 | [P2探针_DroneVehicle特征图](2026-09-05_P2探针_DroneVehicle特征图/README.md) | probe | ⚠️已勘误：S_native 身份错误（VEDAI 模型混入）+ CKA 未中心化，"蒸馏拉力 +0.32"与"P3 饱和"结论作废；FFT 频段对比撤回 |

> 新条目：建目录 → 写 README → 在本表加一行（最新的放最上面）。
