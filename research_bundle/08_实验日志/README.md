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
| 2026-09-06 | ops_GitHub完整审计包（本包未收录：`2026-09-06_ops_GitHub完整审计包/README.md`） | ops | 整理与上传中：新分支research/full-evidence-20260906，补齐诊断图册、逐目标记录、OEv1代码/协议/三seed运行证据及审计入口 |
| 2026-09-06 | [train_RGBIR对象判别蒸馏三seed扩展](2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/README.md) | train | 已扩展：GPU4 paired42约89/200；GPU5 weight0 0、GPU6 paired123运行，其配对臂同卡排队；8项调度检查+4canary通过，三seed终点0/6暂无AP |
| 2026-09-06 | [train_RGBIR对象判别蒸馏首轮](2026-09-06_train_RGBIR对象判别蒸馏首轮/README.md) | train | 已启动：19项CPU检查+两臂24更新canary通过；94仅GPU4，DroneVehicle paired seed42/E200运行、weight0串行排队；暂无性能结论 |
| 2026-09-06 | [train_OS-SSL-IR迁移](2026-09-06_train_OS-SSL-IR迁移/README.md) | train | OS-SSL 三臂 SSL 完成（shuffled loss 高 4-8× 后期反超系固定错配可记忆，判读只看下游）；9 微调队列等待并行会话臂完成 |
| 2026-09-06 | [probe_RGBIR数据特性与可迁移知识](2026-09-06_probe_RGBIR数据特性与可迁移知识/README.md) | probe | 已完成：六baseline/521对图；Drone优先前景判别、LLVIP有定位机会、VEDAI为RGB–NIR；12组特征图及3组配准图已落盘 |
| 2026-09-05 | [audit_跨模态蒸馏全项目复盘](2026-09-05_audit_跨模态蒸馏全项目复盘/README.md) | audit | 完成：FreqMix +9 为错数据集比较（实为 OGSOD 且为负）；HNEWA +12.7pp 系 precision 误作 mAP（真实 paired−shuffled +0.336±0.289）；P2/CGA 依据与实现有误；OS-SSL 有正例，监督 KD 净增益仍弱 |
| 2026-09-05 | [train_CGKD-W1_native](2026-09-05_train_CGKD-W1_native/README.md) | train | 预注册 CGA-KD 后第一个训练臂：**已完成**：N 53.954±0.356 vs L 53.605±0.332（3 seeds 协议匹配）——现有全量 KD 净负 −0.35，昼夜两桶皆负；门 1 通过 |
| 2026-09-05 | [probe_DroneVehicle昼夜分桶](2026-09-05_probe_DroneVehicle昼夜分桶/README.md) | probe | ⚠️已勘误：条件差异观察保留（白天 RGB +2.2、夜间 IR +11.2 AP50），"必然负迁移"表述撤回——代理分桶/标签口径/recipe 混杂 |
| 2026-09-05 | [P2探针_DroneVehicle特征图](2026-09-05_P2探针_DroneVehicle特征图/README.md) | probe | ⚠️已勘误：S_native 身份错误（VEDAI 模型混入）+ CKA 未中心化，"蒸馏拉力 +0.32"与"P3 饱和"结论作废；FFT 频段对比撤回 |

> 新条目：建目录 → 写 README → 在本表加一行（最新的放最上面）。
