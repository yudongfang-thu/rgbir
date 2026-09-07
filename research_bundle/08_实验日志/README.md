# 08_实验日志 · 索引与规范

> 目的：**每个实验（探针/训练/评估/复现/审计）一个目录，结论落在 md 里**，保证新开对话不丢进度。
> 新会话启动时先读本 README 与最近条目，再读 [AGENTS.md]（服务器/本地保留，未包含于本阶段发布：../AGENTS.md）。

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
| 2026-09-08 | [train_分类快速反馈E8](2026-09-08_train_分类快速反馈E8/README.md) | train | 03:31已入队N/C0/C1 seed42共同E8，N在GPU4训练；预算约7.3小时，按实测耗时且看AP前冻结，原E200保持 |
| 2026-09-08 | [ops_训练吞吐诊断](2026-09-08_ops_训练吞吐诊断/README.md) | ops | selected-only通过新旧各24次更新逐位一致性；正常频率N/C0/C1实测0.858/0.891/2.476秒每批，E20三臂仍需约13.2纯训练小时，采用预留E8快速反馈 |
| 2026-09-07 | [audit_数据分析综合复盘](2026-09-07_audit_数据分析综合复盘/README.md) | audit | 串联521对普查、2448对raw探针、全dev AP、定位压力和64批自然流；修正低置信计数代表AP瓶颈、共享标签代表物理配准等解释，附逐类原值重绘图，无新训练 |
| 2026-09-07 | [probe_双数据集证据优先推进](2026-09-07_probe_双数据集证据优先推进/README.md) | probe | 已完成并独立接受：完整dev AP/逐类/定位压力+两组自然64批；LLVIP定位候选207/62批、Drone602/63批；优先LLVIP定位与Drone少数类混淆，非KD收益或L1准入 |
| 2026-09-07 | [probe_Baseline蒸馏机会重诊断](2026-09-07_probe_Baseline蒸馏机会重诊断/README.md) | probe | 已完成2448图：Drone可修复461个中351低置信；LLVIP同anchor教师DFL更好61.13%，Drone39.06%；IR特征未稳定胜独立RGB，且ridge欠拟合已揭示；不改变C1/L1 |
| 2026-09-07 | [audit_定位补救与准入修订](2026-09-07_audit_定位补救与准入修订/README.md) | audit | 原L1为几何证据与覆盖阻塞，尚无定位长训负结果；建议先诊断raw目标稳健性，再另立L2对象身份与目标质量合同，保留DFL、0.70门和GT控制；本次未修改或启动训练 |
| 2026-09-07 | [train_IndependentKD实施](2026-09-07_train_IndependentKD实施/README.md) | train | 2026-09-08 03:37：C1按42/0/123为40/38/40轮、旧控制84/80；五任务持续更新，无新端点或失败，定位仍缺几何准入；E8另立证据链 |
| 2026-09-07 | [audit_独立分类定位新规格](2026-09-07_audit_独立分类定位新规格/README.md) | audit | MD/ZIP一致；采用C1/L1独立不融合计划，先C1三seed；修E_C索引/校准状态，L先核自然64批覆盖上界；核心12次与最终四臂预算分开，无新GPU训练 |
| 2026-09-07 | [audit_native与蒸馏载体](2026-09-07_audit_native与蒸馏载体/README.md) | audit | N−历史native均值+0.469pp但2/3正；CPU证实workers改变第5批起增强，初始化入口重建499张量一致；C是单标量logit证据、L是4×16分布；R<.70明显压缩机会，特征/任务选择文献已核验 |
| 2026-09-07 | [audit_类别与定位最新进度](2026-09-07_audit_类别与定位最新进度/README.md) | audit | 11:19：N/C三seed独立端点齐，C−N为+0.266655±0.144373pp且3/3正；random到163/172/159轮，shuffled31与same-modal15轮；L仍未几何准入，无CL/CGT长训结果 |
| 2026-09-07 | [train_OEv1内容归因](2026-09-07_train_OEv1内容归因/README.md) | train | 按L几何不准入时的C归因分支，四条canary及新旧C/N真实精确等价通过；C-shuffled42已起E200，C-same-modal42因240GiB预约阈值排队 |
| 2026-09-07 | [train_TaskConditional首轮](2026-09-07_train_TaskConditional首轮/README.md) | train | 用户计划已冻结，独立 C+L/GT 模块部署94；N/C真实短测和D1/D2已执行，CL/CGT长训仍受几何、校准及canary门槛约束 |
| 2026-09-07 | [probe_TaskConditional机会诊断](2026-09-07_probe_TaskConditional机会诊断/README.md) | probe | 两数据集真实raw anchor/DFL诊断启动，训练2048与开发200分别统计；无几何版本只作机会诊断，不能授权长训 |
| 2026-09-07 | [probe_TaskConditional几何审计](2026-09-07_probe_TaskConditional几何审计/README.md) | probe | 冻结300对抽样名单，已看48对；仅LLVIP050001局部6点独立接受，范围内D2最终对象为0，不能推广整组 |
| 2026-09-07 | [audit_RGBIR实施起点](2026-09-07_audit_RGBIR实施起点/README.md) | audit | 严格C42−N42为+0.144554pp/AP75−0.327243pp；分析器v2接受受限用途，真实loader六图80框等价通过，历史16.1%解释已勘误 |
| 2026-09-07 | [audit_TaskConditional规格审阅](2026-09-07_audit_TaskConditional规格审阅/README.md) | audit | DRAFT-v1总体可采用，DFL内核14项CPU通过；需补几何容差、R/native owner冲突及GT-DFL温度，建议GT控制前置；原规格/在跑训练未改 |
| 2026-09-07 | [audit_判别与定位蒸馏可行性](2026-09-07_audit_判别与定位蒸馏可行性/README.md) | audit | CPU对象级再分析支持条件定位分支：Drone/LLVIP候选241/3083、117/672；Drone直接搬IR框平均IoU差−.04564，需身份与RGB坐标优势检查；未启动新训练 |
| 2026-09-06 | [ops_单卡并发与计划澄清](2026-09-06_ops_单卡并发与计划澄清/README.md) | ops | 23:59：主方法仍OEv1，梳理原计划与归因缺口；同卡双训练短测通过，GPU2已R42+R0正式双开，R123短测通过后等待N42评估；4卡5正式任务，规则允许多开 |
| 2026-09-06 | [train_OEv1随机选择对照](2026-09-06_train_OEv1随机选择对照/README.md) | train | 23:25：17项CPU及24更新canary通过，30批初始化/输入/名义剂量一致但对象不同；GPU2启动random42/E200，0/123串行，检验整体可靠性/质量选择 |
| 2026-09-06 | [ops_OEv1优先级与对比实验](2026-09-06_ops_OEv1优先级与对比实验/README.md) | ops | OEv1优先、VEDAI暂缓；OS-SSL停止后续及IR-only0第6轮，全部产物保留；补完CCLKD partial三seed独立mAP54.297±.216，历史native差+.344±.551（2/3正） |
| 2026-09-06 | [audit_实验全景与设置对照](2026-09-06_audit_实验全景与设置对照/README.md) | audit | 22:26–22:28：完整方法/设置/逐seed对比；OEv1 3/6端点、0/3配对；OS-SSL新增独立last P123−S123 +0.708mAP/+0.533AP50，仅单seed且native初始化混杂未解除 |
| 2026-09-06 | [audit_RGBIR夜间结果与GitHub更新](2026-09-06_audit_RGBIR夜间结果与GitHub更新/README.md) | audit | 21:46：OEv1 P123=54.637/N0=54.346新增，3/6端点但0/3配对；OS-SSL P123−shuffled123 CSV+0.669mAP/+0.495AP50；已推送c6073b9并回读核验 |
| 2026-09-06 | [audit_RGBIR晚间进度与新结果](2026-09-06_audit_RGBIR晚间进度与新结果/README.md) | audit | 17:30：OEv1首个独立端点P42 mAP54.658，历史参照+0.843但0/3完整配对；OS-SSL 2/9完成，证实旧native检测头初始化混杂 |
| 2026-09-06 | [ops_GitHub完整审计包](2026-09-06_ops_GitHub完整审计包/README.md) | ops | 已推送并回读核验：research/full-evidence-20260906，803文件/104.1MiB；完整诊断图册、代码/原始结果/94快照及模型审计指南 |
| 2026-09-06 | [train_RGBIR对象判别蒸馏三seed扩展](2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/README.md) | train | 22:26：P42/P123/N0独立E200完成；N42/P0/N123完成146/38/31轮；3/6端点、0/3完整配对，详见全景audit |
| 2026-09-06 | [train_RGBIR对象判别蒸馏首轮](2026-09-06_train_RGBIR对象判别蒸馏首轮/README.md) | train | 22:26：19项CPU及真实batch canary通过；P42独立last mAP54.658，GPU4上同代码N42完成146/200；三seed已扩展，净收益待配对 |
| 2026-09-07 | [probe_定位差距分解](2026-09-07_probe_定位差距分解/README.md) | probe | 夜间 16.1% GT 为可恢复定位市场（day 2.0%），师生框 IoU 0.728 可对应——oev2-loc 立项依据成立 |
| 2026-09-06 | [train_OS-SSL-IR迁移](2026-09-06_train_OS-SSL-IR迁移/README.md) | train | 22:28：3/9微调完成，shuffled0完成167轮；P123/S123新增独立last 53.932/53.224 mAP；IR-only42仍仅CSV，native混杂及D3评估偏差详见全景audit |
| 2026-09-06 | [probe_RGBIR数据特性与可迁移知识](2026-09-06_probe_RGBIR数据特性与可迁移知识/README.md) | probe | 已完成：六baseline/521对图；Drone优先前景判别、LLVIP有定位机会、VEDAI为RGB–NIR；12组特征图及3组配准图已落盘 |
| 2026-09-05 | [audit_跨模态蒸馏全项目复盘](2026-09-05_audit_跨模态蒸馏全项目复盘/README.md) | audit | 完成：FreqMix +9 为错数据集比较（实为 OGSOD 且为负）；HNEWA +12.7pp 系 precision 误作 mAP（真实 paired−shuffled +0.336±0.289）；P2/CGA 依据与实现有误；OS-SSL 有正例，监督 KD 净增益仍弱 |
| 2026-09-05 | [train_CGKD-W1_native](2026-09-05_train_CGKD-W1_native/README.md) | train | 预注册 CGA-KD 后第一个训练臂：**已完成**：N 53.954±0.356 vs L 53.605±0.332（3 seeds 协议匹配）——现有全量 KD 净负 −0.35，昼夜两桶皆负；门 1 通过 |
| 2026-09-05 | [probe_DroneVehicle昼夜分桶](2026-09-05_probe_DroneVehicle昼夜分桶/README.md) | probe | ⚠️已勘误：条件差异观察保留（白天 RGB +2.2、夜间 IR +11.2 AP50），"必然负迁移"表述撤回——代理分桶/标签口径/recipe 混杂 |
| 2026-09-05 | [P2探针_DroneVehicle特征图](2026-09-05_P2探针_DroneVehicle特征图/README.md) | probe | ⚠️已勘误：S_native 身份错误（VEDAI 模型混入）+ CKA 未中心化，"蒸馏拉力 +0.32"与"P3 饱和"结论作废；FFT 频段对比撤回 |

> 新条目：建目录 → 写 README → 在本表加一行（最新的放最上面）。
