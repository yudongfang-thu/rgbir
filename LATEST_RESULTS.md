# 最新结果与证据入口（2026-09-06 21:46 +08:00）

**OEv1现有3/6有效独立端点，但0/3完整同seed配对。OS-SSL新增首个同seed paired−shuffled正差，仍只有CSV口径。尚不能宣布三seed净收益或配对归因成立。**

本次追加17:30和21:46两次审计及原始小证据，保留08:53及更早快照。请优先读[21:46完整分析](research_bundle/08_实验日志/2026-09-06_audit_RGBIR夜间结果与GitHub更新/README.md)，再沿来源检查JSON/CSV/receipt；[17:30分析](research_bundle/08_实验日志/2026-09-06_audit_RGBIR晚间进度与新结果/README.md)保留首次端点和初始化问题的发现过程。

## OEv1：新增N0与P123

全部为固定E200 last/EMA、1469图开发val的独立评估，单位为百分数。

| 臂 | seed | mAP50–95 | AP50 | 相对历史同seed native的mAP差(pp) |
|---|---:|---:|---:|---:|
| weight0 | 0 | 54.346185 | 76.992515 | −0.011463 |
| paired | 42 | 54.658162 | 77.069620 | +0.842606 |
| paired | 123 | 54.636847 | 77.304557 | +0.949415 |

**最后一列是历史背景，不是预注册P−N。** P0完成28/200、N42完成135/200、N123完成20/200，均在运行。不得将P42/P123平均后减N0，也不能把缺失的同代码对照替换成历史native。两个paired端点接近且高于各自历史N，说明有值得继续验证的信号；N0接近旧N0，仍不足以证明基线路径在所有seed完全等价。

- [固定端点汇总JSON](research_bundle/08_实验日志/2026-09-06_audit_RGBIR夜间结果与GitHub更新/oev1_snapshot/oev1_endpoints/summary.json)
- [独立完整证据复核](research_bundle/08_实验日志/2026-09-06_audit_RGBIR夜间结果与GitHub更新/review_oev1_endpoints_complete.json)
- [三个完成端点及运行中原始产物](research_bundle/08_实验日志/2026-09-06_audit_RGBIR夜间结果与GitHub更新/oev1_snapshot/raw_runs)

三个完成端点的训练/评估回执、指标副本、源文件引用、1469唯一val条目和跨端点相同清单都已检查。小文件采集器最初漏掉三个超过2MiB的receipt引用manifest，现已完整补采；旧失败检查保留，最终检查以文件名带`complete`的JSON为准。这不是训练或指标失败。

OEv1训练CSV最终行因空validate字典从15列变8列，末3列是学习率。**不要从此CSV读取AP**；progress.json残留`running`也不覆盖已完成的receipt。独立eval JSON有效，原CSV未改。

## OS-SSL：首次有同seed内部比较

下表是E200训练CSV，尚无统一独立last评估，不能直接与上表排名。

| SSL臂 | seed | mAP50–95 | AP50 |
|---|---:|---:|---:|
| paired | 123 | 53.942 | 75.849 |
| shuffled | 123 | 53.273 | 75.354 |
| IR-only | 42 | 54.623 | 76.822 |

paired123−shuffled123 = **+0.669 mAP / +0.495 AP50 pp**。这是单seed初步方向；+0.495不能舍入为通过+0.5门，且预注册要求多seed。paired123−旧native123为+0.244 mAP/+0.091 AP50，受初始化混杂影响。当前3/9微调完成，shuffled0完成131轮，另5个排队。

W1 native在COCO80→5类转换时有类别输出映射；OS-SSL使用预构造的五类模板，因此不能把SSL−W1差值完全归因于SSL。三个SSL臂259个非骨干张量彼此一致，支持后续内部对照。目标检测器输入RGB，IR-only不是RGB自模态SSL，仍需RGB-only控制。三seed微调共用每臂一次seed42 SSL预训练，不能宣称覆盖三次SSL随机性。

- [21:46 OS-SSL分析与原始入口](research_bundle/08_实验日志/2026-09-06_audit_RGBIR夜间结果与GitHub更新/osssl/README.md)
- [21:46 CSV复算](research_bundle/08_实验日志/2026-09-06_audit_RGBIR夜间结果与GitHub更新/osssl/summary.json)
- [17:31初始化混杂实测](research_bundle/08_实验日志/2026-09-06_audit_RGBIR晚间进度与新结果/osssl/initialization_inspection.json)

## 给外部模型的本次新增任务

先复算新端点及同seed比较；区分历史参照、同代码P−N和SSL内部差。再判断已有正面信号是否改变路线优先级，指出最少需要哪些控制来支持“选中了有益跨模态知识”“减少负迁移”。不得按当前结果修改已冻结阈值，不把新补充实验伪装成原先预注册。

在仓库根目录用Python标准库复核OEv1（输出写到自行选择的可写路径）：

```sh
python research_bundle/08_实验日志/2026-09-06_audit_RGBIR夜间结果与GitHub更新/review_oev1_endpoints.py --snapshot research_bundle/08_实验日志/2026-09-06_audit_RGBIR夜间结果与GitHub更新/oev1_snapshot --historical 02_raw_results_dronevehicle --output oev1_recomputed.json
```

方法实现与数据诊断继续从[模型阅读指南](MODEL_REVIEW_GUIDE.md)和[完整审计提示词](REVIEW_PROMPT.md)进入。新增来源映射及发布复核见[本次发布检查](publication_checks/update_20260906_2146/README.md)。当前更新不包含数据集、模型权重、凭据或新GPU试验。
