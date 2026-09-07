# 200dev对象证据与完整AP瓶颈的有限桥接协议

根任务在完整TIDE结果后要求按全部五类补充已有对象缓存，先冻结本协议再运行。它是事后描述性桥接，不修改模型、选择、阈值或方法，不重推理、不训练读出。

输入：`../../2026-09-07_probe_Baseline蒸馏机会重诊断/remote_exports/dronevehicle_full_attempt1/objects.jsonl`。只取dev/val且非background的全部GT对象，预期200图/3084 GT；train和人工背景窗口不入分母。

固定复用原分析器 `new_probe_analysis/analyze_baseline_probe.py` 的 `object_view`、`opportunity_summary`，使用assigned对象候选而非same-anchor。原confidence=.25、同类RGB GT IoU=.50、T的标签关联paired_gt_iou≥.50均不变；no_candidate是原导出粗候选阈值.05之后的无关联候选，不等于网络没有任何响应。

所有五类和总计同一规则列出：GT、N42正确/错误数、N错误互斥state（low_confidence优先，然后class_and_localization/class_only/localization_only，缺候选单列）；T42和独立RGB N0可修复/可损伤数量及各自分母，修复按N state分类。T修复必须已有IR GT关联；N0同RGB GT身份不需IR关联，另报仅IR已配对子集的N0以提供同一eligible分母的描述性比较。

分母同时给出本类全部GT、本类N错误、本类N正确和eligible GT。修复率与损伤率不可直接相减。保留共同配对子集上的T-only/N0-only/both修复计数，避免把独立RGB自然互补当成纯IR贡献。

已知总体原结果（3084 GT，N42正确2386，T修复461/损伤193，N0修复221/损伤192）只作复现交叉检查；全五类全部报告，不按TIDE结果挑类/挑对象。

该200图输入使用统一640方形letterbox，TIDE完整1469图使用native rect544×672。它们有部分图像ID重合且N42模型身份相同，但候选规则、后处理、GT辅助和几何画布不同，不能逐对象冒充同一AP匹配。对象可修复仅表示此候选/阈值下T正确且N错误，不证明可学性；不能外推完整macro AP增益或称跨模态因果贡献。

来源只记路径/size/mtime并存源码副本与直接字节相等核验，不新增hash/SHA。
