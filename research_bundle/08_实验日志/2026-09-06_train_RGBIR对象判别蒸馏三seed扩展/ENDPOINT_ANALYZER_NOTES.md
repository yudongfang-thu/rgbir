# 固定端点收集器使用与证据边界

> 当前两次只读快照均为 0/6 个完整端点；没有产生性能或三seed汇总结论。当前脚本增加了固定seed42教师/reference路径校验，10项CPU构造fixture通过。

`analyze_three_seed_endpoints.py`只读取固定六个run的小型JSON/YAML和receipt所引用的快照，不读取训练CSV、不加载checkpoint、不导入torch、不进行GPU计算。会检查last.pt文件存在，但不读取其张量。

有效端点要求：实际args与launch的seed/arm匹配，200轮完整completion，train/eval的COMPLETED receipt及源/metric快照完整，开发val的`fixed_budget_last_ema`端点与该run实际`weights/last.pt`路径一致。模板config的默认seed42不覆盖CLI指定的实际seed0/123。每run的`single_seed_exploratory=true`被保留，允许规范地做三seed描述统计，但不自动升级证据。

三对完整后，主量按seed计算`100*(paired.mAP50_95-weight0.mAP50_95)`；三个差值及每臂分别报告mean和sample SD（ddof=1）。少于三对时可列出已完成的逐seed结果，`three_seed_summary`保持null。缺seed、缺终态receipt、错误端点与无效身份不会被CSV0或历史结果补位。不计算p值。

## 运行

在94使用现有pinned Python，输出目录必须尚不存在。例如后续可将编号改为3：

```bash
CUDA_VISIBLE_DEVICES='' /mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_expand_20260906/analyze_three_seed_endpoints.py --output /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_expand_20260906/endpoint_snapshot3
```

快照1保存首次9项fixture版本；快照2保存增加固定教师身份校验后的10项fixture版本，两者各有code副本。`endpoint_cpu_tests_attempt1.log`、`endpoint_cpu_tests_attempt2.log`保留全部检查结果。构造fixture明确验证1/2/3百分点的均值2、样本SD1、缺seed不汇总、错误args seed/receipt arm/教师/epoch/端点/metric快照被拒绝，以及已有输出目录不会覆盖。

## 限制

该脚本是描述性端点收集器，不是项目已accepted的完整analyzer。teacher与reference固定seed42，三个学生seed不覆盖教师训练不确定性；本轮只有paired/weight0，仍缺shuffled/same-modal归因。正的P−N也不能单独证明跨模态独特信息或避免负迁移。
