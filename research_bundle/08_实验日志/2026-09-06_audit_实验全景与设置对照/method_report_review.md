# 全景总报告的方法与主张复核

> 2026-09-06：独立复核本目录 README.md，未发现阻断性的算法、超参或数值事实错误；建议补充 pre-NMS / argmax 条件，并将“可迁移知识不同”的总结收紧为“候选可迁移知识因数据集而异”。

## 核对范围

只读本目录 README.md、dataset_and_history.md、oev1_method_settings.md、osssl_new_eval/README.md 与 summary.json，以及此前核实的实际 release_v2 与冻结 OS-SSL 计划；没有改源代码、训练、阈值或 GPU 任务。

## 已核对一致

- OEv1 知识公式是 P3/P4 正确类别 logit 的前景/背景 LME 差，差后除 T=2。
- K 是 eligible 数的一半向上取整，以基础集合 E 归一化；q 只排序，不连续乘入损失。
- lambda0.1、rho0.5、clip8、SmoothL1 beta1、匹配 IoU0.5、RGB conf0.05/IoU0.1、IR conf0.25/IoU0.5 均与实际源码一致。
- P/N 采用同代码辅助路径、仅改变最终 KD 剂量；训练器实际只开放 P/N，局部可选算子没有被冒称已完成归因实验。
- 预训练初始化和部分任务头随机、教师/reference 固定 seed42、首批数据流相同的解释符合实际检查。
- 独立 IR GT 使用、冻结参考见过 train、没有动态学生保护门、尚不能证明负迁移抑制的边界均已保留。
- 新 OS-SSL 独立端点及差值与 summary.json 一致；单次 SSL / 单个微调 seed、历史 native 初始化混杂、评估回执缺口与 D3 均明确标注。

## 建议修改的两处

1. README §3 第三步建议明确：“均为 NMS 前候选；IR 候选还需 argmax 类别等于 GT。” 当前“正确类别”可以理解为已包含此条件，但未写出 pre-NMS 容易与 probe 的 NMS 后一对一命中判定混淆。
2. README §8“不同数据集的教师方向和可迁移知识不相同”，建议写为“教师优势与候选可迁移知识因数据集而异”。数据诊断证明的是模型能力与错误互补差异，尚未直接证明这些知识已成功迁移。

除上述精度建议外，主报告的方法描述和结论强度可供用户阅读。
