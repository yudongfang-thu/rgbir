# 对象证据损失 CPU 算子检查（2026-09-06）

> 13 项 CPU 检查通过；这仅确认算子实现性质，不代表检测收益、方法新颖性或完整 trainer 已正确集成。

## 实现与运行

- 代码：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_object_evidence_v1/object_evidence_loss.py`。
- 测试：同目录 `test_object_evidence_loss.py`，命令为 `python test_object_evidence_loss.py`。
- 已用本机 `D:/Anaconda/envs/KGJ_proj/python.exe` 执行，torch `1.8.0+cu111`，scipy `1.10.1`，测试张量全部为 CPU，未占用 GPU。
- 完整逐项日志：`loss_cpu_tests.log`。13/13 通过；训练服务器仍应使用其实际环境再执行一次。

## 已验证性质

1. 当教师前景相对背景证据更强时，学生前景类别 logit 梯度为负、背景梯度为正；其它类别及 P5 无梯度。
2. 教师、冻结 RGB 参考及学生 DFL 均不收到 KD 梯度。
3. 同一类别各位置加常数后，对象证据及损失不变（候选有效性保持的测试条件下）。
4. 教师独立 IR GT 用于教师 ROI；复制 RGB GT 会改变结果，证明没有默用共同框。
5. 局部背景排除同模态全部 GT，包括其它类别及未匹配 GT。
6. Hungarian 匹配先最大化达阈值匹配数，再最大化 IoU；覆盖“先最大 IoU 和、后阈值过滤”错误反例。
7. 教师候选必须 argmax 类别正确、正确类概率及自身 GT IoU 均达阈值。
8. 空 GT、无有效环形背景、无质量优势均返回可反传的有限零。
9. weight0 与 native 的学生梯度逐元素完全一致（合成标量检测损失级别，非完整 trainer 回执）。
10. same-modal 只读 RGB GT，不读 IR 标注。
11. 随机选择保持 P 的选中数与相同 base 分母；独立 RNG 不改变训练全局 RNG；质量并列按 batch/GT 顺序稳定处理。
12. 基准候选集有两个对象但仅一个满足教师质量条件时，归一化分母仍为 2，防止减少对象后重标定 KD 剂量。
13. 未冻结的 shuffled 与 GT 控制臂显式拒绝运行，避免伪造已实现控制。

## 公式与局限

配置默认 T=2、P3/P4、2 倍同心外框、至少 1 个前景锚点和 4 个背景锚点、同类 GT IoU≥0.5、参考候选 conf≥0.05/IoU≥0.1、教师候选 conf≥0.25/IoU≥0.5、ρ=0.5、目标截断±8、SmoothL1 beta=1。证据取对象正确类别 log-mean-exp 与局部背景 log-mean-exp 之差，再除 T；各对象只平均双侧共同有效层。

q 为 `max(softplus(-e_ref)-softplus(-e_teacher),0)`，它是排序代理，不是已校准错误或信息量。K=`ceil(ρ × eligible_count)`，分母为教师正确性和 q 过滤前的几何有效共同 RGB 参考候选数。该版本采用 batch 全局 K，未实现逐类/尺度分层预算。

“教师正确”是原始 DFL 解码候选支持的代理：未做 NMS，同一候选可以支持多个附近 GT，不能等同于一次正式评估中的一对一 TP。随机臂从整个 base 抽取与 P 同 K 个对象，可能包含教师错误对象，检验的是整个正确性/质量选择策略。uniform 臂使用整个 base，名义剂量高于 P，比较时必须明示。

本模块返回未乘 λ、未乘 batch 的单标量。完整 trainer 须另验 λ=0.1、native loss batch 约定及总 KD 只乘一次，并记录梯度与优化步数。未启动训练或读取任何实验收益。
