# 冻结计划：OEv1 同剂量随机选择对照

冻结日期：2026-09-06。冻结发生在本对照任何训练/评估之前，但已经看到部分主线 P 端点；这是明确登记的后续机制实验，不追溯声称与主线同时预注册。

## 研究问题与 estimand

固定相同数据、学生初始化、教师、参考、loss、训练预算和每batch名义蒸馏剂量时，P 的“教师可靠性＋质量选择”整体是否优于从基础集合 E 中随机取 K 个对象？

- 主比较：三个学生seed的 `mAP50_95(P_s) - mAP50_95(R_s)`，其中 R=`paired_random`。
- 辅助指标：AP50、AP75、precision、recall及逐seed方向；报告mean±样本SD（ddof=1）。指标单位统一为百分点。
- 同时保留 P−N 与 R−N，使用相同seed的现有同代码weight0端点。
- 若未完成三个seed，只能描述已完成seed；不能用跨seed补配，也不能按结果替换seed。
- 正向支持须三seed均正，均值为正，且代码/剂量/初始化/评估审计通过；本初始机制验证不宣称显著性或临床式阈值。不设结果后调参回路。

## 实验臂和精确差异

基础集合 E 完全沿用主线：实际RGB/IR GT同类IoU≥0.5对应，两边有效P3/P4前背景区域，冻结RGB参考存在confidence≥0.05、IoU≥0.1粗候选。

`eligible = E & teacher_correct & (q > 0)`；teacher_correct与q定义不变。

`K = ceil(0.5 × |eligible|)`。

- P：从eligible中按q降序取K。
- R：从**整个E**中，用无放回均匀随机抽样取K。
- 两者均除以 `max(1,|E|)`，因此同batch的K、normalizer、名义剂量K/|E|相同。
- R仍计算teacher_correct与q来确定K；没有删除所有质量信息，它删除的是按这些条件选择“哪个对象”。
- R允许抽到teacher不正确或q≤0对象，因此 P−R 检验**整个可靠性＋质量选对象策略**，不能写成“q排序单独有效”。
- 随机数使用独立CPU `torch.Generator(seed=student_seed+batch_call_index)`，不推进学生训练、增强或global torch RNG。
- 不把不同目标引起的实际loss/梯度幅度变化强行归一化；那会新增与P不同的机制。

## 固定设置

- DroneVehicle RGB→单RGB部署；IR为辅助教师。train17,990，development val1,469，五类。官方test不访问。
- 学生YOLO11n，从与主线同路径通用预训练权重初始化；学生seed0/42/123。教师/参考固定各模态seed42已训baseline。
- E200，640，batch32，nbs64，workers4，AMP，deterministic；SGD lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、warmup3。
- 完整训练配置直接复制真实release_v2 `config_drone.yaml`，字节不改；随机臂通过CLI设置arm/seed。
- λ=.1，T=2，ρ=.5，P3/P4，background scale2，clip8，SmoothL1 beta1，全部沿用。
- 原P/N的teacher/reference与额外IR标签访问均相同，R同样使用IR真实标签。
- 固定last/EMA，经同一个独立evaluate_object_evidence.py全量val评估。训练中不按val early-stop或选best。
- 训练器中仅两个变更：CLI开放paired_random；criterion将weight0映射为paired，其余arm透传。loss/loader/evaluator/config无变更，现行P/N源码不改。

## 三seed矩阵

| seed | 对照臂 | 预算 | 当前状态 |
|---:|---|---|---|
| 0 | paired_random | 200 epochs | 已冻结，未启动 |
| 42 | paired_random | 200 epochs | 已冻结，未启动 |
| 123 | paired_random | 200 epochs | 已冻结，未启动 |

## 执行前关卡

1. 源代码diff和直接字节比较：仅新训练器的两个arm路由变更，其余与release_v2字节相同；远端验证不新生成哈希。
2. CPU：既有loss算子测试；新增真实criterion的AST提取CPU执行，验证P/N损失与梯度不变、R透传；同K/normalizer、包含不合格对象、独立RNG等科学属性。
3. 先seed42，执行24次**真实optimizer update** canary，不是24个batch；AMP跳步另记。
4. CPU对比新R canary与原同seed P canary的initial_student.pt、first_batch.pt完全一致；逐共同batch核验学生文件顺序、base/eligible/K/normalizer/nominal dose相同，选中对象允许不同。
5. 验证非零KD梯度、冻结教师/参考无梯度、weight0原生梯度完全一致、只使用1张GPU、实测峰值≤预留（10,000MiB VRAM，49,152MiB RSS），GPU始终保留≥2GB。
6. seed0/123在各自完整训练前做同样24update canary和同seed历史P canary对照。不因为seed42结果而换参数。
7. 通过后才排三seed全训；使用screen/tmux与全项目guard，共用AGENTS卡数限制，优先保障现有P/N完成。本条目不自动启动任何训练。

## 失败与报告纪律

初始化或数据不等、R仍走paired、无有效随机对象、统计剂量不等、OOM/非有限loss、资源约束失败均为实现/执行失败，保留原attempt并停止，不能记为方法负结果。若协议正确而R优于P，如实报告当前选择策略未获支持，不能看结果后修改阈值。

没有实现 same-modal（教师与参考完全相同会令q=0的陷阱仍需单独设计），没有实现shuffled/GT-only；不把这些未跑对照写成已有实验。
