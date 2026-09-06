# RGBIR 晚间进度与新结果（2026-09-06）

> 新增首个有效 OEv1 端点：paired42 mAP50–95=54.658、AP50=77.070，较历史同seed native描述性高0.843pp，但同代码weight0未完成；OS-SSL已有2个微调完成，同时证实其复用native存在检测头初始化混杂。当前没有三seed净收益或配对归因结论。

## 目的
回答当前整体进度、是否出现新的完整结果，以及结果支持的判断和后续优先级。

## 设置
94 SSH只读检查；不修改冻结方法、阈值、队列或训练。OEv1复用既有固定端点收集器，OS-SSL-IR分别核对训练完成、端点、对照身份和三seed齐备性。OS-SSL初始化只在CPU读已有checkpoint做直接张量比较，不占GPU。本轮应用monitor-experiment与analyze-results工作流，辅以两条路线的独立证据复核。

运行/小产物快照：2026-09-06 17:29–17:30 +08:00；OS-SSL初始化检查：17:31。全项目实际4个训练CUDA进程，GPU1/4/5/6；对应已有4卡放宽安排，仍有3张完全空卡。匹配项目进程RSS和约133.6GiB（含共享页重复统计），各卡余量约16GiB。本轮未调整任何任务。

## 结果
### OEv1：1/6有效端点，0/3完整配对

| student seed | paired | 同代码weight0 |
|---|---|---|
| 0 | 排队 | 完成162/200，当前163 |
| 42 | **E200训练+独立评估完成** | 完成61/200，当前62 |
| 123 | 完成151/200，当前152 | 排队 |

P42固定E200 last/EMA、1469张唯一开发val图：

| 指标 | P42 | 历史native42 | 描述性差值(pp) |
|---|---:|---:|---:|
| mAP50–95 | **54.658162** | 53.815556 | +0.842606 |
| AP50 | **77.069620** | 76.001138 | +1.068482 |
| AP75 | **63.839323** | 62.566462 | +1.272861 |

历史CMDistill42 mAP=53.244840，P42相对其描述性高1.413322pp。历史参照不能替代本轮同代码weight0；workers、loader/trainer和评估契约需分开核对。没有把该背景差写成预注册净收益。

train/eval COMPLETED receipt、E200 completion、指标副本、源码引用、实际checkpoint路径、1469唯一val条目与collector数值已独立检查通过。详见[端点复算](oev1_first_endpoint_context_check.json)、[独立判读](oev1_review_notes.md)。不需要因为日志格式问题重跑这个有效评估。

### OS-SSL-IR：2/9训练完成，暂无paired终点

| SSL初始化臂 | seed | E200 CSV AP50 | E200 CSV mAP50–95 | 相对同seed native CSV的ΔmAP(pp) |
|---|---:|---:|---:|---:|
| IR-only | 42 | 76.822 | 54.623 | +0.825 |
| shuffled | 123 | 75.354 | 53.273 | −0.425 |

paired123完成103/200；其他6个微调排队。上表严格CSV对CSV，没有与独立last评估混表。OS-SSL worker目前只训练，尚未产出独立last评估；stdout最后显示的是best.pt。两个不同seed、不同臂不能直接相减判断配对信息作用，也不能把OEv1的54.658与OS-SSL的54.623直接排名。

**新增初始化混杂证据**：W1 native从COCO80类权重构造5类检测器，日志有3/5类别行映射、451/499张量加载；OS-SSL clean模板已经是5类，类别名为数字占位，加载499/499张量。三SSL臂的259个非骨干张量彼此完全一致，但与W1全检测器初始化不等价。因此相对W1的正差混合了SSL骨干与检测头初始化变化；尚未测定各自贡献。详见[OS-SSL完整审计](osssl/README.md)。

另有两项需补证：目标模型是RGB，所以现有IR-only不是RGB自模态SSL控制；completion receipt沿用native方法标签，不能只读arm字段识别实际实验。冻结迁移阈值为paired−native至少+1.0 AP50且3/3正向，旧README的+0.1是误写，未据结果更改阈值。

### 训练健康与日志发现

OEv1已采集的各run KD日志数值均有限，无零选择batch；weight0实际加权KD恒为0。P42全程入选对象比例约27.2%，模型完成训练和评估。这些是工程与目标拟合证据，不能证明选择到了可迁移信息或避免了负迁移。

P42的progress.json终态仍保留旧`running`字段，不能据此认为还在训练。其CSV最后一轮从15列缩为8列：最后轮触发validate返回空指标字典，框架按当前字典写行；最后3列实际是学习率，会被按旧header读取为precision/recall/AP。真实训练完整200轮，独立evaluation_val.json不受影响。最初健康采集因None字段失败，失败脚本/输出保留；成功快照的健康摘要保守排除了该行，**其199轮摘要已由[进度解释](progress_interpretation.json)纠正为200轮**，不改原CSV或原快照。

## 结论
1. 当前从“仅工程跑通”推进到“已有首个可靠完整性能端点”。对象判别蒸馏值得继续完成已冻结的三seed，但尚不能宣布有效或paper-ready。
2. 历史低效结果不能推出RGBIR没有空间；新P42相对历史参照的正差增加了该方向的可行性信号，但主量仍是同seed P−N。新native如果同样上涨，现有正差未必来自KD。
3. OS-SSL的正数先触发了控制组身份问题，不能据此主张IR预训练有效，也不能宣布配对无用。现有三SSL臂内部共享非骨干模板，因此应保留其可解释的后续比较。
4. 投稿缺口依然是稳定净收益、跨模态/同模态/随机同剂量归因，以及与数据特性对应的机制证据。早晨521对诊断没有因一个新AP自动升级为因果机制结论。

## 产物路径
本地：本目录下oev1_snapshot/（原始小结果、完整P42证据链）、osssl/（70个原始小产物及CPU初始化检查）、独立复算脚本/JSON/说明。OEv1全KD日志仍在服务器，样本片段文件明确标partial。

94新快照：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/audit_rgbir_evening_20260906_1732/`。初次失败快照`audit_rgbir_evening_20260906_1727/`保留。原run、receipt、checkpoint均不移动、不覆盖。

原OEv1路径：`RGBT_campaign/runs/rgbir_object_evidence{_v1,_expand}_20260906/`；OS-SSL：`RGBT_campaign/runs/osssl_ir_20260906/`。早晨GitHub分支保留08:53快照，本轮结论尚未追加到GitHub。

## 局限与下一步
1. 保持OEv1六个run及阈值不变，优先等N42。按最近5轮速度估计，N0还约2.5小时、P123约4小时、N42约8.7小时；首个完整同seed比较大约9月7日凌晨。全六臂还需后续P0/N123各跑200轮，预计9月7日白天，耗时随服务器负载变化，不是完成承诺。
2. OS-SSL对已有及后续完整权重按冻结协议补统一last/EMA独立评估；单独冻结新增“同5类模板/同非骨干初始化、零SSL步native”与“RGB-only SSL”对照。新补充设计不能伪装成原先预注册，也不覆盖旧native。
3. 若OEv1三seed同代码P−N稳定为正，优先增加same-modal、合理shuffled与同剂量随机选择对照，回答“哪些跨模态知识有用、是否减少负迁移”。本轮不根据单seed结果增加机制或调整门限。
4. 本轮只读审计不启动新GPU任务或自动监控；没有完整三seed结果时不计算/宣称mean±SD收益或升级accepted claim。
