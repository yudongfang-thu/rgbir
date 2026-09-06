# OEv1优先级与对比实验（2026-09-06）

> 根据用户本轮明确的研究优先级，集中原创候选OEv1；OS-SSL降为保留现有产物的未完成探索，VEDAI移出近期实验。先补已训练CCLKD的独立评估，并冻结OEv1关键对照与外部方法的公平协议。

## 目的
用户强调快速形成自己的方法投稿，询问OS-SSL是否值得继续、VEDAI是否必要，以及能否并行对比方法。上一轮要求完整跑完OS-SSL矩阵的计划由本次资源优先级调整取代；不将中止后的不完整矩阵报告为成功或失败。

## 设置
保留OEv1原P/N×3正在运行的冻结实验，不改其方法、阈值与端点。暂停OS-SSL未开始任务；IR-only0刚启动且仅几轮，保留last checkpoint后结束该任务并释放其租约；不影响其他用户或OEv1。所有操作以执行时PID/cmdline核验及回执为准。

CCLKD先只读核实实际运行版本、预算与权重，再做固定last开发val独立评估。运行前须使用现有资源守卫与实测显存canary。不能把不完整移植版本或历史配置误记为原论文复现。

## 结果
**已执行并核验（23:25）**：

1. 23:12停止两个精确识别的OS-SSL调度shell及刚启动的IR-only0训练/数据进程，保留guard等待其正常释放租约；无目标残留PID。IR-only0保留第6轮last（10,748,307字节），停止前后大小/mtime一致，ZIP CRC通过。没有删除checkpoint，未伪造训练完成回执。
2. OS-SSL目前4/9微调完成（paired123、shuffled0/123、IR-only42），IR-only0第6轮因优先级变化停止，4个未启动任务暂缓。此矩阵不完整，不判OS-SSL方法成功/失败；不会为完成它继续补同模板native/RGB-only或更多SSL变体。
3. 已完成64张高密度图像显存canary和CCLKD partial三个既有last的完整1469张开发val独立评估，全部由资源guard覆盖。每次GPU2单CUDA PID，峰值1370MiB；62个小证据文件和逐seed统计已保存。
4. 新OEv1同剂量随机选择控制已实现、冻结三seed，17项CPU检查通过；seed42完成24次真实optimizer update，与原P的初始化/首batch逐张量相等，30个共同batch的E/eligible/K/分母/名义剂量相同，30批选择对象均不同。实测GPU峰值6304MiB。
5. 随机控制seed42的E200正式训练已启动，screen=`oev1_random_queue_3seed`，GPU2；seed0/123串行排队，各自先做canary才可进full。当前原P/N在GPU4/5/6继续运行。申请第4卡前确认5张完全空卡，使用用户AGENTS §2.1放宽条款，入场后仍≥2空卡；所有阶段继续经过guard。

| 方法，独立last mAP | seed0 | seed42 | seed123 | mean±SD |
|---|---:|---:|---:|---:|
| 历史native | 54.3576 | 53.8156 | 53.6874 | 53.9535±0.3558 |
| CMDistill-corrected | 53.9003 | 53.2448 | 53.6692 | 53.6048±0.3324 |
| 本次补评估CCLKD partial（LLD+CCL） | 54.0660 | 54.4933 | 54.3326 | **54.2973±0.2158** |
| CCLKD partial−历史native，pp | −0.2916 | +0.6777 | +0.6452 | **+0.3438±0.5505** |

CCLKD partial均值高于主要recipe匹配的旧native，但只有2/3 seed为正，未证明稳定收益。OEv1已完成P42/P123相对这一历史partial端点的描述性差为+0.1649/+0.3043 mAP，较相对旧native的小，说明补更强对比有必要；不能忽略workers、训练器和OEv1额外IR GT的差异而声称公平净因果优势。

CCLKD正式历史训练没有FLD/RLD，且v2历史完整源码快照未找到，不能列为完整原论文复现。新eval源码与receipt不补写成旧训练源码完整。详见[cclkd/RESULTS.md](cclkd/RESULTS.md)。

## 结论
OEv1是本项目提出并实现的候选方法，其文献新颖性和科学有效性仍待独立验证。既有方法值得作基线，但不应以另开研究主线的预算挤占OEv1关键证据。

**OS-SSL现阶段不值得作为平行主线继续扩展。** 其当前角色是保留的探索/潜在外部参照；继续补完初始化控制和9个微调，会投入较大算力却不直接回答OEv1的贡献。调整依据是用户的研究目标和实验价值，不因单seed数值好坏选择性丢弃结果。

**VEDAI近期暂缓。** 它是NIR、目前RGB教师方向更合适、小目标有效token条件不同，加入后会同时扩大模态方向与尺度问题。保留已有六baseline/特征诊断，不新增VEDAI训练。先完成Drone主验证，LLVIP可作为现有资源充分的第二数据集；不用对VEDAI的暂缓推断其长期价值。

**对比方法现在就并行准备，但分清外部方法与机制对照。** 外部：先复用CMDistill、补CCLKD partial端点，下一项优先完整FGD的YOLO11n/RGBIR协议适配；MGD备用、CrossKD后置。机制：random已启动，same-modal、合理shuffled、GT-only随后冻结；它们不能替代论文方法比较。

FGD选择理由是其前景/背景、空间/通道注意力与关系知识可直接挑战“选择对象信息”的贡献，须保留官方结构与审计loss剂量，不能简化成前景MSE后沿用FGD名称。目前FGD尚未实现或启动，详见[候选清单与官方来源](comparison_shortlist.md)。

## 产物路径
本目录保存调度前后只读快照、安全停止脚本与回执、对比方法清单、CCLKD原始小证据及后续独立评估。本地不移动/覆盖原始训练产物；远端小产物存RGBT_campaign/artifacts/oev1_priority_comparators_20260906/。

- [OS停止回执](stop_receipt.json)、[23:25最终状态](final_status.json)。
- [CCLKD完整统计](cclkd/RESULTS.md)、[新独立评估原件与回执](cclkd/eval_raw)。
- [随机对照新实验条目](../2026-09-06_train_OEv1随机选择对照/README.md)。其实际远端路径为`artifacts/rgbir_oev1_random_20260906/`与`runs/rgbir_oev1_random_20260906/`。
- 本次没有GitHub推送；全部新结论已本地落盘。

## 局限与下一步
OEv1同代码三seed净收益优先；random只检验整个可靠性/质量选择是否超过同名义剂量随机，不单独证明q排序。same-modal/shuffled/GT-only尚未完成，尤其不能让同一个RGB参考同时作为teacher导致q恒0，也不能用错配导致空候选的shuffled作假对照。FGD仍需完成忠实的协议适配与canary。此次不扩到VEDAI，也不因CCLKD更强而修改正在运行的OEv1方法或阈值。
