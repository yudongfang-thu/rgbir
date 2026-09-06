# OEv1 三 seed 扩展冻结计划（2026-09-06 07:35 +08:00）

本轮在首个paired完整结果产生前冻结；首轮只见训练日志，未作中间AP评估。方法定义继承首轮EXPERIMENT_PLAN.md，不改损失、几何、候选或选择阈值。

## 研究量与分析
主量：同seed的`100 × (paired.mAP50_95 − weight0.mAP50_95)`，单位百分点。按seed0/42/123报告每臂、配对差值的均值和样本SD（ddof=1），逐seed方向，不把batch数当独立样本。次指标AP50/AP75/precision/recall使用相同端点和单位；不按次指标改选主结论。

本轮主量仅是整套对象判别干预净效果。无四臂归因之前，不得将正P/N解释为“跨模态独特知识”、选择机制收益或避免负迁移。不把已有不同recipe baseline直接拼作新N。

## 已冻结的六个run
| student seed | paired | weight0 | 调度 |
|---|---|---|---|
| 42 | 既有run，约86/200 | 既有队列待执行 | 原GPU4串行，完全保留 |
| 0 | 本轮新增E200 | 本轮新增E200 | 同一卡先weight0后paired |
| 123 | 本轮新增E200 | 本轮新增E200 | 另一卡先paired后weight0 |

不同队列的先后顺序平衡首批方法/对照覆盖，不能消除全部共享服务器时段差异。每个seed两臂共同初始化、训练随机数种子、数据、recipe和预算；canary直接比较初始学生state与首批输入/标签。teacher/ref统一固定seed42，用于隔离student训练随机性。

复用原94 release_v2的train_object_evidence.py、object_evidence_loss.py、paired_rgbir_data.py、config_drone.yaml与evaluate_object_evidence.py，不改文件。训练CLI显式`--seed 0/123`；原config模板中的42是默认值，实际seed由CLI覆盖，并在args.yaml、launch_manifest和终态receipt记录；config副本必须结合CLI解释。eval从完成receipt读seed。

## 端点与资源
所有run：DroneVehicle17990 train、1469 val，YOLO11n从原通用预训练初始化；E200固定last/EMA；中间val禁用；不读取test。禁止按early AP截停，不自动改batch或恢复NaN。原始失败attempt保留，技术问题只影响具体run。

遵守root与工程AGENTS，按工程较严格口径：全项目最多3张GPU、每卡最多2个本项目CUDA PID且默认1个正式训练；本轮每张卡仅1个正式训练。每job显存预约10000MiB、RSS49152MiB；三卡合计预约144GiB低于240GiB admission。已观测原长训显存峰值7632MiB/RSS28886MiB；新增seed各两臂24实际更新canary后再准入。每次起任务重新检查实际空闲≥预约+2048MiB，不使用其它卡回退；不足仅该队列等待。

初步预算新增4×200epochs，按原训练约200秒/epoch粗估44.4 GPU小时，加短程检查与val；两卡串行各两个run粗估约22小时，受共享负载影响。原GPU4现存run不重跑，不另增加第四张卡。

## 后续归因工作范围
本轮只扩展P/N随机种子，不新增方法变体。same-modal、shuffled必须明确区分辅助标注/配准与内容，random必须同基础集合/同K/同剂量；不能直接调用空候选的错配IR GT获得“shuffled无收益”。这些是后续必须补的证据，不在本轮作已实现或已完成claim。
