# 03:20 N/C/random只读状态复核

> 2026-09-07 03:20:11（UTC+8）仍为4/9个完整独立端点，唯一完整C−N配对仍是seed42的+0.1446 pp；五个旧训练继续推进。R42旧队列仍负责训练与随后独立评估，未退役。

## 设置与证据

复用`collect_readonly.py`与`build_audit.py`，保存114份原始小文件，未覆盖02:23快照。分析使用`rgbir-task-conditional-results-v2`，固定三seed、真实receipt绑定recipe、last/EMA独立val端点；源码副本在`source_v2/`。未调用guard inspect、未刷新lease、未启动停止训练或GPU任务。

## 九个运行状态

“已完成epoch”取完整CSV行，completion receipt存在时以其E200优先；“当前epoch”是progress中尚可能进行中的epoch。ETA仅按近期吞吐估计剩余训练，不含独立评估或后续排队。

| 方法/seed | 已完成epoch | 当前epoch | 独立终态 | mAP50–95（%） | 训练剩余约h |
|---|---:|---:|---|---:|---:|
| N0 | 200 | 200 | 完整 | 54.3462 | 0 |
| N42 | 200 | 200 | 完整 | 54.5136 | 0 |
| N123 | 114 | 115 | 训练中 | — | 5.05 |
| C0 | 119 | 120 | 训练中 | — | 4.87 |
| C42 | 200 | 200 | 完整 | 54.6582 | 0 |
| C123 | 200 | 200 | 完整 | 54.6368 | 0 |
| R0 | 47 | 48 | 训练中 | — | 10.64 |
| R42 | 58 | 59 | 训练中 | — | 9.91 |
| R123 | 33 | 34 | 训练中 | — | 9.41 |

相比02:23的完整CSV epoch，C0 104→119、N123 98→114、R0 34→47、R42 44→58、R123 16→33。尚无新完整独立端点，不用中途训练值补齐比较。C42−N42的mAP差仍+0.1445537 pp、AP50差−0.1100336 pp、AP75差−0.3272431 pp，不能声称三seed稳定净收益。

机器结果在`derived/results_analysis.json`、`derived/progress_eta.json`、`derived/endpoint_table.csv`；原始路径及字节长度在`source_manifest.json`。

## R42队列责任

03:20快照和03:21:13专项只读复核均确认以下进程链存在：旧worker PID975898 → guard PID975933 → R42正式trainer PID975949。训练progress已到59，旧queue_status为`training, seed=42`。queue_status本身在阶段切换时写入，不是心跳，因此同时以实时/proc和epoch前进核对。

`queue_responsibility/random_worker.py`原字节副本明确：R42 guarded train返回后，写`evaluating`并调用`evaluate_object_evidence.py --run full_paired_random_s42_attempt1`。`queue/seed0_ownership.json`及`seed123_ownership.json`均记录`old_worker_retains_seed42=true`；seed0/123由并行worker承接。

旧队列的正常预期退役点是**R42训练和独立评估全部完成后**，进入已经存在的seed0 canary时触发重复路径保护退出。现在R42尚无completion/eval receipt，不能标成正常退役；将来必须同时核实完整独立评估receipt及预期退出原因，不把任意failed都解释成正常交接。本次未改变任何队列。

## 共享资源快照

当前6个lease使用物理GPU2/4/5/6；GPU2为R42+R0，GPU4为R123+新C-shuffled42，GPU5为C0，GPU6为N123。新C-shuffled42属于另外的归因实验，不加入上表九个旧run。

GPU1/3/7为空；项目总进程树RSS约169.61 GiB（RSS含共享页重复计数），lease预约208 GiB。卡上最低剩余显存为GPU2约8781 MiB，GPU4约9755 MiB。此处仅报告只读瞬时状态，不代替下一次任务启动时的guard检查。

## 结论与下一步

保持旧训练和R42评估责任；等C0与N123真实E200独立评估后再构成三个C−N配对。此快照没有新收益证据，不触发任何根据中途AP的选择、早停或参数调整。
