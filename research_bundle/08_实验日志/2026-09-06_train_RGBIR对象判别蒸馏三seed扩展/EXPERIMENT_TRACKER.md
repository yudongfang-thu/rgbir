# 三 seed 运行状态（2026-09-06 07:42 +08:00）

| GPU | 当前完整run | 状态 | 接着串行执行 | screen |
|---|---|---|---|---|
| 4 | paired seed42 | 89/200轮，25346实际更新 | weight0 seed42 | rgbir_oev1_queue_s42 |
| 5 | weight0 seed0 | 1/200轮，234实际更新 | paired seed0 | rgbir_oev1expand_queue_s0 |
| 6 | paired seed123 | 1/200轮，223实际更新 | weight0 seed123 | rgbir_oev1expand_queue_s123 |

以上为已保存的启动快照，持续进度读取服务器对应run/progress.json。每个run固定E200训练后完整val，任何一时只有三张项目GPU、每卡一个训练。本轮新增四个完整run，加原两run，共paired/weight0×三个student seed；不是新增四个GPU任务同时运行。

原seed42根目录：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/`。

新增seed0/123根目录：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906/`。

目录模式：`full_{paired,weight0}_s{seed}_attempt1`。新增队列状态：扩展artifacts/workers/full_s{0,123}_attempt1/status.json；总日志：扩展artifacts/full_s{0,123}_attempt1.log。下一个同卡run尚未创建训练目录是正常排队状态。

## 工程验收

- 8/8调度CPU检查；逐臂失败隔离，不重跑失败attempt，资源拒绝仅原卡等待。
- 4/4真实canary，各24实际参数更新；同seed两臂初始模型/首batch等价，资源峰值和梯度检查通过。
- 10/10端点汇总fixture；完整身份/固定last-EMA/val、三配对均值和样本SD算法经独立复核。
- 当前完整端点0/6，主结果待定，不从中间loss或CSV占位AP生成增益。

## 后续处理

各队列自动完成训练与统一val；需要整理终态时调用ENDPOINT_ANALYZER_NOTES.md中的收集命令并用新snapshot编号保留原记录。全六格完成后报告每seed P−N与mean±SD，仍需四臂与同剂量等归因，不能直接升级论文claim。

按现有约200s/epoch粗估，新增两条队列各约22小时，不保证共享服务器负载下的完成时间。这个估算不作为按AP或固定时钟提前截停实验的规则。
