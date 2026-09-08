# 小时级方法筛选重构

**三臂已全部完成：2048图×FT3、共同成熟N42起点，含短验收与完整dev评价的队列实测15分56.6秒。C1-ft比N-ft高0.109pp、比C0-ft高0.160pp；三臂均低于未微调N42，本配方不自动扩展。见[最终报告](FINAL_REPORT.md)及[完整表格](results_1022/COMPARISON_TABLES.md)。30小时E200不再作为新想法首轮的默认成本。**

开始：2026-09-08。本条目承接用户对毕业时间和实验效率的要求，替代“仅监控旧E200、等待其终态后再考虑下一轮”的执行重点。已有原始attempt不改写为连续新协议；服务器GPU/显存/内存和持久lease规则继续遵守。

## 当前工作

1. 核对纯native baseline与当前wrapper的真实耗时，区分方法计算开销、额外数据处理和共享负载。
2. 从既有Drone训练清单/标签构建固定小子集，保留数据与分层依据，不按dev AP或方法输出挑图。原train/dev/test划分不修改；子集写独立manifest。
3. 已选定成熟RGB baseline初始化的微调路线：同协议weight0 N42 last/EMA共同起点，N-ft/C0-ft/C1-ft，2048图×3轮、每臂192批，fresh optimizer/EMA，lr0=.001、lrf=.1、warmup=0。不同时铺开通用初始化短训矩阵。
4. 使用已验收selected-only实现；只做新路径必要的加载/数据/梯度/资源短验收，复用既有损失与评估测试，不再复制整套审计。
5. 完整dev评价本身只需很短时间，优先保留完整dev，训练子集只压训练成本。用于筛选的结果与正式三seed/四臂结论分开。

2026-09-08 10:06，新的三臂短验收及微调队列已通过screen启动，N canary正在执行。源码release_v2，所有GPU阶段复用原全局lease；训练接纳必须先有对应新路径实测峰值、24成功update、共同初始化及前三十批样本流一致证据。配置在读取新AP前冻结。方法筛选结果不自动迁入论文结论；也不把快筛一次负值当成对所有训练协议的否定。

原RGB baseline E200实测3.473小时，原IR baseline3.696小时；当前C1每轮约18–19分钟不能视为YOLO11n正常基础成本。跨时段共享负载、workers和wrapper不同，不能把全部差距归因某一个算子，具体见baseline_runtime/。

## 当前执行入口

- 94子集：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_hourly_screen_20260908/subset_v1/`。
- 94已冻结源码：同级`release_v2/`；run输出：同级`ft_screen_attempt1/`。
- screen：`rgbirhourly_screen_42`；队列记录：`ft_screen_attempt1/queue/`。
- 正式weight0 N42初始化：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/weights/last.pt`。教师和参考仍为原IR42/RGB42，与此N42学生起点不同。
- 仅首个release_v1部署后、发射前检测到队列源码最后修改；字节核对阻止了启动，未占GPU。保留release_v1与launch_attempt1.json，重新冻结release_v2后启动ft_screen_attempt1，不修改已执行源。
- `launch_release_v2.json`记录启动前全卡显存；实际卡选择、入场与持续资源记录由原调度器写queue目录。

## 10:11 实际短验收

三臂均通过24成功update（52批、2次AMP skip），499项学生状态含检测头逐项等于同一N42，optimizer/EMA重新初始化，前三十批学生与教师标签/名单记录完全一致。实际耗时N=46.440秒、C0=42.668秒、C1=72.148秒；这包含训练入口启动，不含守护进程的入场及退出耗时，不能直接当整个队列时间。

实测单任务NVML峰值N/C0/C1=6702/6702/6782MiB，进程树RSS=28167/27842/27914MiB；加入余量后三臂训练各预约7168MiB显存、30720MiB内存。使用GPU4现有空位，项目仍占2/4/5三张物理卡，没有援引第四卡条款。完整3轮微调正在顺序进行。

## 原始日志的读取口径

禁用训练途中val的旧wrapper在末轮写`results.csv`时，metrics字典变短，末行字段数与表头不一致。原始CSV保留；不得按该表头把末行学习率读成AP。训练是否完成看`hourly_training_receipt.json`，AP只读`evaluations/<arm>/hourly_evaluation_receipt.json`中的独立完整dev结果。新的已接受描述性分析器不读取训练CSV，不需要为此布局问题修改模型或重训。

## 分工与产物

- subset/：训练清单/标签来源、分层采样脚本与覆盖统计。
- FAST_SCREEN_PROTOCOL_REVIEW.md：最小匹配对照与筛选范围。
- ENGINEERING_REVIEW.md：现有入口最小修改与验收边界。
- baseline_runtime/：只读原baseline及当前入口的真实时间小证据。
- 新训练入口、配置、实际资源与完成回执在本目录或对应94新artifacts目录登记。原始失败与旧在跑来源不覆盖、不移动。
