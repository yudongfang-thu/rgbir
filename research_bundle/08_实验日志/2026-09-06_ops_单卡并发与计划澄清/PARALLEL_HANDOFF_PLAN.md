# 冻结调度交接：random0/123提前并行

本变更只调整调度，不改研究假设、seed集合、模型、λ/T/ρ、batch32/nbs64、E200或last/EMA评估。旧P/N保持运行，不重启。root先实测同卡双训练，再按完整长训7632MiB/RSS约28900MiB的实际峰值修订资源预约，保留现有guard的全部准入限制。

## 所有权

- 现行random42仍由旧驻留`random_worker.py`控制，保留其训练、日志管道和终点评估。
- pending random0迁移到GPU2，与random42并发；pending random123在GPU4做短测，通过后等待N42完整训练与评估回执，再接用GPU4正式训练。
- 每个新worker先执行自己的canonical24更新canary，与原P同seed直接对比，然后才正式训练与独立评估。
- 不增加物理卡数；GPU4仅先做约30秒同卡短测，正式任务等待N42终点评估；不加入GPU0他人任务。取消原GPU5双开方案，避免P0终点评估因资源预约而等待新长训。
- 新worker创建独占seed ownership回执，输出仍是原计划的canonical seed0/123路径，禁止重复run或重新挑attempt。

## 驻留旧队列的交接边界

旧Python进程没有可热更新的skip/停排入口。直接杀其父队列会破坏在跑42的日志管道，因此保留它。旧队列在42训练和评估完成后准备seed0时，现有“输出必须不存在”的检查会发现已经转交的新seed0 canary，立即退出，不再运行0/123。

该退出可能在旧`queue_status.json`中显示AssertionError/failed。只有42完整训练及eval_evidence/run_receipt.json存在，并核验错误确实来自转交seed0目录已存在的断言，才能判为调度退役。原日志与status保留，不改写。新0/123状态以本目录seed*_status.json为准；新seed0结束时若42已有完整回执则补端点核对记录，否则后续补查。若42没有正常完成训练和评估，不能用交接解释其故障。R42旧评估仍预约10000MiB，可能等R0释放；P/N主实验不受这一预约冲突影响。

## 资源

仅现行GPU2 random42、GPU5 P0的资源预约由10000MiB/49152MiB改为8300MiB/32768MiB，修改必须在guard同锁内核对精确job/owner/PID及完整训练实峰，并保留before/after回执。不释放、重绑或改变GPU；不改全局guard源码。新0/123各预约8300MiB/32768MiB；双任务合计16600MiB低于guard原70%线，物理余量另留≥2048MiB。6任务总RSS预约224GiB，低于240GiB准入线和300GiB硬线。

两卡新任务的同卡canary须实际采到至少3个双CUDA PID样本且始终观察到≥2GiB空闲；正式第二训练传`--profiled-second-train`。每阶段及重试都核查至多3卡，或至多4卡且入场后仍至少2张完全空卡。新短测预约6500MiB，其依据是既有实际6304MiB短测峰值；正式训练预约8300MiB。只重试明确未启动的资源拒绝，所有已启动失败保留attempt并停止。短测通过不意味着两任务获得2倍吞吐；正式完成时间受共享负载影响，后续读取实际训练曲线。
