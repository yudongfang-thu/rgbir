# 首次资源短测允许共享卡的修订

结论：按根执行者 2026-09-07 对工作区规则的明确解释，首次 bootstrap 不再要求整张卡没有其他用户进程。只要求没有本项目已有 active/pending lease 或任务，并继续满足整卡剩余显存大于 16,000+2,048 MiB、项目 70% 上限、GPU 数量和主机内存限制。原先的“完全空卡才能首次短测”是实现额外收紧，现已移除。

- 首次 bootstrap 仍预约并限制 16,000 MiB。其他用户随后改变占用时，监控发现本次任务越界就仅终止本任务并保留失败证据。
- 第四张项目 GPU 的放宽规则没有改变：占用后必须仍有至少两张真正没有任何 GPU 进程且没有项目预约的空卡。
- 测试覆盖整卡 free=20,279 MiB、有其他用户 PID 的可容纳例子；也覆盖 free=18,048 MiB 时拒绝、同卡本项目 pending lease 时拒绝、共享第四卡只有另外两张全空卡才允许。
- 本地 CPU 共 **25/25 测试通过**。只修改 resource_dispatch.py 和 test_resource_dispatch.py；C1、calibrator、compatibility 计算源码未改。
- 另修复 CPU 夹具在服务器 artifacts 部署位置的 guard 定位：本地使用工程 parents[2]/tools；该位置不存在时使用 DEFAULT_REPO/tools 下实际 server guard。release_gpu1 失败保持原样，不被覆盖。
- 调度器作者未自行签署独立接受。根执行者和另一审阅者负责本修订的快审、dispatcher-only 部署与实际执行。

此文件取代 RESOURCE_DISPATCH_IMPLEMENTATION.md 中首次 bootstrap 必须整卡完全空闲的描述；后者保留原始实现与测试历史。第四卡、显存、RSS、计算路径 profile 绑定和两批 PROFILED 身份约束继续有效。脚本副本在 resource_dispatch_cpu_review_v2_shared/，没有 GPU 启动或新训练结果被记入本修订。

