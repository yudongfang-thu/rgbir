# 原文协议优先：CFT／LLVIP 实际复现结果（2026-09-09）

**原作者权重复评已基本对齐；从头训练未准入。新增迁移到我方配置的实验暂停。**

先在原模型、原划分、原标注和原评估条件下验证作者方法，再讨论迁移效果。否则，迁移负结果会混淆原方法可复现性、实现错误与适用条件变化，不能用于判断论文是否造假。

| 指标 | 论文（%） | 本次（%） | 本次−论文（pp） |
|---|---:|---:|---:|
| AP50 | 97.5 | 97.376907 | −0.123093 |
| AP75 | 72.9 | 72.891348 | −0.008652 |
| AP50–95 | 63.6 | 63.537478 | −0.062522 |

90服务器上，CFT作者双流融合模型及权重、官方previous标注、完整LLVIP test 3463对/7931GT、1024输入、作者评估入口，完整进程90.36秒，exit=0。覆盖和指标聚合经有限范围独立复核，同环境CPU重算三个AP完全一致。论文数值来自[CFT Table 3](https://arxiv.org/html/2111.00273v2#S4.T3)，模型源码来自[作者仓库](https://github.com/DocF/multispectral-object-detection)。

身份为PAPER-RECONSTRUCTED checkpoint reevaluation：保留旧checkpoint运行时类兼容、现代环境与历史标签代码无法逐字节确认等限制。没有三项精确吻合、从头训练、多seed或跨模态蒸馏增益结论。CFT推理需要RGB+IR，不能直接充当我方RGB-only部署的公平对手。

原B32/1024训练的单卡及作者双卡DataParallel三次资源检查均在首批前向OOM，0次成功optimizer更新；没有取消项目每卡显存低于70%的限制。E200未启动，没有测得其整程时长，全部本轮lease已释放。保留三次失败原件；这不是方法负结果，也不证明完整24GB或其他硬件不能训练。

当前用户明确授权原论文协议评估，官方test暴露单列记录；我方grouped train/dev原件不变，此结果不用于我方方法选择或调阈值。新旧官方标签test分别7931/8302GT，不能混比。

- [完整记录、实际设置、训练失败与下一步](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/README.md)
- [协议和兼容差异](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/PROTOCOL_REVIEW.md)
- [独立指标复核与局限](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/EVALUATION_REVIEW.md)
- [原执行完成回执](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/server_execution/official_test_attempt1/receipt.json)
- [训练资源检查汇总](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/server_training/training_feasibility.json)
- [Drone原文方法队列：尚缺作者权重/标签，未启动](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/DRONE_ORIGINAL_PROTOCOL_QUEUE.md)

本次公开小原始预测、指标数组、实际脚本和失败日志；不包含模型权重、数据集原图或凭据。作者源码快照附原LICENSE。之前REPRODUCTION_STATUS_20260909.md保留为本轮早期接入状态，不再代表最新结果或排程。
