# 2026-09-09：90 上的实际复现接入

CFT 作者 checkpoint 经显式兼容适配完成三对 LLVIP fit 前向；BCDL 分类算子 6/6、CMD 原 22 项 CPU 测试通过。没有新增 AP、完整训练或论文整表复现。

- [结果、方法优先级与限制](research_bundle/08_实验日志/2026-09-09_repro_RGBIR对比方法优先接入/README.md)
- [证据范围审阅](research_bundle/08_实验日志/2026-09-09_repro_RGBIR对比方法优先接入/EVIDENCE_REVIEW.md)
- [90 数据与环境接入](research_bundle/08_实验日志/2026-09-09_ops_90迁移与训练接入/README.md)

94 仍故障且旧任务未恢复。本轮无 GPU 长训。CMD 真实 IR 教师仍待绑定。融合模型和 OBB 结果不能充当 RGB-only/HBB 的公平主表对比。

本增量保留报告、脚本、失败/成功回执与小原始预测；不上传权重、数据集图像、第三方论文全文和大源码归档。详细排除清单见 REPRODUCTION_INCREMENT_20260909.json；完整资产位置见报告。本增量不刷新旧总 BUNDLE_MANIFEST 的历史范围，不计算新文件摘要。
