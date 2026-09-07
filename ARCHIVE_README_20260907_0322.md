# RGBIR跨模态蒸馏：完整证据与独立审计入口

**最新更新：2026-09-07 03:22。请先读 [阶段状态与证据](TASK_CONDITIONAL_STATUS_20260907.md)，再按 [本轮独立复核请求](TASK_CONDITIONAL_REVIEW_PROMPT.md)检查代码及原始数据。**

当前主线为未改定义的OEv1判别蒸馏C，以及新实现的条件定位L。C42相对同代码N42的mAP净差仅+0.144554pp，AP75为负，仍缺完整三seed。新D1/D2已覆盖两数据集各2048张训练图和200张开发图；定位信息存在，但当前已接受物理配准覆盖内D2=0，所以CL/CGT尚未开始。

N/C与两个C内容对照均完成24次成功更新，新旧C/N真实损失和梯度等价通过。按预设C归因分支，C-shuffled42已启动E200，C-same-modal42在统一资源队列等待；原N/C/random三seed继续，OS-SSL、VEDAI暂停。

- [冻结计划与执行验收](research_bundle/08_实验日志/2026-09-07_train_TaskConditional首轮/README.md)
- [独立模块源码、测试和运行入口](research_bundle/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1)
- [D1/D2完整结果、图表、gzip原件和复算脚本](research_bundle/08_实验日志/2026-09-07_probe_TaskConditional机会诊断)
- [几何标点、独立复核、拒绝案例及适用范围](research_bundle/08_实验日志/2026-09-07_probe_TaskConditional几何审计)
- [实验日志总索引](research_bundle/08_实验日志/README.md)
- [历史全项目审计入口](ARCHIVE_README_20260906_2146.md) · [通用模型复核指南](MODEL_REVIEW_GUIDE.md)
- [本次导出清单](TASK_CONDITIONAL_BUNDLE_MANIFEST_20260907.json) · [之前的完整包清单](BUNDLE_MANIFEST.json)

本仓库用于独立检查一个面向J-STARS的研究项目，保留正负结果、失败attempt、勘误和时间戳快照。训练中任务没有AP，单seed不代表稳定增益，未核验几何的诊断不能授权定位长训。仓库不含凭据、大权重或数据集原图全集；服务器路径与模型身份保留供复核。
