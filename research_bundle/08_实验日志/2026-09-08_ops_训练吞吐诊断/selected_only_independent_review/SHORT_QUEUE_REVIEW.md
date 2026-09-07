# E20 独立短程队列 driver 快审

**READY（driver 技术范围）。** 2026-09-08，loc_stress；只读核对 `short_screen_draft/run_short_screen_queue.py` 与 `screen_common.py`，未运行队列/GPU/SSH、未修改源码、未计算 hash。

- 六阶段固定串行：N train→N eval→C0 train→C0 eval→C1 train→C1 eval。沿用原 `resource_dispatch.run_job`，不新增调度器、不按 AP 改次序或方法。
- 在创建 campaign 前，三臂 train/eval admission、批准的 config/entry 字节、technical/resource review 与真实证据文件均预检；driver 本身不生成任何 admission/PASS。已有 campaign root 一律拒绝。
- train 预约与外供 resource review 的实测训练峰值比较；eval 预约与既有完整 dev evaluator 的 `evidence_metrics.json` 峰值比较。8192/32768 与 2048/8192 MiB 是预约值，不能替代根尚需确认的实际预算/资源证据。
- 每个 run_job 返回后检查该阶段独立 SHORT_SCREEN receipt 的状态、arm/seed/endpoint/E20 与 dev population。异常或缺失/不匹配即写 failure 并退出，保留已完成阶段和失败目录，不自动重试、不伪造完成。
- 入口、配置、admission 和 native profile binding 外供；其真实有效性仍受现有 entry/guard 检查。此审阅不代替根创建实际独立 admission、确认总预算或执行最终入队，也不升级为 E200/论文增益准入。

未发现阻断准备该队列的技术问题。
