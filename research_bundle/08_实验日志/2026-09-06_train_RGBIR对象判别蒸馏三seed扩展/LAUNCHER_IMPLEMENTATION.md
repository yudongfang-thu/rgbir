# 三 seed 扩展调度实现（2026-09-06）

> 结论：仅新增实验调度器，94 固定 Python 环境下 8 项 CPU 控制流检查全部通过；方法 trainer、loss、数据加载器、配置与评估器均复用首轮 release_v2。GPU canary 和正式运行由主 agent 启动并另记实际状态。

## 实现与实验身份

`expansion_worker.py` 接受 stage、物理 GPU、学生 seed、两臂顺序、artifacts/run 根路径和原始 source 路径。默认 seed0 执行 weight0→paired，seed123 执行 paired→weight0。IR 教师和 RGB 参考始终为 seed42；学生通过原 trainer 的 `--seed 0/123` 覆盖 native 训练 seed。

每个 worker 保存实际学生 seed 的 `effective_config.yaml` 和各臂 `*_seed_override.json`，避免把原始配置快照中的 seed42 误读成实际学生 seed。原方法配置及训练源码未作任何改动。

每个 seed/stage/attempt 使用独立 `workers/<stage>_s<seed>_<attempt>/`，保留 plan、status、训练/评估日志和各臂结果。训练结果目录同样包含实际 seed；已有 run 或 worker 不被复用、移动或覆盖。同一扩展根目录按物理 GPU 加文件锁，两臂串行运行。

## 资源与失败行为

所有 GPU 子进程经原工程 `project_resource_guard.py` 准入，单候选 GPU、单 CUDA PID，预约 VRAM 10000 MiB、RSS 49152 MiB，并预留空闲显存 2048 MiB。原工程 guard 负责项目全局三卡上限等准入限制，主 agent 另行核对既有进程。

只有 guard 返回 2 且记录 QUEUED、从未记录 LAUNCHED 时，才等待30秒并重试同一物理 GPU。已启动训练的任意非零退出码均记录失败且不重跑。单臂训练或评估失败后，继续另一独立臂；最终状态为 `partial_failed`，保留失败日志和原始 attempt。

Canary 每臂完成24次真实参数更新，再于 CPU 直接比较两臂初始 student state 和首批数据张量。另检查实际 seed/arm、非零 KD 梯度、weight0 精确等价及教师/参考无梯度。每个 canary receipt 的实际 GPU、CUDA PID、VRAM 和 RSS 须落在预约内；通过后按 seed 保存 comparison。正式队列仅接受同 seed、同 GPU、同原始 source、已通过资源检查的 canary comparison。

## 验证结果与路径

初版7项检查通过；独立审查指出 canary 实测资源应显式限制在预约内，随后补充该门和边界检查。最终8项全部通过，覆盖准入重试、子进程退出码2不重试、训练/评估失败继续后臂、已有 attempt 保留、seed 专属 canary 门、命令/目录实际 seed，以及资源边界拒绝。

- 本地最终代码：本目录 `expansion_worker.py`；工程中新增同名副本。
- 本地检查：`test_expansion_worker.py`、`launcher_cpu_tests_attempt1.log`、`launcher_cpu_tests_attempt2.log`。
- 94 CPU 检查快照：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_expand_20260906/launcher_cpu_attempt{1,2}/`。
- 固定 Python：`/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python`；CPU 测试设置空 `CUDA_VISIBLE_DEVICES`，未启动训练或 GPU 实验。

这些检查说明调度控制流按设计运行，不提供性能收益证据。真实 canary、资源峰值及正式队列状态以主实验 README 和新产生的运行回执为准。
