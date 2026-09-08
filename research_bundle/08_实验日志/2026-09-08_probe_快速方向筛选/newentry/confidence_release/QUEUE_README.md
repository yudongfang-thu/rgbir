# LLVIP N/C0 独立 mini 队列

**代码就绪，6/6 CPU 合同和两份实际配置检查通过；本任务未启动队列或 GPU。**

运行 `run_confidence_queue.py --release-dir <confidence_release_v1> --output <confidence_attempt1>`；可显式给 `--reference-dir` 与 `--wait-for`。固定 Python 保留 venv 的 `bin/python` 路径，不 resolve 到底层 Python。所有训练/评价都交给原 `resource_dispatch.run_job` 和同一 globallease，不创建资源池。

默认等 `rgbir_direction_screen_20260908/screen_attempt1/queue/completion.json`；每 30 秒仅 CPU 检查，等待期不拿 GPU lease。只接受 `DIRECTION_MATRIX_COMPLETED` 或 `DIRECTION_MATRIX_COMPLETED_WITH_BLOCKED_ARMS`。若原队列为 PARTIAL_FAILURE，写本 mini 队列 failure 并停止，明确等待共同实现问题复核，不因完成文件存在就继续。旧入口及运行产物均不修改。

固定执行顺序：新 N canary24 → 新 C0 canary24 → 两臂首 30 批 RGB/IR/GT 字段 exact、共同初始 checkpoint stat exact → N train/eval → C0 train/eval。完整训练仍需首 30 批等于各自 canary，3轮/192批及实际 BN buffers unchanged。配置复制为 `frozen_configs` 并在各步检查字节未改；C0 固定原置信度分支 λ=0.1，N=0，无定标或 AP 自适应。N 必须新跑，不替用旧 N。

Canary ceiling 为 VRAM8192/RSS32768 MiB；完整训练按实际 NVML/torch峰值与进程树 RSS 加原 margin、向上取整后预约，超 ceiling 直接停止。独立评价固定2048/8192 MiB。所有实际资源余量及全项目上限仍由原 globallease 校验。

输出布局是 `canaries/{N,C0}`、`runs/{N,C0}`、`evaluations/{N,C0}`，没有额外 llvip 层。成功评价文件保留 `direction_evaluation_receipt.json` / `DIRECTION_EVALUATION_COMPLETED`，但身份必须是 `LLVIP_CONFIDENCE_FT3` / `LLVIP_CONFIDENCE_FT3_LAST_EMA`，完整 dev2406/7879。队列全部六阶段通过才写 `queue/completion.json` / `CONFIDENCE_MATRIX_COMPLETED`。失败保留阶段 job 和原回执，写 queue/failure，不自动重试、继续 C0 或进入 E200。

证据：[QUEUE_CPU_attempt2.json](QUEUE_CPU_attempt2.json)、[QUEUE_ACTUAL_CONFIG_CHECK.json](QUEUE_ACTUAL_CONFIG_CHECK.json)。原 attempt1 及前置失败门修订前源码保留；当前版本已拒绝主队列 PARTIAL_FAILURE。独立审阅由协作任务另写，作者不自行宣称 accepted。
