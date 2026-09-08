READY_FOR_CANARY

限定源码审阅通过。已读 `release/export_drone_teacher.py`、冻结 `drone_teacher_spec.json`、根及 release 的同字节 `run_capture_queue.py`，并核对原生 capture/profile helper。独立复跑 6 项纯 CPU 合同测试通过，回执在 `independent_review/CPU_REPLAY.json`；root 提供的部署回执记录同源字节一致与 pinned CPU 通过。尚未据此接受实际 canary、完整 capture 或新 AP。

- 固定历史 IR42 E200 last/EMA 身份、checkpoint stat、seed/epochs/data modality 和五类顺序；新输出目录不覆盖旧结果，不写权重或训练状态。未伪造现行 N 的训练合同。
- full dev 1469 图、24490 IR GT、逐图计数、五类计数及 loader 原标签数组检查均在实际入口。保留 processed image alias 查标签，canonical roster 另作身份绑定；两幅空 GT 图保留。固定首 32 图为 514 GT，仅类 0–3 出现，允许四类实际 AP，模型和数据仍是五类。
- 同一原生 YOLO.val 路径：FP32、640/B32/workers4/rect、conf .001、NMS IoU .7、max_det 300。capture 在原生 metric 更新后读出每图预测与 own-IR GT，包含空预测图，并保留实际 canvas/original shape。canary 为两个 fresh model 的 native/capture 评价，要求总体与实际各类指标 exact、effective kwargs 和 loader 顺序一致；full 只有一次 capture。
- 队列使用原 global resource dispatcher 与原 pinned Python 路径，先 canary 后按实际 NVML/torch 峰值和进程树 RSS 留余量预约 full；实际成功回执、514/24490 GT 和缓存存在性均硬检查，异常退出保留 attempt。原 dispatcher 负责共享资源余量及项目约束，未新增资源池或自动重试。

几何边界正确：独立 IR 标签为 24490，不能与 RGB 的 22462 GT 要求逐行相等；同名 1469 帧只证明图像名单对应。之后如需跨模态实例关联，须独立限定其规则，不能把本 capture 视为物理配准或 L1 几何放行。本审阅无 GPU、SSH、模型加载或新 hash。
