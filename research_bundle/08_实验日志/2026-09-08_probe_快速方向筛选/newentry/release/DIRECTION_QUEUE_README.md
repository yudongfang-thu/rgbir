# 方向筛选统一队列：CPU 已检查，未启动 GPU

入口 [run_direction_queue.py](run_direction_queue.py)。只调用现有 `rgbir_task_conditional_v1_20260907/release_v8/resource_dispatch.py::run_job`，不创建新资源池、不导入即执行 GPU、不计算新 hash。7 项 [CPU mock 检查](DIRECTION_QUEUE_CPU_CHECKS.json)通过，含七个实际配置、两数据集顺序、失败分流及完整前三十批合同；mock PASS 不是实际 canary/训练通过。

```sh
/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/release_v1/run_direction_queue.py --release-dir /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/release_v1 --output /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/screen_attempt1
```

root 通过既有持久 screen/lease 运行；本入口拒绝已有 output，不自动覆盖或重试。

固定顺序为 LLVIP 后 Drone，每个数据集：calib8→所有有效臂各 24 成功 update canary→每臂 train 后立即 eval。LLVIP N/L2-box/L2-GT；Drone N/C1/C2/F-rel。校准与 canary 使用原测量所支持的 8192 MiB VRAM/32768 MiB RSS 上限；正式训练按实际 canary NVML/allocated/reserved/RSS 峰值加原固定余量预约，上限不自动扩展；eval 2048/8192。继续接受原调度器的共享 GPU、全卡余量、项目总内存与任务槽检查。

校准仅接受固定8批、每批恢复该数据集共同学生起点、无 optimizer/EMA 更新且 BN buffers 未变的回执。N=0，C1=0.09227393550836771；新候选仅接受有限 (0,1] 或带明确原因的 null。L2-GT 与 L2-box 保持相同校准有效性/系数。全部有效系数写入 `effective_configs/` 后冻结，canary/train 使用同一份 YAML；F-rel 仅 `kd_coefficient` 非零，classification/localization 系数为0。缺字段、NaN、无原因 null 均是技术错误，不能当作低支持自动跳过。

同数据集有效臂 canary 的初始化 checkpoint stat 与前三十批 RGB/IR 名单、标签和增强后框逐 record exact；每个正式 train 的前30批再与其 canary exact。BN 实际 running buffer 不变、affine 可训练、完整 head warmstart、fresh optimizer/EMA 都按真实回执验证。独立 eval 紧接通过 train，并核完整 dev1469/22462 或2406/7879；LLVIP 使用显式 `llvip_native_evaluation.yaml`，不复用 Drone profile binding。

显式无效校准只跳该候选，并保存原因；一个数据集的技术错误停止它的后续阶段、保存 traceback 与已完成/已验证阶段，再允许另一个数据集继续。最终若有技术失败，写 `DIRECTION_MATRIX_PARTIAL_FAILURE` 并返回非零，不把部分完成写成完整成功。无 AP 读出后的动态调参、挑臂或延长。
