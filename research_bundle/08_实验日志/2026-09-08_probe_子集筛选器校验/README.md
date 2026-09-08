# Drone 固定2048子集筛选器校验（2026-09-08）

**已完成并独立验收：两臂canary、E8训练及完整dev评价总墙钟9分42.24秒；N/C0 mAP=27.588046/26.459942，C0−N=−1.128105pp。固定2048子集未保留已知全量E8正方向，按预设规则结束本版筛选器，不在其上扩矩阵，也不据此判C0无效。**

优先阅读：[最终结果与去留](FINAL_REPORT.md) · [原值和逐类读出](analysis/output_attempt1/README.md) · [独立实际验收](independent_review/FINAL_EXECUTION_REVIEW.md) · [曝光与训练记录对照](SCREEN_VALIDITY_REPORT.md) · [下一步具体任务](NEXT_ACTION_REVIEW.md)。

任务依据为[上一阶段冻结的具体任务](../2026-09-08_probe_对象坐标分布目标/NEXT_FAST_SCREEN_TASK.md)。范围固定`DRONE_SUBSET2048_PRETRAIN_E8_CHECK`，Drone原2048自然分层训练子集、完整dev1469图/22462GT、seed42、N/C0 λ=0/.1、原通用yolo11n、原E8优化/增强与正常BN。每臂8轮×64批=512批，固定last EMA，结束后各一次完整dev独立评价。

先通过新接口小检查与各24次成功更新的真实canary（最多48次optimizer尝试，并记录前30批），再用本路径实测成本核验45分钟执行预算；排队时间单列。预计超预算就阻塞，不降低batch/workers或轮数凑预算。训练/评价均使用原全局lease，不接管旧C1/归因任务或其提速切换。

不重抽子集、扫λ或日程。C0>N只说明已知对照在本子集/seed/短日程保留；非正向则本筛选器未通过校验，不推出C0无效。任何结果均不验证一般方法排序或E200预测性。

实现/实际证据保存在本目录。计划远端根为`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_subset_e8_check_20260908`，新release与attempt，不覆盖原产物；不访问test或计算新证据hash。

## 实际准入与执行

`evidence_canaries/queue/canary_checks.json`记录两臂通过，N/C0各30/31个实际batch。两臂NVML峰值均6692MiB，进程树RSS峰值18120/18000MiB；正式训练分别按7168MiB显存、20480MiB RSS预约。初始化与输入张量的私有参考仅保留在94，未下载。

`evidence_canaries/queue/budget_decision.json`记录正常batch平均0.578092/0.547157秒；计入已用执行时间、初始化、实际输入核对I/O、20%训练余量及两次评估，预计总928.842秒，低于2700秒硬限。该数是启动前预算，不冒充最终实测总时长。

实际六阶段全部正常退出：N/C0训练执行230.134/219.300秒，完整dev评价22.610/22.627秒。累计执行571.548秒、入场等待10.687秒，总墙钟582.236秒；源码实现、审阅、采集及发布不在该队列时间内。两臂各512批/304次尝试，成功更新298/297次、AMP skip6/7次、EMA各304次。独立验收通过后，单次读出认定方向未保留。

当前不新增本子集E8训练。旧正式任务继续原责任队列；本条目原始证据镜像至94同根`review_v1/`，发布到既有GitHub分支`research/full-evidence-20260906`，实际状态以镜像/Git回执为准。私有像素参考、权重和完整主机进程资料不进入发布包。
