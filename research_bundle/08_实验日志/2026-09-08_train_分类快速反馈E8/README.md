# 分类快速反馈：匹配 N/C0/C1，seed42，E8

**04:36核对：N已完成E8和完整dev独立评估，mAP50–95=43.376542；C0第4/8轮，C1待运行。N端点交叉核对通过，目前只有短程参照原值，尚不能判断蒸馏收益。** [N完整端点](endpoint_N_20260908_043812/README.md) · [回执复核](endpoint_N_20260908_043812/ENDPOINT_REVIEW.md)。

## 目的与设置

原 C1 E200 的等待过长，正常日志频率实测 N/C0/加速 C1 分别为 0.858475、0.890618、2.475788 秒/批。三臂 E20 纯训练估计13.21小时，故采用此前预留的 E8 早期筛查方案。吞吐证据及失败优化候选均保留在 [训练吞吐诊断](../2026-09-08_ops_训练吞吐诊断/README.md)。

三臂共同设置：Drone train17990/dev1469，22462个dev GT；YOLO11n原通用预训练初始化；seed42、640、batch32、nbs64、workers4、SGD、AMP、原数据顺序与增强；**共同8轮及8轮学习率日程**，warmup3轮保持不变。系数N=0、C0=0.1、C1=0.09227393550836771。教师与参考、对象选择、阈值及归一化均沿用原定义。

C1仅采用已验证的selected-only实现：只为选中对象计算参与学习的S/T全类相对logit，原pool算术与FP32 cast位置不变。新旧各24次真实更新的学习损失、梯度、参数、optimizer及EMA状态逐位一致；该验证不等于E200全轨迹保证。

## 固定比较与顺序

顺序为 `N训练→N独立评估→C0训练→C0独立评估→C1训练→C1独立评估`。每臂固定E8 last/EMA，完整dev，沿用已接受native evaluator；不读取封存test，不根据中途AP改变轮数、系数或剩余臂。

比较C1−N、C1−C0、C0−N，报告mAP50–95/AP50/AP75/逐类AP和更新、AMP跳步。单seed没有跨seed方差；3轮warmup占E8的37.5%，结果仅反映早期学习，不能证明最终收敛增益，不能与旧E200 N/C0直接相减，也不用于终止现有E200。

## 预算与资源

按每轮563批，三臂纯训练约5.29小时；加20%训练波动余量及1小时启动/评估/调度准备量，总计约7.34小时。此为短窗口外推，不是截止时间保证，未知排队或共享负载变化可能超出。

正常频率24更新实测NVML为6304/6304/6368MiB，进程树RSS28738/28737/28703MiB；训练预约8192MiB/32768MiB。复用的完整dev评估实测NVML1370MiB/RSS6260MiB，预约2048/8192MiB。所有阶段由原全局lease动态选卡；项目常规最多3张卡、当前每卡最多2个项目CUDA任务、整卡至少2GiB余量、项目显存<70%、总RSS≤300GB。失败保留attempt并退出，不自动重试或改配方。

## 代码与证据位置

- 新E8代码：`../2026-09-08_ops_训练吞吐诊断/short_screen_E8_release/`，独立于原E20草稿及在跑release。
- 24更新一致性：[SELECTED_UPDATE24_RESULT.md](../2026-09-08_ops_训练吞吐诊断/SELECTED_UPDATE24_RESULT.md)。
- 正常频率实测：`../2026-09-08_ops_训练吞吐诊断/remote_short_profile_attempt1/`。
- 计划远端release：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_throughput_20260908/short_screen_E8_release_attempt1/`。
- 计划远端campaign：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_short_screen_E8_20260908_attempt1/`。

已有5个E200任务未修改。screen为 `rgbirfast_matrix_42`；N启动PID3395157，lease `short_N_s42_E8_train-2bd398c13898`，启动后项目物理卡为2/4/5，未援引第四卡条款。持久队列顺序负责后续五个阶段，禁止另开重复任务。

实际冻结、启动、准入、源副本与独立差异审阅已收至 `launch_evidence_attempt1/`，采集入口为 `collect_launch.py`。既有每小时heartbeat已改为优先跟进这条短程队列及故障/完成回执，无变化时静默。小产物同步本目录，大权重仅记录服务器路径；未计算新hash。

03:37只读跟进：[状态与资源正常](heartbeat_20260908_033711/README.md)。03:42 N已进入第2/8轮，首轮530.879秒，比短窗口外推高约9.8%，仍在原20%训练预算余量内；尚无独立评估，不改变预算与排程。

04:36跟进：[N完成、C0训练、七项状态/资源检查通过](heartbeat_20260908_043626/README.md)。N实际训练50.963分钟，独立评估14.542秒，AP50/AP75为64.983495/50.301986；共享负载变化，不能把阶段耗时下降归因于代码加速。仍按原顺序完成C0/C1，不按N原值改变科学配置或E8日程。原训练CSV末行列短，禁止从训练CSV读取AP，详见端点报告。
