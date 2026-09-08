# 阶段报告勘误：实际GPU与四卡放宽回执

`STAGE_REPORT.md` 中“均由现有全局lease在GPU4动态获准，沿用3张项目物理卡”的概括不准确。主矩阵实际使用GPU4和GPU7；其中Drone N完整训练在GPU7，当次按工作区允许的四卡例外获准。保留原阶段报告，不覆盖历史原文。

已公开的 [本任务准入与guard必要字段摘要](RESOURCE_ADMISSION_EXCERPT.json) 保留原来源路径、字节数和采集字节核对结果。原Drone N_train admission明确记录：

- 加入后项目active GPU为 `[2,4,5,7]`；完全空卡为 `[1,3]`。
- `four_gpu_exception=true`，`minimum_empty_required=2`。

因此此次GPU7加入具备“最多4张且加入后仍有至少2张完全空卡”的明确放宽依据，不是无回执扩卡。原 [job预约](results_1203_snapshot/queue/direction_screen_attempt1_drone_N_train_job.json) 为7168MiB显存、30720MiB进程树RSS；原 [训练完成回执](results_1203_snapshot/runs/drone/N/completion_receipt.json) 记录GPU7单CUDA进程、实测NVML峰值6776MiB、进程树RSS峰值27822MiB。

上述[摘要](RESOURCE_ADMISSION_EXCERPT.json)中的该stage guard为COMPLETED、exit_code=0、monitor_errors为空；55条采样中全卡最小空余13474MiB，项目总RSS采样最大169909MiB。余量和项目RSS采样均满足工作区门限；这是已有guard的实测范围，未声称连续每一时刻均被本次复核独立观测。

完整admission与resource_profile原回执仍保留在本地 `results_1203_snapshot/queue/`，按发布过滤规则不上传；公开摘要仅含本任务GPU7准入、资源预算/峰值及guard必要字段，不包含其他用户进程、命令/PID或完整主机快照。

本次只读复核使用本地12:03已收的小回执；admission实际上已经存在，因此没有再SSH、启动GPU或读取其他用户进程信息。没有修改旧结果或计算新文件hash。综合草稿采用上述纠正后的资源表述。
