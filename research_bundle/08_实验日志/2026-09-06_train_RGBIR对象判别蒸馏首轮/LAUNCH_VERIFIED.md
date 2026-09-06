# 完整训练启动核验（2026-09-06 02:49 +08:00）

paired已完成第一轮并进入第2/200轮；第570batch记录563次实际更新、7次AMP skip，KD仍非零。weights/last.pt已生成（约11MiB，含训练状态）；最终比较只使用固定E200 last/EMA，当前best.pt并非基于验证指标选出的模型，不使用它作性能结论。

94 screen：`2654179.rgbir_oev1_queue_s42`；队列PID2654180，guard PID2654181，实际训练CUDA PID2654191。weight0由同一队列在paired训练及统一val完成后开始，当前没有第二个本轮训练进程。

resource guard实测：仅物理GPU4、1个本轮CUDA PID，显存峰值7620MiB，RSS峰值28792MiB。GPU4实际剩余显存16442MiB，满足至少2048MiB余量。预约10000MiB显存/49152MiB RSS。canary较短，完整首轮见到的显存峰值高于canary6304MiB，仍在预约和项目70%上限内。

local `paired_progress_at_launch.json` 是较早的第440batch启动快照；本文件记录随后首轮checkpoint与第2轮的独立核查，不将动态进度覆盖成历史事实。94动态真实状态以full_queue_status.json和对应run/progress.json为准。

结论：授权的两个完整实验已按“paired运行、weight0同卡排队”落地，不是只准备了启动脚本。首轮性能结果仍待E200两臂完整val。
