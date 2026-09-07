# 06:36 核对：E8 三臂完成

**N/C0/C1 均完成固定 E8 及完整 dev 评价，持久队列六阶段完成；七项状态/资源检查通过。接受分析器后的三臂判断见 [阶段报告](../three_arm_summary_20260908/README.md)。**

C1 完成 4504 批、2667 次成功更新、7 次 AMP skip、2674 次 attempt/EMA。mAP50–95/AP50/AP75 百分制原值为 43.408107/64.712517/50.145839。训练 4391.556851 秒，独立评价 14.738947 秒。已完成后的旧 progress 保留最终保存前状态，完成身份以 training/evaluation receipt 及 queue completion 为准。

当前剩余五个项目 E200 CUDA 任务：GPU2/4/5 各 2/1/2 个，项目实测 RSS 141993MiB；项目显存<70%，各项目卡全卡余量≥2GiB，实测及预约总内存均低于300GB。本次不启动新任务，E8释放的资源保持可用。

`snapshot.json`为原只读状态与资源证据，`checks.json`为七项检查，`collect_heartbeat.py`为采集入口。完整主机 snapshot 不进入GitHub公开增量。C1新小产物存 `../endpoint_C1_20260908_063705/`，原权重与压缩预测留94；本次无新hash、训练或重复评价。
