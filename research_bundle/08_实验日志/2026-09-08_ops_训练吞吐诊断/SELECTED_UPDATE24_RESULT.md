# Selected-only 原池化：24 次更新结果

**本次旧/新 C1 各 24 次真实更新的训练轨迹字节精确，原固定容差也通过。** 这是 seed42 的一次受限短程验证；不推导 E200 等价、正式 epoch 吞吐或检测增益。旧 block16 的真实轨迹失败仍独立保留。

实际输入是 `remote_selected_update24_probe_attempt1/comparison_attempt1/`。已运行只读汇总器，独立将 `comparison.json` 的逐文件记录重新聚合，与所有分类汇总完全一致，并核对两份 worker/coordinator 回执、逐批 KD 与共享梯度/sanity JSONL。这里没有重新打开远端完整 state `.pt`；张量字节/容差结论来自已执行的完整比较器。

| 核对项 | old | selected-only |
|---|---:|---:|
| 实际训练批次 | 30 | 30 |
| optimizer 尝试 / 真正成功更新 / AMP 跳步 | 30 / 24 / 6 | 30 / 24 / 6 |
| 薄学习批次 | 0 | 30 |
| 完整诊断批次 | 30（原全量路径） | 30（另行 no_grad 统计） |
| 回退批次 | 0 | 0 |

新臂的 30 批全部使用真实 selected-only 学习图；sanity=True 只令每批另外采集完整统计，没有让学习 loss 回到旧全量图。原 target/off-target 组件与共享参数梯度观察作用于真实薄学习图。

初始 student/EMA/optimizer/scaler/RNG、首批真实 RGB/IR 像素、30 批 source/双标签/增强元数据/worker RNG、所有选取身份/有效尺度/quality/C0、selected S/T delta、每批 native/KD/加权 KD/总损失与 target/off-target 组件、scaled/applied 全梯度、逐次 student 状态/optimizer/EMA/scaler/计数均 **exact，max abs=0**。这里的 student 状态包含参数和 buffers。六次 AMP 跳步及对应非有限梯度位模式也一致；不能把匹配的非有限位模式称为可用有限梯度。共享梯度观察 JSON 与 sanity JSON 也完全一致。

学习快照没有采集未选对象 S/T delta 或 R delta，故不把它们的私有占位当零测量或比较失败。完整统计日志另由本批同 raw 的原 no_grad 实现产生。后续 29 批没有额外保存全部像素，因此“flow exact”限定为实际已比较的 source、双标签、增强元数据和 RNG。

## 计时：完整 wall 与诊断开销分开

| 秒 | old | selected-only |
|---|---:|---:|
| 完整 trainer span（含初始化等） | 134.085289 | 187.673852 |
| 其中审计复制/保存/检查 | 10.832210 | 10.142454 |
| 新增 no_grad 完整诊断 | 0 | 81.048239 |
| span 扣审计 | 123.253079 | 177.531398 |
| span 再扣额外完整诊断 | 123.253079 | 96.483160 |
| 去前 6 批后，完整 batch wall median | 3.560012 | 5.424252 |
| 去前 6 批后，扣审计 batch median | 3.237961 | 5.071938 |
| 去前 6 批后，再扣额外完整诊断 batch median | 3.237961 | 2.283448 |

完整诊断 canary 的新臂实际总 wall 更长；它每批额外执行完整统计来保留验收证据。将明确测量的审计与额外统计分列后，观察到的 batch 中位数约为旧路径的 **1/1.42**。这只是插桩后的时间分层：callback 不含 loader fetch，额外诊断改变缓存和重叠，**不能当作正式 epoch 速度承诺**。24 个热身后样本逐项保留在 summary.json，没有只报告最快批次。

## 输入、源码与实测资源

两臂都从原 `yolo11n.pt` 初始化，保持正式 C1 seed42 的 E200 recipe、B32/nbs64/workers4/AMP 和实际 λ=`0.09227393550836771`。教师是旧 DroneVehicle `infrared_seed42_native_b32a2/weights/last.pt`，参考是旧 `rgb_seed42_native_b32a2/weights/last.pt`；不是从本次 raw capture 的已训练学生继续更新。实际完整路径保留在 [source_input_review.json](selected_update24_readout_attempt1/source_input_review.json)。

独立逐字节核对两臂 **65 对源码快照**、配置副本，以及输入 model/teacher/reference 的 path/size/mtime；全部一致。实际执行的 `selected_only_v1.py` 和 `compare_24_selected_only.py` 还与当前冻结本地文件逐字节一致。未加载权重内容，没有计算新 hash。

GPU4 上两臂 worker 回执的 NVML 显存峰值均为 **6510 MiB**；进程树 RSS 峰值分别 **29367 / 29397 MiB**。PyTorch allocated 峰值均为 5087.686 MiB，reserved 为 5976 / 5866 MiB。原 guard 正常退出，无 monitor error；整卡最小剩余显存 **4236 MiB**，54 个监测样本中的本项目总体 RSS 峰值 **173374 MiB**。这些实测值与 8192 MiB 显存、32768 MiB RSS 的预约区分记录。

本次证据支持继续受限反馈试验时使用这个已核对版本；任何 E20/E200 排程、效果评价或长期替换还要按各自冻结协议与实际结果判断。这里没有停止旧 E200、启动新训练或作长期准入。

产物：[JSON 汇总及短报告](selected_update24_readout_attempt1/README.md)、[完整 summary](selected_update24_readout_attempt1/summary.json)、[源码/输入/guard 核对](selected_update24_readout_attempt1/source_input_review.json)。只读入口为 `performance_candidate/update24_selected_only/summarize_completed.py` 与 `verify_collected_sources.py`；旧脚本、旧失败和远端 state 文件未改。
