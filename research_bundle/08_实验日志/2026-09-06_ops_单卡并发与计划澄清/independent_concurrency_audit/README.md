# 同卡双训练短测独立核验

> 通过：GPU2 上原 R42 正式训练与 seed0 随机对照短测确实同时发生；短测完成24次真实参数更新，原训练持续前进。该证据支持同卡双开准入，不是两次完整200轮长期共存的峰值/吞吐保证。

## 原始采样核验

| 项目 | 独立读取结果 |
|---|---:|
| GPU2 出现2个 CUDA PID 的外部采样数 | 30 |
| 原 R42 CUDA PID | 975949 |
| 短测 CUDA PID | 1097224 |
| 外部全卡峰值占用 | 13,973MiB |
| 外部最小空闲 | 10,111MiB |
| 短测进程返回码 | 0 |
| 短测 batch /真实 optimizer update /AMP跳步 | 30 /24 /6 |
| 短测 KD scores 梯度 L2 | 0.0037815419491380453 |
| 总梯度线性组合最大误差 | 7.62939453125e-06 |
| 教师、RGB参考是否有梯度 | 均无 |
| 短测进程子树 GPU峰值 /RSS峰值 | 6,304MiB /28,716MiB |

初始学生和首批数据与原 P0 canary 逐张量一致；30个共同batch的E/K/分母与名义剂量一致，30批均选了不同对象。loss/loader与原实现逐字节相同，trainer改动仅预期的random路由。

## 原长训未被中断

外部采样前后 R42 的 optimizer_updates 从2313到2373、batches从3795到3915，CUDA PID 始终为975949。短测进程退出后GPU2回到单PID、占用7652MiB、空闲16432MiB。之后独立读取进度已进入epoch8、2493次真实更新；没有该run的完成回执，符合仍在正常长训。

## 审计注意

1. 原 canary completion 的 `resources.cuda_pid_counts={2:1}` 是其资源采样范围，并不表示物理GPU仅一个进程。必须用同目录外部 `concurrency_profile.json` 中每次 nvidia-smi 的 PID 列表证明双开。
2. `minimum_observed_free_mib` 是约1秒粒度离散采样所见最小值，不是严格连续监测下界。
3. 6304MiB 是短canary峰值；完整原训练已经约7630MiB。不能据此以6500MiB预约完整200轮，也不能据此断言三开可行。
4. 检查确认共存与梯度有效，但没有随机化单开/双开吞吐对照，不能报告明确加速倍数。

## 来源与操作范围

`fetch_concurrency_audit.py`只通过scp和远端Python读文件、nvidia-smi采样；没有写服务器、启动或停止进程。远端原始路径为 `RGBT_campaign/artifacts/oev1_concurrency_20260906/`。本目录保留原JSON/log、副本提取表与短测后的独立快照。
