# 协议补充与来源记录勘误（2026-09-07）

根任务补充此前用户约束：**不新增 hash/SHA 计算**。首次 PROTOCOL.md、官方源下载回执与已完成 Drone `drone_results_v1` 原样保留；其中的 hash 是未继承该约束时已经产生的记录，不据此扩展计算。首轮 runner 原始副本保存在 `drone_results_v1/runner_source.py`。

后续运行入口 `run_tide_audit.py` 已移除 hash 计算，改用路径/size/mtime、源码副本及逐图完整 GT 直接相等检查。不再重新下载或计算官方源散列。此变更只影响来源元数据，冻结的预测、匹配、阈值、误差 oracle 与统计口径不变。

已核实 Drone 原 pinned evaluator 使用同画布直接比较，代码未在 AP 主路径做缩放/clip。实际与 TIDE 的差异包括：pinned IoU 优先全局候选去重匹配和 np.interp+梯形积分；TIDE 分数优先逐检测匹配和101个 recall 采样均值。因此同一物理 GT 集可核一致，但 TP/FP、error oracle 的有效分母和AP积分不与原 pinned 口径等同。

已核实 LLVIP 旧200图缓存最低分约.05，不能重建完整dev低阈值AP。之后接入的新全量LLVIP预测必须独立记录预测协议、模型身份及单seed/跨模态模型差异；不当作匹配三seed KD效果。
