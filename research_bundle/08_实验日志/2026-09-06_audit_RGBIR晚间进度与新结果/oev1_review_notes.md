# OEv1 固定端点与可比性独立复核

> 17:30 +08:00 快照首次得到有效 OEv1 端点：paired seed42 的 E200 开发集 mAP50–95 为 54.6582，高于历史同 seed native 0.8426 pp；本轮同代码 weight0 尚未完成，不能把这个历史差值当作蒸馏净收益。当前完整端点 1/6、完整同 seed 对 0/3。

## 0. 本次新增的实际结果

原始独立 evaluator 的 [evaluation_val.json](oev1_snapshot/raw_runs/full_paired_s42_attempt1/evaluation_val.json) 和 train/eval terminal receipt 均已读取。固定 E200、last/EMA、DroneVehicle RGB val，无 official test，保留 single_seed_exploratory 标记。评估 receipt 创建于 13:55:27 +08:00，采集器 17:30:20 +08:00 确认为 completed。

本地 CPU 复核通过：train/eval receipt 均 COMPLETED，引用的源文件齐全，两个 metric_snapshot 与对应原始 JSON 逐内容一致，eval roster 为 1469 条且全部唯一，collector 指标与原始 eval 精确相等。详见 [复算产物](oev1_first_endpoint_context_check.json)。

| 指标 | OEv1 paired42 | 历史 native42 | 历史 CMDistill42 | 对历史 native 差值 | 对历史 CMDistill 差值 |
|---|---:|---:|---:|---:|---:|
| mAP50–95 (%) | 54.658162 | 53.815556 | 53.244840 | +0.842606 pp | +1.413322 pp |
| AP50 (%) | 77.069620 | 76.001138 | 75.164422 | +1.068482 pp | +1.905198 pp |
| AP75 (%) | 63.839323 | 62.566462 | 62.493584 | +1.272861 pp | +1.345739 pp |

这给出了值得继续等待完整对照的正面初步信号，尚未产生预定 P−N 主量。历史 N 文件虽然名为 `N_llvip_seed42_metrics_record.json`，其内部 dataset/data_yaml/checkpoint 明确属于 DroneVehicle；本次按内部身份读取，未当作 LLVIP 结果。

同时间进度为：weight0 seed42 完成 61/200、当前 62；weight0 seed0 完成 162/200；paired seed123 完成 151/200；paired seed0 和 weight0 seed123 排队。下一项最有信息的结果是 **paired42−weight042**，然后是三 seed 的方向和幅度；继续保持冻结阈值与 E200/last 规则，不因这个先到的正面历史差值改方法。

## 0.1 训练 CSV 末行列数异常的独立源码解释

当前 P42 CSV 的 header 和 epoch199 均为 15 列，epoch200 为 8 列。这不是本次复制截断的推断：固定 Ultralytics 源码给出了对应写入路径。

1. `engine/trainer.py:408` 将 4 个检测指标和 3 个 val loss 初始化为占位 0。
2. `trainer.py:593-596` 即使 `val=False`，最后一个 epoch 也调用 `self.validate()`。
3. OEv1 `train_object_evidence.py:234-235` 的 override 返回 `{},0.0`，使 `self.metrics` 变成空 dict。
4. `trainer.py:604` 只合并当前 train loss、当前 metrics 与 lr；`919-927` 按当次 dict 的长度写 CSV，但已有文件不更新 header。

因此最后一行只剩 epoch/time、3 个 train loss、3 个 lr；按旧 header 用 DictReader 读取会把 lr 错位到 precision/recall/AP50，后续列出现 None。独立 `evaluate_object_evidence.py` 另行执行完整 `model.val`，写出的 JSON 含完整 5 个指标并有 receipt，端点不依赖训练 CSV。

本次不修改原 CSV 或运行源码；监控器不得把这三个 lr 当成检测指标。已存在的固定端点收集器完全不读取 CSV，因此该工程记录缺陷不使独立 AP 端点作废。首轮 completion 还记录实际 optimizer updates 56695、尝试 56722、AMP 跳过 27；对照完成后应并列核这些实际更新信息，不能把 E200 等同于所有 run 更新成功次数必然相等。

## 1. 什么才算新结果

| 证据状态 | 收集器状态 | 允许解释 |
|---|---|---|
| 尚无 run 目录 | pending_not_started | 尚未找到本轮正式 run |
| 无 training completion | pending_training | 训练终态证据尚未出现；需另外看 PID、队列和退出码 |
| E200 completion、last.pt 存在，但 train receipt 缺失 | pending_training_evidence | 训练自报完成，审计证据待齐 |
| train 证据有效，尚无 evaluation_val.json | pending_evaluation | 训练完成，检测性能尚未完成评估 |
| 有 eval JSON，但 eval receipt 缺失 | pending_evaluation_evidence | 有待核实指标，暂不进入正式端点表 |
| E200、last/EMA、val、train/eval receipts 和指标快照全部通过 | completed | 可报告该单 run 的开发集性能 |
| 身份、预算、端点、数值、快照不一致 | invalid_evidence | 不进入性能比较，先诊断证据链 |
| 存在 failure_receipt.json | failed_training | 训练失败；不得拿其历史中间值补终点 |

收集器只在同一个学生 seed 的 paired 和 weight0 均 completed 时计算差值；只有 0/42/123 三对齐全时才计算三 seed mean ± sample SD（ddof=1）。主量固定为 `100*(paired.mAP50_95-weight0.mAP50_95)`，次指标为 AP50/AP75/precision/recall，不在看见结果后换主指标或改用 best。

## 2. 收集器本身的边界

- `pending_training` 和 `pending_evaluation` 是证据状态，不是进程活性判定；评估异常没有单独的 failed_evaluation 分支。应合读 queue status、worker stage、return code、最新日志和实际 PID。
- 当前收集器验证 roster/source 文件存在、指标副本一致，检查实际 seed/E200/batch32/imgsz640 和固定 seed42 teacher/reference 路径；它不逐项验证全部超参、不检查 checkpoint 张量身份、不自行比较六臂 roster 内容和每个 ID。
- 原 evaluator 明确要求 train/val-only YAML、1469 张 val、固定环境和该 run 的 last.pt。因此实际源快照、实际配置与 roster 的直接内容核对仍有价值；不是发现了训练错误。
- 队列执行期间分文件读取不是原子快照。刚写到 completion 或 eval JSON 时可能暂处于 pending；重新采集应写新快照目录，保留旧快照。
- 当前脚本明确不是 accepted analyzer，不将单 run exploratory 标记升级为 confirmatory，也不从三个 seed 输出 p 值。

## 3. paired seed42 对历史 baseline 的可比边界

可以把历史 DroneVehicle RGB baseline 约 53.82 mAP50–95 作为背景，说明新端点处于什么性能量级；必须标明“历史参照”。原 N/L 三 seed 的实际主要 recipe 已核同为 batch32/nbs64/E200、同初始化路径、增强和优化器；不能依据目录名 `b32a2`/`b32_e200` 再次认定它们预算不匹配。

但历史 N 不能替代本轮 weight0：新 OEv1 的 workers 从历史 8 改为 4，使用配对数据流水线和自定义 trainer，还增加 IR 独立 GT 对应、冻结参考筛选和相对背景类别证据。两臂当前同代码、同 seed、同学生路径且完整 E200 的 P−N 才是本轮预定主量。即使 paired42 高于历史 N 或历史 CMDistill，也不能提前称为 KD 净增益。

## 4. 最有信息的下一个判断

1. 首先等同 seed 的 paired/weight0 固定端点配齐，观察方向和幅度；任一单独端点只能报告性能，不能定成功/失败。
2. 三对齐全后报告逐 seed 差值和 mean ± SD，并明确 teacher/reference 固定 seed42、已测数据顺序及首批增强跨 seed 相同，重复主要覆盖初始化变化。
3. 如果 P−N 在三 seed 稳定为正，最多支持“对象判别证据干预相对同代码 weight0 有净收益”。这还不是跨模态独特知识、选择规则有效或避免负迁移的因果证明。
4. 随后的归因应独立冻结 shuffled/same-modal、同 E/同 K/同 λ/同分母的随机选择，以及必要的同 mask GT-only 内容控制；不能用明显破坏坐标的整图随机错配单独证明实例知识。
5. 如果 P−N 不稳定或为负，如实收束当前冻结版本；不要根据一个早到端点改 λ、ρ、候选阈值、checkpoint 规则或追加 loss 挽救原协议。

从 521 对诊断仍可保留的设计依据是：DroneVehicle 的 IR 独有命中有 118/360 个属于 RGB 已有正确位置的低置信候选，但双命中目标的 IR 定位优势中位数接近零；这支持先验证对象相对背景的判别信息，不保证蒸馏能改善 AP。LLVIP 定位机会更明确；VEDAI 为 RGB–NIR，教师方向不能默认 IR→RGB。

## 5. 复核来源

- [当前收集器](../2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/analyze_three_seed_endpoints.py)与[判读说明](../2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/ENDPOINT_ANALYZER_NOTES.md)。
- [首轮冻结协议](../2026-09-06_train_RGBIR对象判别蒸馏首轮/EXPERIMENT_PLAN.md)、[实际 evaluator 副本](../2026-09-06_train_RGBIR对象判别蒸馏首轮/code/evaluate_object_evidence.py)。
- [跨 seed 随机性核验](../2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/cross_seed_realization.md)。
- [最新 RGBIR 诊断与旧结论勘误](../../07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md)。
- [固定 Ultralytics trainer 源码快照](../../remote_snapshot_20260906/framework_snapshot/ultralytics/engine/trainer.py)。
- 本目录 [原始端点 CPU 复算脚本](recalculate_oev1_context.py)，历史数值输入为 `09_外部审计_rgbir/02_raw_results_dronevehicle` 中同 seed N/L 原始 JSON。

本复核仅阅读本地证据和源码；没有启动、停止或改动训练，没有访问 GPU、检索文献或生成 hash。
