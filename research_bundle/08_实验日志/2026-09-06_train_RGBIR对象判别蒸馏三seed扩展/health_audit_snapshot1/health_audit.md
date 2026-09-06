# RGBIR OEv1 seed42 运行健康只读审计

> 结论：当前训练与蒸馏信号没有出现日志可见的数值崩溃或选样归零；训练尚未完成，尚无检测性能或跨模态净收益结论。

采集时间（UTC）：2026-09-05T23:34:16.909357+00:00。原 run 只读，无 GPU 计算、无权重读取、无训练状态修改。

## 训练进度与数值

CSV 已完整写出 86 / 200 轮；progress 当前第 87 轮。真实更新 24646，尝试 24660，AMP skip 14（0.0568%）。更新计数可核对：True。
已记录训练 batch 48475，累计入选对象实例 5895799（同一对象可在不同 epoch / 增强中重复，不能当成独立对象数）。

| 原生训练损失 | 首5个完整epoch均值 | 末5个完整epoch均值 |
|---|---:|---:|
| train/box_loss | 1.333260 | 1.059646 |
| train/cls_loss | 1.115328 | 0.561942 |
| train/dfl_loss | 1.173146 | 1.022122 |

CSV 数值全部 finite：True；KD 日志数值 nonfinite：0；失败回执存在：False。

## KD 运行信号

固定首5轮与末5个完整轮次比较；每100 batch采样一次，前3 batch额外记录，不是全量均值。

| KD 日志指标 | 首5个完整epoch | 末5个完整epoch |
|---|---:|---:|
| loss_unweighted 均值 | 0.473755 | 0.219758 |
| weighted_kd_total 均值 | 1.516017 | 0.703225 |
| selected_count 均值 | 116.548387 | 112.607143 |
| nominal_dose 均值 | 0.270076 | 0.267165 |
| student_evidence_selected_mean 均值 | 1.649802 | 2.917021 |
| teacher_evidence_selected_mean 均值 | 3.641660 | 3.559570 |
| reference_evidence_selected_mean 均值 | 1.734659 | 1.637240 |
| quality_selected_mean 均值 | 0.173666 | 0.178429 |

全部 487 个已记录 batch 中，选样数为0：0；KD为0：0。末5轮加权KD/native损失比均值 0.818%。

入选数与质量仍非零，KD仍参与总损失；这些日志支持‘辅助分支在工作’，不能推出‘有效避免负迁移’。长训未逐步记录梯度范数，不能仅凭loss非零断言后期梯度强度未退化；canary只证明初始实现梯度非零及零权重等价。

## 资源与身份

runtime 可见设备=4，batch=32，train workers=4；frozen模型不进入optimizer=True；学生独立导出=True。

相关进程 RSS 相加 28.09 GiB；RSS包含共享页面重复计数，实际子进程也包含预建验证loader。GPU快照与guard租约原值见 summary.json。

## 结果口径与下一步

results.csv 的 precision/recall/AP/val-loss 均为关闭中间验证后的占位0，不能报道为AP=0。按冻结协议完成200轮，再由标准 last/EMA full-val 评估读取性能；weight0仍在同卡队列，未得到净收益比较。保持方法与阈值冻结，扩展到seed0/123只检验重复性；三seed paired−weight0仍不替代shuffled/same-modal归因。

## 证据路径

原始run：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/full_paired_s42_attempt1`。本报告脚本 `audit_running.py`；完整机读摘要 `summary.json`；本次原始小文件与KD标量快照见 `sources/`。不同文件在训练继续期间分别读取，少量batch时差属于采样时差。
