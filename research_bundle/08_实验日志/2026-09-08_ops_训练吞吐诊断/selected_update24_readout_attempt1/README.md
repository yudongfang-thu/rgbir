# Selected-only：24 次更新结果读出

**本次训练轨迹：字节精确=True；原容差通过=True。** 这是已完成 JSON/JSONL 的只读汇总，不是正式长训准入。

| 路径 | 实际批次 | 成功更新 | 尝试 | AMP 跳步 | thin / full diagnostics / fallback |
|---|---:|---:|---:|---:|---|
| old | 30 | 24 | 30 | 6 | 0 / 30 / 0 |
| selected_only | 30 | 24 | 30 | 6 | 30 / 30 / 0 |

实际 λ=`0.09227393550836771`；原初始化、E200 配方、B32/workers4/AMP。原生损失与 KD 组件按同一容差分别核对。

| 比较项 | bitwise exact | 原容差 | max abs |
|---|---|---|---:|
| initial | True | True | 0 |
| first_batch_pixels | True | True | 0 |
| flow_worker_rng_gt | True | True | 0 |
| selection_identity | True | True | 0 |
| selection_floating | True | True | 0 |
| per_batch_losses | True | True | 0 |
| gradients_scaled | True | True | 0 |
| applied_gradients | True | True | 0 |
| student | True | True | 0 |
| optimizer | True | True | 0 |
| ema | True | True | 0 |
| control | True | True | 0 |

student 包括参数和 buffers；scaled gradient 的 AMP 非有限模式另列，不能称为有限可用梯度。学习 delta 仅比较 selected S/T；未选与 R delta 未采集到学习快照。

| 逐批 loss / 组件 | JSON exact | 原容差 | max abs |
|---|---|---|---:|
| native_total | True | True | 0 |
| loss_unweighted | True | True | 0 |
| weighted_kd_total | True | True | 0 |
| total_loss | True | True | 0 |
| target_loss_unweighted | True | True | 0 |
| off_target_loss_unit | True | True | 0 |
| off_target_loss_unweighted | True | True | 0 |

| 时间（秒） | old | selected-only |
|---|---:|---:|
| training_span_seconds | 134.085289 | 187.673852 |
| training_audit_copy_save_check_seconds | 10.832210 | 10.142454 |
| extra_full_diagnostics_seconds | 0.000000 | 81.048239 |
| training_span_minus_audit_seconds | 123.253079 | 177.531398 |
| training_span_minus_audit_and_extra_diagnostics_seconds | 123.253079 | 96.483160 |
| 去前 6 批后 median：wall_seconds | 3.560012 | 5.424252 |
| 去前 6 批后 median：wall_minus_audit_seconds | 3.237961 | 5.071938 |
| 去前 6 批后 median：wall_minus_audit_and_extra_diagnostics_seconds | 3.237961 | 2.283448 |

完整 wall、审计、额外完整诊断均保留；每个热身后样本在 summary.json。回调时间不含 loader fetch，扣除诊断会受到缓存与重叠影响，**不能当作正式 epoch 吞吐**。

首批像素与后续 source/GT/增强元数据/worker RNG 的证据范围分开；没有重新打开大规模 state .pt。共享梯度观察 JSON exact=True、容差=True；sanity JSON exact=True。

同公式或同 raw 通过不推导训练轨迹通过；本次结果不覆盖旧 block16 的失败，也不外推 E200 或多 seed 增益。

输入：`E:\SHARE\光sar\08_实验日志\2026-09-08_ops_训练吞吐诊断\remote_selected_update24_probe_attempt1\comparison_attempt1`。输入路径逐项保留于 summary.json；原始结果未改，无新增 hash。
