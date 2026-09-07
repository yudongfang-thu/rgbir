# Selected-only 实际 24-update 读出独立审阅

**PASS，限定已执行 seed42、30 批 / 24 次成功更新的读出范围。** `SELECTED_UPDATE24_RESULT.md` 与当前小型实际产物一致；没有把插桩时间比当生产 epoch 吞吐，也没有把旧 block16 失败称为修复。

2026-09-08，loc_stress。只读 JSON/JSONL 和小文件字节副本；独立脚本为 [check_result_readout.py](check_result_readout.py)，[receipt](result_readout_receipt.json) 为 `PASS_JSON_READOUT_SCOPE`。未加载 state `.pt` / 权重，未调用 GPU/SSH，未计算 hash，未改作者报告或旧结果。完整张量 byte/numeric 结论来自前述已审代码实际执行的 comparator，本审阅没有声称再重跑它。

- 逐 record 重聚合全部 category，与 `comparison.json` 及作者 `summary.json` 全部相同；初始、流、选择、selected S/T、每批 loss、参数/buffer、optimizer、EMA、控制均按该已执行比较器 exact。6 次 AMP skip 所在 scaled gradient 非有限位模式一致，不能称这些位为有限有效梯度；applied gradients 的记录均有限。
- 两 worker 的实际计数都是 30 批、30 attempts、24 optimizer calls、6 skips；新臂 `thin_learning_batches=30`、`full_diagnostics_batches=30`、`fallback_batches=0`。**实际 EMA updates=30，两边一致**；不能将“24 次成功更新”误写成 24 次 EMA 更新。
- 独立比较 30 条 KD JSONL 的 native/KD/weighted/total/target/off-target loss，及 shared-gradient/sanity JSON，全部相同。学习快照仅 selected S/T 内容；其余学习 delta 未采集的限定正确。只有首批保存像素，后续 flow exact 的 source/GT/元数据/RNG 限定正确。
- 65 对源码副本、输入 config 副本和模型 stat 独立闭合；实际执行 candidate/harness 与当前文件字节相同。这里没有模型内容 hash 或权重内容比较声明。
- 完整 trainer span 为 old **134.085289 s**、new **187.673852 s**，新臂确实更慢。其额外 no_grad 完整诊断为 **81.048239 s**。逐批扣除 audit / 额外诊断的算术，以及去前六批后的全部中位数，均与原 worker rows 一致；该调整后比值 **1.4180135581**。报告保留更慢的完整 wall，并明确这是受插桩与缓存/重叠影响且不含 loader fetch 的分层，口径合适。

没有必须修正的定量或范围表述。本证据支持该冻结版本的受限路径一致性；不推出 E200 identity、检测收益、未观察长期日志开销，亦不取代后续 E20 配方吞吐或开发集评价。
