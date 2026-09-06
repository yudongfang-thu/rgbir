# 分析器v2与接线修复复核

> **分析器现在比较实际训练receipt绑定的共同recipe，manifest protocol_id相同也不能掩盖batch或学习率差异；20项CPU fixture通过，原快照v2重算仍为C42−N42 +0.144554pp。**

## 分析器改动

- 从`run_evidence/run_receipt.json`声明的config snapshot解析完整实验协议，必须含模型、paths和固定recipe字段。新run的`protocol_config.yaml`存在时，与绑定配置的共同recipe逐字段一致才可接受。
- 实际旧OEv1终态没有单独protocol_config.yaml，直接使用其已完成训练receipt保存的配置；结果明确标`legacy_receipt_bound_protocol_only`，不是仅信任manifest标签。
- 比较student model、imgsz/epochs/batch/nbs/workers、optimizer/lr0/lrf/momentum/weight_decay、三项warmup、cos_lr/close_mosaic/patience、AMP/deterministic、augmentation、student_data_yaml、Torch/Ultralytics版本。配对与同臂汇总都检查这些字段。
- teacher/reference、KD λ、method_id和geometry不是共同recipe比较列，因为它们可以是干预本身。seed始终取实际completion/eval/run receipt，忽略配置文件默认seed。
- 新增两项fixture覆盖：相同manifest ID下batch或lr不同拒配；launch protocol与bound config不符拒收，而教师/λ/method ID及YAML seed不同不误伤已正确绑定的seed配对。

执行20项CPU单元测试，全部通过；没有GPU、数据集推理或远端写入。源文件副本为`analyzer_source_v2/`；原v1源码和报告保留。新重算报告：`snapshots/2026-09-07T022337.461405_0800/derived/results_analysis_v2.json`，四个完成端点仍通过，其他五个incomplete，没有新增研究结果。

## root接线修复只读复核

已确认此前三项主要修正：C/N直接调用legacy.combine_loss保留乘法顺序；formal加载geometry并检查verified与非空、校准和D2模型/数据/geometry绑定、canary的24成功update/equivalence/λ绑定；C canary增加非零score梯度要求。

进一步提醒已发送root：梯度抽样条件应包含“C selected且此前没有正C梯度”，否则首batch C为空但L非零会使后续真实C梯度没有被记录，造成误判失败。该点修复状态以root最终代码和实际canary为准。仍需真实loader/初始化/小batch等价与GPU canary；静态复核不取代这些测试。

分析器仍待root独立接受，本条目不自评accepted，不改原始实验结果或旧review。
