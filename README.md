# RGBIR跨模态蒸馏：完整证据与独立审计入口

本仓库用于让外部模型独立诊断一个面向J-STARS的跨模态目标检测项目。研究经历了RGB→SAR历史方法、RGB–IR数据诊断和首版对象判别蒸馏；这里保留正负结果、勘误、源码、协议、图册及可复算的小型证据。

**请先读 [MODEL_REVIEW_GUIDE.md](MODEL_REVIEW_GUIDE.md)，再按 [REVIEW_PROMPT.md](REVIEW_PROMPT.md)独立审计。** 不要把已有分析当作正确答案；请复算指标并核查实现。上传分支为`research/full-evidence-20260906`，旧入口原文保留在[ARCHIVE_README_v1.md](ARCHIVE_README_v1.md)。

## 最新状态及判断边界

- **新方法方向是DroneVehicle的IR教师→RGB学生**，训练期使用独立IR标注辅助对象对应/教师质量判断，推理仅RGB。LLVIP也诊断了IR→RGB；VEDAI是RGB→NIR，不能统称热红外。
- 六个baseline、三个数据集共521对图像的诊断已完成，含逐图/逐目标记录、特征图和配准可视化。配对相似性、教师更强或局部互补都不能直接推出可蒸馏增益。
- OEv1用对象前景相对局部背景的正确类别证据做选择性蒸馏，方案在运行前冻结。现有paired/weight0×student seed0/42/123正在执行，teacher/reference固定seed42；已测原生数据流固定，seed重复主要覆盖初始化变化。
- **2026-09-06 08:53:57 +08:00只读快照**：paired42已完成110轮，weight0 0完成21轮，paired123完成20轮；其对应另一臂排队。OS-SSL-IR的shuffled123微调完成50轮。**尚无完整E200终点，不能报告OEv1性能增益**。中间训练CSV的AP=0是禁用验证后的占位。
- 历史协议匹配CMDistill相对新native为−0.349±0.292 mAP百分点（3seed），但不能推广为“所有监督KD都无效”。HNEWA等已有小幅条件差异，需结合配对归因和方差；OS-SSL也有另一条独立证据线，不能用“唯一正例”替代逐协议判断。

最新运行来源见[94只读补采](research_bundle/remote_snapshot_20260906/README.md)。这是带时间戳的审计快照，不是实时仪表盘；旧快照继续保留。

## 从结论到证据

| 想审查的问题 | 优先入口 |
|---|---|
| 为什么从SAR转向RGBIR、有哪些历史误判 | [全项目复盘](research_bundle/07_研究分析/全项目复盘与研究诊断_20260905.md)；[历史审计原始记录](01_audit_20260905) |
| 数据配准、教师互补和可迁移知识 | [RGBIR诊断](research_bundle/07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md)；[521对完整产物](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识) |
| 直接查看图像和特征 | [特征图册](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/特征图册.md)；[配准图](research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/registration_panels/README.md) |
| OEv1具体蒸馏什么、如何筛选 | [冻结方案](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏首轮/EXPERIMENT_PLAN.md)；[源码与测试](research_bundle/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_object_evidence_v1) |
| 实现是否生效、有没有三倍KD/标签/EMA问题 | [首轮检查](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏首轮/README.md)；[独立代码审查](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏首轮/EXPERIMENT_CODE_REVIEW.md) |
| 多seed与当前证据强度 | [三seed扩展](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/README.md)；[端点收集器说明](research_bundle/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏三seed扩展/ENDPOINT_ANALYZER_NOTES.md) |
| 历史N/L与HNEWA能否独立复算 | [N/L原始指标](02_raw_results_dronevehicle)；[HNEWA多臂记录](04_hnewa_eval_records) |
| OS-SSL-IR并行路线 | [预注册](research_bundle/07_研究分析/方法预注册_OS-SSL-IR_20260906.md)；[执行记录](research_bundle/08_实验日志/2026-09-06_train_OS-SSL-IR迁移/README.md)；[补采来源](research_bundle/remote_snapshot_20260906/README.md) |
| 文献背景与待核验的创新性 | [调研笔记](research_bundle/01_文献/RGB-IR_20260905新增)；文献旧判断不等于本次核验结论 |

**历史命名陷阱**：`02_raw_results_dronevehicle/N_llvip_seed{0,42,123}_metrics_record.json`内部dataset/data_yaml实际属于DroneVehicle。文件名保留历史原样，辨认数据集须读取内容，不能把它们误作LLVIP原生对照。

## 包的范围

新增`research_bundle/`镜像相关本地资料，保留原目录便于追溯；`remote_snapshot_20260906/`补实际运行源码、协议、manifest、完整小型结果、部分注明截断的日志、固定依赖版本及Ultralytics源码/原发行LICENSE。大manifest有无损gzip和可读配对TSV；图册、PNG、CSV/JSON与导出NPZ均可检查。

未上传原始数据集、模型权重、凭据、第三方论文全文或环境缓存。源代码/原始数值保持原字节，导出Markdown做相对链接适配。来源见[BUNDLE_MANIFEST.json](BUNDLE_MANIFEST.json)，省略与运行边界见[BUNDLE_SCOPE.md](BUNDLE_SCOPE.md)。完整训练仍需数据、权重、相容环境和路径适配；本包没有宣称可离开94一键复现。

复制的AGENTS仅作为研究规范和当时资源约束的证据，不要求审计模型连接服务器或执行训练。请区分已接受的历史分析、描述性复算、工程检查、待验证假设与尚未完成的正式端点。

发布前的来源保持、数值复算与导航检查见[发布检查记录](publication_checks/README.md)。
