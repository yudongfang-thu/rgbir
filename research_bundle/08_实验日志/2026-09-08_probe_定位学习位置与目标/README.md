# LLVIP 定位学习位置与教师目标（2026-09-08）

**已完成并独立接受：11个定位机会的历史R候选都连接到native学生anchor；120条边中110条教师/GT输出导数同向，47条因饱和完全相同。暂停本版L2-box仅放门/加系数扩展，不否定所有定位或DFL内容。** 完整去留判断与证据范围见[FINAL_REPORT.md](FINAL_REPORT.md)。本轮0 GPU、0新前向、0训练。

## 目的与固定范围

执行[已冻结的下一项](../2026-09-08_probe_Drone教师完整推理/NEXT_BOUNDED_ACTIONS.md)。首32图的80个GT整体保留，固定其中11个双方粗检出、仅T达IoU0.75的定位对象和1个反向对象，不按dev名单挑选。历史8批L2记录逐批分析，重复图/对象另计，不假设是独立样本。

原门、原selected、未执行教师后门的null均保持原身份。原生NMS检测框、冻结R选点、实际L2学生学习anchor分别记录，不用数值相近替代预测ID/anchor身份。输出坐标差/导数与共享参数梯度分开解释。

## 证据输入

- `../2026-09-08_probe_训练侧定位覆盖/{output_attempt1,bridge/output_attempt1}`。
- `../2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe/`。
- `../2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/llvip/`及原执行源码。

## 产物与局限

本目录保存作者分析、原字段小副本、真值检查和独立复核，不覆盖任何旧结果。有效产物为[anchor attempt3](anchor_join/output_attempt3/README.md)和[target attempt1](target_audit/README.md)，[独立审计](independent_review/EXPERIMENT_AUDIT.md)限定接受缓存诊断；原解析失败及审阅浮点exact断言修正保留。R代理不是实际student参数梯度，未形成新方法效用结论。

94镜像目标为`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_localization_learning_target_20260908/review_v1/`，同步回执见本目录。下一项仅限[DFL信息读出](RAW_DFL_INFORMATION_PLAN.md)，新短训尚未准入。
