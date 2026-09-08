# Drone IR42 完整 dev 输入清点

**本次只读清点未找到可直接绑定目标 IR42 的完整1469图 native post-NMS capture；已准备独立完整推理入口。IR dev 实际为24490 GT，不是 RGB 的22462。此处没有新教师推理或新AP结果。**

实际远端清点源及原始输出为 `audit_remote_inputs.py`、`remote_input_audit.json`。通过 `ssh 94` 只读项目 `RGBT_campaign/artifacts` 和 `runs`，检查18445个文件，目录不穿透symlink，排除weights和源码副本；相关JSON/YAML/MD元数据只读≤2 MB。检查目标训练目录清单、原完成/评价回执，按精确checkpoint引用及预测/capture候选名查找。这是明示目录与筛选范围，不声称服务器所有位置绝无缓存。未读取checkpoint内容、未下载权重、未使用GPU、未计算新hash。

目标 checkpoint：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt`。
原stat为size=5483610，mtime_ns=1788133393714042412；训练完成回执为completed、seed42、epochs200、student_modality=infrared、native。原训练目录只有args、completion、method_stats、metrics_record、queue日志、results和weights，未见逐图完整预测。

原 `metrics_record.json` 仅有总体旧指标：AP50=.8083569915490951、AP75=.7119863600014217、mAP50_95=.5963198259876477，原evaluation 9.464867846108973秒；不具逐GT/pred身份，不能据此重建检出四桶。这些数字仅标旧端点来源，未当新capture结果。

另一个名字包含 `dronevehicle_full` 的候选：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_transfer_diagnosis_20260906/dronevehicle_full/prediction_records.json`（2089865 bytes），在本地 `2026-09-06_probe_RGBIR数据特性与可迁移知识/dronevehicle_full/` 有对应副本。实际列表只有200图；input_manifest固定sample_seed20260906，summary明确n_images200/GT_RGB3116/GT_IR3276，completion亦n_images200。虽然引用目标IR42，仍不能当完整1469dev。baseline_information 中train1024+dev200原raw对象/特征、自然训练批次以及task-conditional dev200同样不是完整教师native capture。

原data YAML：
`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/infrared.data.yaml`。
原processed root为 `.../data/processed/dronevehicle/yolo/hbb_v1/infrared`；val=`images/val`，标签为同root的 `labels/val`。完整1469个processed alias和canonical身份、逐图label count已写进 `release/drone_teacher_spec.json`，运行必须一致，loader保留processed alias以正确推导标签。

| 原标签/图像事实 | 数值 |
|---|---:|
| 完整IR开发集图像 | 1469 |
| 完整IR GT | 24490 |
| 空GT图像 | 2 |
| car / freight car / truck / bus / van | 20588 / 918 / 1470 / 789 / 725 |
| 原图W×H（全量图片header） | 640×512 |
| 固定first32 GT | 514 |
| first32五类GT | 447 / 24 / 30 / 13 / 0 |

RGB/IR同名图像清单1469项一致；二者有独立标签，GT数不同，不进行LLVIP式逐行GT数组exact绑定。未来跨模态GT关联另有CPU协议；capture不把同名帧或标签匹配视为物理对齐。原图shape不替代native rect canvas；输出记录真实canvas/original shape，预测框不另裁剪。

原RGB N42完整capture参照位于 `2026-09-07_train_IndependentKD实施/legacy_diagnostics_snapshot_20260907_161459/N_s42_attempt1/`。对应模型是新协议weight0 `.../runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/weights/last.pt`，不是旧RGB reference `.../formal_native/dronevehicle/rgb_seed42_native_b32a2/weights/last.pt`。本次补的是历史IR42 teacher，三个身份分开记录。

实现复用 `2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/export_full_dev.py`（已接受attempt2），原源逐字节副本保留为本目录 `accepted_llvip_export_full_dev_source.py`。运行仍导入 `94:.../artifacts/rgbir_independent_kd_v2_20260907/release_gpu5/{evaluate_independent,evaluator_profile}.py` 的纯capture helper，不调用formal run/publish或hash绑定。固定原生FP32、imgsz640、B32、workers4、rect、conf=.001、NMSiou=.7、max_det300。首32原生/capture双运行验证接口与峰值；通过后仅capture全量一次。新入口合同及portable CPU命令见 `release/README.md`，6项CPU测试通过，不能代替真实canary。
