# Drone IR42 完整 dev 预测补全

**已完成：Drone教师完整dev推理、双模态21,568对GT的四组CPU匹配及固定框cross-GT补查；支持优先LLVIP条件定位、Drone高度一致对象定位，并收窄置信度与特征主张。** 结论和局限见[FINAL_REPORT.md](FINAL_REPORT.md)；前置范围见[上一阶段固定后续](../2026-09-08_probe_开发集真实检出机会/NEXT_BOUNDED_ACTIONS.md)。

目标模型为既有 `formal_native/dronevehicle/infrared_seed42_native_b32a2/weights/last.pt`；完整dev1469图/24490 IR GT，复用新协议RGB N42完整缓存/22462 RGB GT。教师producer28.69秒，含canary和调度的队列212.53秒；配对CPU19.67秒、cross-GT1.90秒。未重复RGB推理或新增训练。

实际复用已接受native capture，先固定首32图验证双路径一致与资源，再由唯一global lease在screen中执行完整dev。固定源码`release/`、执行输出`evidence_1401/`，原GPU产物位于94项目`artifacts/rgbir_drone_teacher_capture_20260908/attempt1/`。所有源、失败attempt、回执保留，不读取test或下载权重。

跨模态GT执行原同类一对一IoU≥0.5关联，未配对RGB/IR为894/2922；标签关联不证明物理配准。CPU原值和失败attempt见`cpu_analysis/`，cross-GT固定框原值见`cross_gt_attempt1/`；独立复核在`independent_review/`，以外部接受文件解释原PENDING状态。后续范围见[NEXT_BOUNDED_ACTIONS.md](NEXT_BOUNDED_ACTIONS.md)，同步状态见本条发布回执。
