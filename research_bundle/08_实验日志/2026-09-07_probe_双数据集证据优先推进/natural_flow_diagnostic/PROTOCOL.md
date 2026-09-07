# 固定自然 64 批选择诊断

执行前冻结：LLVIP 先、DroneVehicle 后。复用 IndependentKD release_gpu5 的 `coverage_probe.build_natural_loader(cfg, seed=20260907)`，B32/workers4/imgsz640，必须与已有 release_cpu1 的完整 64 批 natural_batches.jsonl 中的源图、顺序、worker seed、变换矩阵/trace、增强后双侧 GT/类别、shape/dtype 逐字段 exact。禁止换流、补批或重采样；原 trace 未保存像素，因此不能声称像素字节 exact。

每批仅对 cfg 的冻结 T42/R42 各做一次 eval FP32 前向，使用 `runtime.to_device` 预处理一次。使用真实 `selection_adapter.build_classification_selection` 的整批 C0/C1 公共选择；student 槽仅复用 R 的 shape/区域有效性，不报告或使用学生学习/损失（原分类适配器内部仍计算未使用的C0标量）。C0/C1 selected 是同一集合，不是两种独立选择器。该 cfg 的 Drone R 是原 formal_native RGB42，不能称最近重诊断中的新 N42。

整批 `localization_loss.build_localization_selection` 是唯一定位主结果：mode=teacher、geometry_eligible=None、geometry_verified=False、return_records=True，保留冻结 reference confidence-first / IoU / anchor-id tie break。记录全部累计 gate、base 对象/anchor、选中对象、类别、支持域、source 图/组。为了给 base 之前的 gate 归属图/组，只对同一前向缓存逐图重放该原选择器；每个真实批必须断言所有累计 count 和主选择器的 base/selected GT identity/anchor exact，逐图统计不替代整批分母或选择。

全程状态为 `UNVERIFIED_GEOMETRY_DIAGNOSTIC`。这里是 geometry 全 true 放宽后的诊断；因为正式 geometry mask 也可能改变候选 anchor，这不是严格数学保证的最终 L1 selected 数量上界。只可把它看成未验证候选支持/机会诊断，不是原 L1/D2 已准入、自然学习信号或校准剂量。没有 backward、训练、优化器、lambda 校准、AP、validation/test 评价。只统计完整自然前 64 批；`--batches 2` 仅技术 canary，不据其改变阈值或流。

从旧 coverage receipt 读取同一 geometry roster，仅用于重建旧 trace 的 roster_hit 元数据，不向定位选择器供应该 roster 作为已验证 geometry。所有模型/配置/模块路径和文件大小/mtime 保存，不计算新哈希。每次输出目录必须不存在，失败 attempt 保留。实际运行要求现有 global GPU lease；每模型双批前向和双批选择器路径必须真实测显存/进程树 RSS，再为64批预约并持续监测。无几何/训练准入回执由本脚本生成。
