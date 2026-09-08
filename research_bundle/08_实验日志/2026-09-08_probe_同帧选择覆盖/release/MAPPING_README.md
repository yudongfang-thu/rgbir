稳定 GT 映射包装已通过 8 项 CPU 检查；它记录原 native 标签行经实际增强过滤后的身份，不重建第二条数据流，也不以浮点最近邻猜测对象对应。

root 在原 builder 中注入 `make_traced_dataset(runtime.TrackedDualLabelRGBIRDataset)`。返回 callable 的参数与原 PairDataset 相同，先构造原 Pair，再在 loader 创建及 worker 启动前安装 metadata tap。原 `PairDataset.__getitem__`、collate、RGB 一次增强及 IR 的原 RNG replay 均继续执行。禁止对已启动 worker 的 loader 补装。

预增强身份为 canonical 原图路径 + `::native_gt:` + 原 `dataset.labels` 行号。原表是 native 已验证/缓存的标签表，不声称行号等于原 XML 对象号或 YOLO 文本行号。输入 `Instances` 的 cls/bboxes 必须与该原表 exact。唯一目标过滤截获原 `RandomPerspective.box_candidates` 实际 bool mask，并确认运行时 `apply_instances` 的最终语句直接用同一 mask 索引 `new_instances` 与 `cls`。不添加随机调用，不修改 mask、标签、图像或坐标。

增强输出 metadata 存入原 `pair_info.gt_identity_trace`。`batch_identity_contract(batch, frame_id)` 对照实际 collated RGB/IR 的 batch_idx、cls、bboxes exact 后返回 `STABLE_GT_IDENTITY_VERIFIED`、`frame_id`、按各自 global GT 行排列的 `rgb_rows/ir_rows` 和原始逐图记录 `frames`。每个增强后对象包含 `stable_gt_id`、`source_gt_row`、原始与增强 bbox；被裁掉的原对象单独记录在 `dropped_source_gt_rows`。没有事后 GT 配对，RGB/IR 原 ID 各自独立。

缺源标签、未知过滤语义、重入/重复 transform、非 HBB、启用混合增强、类别或数量不闭合、batch 标签变化时明确失败。空 GT 允许输出已验证的空行表。真实 probe 还必须通过与已完成 LLVIP N 首批 `sample_stream` 的逐字段 exact 检查；CPU 合成测试不能替代该实际检查。

`MAPPING_CPU_attempt2.json` 为 8/8 PASS：复用真实 pinned Pair/Tracked 类，执行已采集 native `RandomPerspective.apply_instances/box_candidates` 函数体；覆盖同类中间 GT 被过滤后的 `[0,2]` 身份、空集、负例、原/包装 payload、实际 DataLoader shuffle/generator 及三种 CPU RNG exact。首轮本地 Python 3.8 的现代注解编译失败已保留在 `MAPPING_CPU_attempt1_IMPORT_FAILURE.json`，仅修正测试编译 future annotations。远端测试可在 release 内执行 `python test_mapping_trace_cpu.py --reference-dir <release_gpu5> --receipt <fresh.json>`，自动使用该环境真实 installed native augment 源码。没有 GPU、SSH、权重加载或新 hash。
