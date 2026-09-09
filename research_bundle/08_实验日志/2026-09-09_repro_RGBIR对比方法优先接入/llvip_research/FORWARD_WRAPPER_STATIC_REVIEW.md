# CFT 前向包装静态审阅（2026-09-09）

本地审阅父任务 `../run_cft_author_cpu.py`，未运行模型、未访问服务器。原作者源码调用链与本项目 fit manifest 字段一致，未发现阻断本次机械验证的静态问题；实际权重加载与前向成功仍以90运行回执为准。

- 包装读取 `data_attempt1/processed/llvip/splits/grouped_v1/fit.tsv`。`prepare_data90.py` 生成的列确为 `stem, sequence_prefix, visible, infrared, annotation`；visible/infrared 指向已抽取的 official train 原图。这些图同时是本项目 processed fit 视图的 symlink 源，使用 raw 路径不改变所选样本。
- 从原始完整 fit TSV 独立读得所选索引0/4809/9618为 `020001/100118/180408`，前缀02/10/18均不属于 dev 前缀01/04/07/12/25。只读取训练名单内这三对图像，不需要读任何标签、dev图或封存test图。
- 原 `models.experimental.attempt_load` 使用作者 checkpoint 中 EMA 或 model 并转 FP32/fuse/eval。原双输入 `models/yolo_test.py` 的 forward 签名适配 `model(rgb, ir, augment=False)`，输出解包与作者 test.py 一致。
- 使用原 `letterbox(..., new_shape=1024, auto=True, stride=stride)`、BGR→RGB、CHW、float32/255、原 NMS，保留预测有限性/类别维度检查和原输出副本。`auto=True` 可产生非1024方形输入，包装记录实际形状，故1024表示目标长边尺寸。
- 没有调用 test.py 主入口或其 dataloader；尤其规避 `test.py --task train` 仍访问 YAML val/official test 的问题。
- 包装记录4个 Torch CPU线程、1个inter-op/OpenCV线程；与我最初提出的单线程建议不同，但设置和记录一致。此前数据准备单线程约束不因本包装发生改变。
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 为受信任作者旧模块 pickle 的兼容设置，已写入 plan；仍需90实测确认 checkpoint 类与当前原仓库及Torch2.10兼容。

此次可支持的完成状态只有作者checkpoint恢复与三对fit图像前向/NMS成功。作者 official train 若覆盖本项目dev，后续dev AP不能视作未见数据泛化比较。CFT推理为RGB+IR，不进入RGB-only KD胜负表。
