# LLVIP full-dev attempt2 独立修复审阅

**ACCEPTED_FOR_CANARY：当前 attempt2 代码可执行新的真实 canary；没有已知阻塞代码问题。** 实际 GT、指标、资源和完整评价仍须由新 canary/full 回执证明，不复用 attempt1 的接受状态。

## 失效链与本次范围

初版独立静态审阅漏掉了 processed image symlink 被 resolve 成 raw 路径后改变 YOLO 标签推导路径的问题。attempt1 中 native/capture 同为零 GT，零指标相等不能证明有效评价；该 attempt 全部评价 INVALID，原审阅与原产物按 `ATTEMPT1_INVALID.md` 保留。这里明确纠正原审阅遗漏，不把它解释为模型零 AP。

本次阅读当前 `export_full_dev.py`、`run_campaign.py`、`prepare_remote.py`、冻结协议和失效说明，对照已执行 pinned 数据集实现 `evaluator_profile_attempt2/evaluation_profile_binding/sources/003_dataset.py` 的 `get_label_files`、`get_labels`，以及 release_gpu5 的 `capture_contract` / `verify_population` / `metric_record`。只做本地静态与 CPU AST 逻辑验证，未 SSH、GPU、权重写入或新哈希。

## 修复核查

1. `evaluation_aliases` 保留 processed 图像路径，评价 txt 写 alias，`img2label_paths` 因此从 processed image 路径推导其对应 label。canonical resolve 只用于唯一 2406 dev 名单、同一图片身份与尺寸核对；原生 loader 和 capture 的路径继续保留 alias。`alias_to_canonical` 随真实 loader 保存，供 Windows 后处理映射，不在 Windows 错用 Linux symlink resolve。
2. 每个 processed 标签文件必须存在；全部非空行要求 5 列、类别 0、float32 可解析、finite、xywh 在 [0,1] 且宽高正。完整 GT 数必须正。canary 使用前 64 个 alias 及同序对应 GT 计数；full 用全部 2406。
3. 两条 canary 路径及 full 的 `on_start` 先核对真实 loader roster，再要求 GT 总数 exact，并将每图 loader `cls` / normalized `bboxes` 与该 alias 的原 label float32 数组逐元素比较。相同 GT 数但错误坐标/类别会拒绝，旧零标签缓存会拒绝，不能再只靠两个零指标相等通过。该比较发生在 dataloader iteration 前，符合已读 pinned 数据流。
4. capture 仍在原生 update_metrics 后读出，不改 metric/GT/pred tensors；`on_end` 验完整唯一图像集合与 captured GT 总数。**captured 坐标已经过真实变换，本检查是总计数 exact，不是原标签坐标字节 exact。** person 逐类指标必须存在且 class_id=0；native/capture 的五汇总、逐类指标、实际参数和 loader 顺序仍全部 exact 才接受 canary。
5. `quantize=None` 与 actual FP32 断言、固定 conf=.001/iou=.7/max_det300/B32/640/workers4/rect、完整 dev 原尺寸检查、旧权重和 args/data/names 身份、防旧权重写入、已有全局资源 lease 均保留。runner 额外拒绝 canary GT<=0；full 仍依本模型新 canary 实测峰值加余量预约。
6. 部署目标改为新的 `llvip_full_eval_attempt2`，且要求本次实际审阅文件。目录 exclusive 创建，保留原 attempt1。内部 `queue_attempt1` / `N42_canary_attempt1` 等是新外层 attempt2 下的局部名称，不会覆盖外层 attempt1；回执应始终携带完整路径避免读者混淆。

## 已执行 CPU 检查

`independent_attempt2_cpu.py` 从实际 exporter 提取 alias 函数、标签解析块与 `on_start`，不导入网络或执行 `run`。10 项检查全部通过：alias 函数不调用 resolve；合法 float32 标签；拒绝 NaN、越界、零宽、错误类和空总 GT；逐图 exact 接受；相同数量但不同框拒绝；零 loader GT 拒绝。

回执为 `independent_attempt2_cpu_receipt.json`，绑定当前 source 大小 12260 bytes 和记录的 mtime。路径 fixture 与 capture_contract 替身只验证源逻辑，不是实际 Linux symlink/Ultralytics 数据加载器或 GPU canary。新真实 canary 必须观察非零 GT、逐图 loader 标签 exact、person 指标完整与资源实测；通过后才按已审流程运行 full。

接受范围仅旧 LLVIP N42/T42 完整 dev 诊断的技术路径；不接受新 N 训练身份、L1 几何、校准、训练或 KD 增益。独立审阅者：`baseline_feature_analysis`。
