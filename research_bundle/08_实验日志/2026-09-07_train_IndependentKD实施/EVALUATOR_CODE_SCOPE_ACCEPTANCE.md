# Evaluator 独立代码范围复审

**结论：CODE_SCOPE_ACCEPTED。当前 evaluate_independent.py 未发现必须修复的代码阻塞，可以进入统一 lease 下的真实固定 checkpoint 原生/扩展评估校验。此处不接受尚未执行的 GPU AP 数值等价，也不替代正式端点/旧结果桥接验收。**

日期：2026-09-07。审阅者 `/root/review_c1_spec` 未编写或修改该 evaluator；只读审阅当前源码、10 项 fixture、冻结旧 evaluator、已留存 pinned Ultralytics BaseValidator / DetectionValidator 源码，以及 run receipt 的副本语义。没有 SSH/GPU，没有新 AP。

## 核查结论

- **原生调用保持。** 对照旧 evaluate_object_evidence / evaluate_task_conditional，model.val 的 data/split/imgsz/batch/workers/device/plots/save_json/verbose/project/name/exist_ok 参数保持同义；未新增 half/conf/IoU/NMS 设置。新项只指定继承原 DetectionValidator 的对象捕获扩展。
- **捕获顺序和坐标正确。** 扩展先原样执行 super.update_metrics，再读取预测和重新准备的 GT。pinned `_prepare_batch` 使用布尔索引后生成 canvas bbox，不原地改写原 batch；预测与 GT 都在同一 canvas。主指标仍直接来自原生 metrics 对象。
- **实际精度记录有依据。** pinned BaseValidator 在 AutoBackend 构建后将实际 fp16 状态写回 args.quantize，随后才触发 on_val_start；故 capture_contract 的 quantize 与 `half = quantize == 16` 来自真实执行状态。
- **名单不是仅靠预期推断。** 起始核查实际 dataloader.im_files 数量、唯一性和集合；允许 rect 排序，同时保存实际顺序。结束时再核 seen 与对象一图一条的数量/唯一性/集合，缺图和重复图不会发布成功端点。
- **训练身份先行绑定。** 请求配置必须等于 run 内 protocol_config；后者必须等于实际训练 receipt 的有效配置副本，completion 也与绑定副本一致。E200、last/EMA 路径、dataset/arm/source/seed/模型身份/系数均先核对。
- **对象证据有字节绑定。** evaluation_val.json 与 objects.jsonl.gz 都加入实际 eval receipt 的 metric snapshots；发布前逐字节核对其原始产物。评估配置、合同、规范 roster 和原生 evaluator/metrics/NMS 源码另有 receipt 副本。
- **重试保留原结果。** 完整原始 attempt 与 receipt 先形成，canonical receipt 副本逐文件不覆盖发布，canonical metric 最后出现。相同 attempt 发布中断可补齐，无需再次 GPU 计算；不完整推理 attempt 留存并用新编号。既有异内容拒绝替换。

## 验证与边界

`D:/Anaconda/envs/KGJ_proj/python.exe -m unittest test_evaluate_independent -v`：**10/10 通过**。原始输出：`evaluator_independent_rereview_cpu.log`。

测试中 NativeFixture / ValueTensor 明确是合成对象，只验证 wrapper 顺序、字段不变和文件状态机，不冒充 Ultralytics 推理或 AP 测试。下一步仍须用同一个真实 checkpoint、相同完整 dev、旧/新 evaluator 实际输出完成数值等价与旧端点合同桥接；未完成不妨碍核心技术验证启动。

审阅源码副本：`evaluator_code_scope_review/evaluate_independent.py`。本接受只对应该副本字节，不自动覆盖未来修改。
