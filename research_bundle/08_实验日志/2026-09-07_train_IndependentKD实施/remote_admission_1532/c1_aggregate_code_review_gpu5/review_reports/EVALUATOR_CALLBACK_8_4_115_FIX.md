# 8.4.115 验证器回调修正

结论：release_gpu4 的旧 N42 完整 dev 评估资源探针在首批推理前因错误访问 `validator.model` 失败；已按 94 实际安装源码修复接口，CPU 回归通过，真实 AP 等价和资源峰值仍须由新 attempt 完成。

## 目的与实际故障

只修独立评估探针的源码记录。原失败目录保留在 94：
`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/admission_queue_attempt4/`。
原始 `evaluator_profile_a1.log` 的只读副本位于本目录 `evaluator_callback_fix_v2/failed_evaluator_profile_a1.log`。

日志显示 Ultralytics 8.4.115、Python 3.10.20、Torch 2.10.0+cu128，1469 张 dev 图像扫描成功；`on_val_start` 回调内的 `type(v.model)` 抛出 AttributeError。这不是 AP 结果，也没有生成成功的资源 profile。

## 实际安装接口证据

本轮通过只读 SCP 获取以下实际文件，保存到 `evaluator_callback_fix_v2/pinned_8_4_115/`，未依赖凭空构造的 mock 接口：

- 安装根目录：`/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/lib/python3.10/site-packages/ultralytics/`。
- `engine/model.py`：`Model.val` 将 `self.model` 作为 `model` 参数传给 validator（实际行 589）。
- `engine/validator.py`：`BaseValidator.__call__` 在局部变量中建立 AutoBackend（实际行 183），完成 dataloader 和 warmup 后触发 `on_val_start`（实际行 239）；没有赋值 `self.model`。`init_metrics` 随后执行，故 `.seen` 的读取保持在 `on_val_end`。
- `models/yolo/detect/val.py`：postprocess 返回带 `bboxes/conf/cls` 的字典；`_prepare_batch` 返回 `cls/bboxes/ori_shape/imgsz/ratio_pad/im_file`。与 `make_evidence_validator` 当前只读对象记录的字段一致。

## 修改

`evaluator_profile.py` 新增 `capture_runtime_sources(validator, loaded_model, sources)`，从回调闭包内实际已经加载的 YOLO 对象 `loaded_model.model` 记录网络类，从实际 validator 记录 dataloader/dataset 类。三个来源的类名和文件路径写入 contract，实际源码参与既有 bytes 绑定。AutoBackend 源码另行显式加入源码集合，以覆盖 native validator 局部持有的后端。

`evaluate_independent.py` 经核对没有 `validator.model` 或 `v.model` 访问；其 start 回调只取实际 args/dataloader，end 回调才取 seen，不需改动。分类、校准、兼容轨迹、调度和旧结果均未修改。

## 验证

- 本地 `D:/Anaconda/envs/KGJ_proj/python.exe -m unittest test_evaluator_profile test_resource_dispatch test_evaluate_independent -v`：**48/48 通过**（10 + 28 + 10）。新测试覆盖不存在 validator.model 时仍能记录实际网络源，以及无法记录实际类源时明确拒绝。
- `evaluator_callback_fix_v2/test_pinned_validator_contract.py`：**4/4 通过**，直接解析上述 94 原始安装源码副本，检查真实模型传参、局部后端、回调时序、检测字段和本次接口修复。
- 独立审阅者 `/root/review_matrix_spec` 只读复核安装源码和本次修改，并独立重跑 10 项 probe CPU 测试通过，接受本次代码范围；未接受尚未执行的 AP 或资源实测。

这些是 CPU/安装源检查，不能替代完整 dev1469 的原生与扩展评估。根调度器下一 release 的新 probe attempt 必须重新完成五指标及逐类 AP 精确相等、loader/seen 检查和共享 guard 下的实际资源测量，才能用于后续正式评估预约。

## 产物

- `evaluator_callback_fix_v2/source_copies/`：修正后的 probe 源码、测试，以及本轮只读审查的正式评价源码。
- `evaluator_callback_fix_v2/implementation_receipt.json`：本次实现和验证范围；不是训练/校准/资源成功回执。
- 本轮未启动任何 GPU 任务，94 仅执行日志/源码读取与下载。
