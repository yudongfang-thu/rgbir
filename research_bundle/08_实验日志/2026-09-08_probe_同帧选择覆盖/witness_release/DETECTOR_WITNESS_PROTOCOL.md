# 固定首批 detector witness

本实现于新 witness 结果前固定。scope 为 `SAME_FORWARD_DETECTOR_WITNESS`，复用前次相同首批、GT 身份与 S/T/R raw；没有新训练或模型 forward。原 `release/`、selector 和旧输出保持冻结。

入口为 `analyze_witness(batch, raw_by_name, model_by_name, evidence_config, strides, existing_export)`；raw/model 字典必须恰含 S/T/R，model 为现有 DetectionModel。driver 在原 no_grad/AMP 上下文内调用，并在此之前核对原 exporter 全记录与旧结果 exact。返回 `records`、`ir_records`、`frames` 与合同元数据。

每个 RGB record 保留原 stable IDs、states、gates 等字段，新增 `detector.S/R/T`。T 使用自己配对的 IR GT，未配对时为 null。全 IR 另由 `ir_records.detector_T` 保留。所有坐标位于同一增强输入画布，未反变换成原图。

固定区分四种观测：

1. 原 assigned state：原输出原样保留，不重新分配。
2. `dense_any_correct/count/witness`：原 OEv1 `_decode_boxes` 对 raw 做 detach/float32 解码；own GT IoU≥.5、最高概率类别正确、概率≥.25。count 数所有满足者；witness 固定取最高概率，tie 取先出现 anchor。另列该框对同图其他 GT 的 IoU≥.5 重叠，不把 dense witness 当作一对一检测。
3. `native_pre_nms_any_correct/count/witness`：pinned `Detect._inference(raw)` 原始 raw dtype/caller AMP 解码，再按原生 NMS 的严格概率>.25边界查询。限定 LLVIP 单类 standard Detect，不支持 end2end/export/xyxy 模式。记录原 FP32 与 native 坐标最大差，不能把前两种观测差全部归因 NMS。
4. `native_postnms_matched/native_witness`：同一 native decode 经 pinned NMS，再进入实际 `DetectionValidator._process_batch`、`BaseValidator.match_predictions`（iouv=[.5]）。用窄 Python return-frame capture 取得该原函数实际 GT/prediction matches，逐个 TP 位 exact 核对；非空 GT/pred 必须恰捕获一次。原排序不被自写 greedy 或 Hungarian 替代。结束后恢复原 profile callback。

NMS 固定 conf=.25、iou=.7、max_det=300、agnostic=false、multi_label=true（单类内部等效单标签）、end2end=false；`return_idxs=true` 给原 anchor 身份。先 clone native decoded tensor，避免原位 xywh→xyxy 污染 raw。逐图调用避免 native batch 计时上限跳过后续图，全部 frames 含空图均保留；记录实际 TorchNMS/torchvision 分支。native 原生函数本身的严格>.25与 raw witness≥.25边界都在合同中明示。

已有 LLVIP native profile 实测是 conf=.001、iou=.7、max_det=300、FP32完整dev。**这里的 .25 是新固定阈值首批诊断**，且原始 raw 来自训练模式/冻结 BN/AMP；使用原生解码和后处理，不代表重做正式 evaluator 推理或 AP。不能计算全 dev AP、可蒸馏上限或负迁移。

每个 `native_witness` 的 prediction_index 仅在同 frame/同模型内有效，并带 anchor_index、box、conf、class、own GT IoU。`frames[].native_detections[model]` 保留该图全部 post-NMS 框及实际 TP 位。T raw dense-any 必须逐对象 exact 等于旧 `gates.teacher_correct_own`。

CPU 命令：

```text
<pinned-python> test_witness_cpu.py --reference-dir <release_gpu5> --output <new-receipt.json>
```

默认用 installed pinned native NMS/evaluator/Detect，仅对小 raw 调用 `_inference`，不执行模型 forward。Windows 本地可额外 `--native-source-dir <旧接受LLVIP full sources>` 使用已存原 native NMS/evaluator 函数 AST；这种模式的 head 是小替身，回执明确 `native_head_actual=false`，不能冒充 installed-native 验证。

本地 `CPU_attempt1.json` 与补强 profile 漏捕获负例后的 `CPU_attempt2.json` 均 6/6 PASS，torch1.8 CPU；原 attempt 保留。测试覆盖≥/>边界、最高概率 witness、重复 GT dense overlap、真实 native 非贪心次序、原 NMS anchor返回/空图/clone、全 API/原门矛盾拒绝/raw不变。loc 的独立只读修订复核无阻断；真实 pinned CPU 与唯一 lease 下的实测由 root 独立执行。
