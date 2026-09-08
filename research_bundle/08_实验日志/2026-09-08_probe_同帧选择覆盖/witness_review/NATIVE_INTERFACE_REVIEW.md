原生接口支持在同一批 raw 输出上增加检出与 GT 身份见证，无需第二次完整模型 forward；最终 producer 接受仍以新源码和真实 native CPU 接口检查闭合为准。

已读现有 pinned 8.4.115 源副本：`2026-09-06_ops_GitHub完整审计包/remote_snapshot_20260906/framework_snapshot/ultralytics/nn/modules/head.py` 的 `Detect._inference/_get_decode_boxes`，以及 `2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/remote_completed_attempt2/N42_full_attempt1/sources/{3_val.py,4_validator.py,6_nms.py}` 的 validator/NMS。未启动 GPU 或读取新检出结果。

`Detect._inference(raw)` 复用 `boxes/scores/feats`，仅做 DFL 解码、anchor/stride 解码及 sigmoid，不重算 backbone 或 cv2/cv3 分类回归头。DFL 内部会执行固定卷积的 forward，因此准确表述为“无额外完整 model.forward，新增原生 head decode”。`_get_decode_boxes` 在 shape 不同或 dynamic 时可能写入普通属性 `head.shape/anchors/strides`；这些不在 state_dict 中，不能以参数/BN state 相等推导整个对象所有属性未变。

NMS 应直接用原函数及固定 conf=.25、IoU=.7、max_det=300，保留 native validator 的 multi_label=True、agnostic=False、end2end=False、rotated=False、labels=()。LLVIP nc=1 时原函数内部会关闭 multi_label。`return_idxs=True` 可以直接得到原 anchor 索引。原 NMS 会原位把输入 decoded xywh 转为 xyxy，保留 decoded 证据时应传 clone。GT 使用原 `_prepare_batch` 的增强输入 canvas 像素 xyxy，匹配不回映射原图坐标。固定 conf=.25 是本次局部检出见证口径，不是完整 dev AP 评价。

`match_predictions(use_scipy=False)` 的真实顺序是：同类 IoU 达阈值 → IoU 降序 → `np.unique(det, return_index=True)` → `np.unique(GT, return_index=True)`；两个 unique 之间没有再次按 IoU 排序。GT 身份必须取自该原函数实际生成的匹配对，并逐位核对其 TP 布尔输出，不能用独立近似匹配器或只看 TP 数量。

独立 CPU 使用捕获源码的原函数和 return-frame locals 记录实际 `matches`，4 个已知真值通过（`NATIVE_MATCHING_CPU.json`）。特别是一个 GT、两个预测的 IoU 为 `[.8,.9]` 时，native 最终选预测索引 0；常规最高 IoU greedy 会选索引 1。另测一预测争夺两 GT、错类、等 IoU。该小样例说明身份见证必须保留 native 的实际排序语义。

实际实现还应确认 NMS 没有 timeout 提前结束而将后续图像留为空输出，并按原映射将每图局部 GT 索引连接到 `identity_contract` 的 stable GT ID。以上不增加训练或新评估任务要求。
