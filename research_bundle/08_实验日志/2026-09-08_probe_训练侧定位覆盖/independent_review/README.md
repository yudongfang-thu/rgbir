# 独立审阅

**ACCEPTED：仅接受固定首 32 图缓存的定位正确性分层与原 C 门覆盖描述；不构成 L1/L2 准入或 KD 收益证据。**

作者 4 项 CPU 真值独立复跑通过（`CPU_REPLAY.json`）。另直接执行固定原生 `box_iou` / `match_predictions` 函数，独立复核 32 图 × S/R/T × 两个 IoU，共 192 次匹配；逐 GT、原预测 ID、完整 TP 位及 IoU 与保存输出一致。与原 IoU .5 见证闭合 240 个 GT 状态、232 个正匹配、244 个预测 TP 位，IoU 最大差为 0。源副本和已接受 helper 按字节一致；未计算新 hash。

两阈值独立匹配，未从 .5 匹配过滤生成 .75。当前样本中跨阈值均正确的 GT 未改变匹配预测 ID；这只是本批事实，不能推广为原生匹配的嵌套保证。原稳定 RGB/IR GT 身份及全部 C 门逐行未变；七桶对象数、图数、ID 列表和分母均独立闭合。

双方 IoU .5 正确的 77 个对象中，只有 T 达到 .75 的为 11 个（8 图），反向为 1 个。前者原 C base/eligible/selected 为 11/10/8；这说明固定分类选择对该定位模式的覆盖，不证明 C 或 L2 已学到定位收益。

附加 native-box 代理表独立核对为：S IoU < .70 共 4 个、T own-GT IoU 领先 > .05 共 10 个、交集 4 个且原 C 均 selected。它使用保存的原生检出框及各自 GT，不等同于真实 L2 固定 R/T anchor、支持域、目标映射及门链；`actual_L2_selector=false` 的范围正确。

本次是 local Torch 1.8.0 CPU float32 执行固定 Ultralytics 8.4.115 原生函数的缓存重匹配，不能表述成原 GPU TP 的跨运行时逐位复现。没有模型 forward、训练、GPU、权重加载或新增 AP。32 图不能代表完整训练分布，也不能由此宣称过拟合、物理配准或蒸馏增益。

复算脚本：`review_actual.py`；实际回执：`ACTUAL_REVIEW_RECEIPT.json`。原核心源码及 output_attempt1 均保留未改。
