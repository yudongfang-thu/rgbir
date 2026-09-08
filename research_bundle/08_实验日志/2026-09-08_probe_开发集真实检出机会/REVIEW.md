**ACCEPTED — 仅接受固定原生源码在本地 CPU float32 上对既有 LLVIP 完整 dev post-NMS 缓存的描述性重匹配。** 以下保留协议预审边界，文末补实际源码、真值与全量输出验收。这个结论不表示重新复现原 GPU TP 的 bitwise parity，也不升级为新 AP 或 KD 收益。

已读冻结 `README.md` 及原 N42/T42 缓存结构、`verified_pair_manifest.jsonl` 的字段。两个缓存是各 2406 图/7879 GT 的既有低阈值 post-NMS 输出，模型身份是旧 native RGB42/IR42，不是新 FT3 端点。输入含 image、canvas_shape、original_shape、GT/预测框类别和预测置信度；实际 native canvas 可为 544×672，不能硬改成 640×640 或原图坐标。新实现须保持原浮点框与预测顺序，并保留 confidence>.25 子集到原缓存 pred ID 的映射。

逐图配对应通过已验证 manifest 的完整模态路径、相对 pair key、图像唯一性、canvas/original shape、GT 数量及 GT 数组 exact 闭合；匹配分别使用模型自身 GT，再通过已验证的 native GT 行身份连接。缺缓存或身份不闭合应显式失败/UNKNOWN，不通过最近邻补数据。LLVIP 共用 annotation 坐标不能被称作两套独立物理标注或真实像素配准。

预测框的有限值、宽高方向和 canvas 范围需要检查，但原生未 clamp 预测可以越出 canvas；这种情况应记录而不擅自截断或改框。confidence>.25 是对已保存 post-NMS 序列的保序筛选，不能冒充重新运行了 confidence=.25 的网络/NMS，亦不能覆盖原 accepted AP。

四组 confidence/IoU 应各自调用真实 pinned `_process_batch/match_predictions(use_scipy=False)`，直接捕获原 GT/pred 匹配对并逐位核对 TP，保留 native 的 IoU 排序→预测 unique→GT unique 顺序。IoU=.5 与 .75 应保留独立结果和每 GT 的联合状态，不能靠正确数量相减、强制子集或复用低阈值分配推导定位谱。即使本数据没有非嵌套 GT 状态，也不能因此省略独立匹配。

“仅 T 正确”“仅 N 正确”等四桶是固定模型、缓存阈值、own-GT 匹配定义下的检出状态。它们不等于蒸馏可迁移知识量、KD 可达 AP、实际学习修复或负迁移；“双方 IoU.5 正确、T IoU.75 正确而 N 不正确”只能称该定义下的定位差异。结果不恢复原 C selector、候选分母、实际梯度或剂量；不能据此给损失载体排名。单类数据也不提供分类蒸馏已解决的证据。

Drone 尚无已确认的完整 IR42 dev post-NMS 缓存，本轮应保留缺口；200dev/1024train raw 与完整 N/C0 缓存均不能补成完整 teacher 输出。当前协议没有授权据该缺口追加 GPU。

审阅范围：只读既有源和少量输入字段，未计算本轮新全量统计，无 GPU、权重加载或新 hash。本文件和后续 `independent_review/` 由独立审阅者单独维护，以避免与作者 README/测试报告同名覆盖。

## 实际源码与输出验收

上述“尚未计算新全量统计”是协议预审时的历史范围。随后实际审阅 `analyze_dev_cache.py`、`native_cached_match.py`、`test_dev_cache_cpu.py` 与 `output_attempt1/`。作者执行保存源码与当前源逐字节相同，三份原生函数源副本也与既有接受输入逐字节相同。独立复跑现有 6 项 CPU 真值全部通过，涵盖保序与 confidence>.25 边界、重叠 GT/重复预测、类错、空集、合法预测越界及两 IoU 下预测身份变化。原 attempt1 合成期望错误及后续 fixture 修复保留；没有借此修改 native 匹配逻辑。

**全量独立复核闭合：2406 图、7879 GT、41335 条缓存预测记录、四条件×两模型共 19248 次匹配。** 审阅脚本直接执行已接受 `box_iou/match_predictions` 函数体，独立保序 confidence 筛选并从函数返回帧捕获实际 GT/pred 对；没有调用作者 `match_row` 或汇总函数。每个输出 GT ID、原/筛选后 prediction ID、included 标志、匹配方向、完整逐预测 TP 布尔、witness IoU 均与输出 exact；IoU 最大绝对差为 0。N/T 逐图 GT 数组与 canvas/original shape exact，四桶计数/比例/图像覆盖、两张 16 格置信度转移表及 16 格联合 IoU 状态表全部独立重算一致。

在 score>.25 时，IoU.5 四桶为双方正确 **5351**、仅 T **1635**、仅 N **238**、双方未匹配 **655**；IoU.75 为 **2258/2297/659/2665**。仅 T@.75 的 2297 个对象由逐 GT 联合状态分为：**1304** 个 N/T 在 .5 都匹配、T 在 .75 匹配而 N 不匹配；以及 **993** 个 N 在 .5 未匹配、T 在 .75 匹配。这是联合表读取，不是用两阈值总数相减。实际四组始终独立匹配，未强制 GT 嵌套；本数据中没有出现高 IoU 正确而低 IoU 不正确的 GT 状态，不等于可以复用匹配或假定预测 witness 身份不变。

**运行身份必须保持限定。** 原缓存来自已接受 torch 2.10.0+cu128/Ultralytics 8.4.115 的历史评价；本次执行这些固定原生函数体的是本地 torch 1.8.0+cu111 的 CPU float32 与本地 NumPy。所谓 exact 是作者输出与本次独立本地 CPU 重匹配及 ID/汇总之间 exact，不是与原 GPU 评价 TP 的逐位复现。边界运算和 NumPy 的排序/并列处理可能随运行库不同，本次没有声称跨运行库 parity，也没有为此追加 GPU。原 AP 缓存和评价回执不改变。

结果用语范围内合适：“仅 T 匹配”是当前固定模型、阈值与缓存定义下的互补模式；“N 在 .5 匹配但在 .75 未匹配”是定义明确的定位差异。它们不能给出可蒸馏上限、KD 增益、梯度目标优劣、真实负迁移或物理配准结论。预测越 canvas 的实际记录 N=25/T=15 被保留且未 clamp；未恢复被原 NMS 抑制的候选。Drone 缺口没有被其他缓存替代。

审阅小产物为 `independent_review/CPU_REPLAY.json`、`review_outputs.py` 和 `ACTUAL_REVIEW_RECEIPT.json`。独立全量 CPU 复核耗时约 6.31 秒；没有 GPU、模型前向、NMS 重跑、训练、checkpoint 加载或新 hash。作者 `output_attempt1` 保持原 `COMPLETED_PENDING_INDEPENDENT_REVIEW` 回执不覆写，本独立 ACCEPTED 文档提供后续接受状态。

已另回读 root 的 `FINAL_REPORT.md`：1304 个定位型模式及 610 个反向模式与逐 GT 联合表闭合；1635 个高阈值 T-only 确实分为 771 个低阈值双方匹配、864 个低阈值仍 T-only。FP1998/519 被正确定义为该条件下一对一匹配未获 GT 的预测数，未全部等同为背景误检。后续研究优先级是有边界的工作判断，报告没有据此宣称现有 L1/L2 几何或 KD 收益已通过，也未将 dev 对象用于训练；范围内无确定修正项。
