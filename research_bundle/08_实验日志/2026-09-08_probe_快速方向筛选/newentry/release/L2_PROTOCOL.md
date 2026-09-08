# L2-box / L2-GT 固定算子协议

**新对象坐标回归诊断候选；不解除旧 L1 几何阻断。** 本协议来自根任务在计算新结果前的固定定义，适用于同画布、配对增强后的 RGB/IR GT 实例关联。未启动 GPU、训练或评价。

接口为 `compute(student, teacher, reference, batch, strides=(8,16,32), variant='L2-box') -> (loss, stats)`；另一个合法 variant 是 `L2-GT`。调用前使用原 runtime 的 pinned 路径：`rgbir_independent_kd_v2` 和其 `task_conditional_reference`。输入 `scores[B,C,A]`、`boxes[B,4*reg_max,A]`（DFL logits）、三个 `feats`；RGB 标签为 normalized xywh 的 `batch_idx/cls/bboxes`，IR 独立标签为 `teacher_batch`（兼容已有 `ir_batch`）。学生与 R 的网格、bin 数和类别一致；T 独立选 anchor。画布坐标只支持标签关联，不构成像素配准证据。

1. 复用 pinned `selection_adapter.original` 的标签、布局、IoU、解码及同类最大基数/最大总 IoU 一对一匹配：匹配门 0.5，随后标签对 IoU ≥ 0.8。
2. R 只在 P3/P4 中选择。复用 pinned `localization_loss.native_candidate_mask`，包括小 GT 扩张规则；使用 RGB 全部 GT（含异类、未匹配对象）要求 anchor 只有一个 native GT owner。保留原 unclamped RGB GT DFL 支持域 `[0, reg_max−1−0.01]`，不靠 clamp 变成合法。粗条件 R 最大类置信度 ≥ 0.05、预测对 RGB GT IoU ≥ 0.1；按置信度、IoU、升序 anchor ID 决定单 anchor。
3. **在粗 R 选择完成处固定 base_count**，早于 R 可靠性和 teacher quality。根任务确认此分母位置。后续 R 门为预测类正确、置信度 ≥ 0.25、RGB GT IoU < 0.70；失败对象仍在分母。
4. T 在 IR 侧 P3/P4 独立寻找该配对 GT 的 anchor，也由 IR 全部 GT 检查唯一 native owner；预测类正确、置信度 ≥ 0.25、own IR GT IoU ≥ 0.5。按 own IoU、置信度、升序 anchor ID 决定单点。T 不继承 R anchor，也不要求跨模态 DFL bin 对齐。
5. 将 T decoded xyxy 用 IR GT→RGB GT 的逐轴平移和比例变换，得到 canonical object-relative target。映射后的框须对 RGB GT IoU ≥ 0.60 且严格大于 R IoU + 0.05。该变换不估计相机或物理配准；正轴向仿射下映射前后的 own GT IoU 应保持（两道质量门保留为明确审计字段）。没有裁剪目标。
6. 学生在固定 R anchor 按原 native float32 softmax/bin 期望解码，保留梯度，转 RGB GT 归一化 xyxy。L2-box 目标为同样归一化的映射 T box；L2-GT 目标为 `[0,0,1,1]`。二者对象、R/T anchor、门和分母完全相同。损失为每对象四边 SmoothL1（beta=0.1）均值，再对选中对象求和除以 `max(1,base_count)`。

T/R 解码、所有选择和目标都在 no-grad 中，学生数值不参与选择。非法 GT 宽高、非有限 raw、错误布局/variant 会报错；预测宽高不合法者不进入候选，选中学生框非法时报错。重复/重叠 GT 的 anchor 归属歧义被全部 GT 的唯一 owner 检查拒绝；GT 关联本身沿用原确定性一对一 Hungarian 规则，没有新 physical identity 标注。

空集返回与学生 boxes 相连的有限可微零。算子不读/修改 native loss、不计算分类 KD、不施加 lambda 或 batch-size multiplier；外层必须保留完整 native loss，并只施加一次已冻结 lambda。lambda 的 8 固定 batch/native 10% 梯度定标由根任务负责，本算子不自行定标或放宽门。

LLVIP 等共享 RGB/IR 标签的输出明确属于 **annotation-coordinate target**，不能称为独立物理标注、真实像素对应或旧 L1 几何准入。所有当前小样例仅为 CPU 算子真值，不是新数据集结果。
