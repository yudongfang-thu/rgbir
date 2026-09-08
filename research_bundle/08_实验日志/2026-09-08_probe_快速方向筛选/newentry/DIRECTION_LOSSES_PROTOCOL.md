# C2 与 F-rel 冻结计算定义

本文件冻结两个快速方向筛选的纯 loss。CPU 真值通过不代表训练收益、GPU 准入或正式 E200 证据；不改变原 selector、样本流、优化器或训练预算。C1 对照仍调用原 selected-only API 的 `api.loss`。

## 共用对象与选择

输入 `selection.learning`（也接受解包后的 selection）继承原 C1 全部筛选前 base 对象、`selected`、`valid_levels` 和 `base_object_ids`。base 是 teacher eligibility/quality gate 之前已有的对象集合；分母不改成最终 selected 数量。无新筛选阈值，不解释 selected-only 非学习统计占位为真实统计。

## C2

入口 `class_balanced_loss(selection)` 调用原 `classification_logit._classification_terms`。温度 T=2，teacher raw relative-logit clip=16，非目标系数 eta=0.25；teacher detach。每对象损失为有效层平均的 `(target KL + 0.25 × non-target KL) × T² × selected`。先对每个 GT 类别的对象损失求和，除以该类**筛选前 base 数量**，再对**base 中出现的类别**取平均。某类完全未选中也保留在类别平均分母。未出现类别不加入平均。

若 base 仅含一个类别，直接复用原 C1 loss 的同一浮点归约，保证同输入 loss/gradient bitwise 一致。多类别归约的含义是改类别权重；不能称原 C1 相同梯度剂量。空 base 或无 selected 返回与 student raw delta 图连接的零。

## F-rel

入口 `feature_relation_loss(selection, student_raw, teacher_raw, batch)` 使用确认的 Detect head 输入 `raw['feats']`，仅 P3/P4（列表索引 0/1）。student 使用 `batch.img` 和 RGB GT，teacher 使用 `batch.strong_img` 和 `teacher_batch`（兼容 `ir_batch`）GT。`base_object_ids=(image_index, RGB global GT row, IR global GT row)` 必须与各自图号/类别匹配；不同模态不共享 ROI 框。

每个 GT 的 normalized xywh 按**实际对应输入图 H/W**转 pixel xyxy。每边按 `(i+0.5)/3` 取 canonical 3×3 cell centers；grid 为 `2*x/W-1, 2*y/H-1`，`grid_sample` 使用 bilinear、zero padding、align_corners=False，原特征空间尺寸参与实际插值。每个 image/level 将多个 ROI 网格沿 grid 高度拼接，一次调用采样单张 feature map，不复制 feature map 为每个 ROI 的 batch。

采样 token 为 9×C，按 channel 维 L2 normalize，固定 epsilon=1e-6，再计算 9×9 cosine Gram。student/teacher 可有不同 channel 数，不做通道配对或投影。删去对角线后对 72 个 relation 的平方差取均值；对每对象共同有效层取平均，最后 selected 对象求和除以筛选前 base 数量。仅 selected/valid 的 ROI 被采样。采样、normalize、Gram/MSE 均 FP32，teacher 在 no_grad 下 detach。无 trainable projection。

空 base 或无 selected 返回与 P3/P4 student 特征图连接的零。只有计算结果有限性/身份断言，不加入新的数据依赖 gate。F 的系数由根任务已冻结的非 AP calibration 协议处理，本文件不调 epsilon、系数或拒绝后补救。

## 验证与范围

`test_direction_losses_cpu.py` 在本地 torch 1.8 CPU 上调用保存的真实 pinned classification 源码，12/12 真值通过：单类 C1 loss/梯度精确、未选中类别分母、混合有效层、空集梯度、已知坐标 ramp、合并/逐 ROI 采样相等、通道正交变换/不同 channel 数 Gram、独立双 GT/global row、有效层/base 归约、teacher 无梯度、错误归属拒绝与输入不变。

记录：`DIRECTION_LOSSES_CPU_attempt2.json`。首次测试因 fixture 目录祖先写错，在 import 前失败，单独保留 `DIRECTION_LOSSES_CPU_import_attempt1.json`；未改 loss 定义修复该测试问题。没有计算新 hash、GPU 测试、AP 读取、权重写入或优化器步骤。实际 pinned CUDA / AMP 由根任务受限 canary 继续验证。
