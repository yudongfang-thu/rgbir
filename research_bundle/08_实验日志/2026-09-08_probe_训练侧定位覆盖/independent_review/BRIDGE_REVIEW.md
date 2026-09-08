# 历史 L2 记录桥接独立审阅

**ACCEPTED：限定接受原首批 L2 记录的稳定对象关联及真实门覆盖，不将其表述成本次 witness forward 重算 L2。**

独立复跑 3 项 CPU 真值通过。源码重放与原 bridge 输出全部一致；另独立核对 79 条原始 base 记录到 80 对象的图序、RGB/IR 全局 GT 行、类别、双框、pair IoU、稳定 ID 及完整 C 门。原 L2-box/L2-GT 的 base 记录和 selected IDs/anchors 一致，未进 base 的一对象及下游未执行的 null 保留，未伪造负门。

完整配置除校准回执路径字段外一致；学生初始 checkpoint path/stat 与当前回执 exact，原校准声明完整初始化核验、逐批恢复参数/buffer、BN 不变且无 optimizer/EMA 更新。独立字段和计数复核确认：历史 base/reliable/gap/teacher quality/mapped quality/selected = 79/78/7/7/7/7。

11 个双方 .5 正确而仅 T 达 .75 的对象中，真实历史 L2 的门计数为 11/11/4/4/4/4；其余 7 个均在 `< .70` 参考 IoU 门退出，不能由未执行的 teacher 字段归因教师质量失败。全 80 对象 C/L2 同选 6、仅 C 26、仅 L2 1、均未选 47；11 对象内同选 4、仅 C 4、均未选 3。所有桶的对象/图数、ID、null 计数、分母和交叉格已核。

有限输出重叠另明确区分见证定义：历史 R 与当前 dense witness 同 anchor 72 条，box/confidence/class/IoU 均 exact；历史 T 与当前 dense witness 同 anchor 为 0。追加独立检查发现历史 selected T 的 7 条中，5 条与旧 assigned witness 同 anchor，box/confidence/class/own-GT IoU 四字段均 exact（`BRIDGE_TEACHER_OVERLAP.json`）。这支持部分保存输出的身份一致，仍不能补造历史 T 单独 checkpoint stat、全 raw 张量或训练像素 exact 证据。

原 native-box 代理与实际 L2 碰巧均得 4/11，来源仍分别保留。此审阅不支持放宽门、改变剂量、L1 几何放行、KD 增益或扩大 GPU 矩阵；单批覆盖不代表完整训练分布。

回执：`BRIDGE_CPU_REPLAY.json`、`BRIDGE_ACTUAL_RECEIPT.json`、`BRIDGE_TEACHER_OVERLAP.json`；复算脚本 `review_bridge.py`、`review_bridge_teacher_overlap.py`。原源码与产物未改，无 GPU/forward/新 hash。
