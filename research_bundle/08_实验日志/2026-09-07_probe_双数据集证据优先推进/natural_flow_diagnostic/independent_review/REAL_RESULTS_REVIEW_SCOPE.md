# 真实自然流结果 CPU 验收范围

冻结于下载 completed attempt2 前。保持原固定 20260907、2/64 batch、B32、T/R 与所有阈值；不重新前向、不训练、不读 AP 决定规则，不访问 GPU/SSH，不新增 hash。

读取两数据集 canary/full 的原始 summary、逐批 trace/selection、source 副本、input/model identity、global lease admission/resource profile。与本地旧覆盖原始 64 流和已审定源码直接 byte identity 对照；每一真实 batch 的旧 source/augmentation/双标签等记录字段必须完全一致。

分类从每个 base record 独立复算 eligible 数、固定 rho 的整批 ceil 及稳定质量排序、normalizer、class/image/group 集合与 summary 累积计数。定位从所有 per-image gate 重积 batch/summary；从 base record 的原始可靠性、IoU 与 margin 字段重算全部后段 gate、selected 身份、class/image/group 闭合。GT 决定的 common/pair/geometry/inside/support/unique-owner 前段 gate 用真实双标签、固定 640 格点和原函数在 CPU 重算（模型值置为无候选，只读 GT 前段统计）。

未保存所有 anchor 的真实 T/R logits，因此不能独立离线重算 reference-candidate 存在性、最高置信 anchor 或分类 teacher quality 本身。对于这些模型值依赖部分，仅能核验输出记录、逐图/整批原调用的运行时 exact 断言、已审源码与上游 frozen 模型身份；不得把这条边界写成全预测重算通过。

资源核验只使用真实采样：峰值须有限正、monitor 无错误、GPU free ≥ 2048 MiB、project RSS ≤ 300 GiB；四卡例外必须给出占后至少 2 空卡的 admission。canary 与 full 不变成 L 训练准入。geometry None/false 不保证最终 selected_count 严格上界。
