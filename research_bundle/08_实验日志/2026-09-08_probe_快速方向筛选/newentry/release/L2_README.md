# L2 对象坐标回归算子

**代码已实现，13/13 作者 CPU 真值通过；仍是待独立审阅和受限实跑的 L2 诊断候选。旧 L1 保持 blocked。**

固定定义见 [L2_PROTOCOL.md](L2_PROTOCOL.md)，实现见 [localization_box_v2.py](localization_box_v2.py)。它通过配对 GT 实例把 IR teacher 的预测框转成 RGB GT 相对坐标；T 与 R 分别选 anchor，同 mask 的 L2-GT 只替换回归目标为 unit box。完整 native 检测损失仍由外层保留。

[CPU 回执](L2_CPU_CHECKS_attempt1.json) 与 [测试源码](test_localization_box_v2_cpu.py) 包括：仿射/不 clamp 坐标、与 pinned decode 数值 exact、学生选中 anchor 的有效梯度且 T/R 无梯度、L2-GT 同集合与真值损失、非完美 teacher 的不同目标、学生不影响选择、teacher 前分母、两侧异类/未匹配 GT 归属歧义、配对标签门、R 原支持域、非法输入、空集、T 排序 tie。全部为本地 CPU 合成数据，未读新 AP、运行 GPU/SSH 或新算 hash。

复跑示例（`--receipt` 必须指定尚不存在的文件以保留旧 attempt）：

```text
D:/Anaconda/envs/KGJ_proj/python.exe test_localization_box_v2_cpu.py --pinned <pinned_rgbir_independent_kd_v2> --receipt <fresh_receipt.json>
```

来源仅复用当前 pinned selection/native geometry helpers；学生独立的可微 decoder 已与原 detach decoder CPU exact 对照。缺失的 T 读出为 null，未伪造为零；base_records 保留所有粗 R 对象及后续筛选。返回 loss 为 unweighted scalar，lambda 由根任务固定 8 batch 梯度定标，本文件不写新训练器或修改旧 run。
