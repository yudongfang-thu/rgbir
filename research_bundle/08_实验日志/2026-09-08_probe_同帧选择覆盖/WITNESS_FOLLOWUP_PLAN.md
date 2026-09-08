# 同批次原生检测见证：固定后续诊断

在首个同帧覆盖探针完成后固定本计划。原结果与release_v1保留，不改变旧错误状态定义或C选择规则。

已观察到：原GT辅助分配中教师未达“正确”的17个对象，有15个仍通过实际教师any-candidate正确门。其中一个对象已有具体替代高置信候选见证，但这也涉及邻近GT；因此不能从旧assigned-low-confidence直接判断最终检测是否漏检。

只重现同一首32图训练批次、相同S初始化/R/T、AMP和BN状态，一次无梯度S/T/R前向。首先要求旧80对象、标签、选择链及原值与前次逐字段一致；然后同时导出：

1. 各模态旧FP32 raw解码下，同类confidence≥0.25、own-GT IoU≥0.5的候选数和最高置信候选，附anchor ID、框、与其他GT的重叠。
2. pinned Detect._inference的原生解码，再用原生NMS（confidence=0.25、IoU=0.7、max_det=300、class-aware），以及原生验证器IoU=0.5匹配的GT↔预测身份。保留NMS输入副本，防止其原位xywh→xyxy操作污染解码证据。
3. 对照旧assigned、dense-any、原生post-NMS三个正确性定义，重新计算教师正确/学生错误和反向风险桶与原selected的交集。

这里0.25继承既有对象状态阈值；正式AP评价的conf是0.001。本诊断不是AP评价、召回曲线或重新挑选评价阈值。teacher始终针对自身IR GT；S/R针对RGB GT，不把GT相等当物理配准。

不用optimizer step/反向/EMA update，不计算KD损失，不开启新训练矩阵。原生head解码允许更新shape/anchor缓存等非state属性，参数和registered buffers仍必须不变。新scope `SAME_FORWARD_DETECTOR_WITNESS`，输出94 `rgbir_selection_coverage_20260908/witness_attempt1/`，原有global lease与资源纪律不变。先通过原生匹配/候选真值CPU检查再执行。

结论仅限这一批成熟起点。无论结果如何，不由本次单批诊断自动扩E200、扫门或扫系数。
