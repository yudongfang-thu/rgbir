# 首32图 native框与历史L2学习anchor连接（结果前冻结）

scope=`LLVIP_FIRST32_HISTORICAL_L2_NATIVE_ANCHOR_JOIN`。仅CPU读取旧完整80GT、32图（含空图）、原native见证、已接受定位状态和历史L2首批79条base记录；不重跑模型/NMS/匹配/selector，不读取checkpoint或计算hash。

主输入为上阶段 `训练侧定位覆盖/output_attempt1/objects.jsonl` 与 `bridge/output_attempt1/{objects.jsonl,original_first_batch_L2_stats.json,summary.json}`，及 `同帧选择覆盖/witness_evidence_1315_final/probe/{witness_objects.jsonl,witness_frames.jsonl,export_contract.json}`。原接受回执仅证明输入桥接身份；本次新增anchor连接仍待独立验收。

固定保留全80GT，另外报告11个“双方native IoU.5正确、仅T达.75”对象、1个反向对照、历史实际L2 selected7以及它们的交集。原C selected32和L2 selected7分开；历史7/11未执行T后门的null不得写成true或false。

以稳定RGB/IR GT ID、frame/image index、增强后global GT行、双GT框及class精确连接。native prediction ID须指向同frame、同模型 `witness_frames.native_detections` 的真实记录，并与core保存的anchor/conf/box/classexact；不能用框相等推定同anchor。native IoU.5原匹配固定，IoU.75仅记录原已有身份，不重新选择。

历史reference_anchor是L2候选R点；只有selected为true且能与原selected_anchors五元组(image index, R anchor, T anchor, RGB globalGT, IR globalGT)exact闭合时，才称实际L2学生学习anchor。源码 `localization_box_v2.compute` 使用 `decode_selected(student, ids[:,0], ids[:,1],...)`，ids[:,1]正是R选点。非selected对象仅有reference候选位置，actual_learning_anchor=null。

用原raw_shapes和strides8/16/32恢复扁平anchor的level/row/col/center，并保留原ID。历史与当前两次forward的范围局限沿用原bridge：首批配置/文件/GT/学生初始化stat及部分FP32字段exact，但没有历史完整像素/raw张量和独立T stat全量比较。same/different只指在已绑定同frame/grid下anchor整数ID是否一致，不声称跨运行所有数值相等。

逐对象报告：(1)历史R候选对native S/R IoU.5匹配anchor；(2)实际学习anchor对native S/R匹配anchor；(3)历史T目标候选对native T匹配anchor；(4)历史R候选对当前FP32 dense witness，另核同ID时box/conf/IoU字段，不把AMP native decode混作FP32。缺记录、未选、native未匹配、T未执行分别列unavailable及原因；不靠另一GT或最佳框补值。真实native检测集合中是否存在同anchor另列，以区分“不是此GT匹配框”和“当前NMS集合没此anchor”，不归因为抑制/低置信，除非缓存直接证明。

输出全80条objects、整数分母及same/different/unavailable，固定11与actual selected7的交叉，缺字段表、源/输入stat及简明README。小真值包含同框不同ID应different、不同框同ID仍same（数值另记）、跨frame拒绝、未selected不能称实际学习、未执行T保持null。当前缓存能回答位置连接，不能提供未保存anchor的学生DFL或判断学习梯度/负迁移；不改L门。
