# 固定匹配框的 cross-GT CPU 补项（结果前冻结）

scope=`DRONE_FIXED_NATIVE_IOU50_WITNESS_CROSS_GT`。只读已独立接受的 `cpu_analysis/output_attempt2`，不再推理、NMS或重新匹配；无GPU、训练、checkpoint、hash、AP计算，也不改L门。

固定仅score>.25的两个GT状态层：双方native IoU.5正确、只有T相对own IR-GT在独立native IoU.75匹配中正确的1869对，及只有N相对own RGB-GT在IoU.75正确的950对反向对照。人数必须exact；不存在/错误输入即失败，不给缺项填0。

每对始终取N/T各自原生IoU.5匹配的原预测ID和原缓存框。单独读取同GT在IoU.75匹配的ID，并分为same/changed/unmatched；不能把高阈值换到的更佳框用于交叉GT主统计。固定IoU.5框自身的ownGT IoU也重新计算并与原见证exact核对；独立IoU.75的GT布尔不能代替该固定框的IoU。

共同缓存坐标为native canvas H544×W672，original_shape为H512×W640。直接在共同native坐标计算两个原匹配框对两份独立GT的IoU，主要比较 `IoU(T固定框,RGB GT)` 与 `IoU(N固定框,RGB GT)`。不隐式缩放到原图、不裁剪框、不计算最佳候选、不重新做GT关联。该操作是假定同像素位置的交叉标签读出，不是物理配准或跨模态定位蒸馏准入。

每个固定cohort整体及原GT关联IoU的[.5,.8)、[.8,1]两层，报告整数分母/图数、T固定框对RGBGT覆盖>=.5和>=.75、固定N框对RGBGT对应覆盖、IoU差(T−N)>0、>0.05、<0、<−0.05、==0。另报T-own比N-own更大但T-to-RGB不再更大的反转数。阈值和分层不根据结果更改；空分母比例为null。原native.75 ID变更按N/T各自单列，same/changed/unmatched合计闭合到各层分母。

逐对象保留N/T GT坐标与ID、原预测ID/框/分数、.5/.75匹配身份、fixed框own与cross IoU、差值、原标签IoU/分层、实际canvas/originalshape。输出新 `cross_gt_attempt1`（源码/CPU小真值在cross_gt_source）；summary.json、objects.jsonl.gz、README及input/source stat/copies。失败attempt保持原样。

CPU小例至少覆盖独立GT错位让teacher-own优势在RGBGT下反转、.75换框但主读出固定.5原ID、负向与零差、分层.8边界。执行后交loc独立验收。该局部固定分母回答几何优势是否能迁移到配对RGB标签，不等于训练剂量、实际蒸馏收益或新学生收益。
