# 同帧检测见证聚合：固定输入与边界

[witness_analyzer.py](witness_analyzer.py) 为CPU只读聚合器；[4组真值](test_witness_analyzer_cpu.py) 已通过，见 [WITNESS_ANALYZER_CPU_attempt1.json](WITNESS_ANALYZER_CPU_attempt1.json)。原80对象、原状态、实际C gates和selected必须与前次probe逐行exact；本分析不重新选对象，不将新检测诊断替换成C训练门。

```text
python witness_analyzer.py --previous-probe evidence_1255_final/probe --input NEW_WITNESS_PROBE --output NEW_READOUT
```

读取前次 `objects.jsonl`，以及本次 `objects.jsonl`、`witness_objects.jsonl`、`completion_receipt.json` 和 `witness_contract.json`。本次原objects需与前次exact；witness对象去掉唯一新增detector字段后也需与原对象exact。scope固定 `SAME_FORWARD_DETECTOR_WITNESS`，32图80个增强后RGB GT，模型forward S/T/R各1次、无反向/optimizer/EMA更新、原流和原对象exact，失败或未完成不出正式读出。

每模态明确区分四种正确性定义：

1. **old_assigned**：原FP32辅助解码后，GT辅助一对一空间分配得到的那个anchor是否正确。
2. **dense_any_correct**：原FP32 decode中是否存在任意同类、confidence≥.25、own-GT IoU≥.5的anchor。T列必须逐对象等于原实际C教师any-candidate门；不因为诊断分配框低confidence就改写这个门。
3. **native_pre_nms_any_correct**：原native head对同一raw输出解码后、NMS前，是否存在同类、confidence>.25、own-GT IoU≥.5的anchor。它把原FP32与native decode/边界差异单列。
4. **native_postnms_matched**：原native NMS `conf=.25, iou=.7, max_det=300, multi_label=True, agnostic=False`，再由原native `_process_batch` / `match_predictions` 在IoU=.5匹配；读取实际匹配的GT/prediction见证。LLVIP nc=1下multi_label内部等效单类，但记录原实际传参True，不能伪写False。

主S/R相对RGB own GT，T相对IR own GT；IR到RGB不作物理映射，也不把own-GT正确当作配准认证。旧FP32门使用 `>=.25`，native候选过滤使用 `>.25`，两者保留真实边界，不统一改阈值。

输出每模态四种定义两两的2×2混淆表，以及各定义下T正确/S错误机会、S正确/T错误风险、两者正确、两者错误和教师未知桶。原selected覆盖同时给对象数、selected/桶和该桶selected/全部selected，零分母为null。原生匹配见证按(frame_id,modality,prediction_index)不能重复分配给多个GT；dense-any见证允许同一个anchor覆盖多个GT，两者语义不可互换。

每个true判定必须有同一own-GT的阈值合格见证，false时见证必须null；dense数量与布尔项闭合，native真实匹配来源和profile必须声明通过。输入stat记录路径/字节数，不计算hash。原始完整JSONL保留。

这不是AP评价：只用confidence .25、单IoU .5、一个train批次，不是完整dev低阈值PR积分。混淆变化可以说明旧assigned诊断与实际检测匹配是否一致；风险桶不证明负迁移已经发生，机会桶也不证明KD能修复。没有训练/梯度、全dev、旧200dev修复率或E200效果结论。
