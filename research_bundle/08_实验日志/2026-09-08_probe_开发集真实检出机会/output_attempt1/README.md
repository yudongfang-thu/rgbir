# LLVIP完整dev缓存原生匹配读出

2406图/7879个配对GT；旧native RGB42/IR42。只在原缓存上保序筛分和独立原生匹配，未推理/训练/NMS重跑/AP计算。结果待独立验收。

|缓存子集|IoU|双方正确|仅T正确|仅N正确|双方未匹配|N TP/FP/FN|T TP/FP/FN|
|---|---:|---:|---:|---:|---:|---|---|
|原低阈值|0.5|6326|1138|184|231|6510/21896/1369|7464/5465/415|
|原低阈值|0.75|2515|2256|747|2361|3262/25144/4617|4771/8158/3108|
|score>.25|0.5|5351|1635|238|655|5589/1998/2290|6986/519/893|
|score>.25|0.75|2258|2297|659|2665|2917/4670/4962|4555/2950/3324|

## score>.25的逐GT联合定位状态

|N@.5|N@.75|T@.5|T@.75|GT|图|
|---|---|---|---|---:|---:|
|False|False|False|False|655|504|
|False|False|True|False|642|489|
|False|False|True|True|993|739|
|True|False|False|False|189|177|
|True|False|True|False|1179|819|
|True|False|True|True|1304|994|
|True|True|False|False|49|48|
|True|True|True|False|610|490|
|True|True|True|True|2258|1309|

四组分别调用实际native匹配；高IoU匹配不假设是低IoU子集。低阈值→>.25的16格转移、比例、图像覆盖见summary.json；对象和原预测ID在objects.jsonl.gz / prediction_matches.jsonl.gz。

仅T匹配是检测互补线索，不是可学习的KD收益或教师oracle。两模态GT数组相同来自LLVIP标签登记，并非独立物理配准证明。缓存不能恢复dense候选、NMS被抑制框、原C门或训练剂量；旧200dev的assigned低置信不是这套真实post-NMS未匹配定义。Drone IR完整缓存缺口仍单列。
