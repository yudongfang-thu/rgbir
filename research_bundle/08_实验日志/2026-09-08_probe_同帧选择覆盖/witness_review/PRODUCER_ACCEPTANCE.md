源码限定 READY：新 `detector_witness.py` 与 `run_witness.py` 已核对原生解码/NMS/GT 身份链；本地 6 项 producer CPU 真值独立复跑通过，另有 4 项原匹配排序见证通过。实际 installed native head/API 的 CPU 检查由 root 在部署环境继续执行。

producer 直接调用真实 `DetectionValidator._process_batch`，通过原 `BaseValidator.match_predictions` 的 return frame 记录实际 GT/预测索引对，并逐位验证返回 TP 矩阵。互审发现并修正“非空输入且全 false TP 时可能未捕获原函数仍通过”的防漏问题：现在非空 GT/预测要求恰好 1 次捕获，任一为空要求 0 次，且已有对应负例。旧 CPU attempt1/2 均保留。

原 `Detect._inference(existing_raw)`、NMS clone、conf>.25/IoU=.7/max300、multi_label=True（nc1 内部退化）、逐图 NMS、input canvas GT 与保留 anchor 索引接口相符。逐图调用使本张结果在 timeout 检查前写入，避免批量提前退出把后续图像当空检出。raw tensor version 有检查；原始 OEv1 FP32 dense `>=.25` 定义和 native dtype/pre-NMS `>.25` 定义分别标记，不混为同一门。

root driver 仅 S/T/R 各一次完整模型 forward，额外为同 raw head decode，检查参数/BN/optimizer/EMA 未变，并明确 `head_decode_nonstate_caches_may_change=True`。新 scope 已统一为 `SAME_FORWARD_DETECTOR_WITNESS`。旧对象状态/gate 与此前 80 对象实际导出 exact 是执行前置条件。

本地 `PRODUCER_INDEPENDENT_CPU.json` 诚实标记 saved native source AST、synthetic head，不能当作 installed head 检查；独立排序回执为 `NATIVE_MATCHING_CPU.json`。本次无 GPU、SSH、额外训练、权重加载、新 hash 或新结果读出；接受仅限同一批 raw 检出与身份见证，不支持完整 dev AP、训练收益或总体覆盖率。
