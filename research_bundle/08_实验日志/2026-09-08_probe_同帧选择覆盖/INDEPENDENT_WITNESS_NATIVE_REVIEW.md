# Native检测见证源码：限定独立复核

只读 `witness_release/detector_witness.py` 与 `run_witness.py` 的接口/映射和固定参数；接受有界同帧见证准备，未使用GPU、未读取真实新见证结果。

原FP32 dense见证调用原OEv1 decode和_has_candidate核对；T映射按原IR global GT行，S/R按RGB global行。原states/gates深拷贝保留；root driver先把当次完整原objects与前次逐行exact比较。新见证只能新增detector，不重配原GT身份或选择。

Native路径由真实 `Detect._inference(existing_raw)` 完成head解码，没有再次调用模型预测forward；不同于原FP32 surrogate，dtype与最大box差单列。原输入tensor版本保持不变，head非state的动态anchor/shape缓存可能变化，driver已经明确记载该例外。没有把这种缓存例外写成权重或BN可变。

原native NMS逐图运行，固定conf=.25/iou=.7/max_det300/multi_label=True/agnostic=False，返回原anchor indices；逐图调用避免batch总超时跳过后续图。GT匹配直接运行native DetectionValidator._process_batch和BaseValidator.match_predictions，IoU阈值数组仅[.5]。窄profile读取该函数实际return时的matches，不自行实现另一套匹配规则；非空GT/预测要求恰一次捕获，空输入允许零次，且完整TP布尔位与GT/prediction见证及双方唯一性核对。profile在finally恢复。

已读作者6项CPU结果与测试边界：本地保存native源码AST模式不宣称测试了真实installed native head；root另负责94的installed-native CPU入口。不会把本地合成检查冒充实际GPU正确性。新聚合器也明确绑定唯一scope、真实native profile、捕获来源与原对象exact；本复核不要求增加训练或AP实验。
