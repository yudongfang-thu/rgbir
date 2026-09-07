# LLVIP完整dev TIDE接入协议（有效attempt2，结果前记录）

只读取根任务已核验的`../llvip_full_eval/remote_completed_attempt2/{N42,T42}_full_attempt1/`。这两个子目录的attempt1是第二轮有效campaign内部的任务编号，不是先前GT丢失的无效campaign。无效远端attempt1不读取、不参与指标解释。

每端点预期2406张dev图、7879个person GT。逐图记录路径必须经`capture_contract.json`的显式`alias_to_canonical`双射绑定canonical roster；GT数与expected_gt/loader_gt/captured_gt三项同时一致。两模态路径不同，分别核完整性，不按basename自行建立跨模态配对。

运行已验证的`run_tide_audit.py`，保持此前冻结的官方TIDE参数：正IoU=.50/.75、背景IoU=.10、max_det300，原预测conf=.001、NMS IoU=.7、FP32、native rect；逐图全GT、空预测图均保留。输出六类main error及special FP/FN独立oracle，不累加、不视为可达KD收益。

模型为旧历史visible_seed42_native_b32a2与infrared_seed42_native_b32a2的E200 last端点，**不是新协议N**，没有三seed方法增益归因。原pinned AP50/AP75/mAP50–95另列为参考，不当作TIDE同口径；两端点AP差只说明既有RGB/IR检测器差异，不等于IR知识可学性或KD收益。源数据组配对由根任务另行核验，本CPU入口不认证物理配准。

每端点另有独立NumPy score-greedy/AP101逐类交叉核验及先前已知真值测试。源只记path/size/mtime，保存运行源副本，不新增hash或训练。
