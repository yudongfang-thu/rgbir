# 独立审阅范围与执行边界

审阅者：`/root/dfl_readout_review`；可观察模型身份：`unavailable`。本审阅由同级执行者独立读取原始源码与小产物，不把角色名称当作跨模型证据。

当前状态：`PENDING_SOURCE_AND_EXECUTED_CPU_RECEIPTS`。尚无本批新输出验收，不准把本文件当作 GPU 启动 READY。

固定依据：`../2026-09-08_probe_定位学习位置与目标/RAW_DFL_INFORMATION_PLAN.md`（相对实验条目父目录）。仅原自然流首 32 图、80 GT；S/R/T 各一次本批前向，零更新。审阅者不启动 GPU、模型前向、训练、官方 test 或新 hash。CPU 合成真值和既有小产物解析在本任务范围内。原始产物及失败 attempt 不改动。

## 已核原始接口

- 原 `witness_release/run_witness.py`：S 为 train + BN frozen；R/T 为 eval；AMP 沿原 trainer.amp；原样验证 sample stream、GT 稳定身份及初始化。
- 原 `witness_evidence_1315_final/probe/export_contract.json` 的 raw_shapes：三模型 boxes=[32,64,8400]，scores=[32,1,8400]，feats 的空间尺寸 80×80、40×40、20×20。
- 旧 OEv1 `_layout/_decode_boxes`：按 level 串接、level 内先行后列；中心=(col+.5,row+.5)×stride；boxes reshape [B,4,16,A]，边序 left/top/right/bottom；softmax 沿 16 bins，bin 支撑 0…15。
- 原 `Detect._inference(existing_raw)` 解码与 OEv1 detach/FP32 解码是不同数值路径。旧实际 witness 已记录 raw float16、native float32，S/R 全 anchor 最大框差 0.126129150390625 px，T 0.0975341796875 px；此次不得把两者设为必须逐位相等。
- 旧 `DFLoss.__call__` 会原地 clamp GT 到 [0,14.99]。本读出必须保留未 clamp 距离，合法域外的无 clamp GT 读出为 null；若另报原生 clamp 读出，须明确标注转换，不能将其混写为合法原 GT 读出。

## 源码验收检查项

1. 稳定 image/GT 身份和历史 anchor index 连接，不以近邻框补身份；新 forward_id 与旧前向明确区分。
2. raw4×16 logits 真实取自原 raw boxes，不以 NMS box/期望构造分布；每记录保留模型、图、GT、anchor、level、stride、center、dtype、bins、GT 未 clamp 距离及两种实际解码框。
3. S/R/T 全部 state_dict 在该批前后相同；无梯度/optimizer/EMA 更新；本批 backbone/head prediction 各 1 次有实际计数，decode-only 单列。
4. native/FP32 闭合分别验证；AMP 模式必须明确，不能把再次 decode 写成新增 backbone 前向。
5. 熵为 nats，方差注明 bin²；不同 anchor/stride/bin 语义不能直接做 KL；对本模型本 anchor 的 GT 合法性逐边记录，缺项保持 null。
6. 固定全 80 GT 与既有 11 个正向、反向对象；不从输出选择新阈值/λ，不外推为物理配准、可训练 KD 目标或 KD 效用。
7. 必须读取实际运行的 CPU 回执和随后本批输出；函数定义、测试文件存在及 literal PASS 字段不等于执行证据。

Hash：按明确任务边界，`audited_input_hashes: not computed`。以路径/文件大小/时间及直接内容审阅记录版本，不声称 SHA-256 锚定。
