# 历史 L2 首批记录与当前定位候选的桥接

**实际历史L2在11个“双方IoU≥.5、仅T≥.75”对象中选中4个；另外7个均因真实参考框IoU已达到.70，被原reference-gap门排除。** 这项结论来自旧L2逐对象记录，没有将native-box代理4/11代替真实选择，也没有重新运行L2、前向或训练。原核心输出保持冻结。

本项在 [L2_RECORD_BRIDGE_PLAN.md](../L2_RECORD_BRIDGE_PLAN.md) 冻结后执行，范围是旧 `EXPLORATORY_FIXED8_BNFROZEN` 校准第1批与已接受首32缓存读出的连接，不是新的同一次forward L2测量。CPU程序及小真值见 [bridge_l2_records.py](bridge_l2_records.py)、[test_bridge_cpu.py](test_bridge_cpu.py)，3/3通过 [CPU_attempt1.json](CPU_attempt1.json)。实际原值见 [output_attempt1/summary.json](output_attempt1/summary.json)、[80条逐对象桥接记录](output_attempt1/objects.jsonl)，全部79条历史base记录另完整保存在 [original_first_batch_L2_stats.json](output_attempt1/original_first_batch_L2_stats.json)。

## 身份与关联检查

旧校准输入为 [calibration/llvip](../../2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/llvip/)，当前见证输入为 [witness probe](../../2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe/)。输入stat记录在summary；不计算新hash。

- 两份完整配置逐键一致，唯一差异为 `calibration_receipt` 从校准时null变为其完成回执路径。模型、参考、教师路径，seed42、batch32、数据/增强、BN冻结、AMP以及所有其他配置均一致。S/R为已完成LLVIP visible42，T为infrared42；具体完整路径保留于原配置与输入回执。
- 校准 `student_initial_checkpoint` 的path/bytes/mtime与当前见证模型stat完全一致。原校准完成回执记录每批恢复全部参数/buffer、完整初始化state核过、BN不变、无optimizer/EMA更新；原代码使用冻结R/T。当前见证另记录S/R/T初始化路径及stat、学生state不变。**旧校准回执未单独保存T checkpoint stat**，本桥接不伪称曾做两次T权重文件全字节比较。
- 旧校准第1行32图文件列表与当前 `first_batch_stream.json.im_file` 顺序exact。79条历史base记录的图内batch索引、RGB/IR全局GT行、类别、双GT框和pair IoU逐字段exact，并绑定当前稳定GT ID。原C完整门字段与核心对象记录exact；未进历史L2 base的1个对象保留为明确空记录。
- L2-box与L2-GT原首批79条base_records逐字典exact；selected IDs及五元anchor列表也和历史记录重建exact。共有base79、reference_reliable78、reference_gap7、teacher_own_quality7、mapped_rgb_quality7、selected7，全部与历史计数闭合。
- 额外只核保存输出的实际重叠：历史R anchor与当前FP32 dense witness相同的72条，其box/confidence/class/IoU全部exact。历史T选取优先序不同，与当前dense witness没有同anchor条目，因此这一交叉没有T重叠字段证明。**不据此声称整张raw张量exact，也不把native AMP框当旧FP32框。** 历史训练像素没有保存做逐元素比较，当前身份闭合来自文件序列、GT、配置、回执和上述有限输出重叠。

## 11个定位候选的实际门

分母固定为这11个对象、8张图。下游门未执行用null表示，不能把它读成“教师质量不合格”。

|历史L2阶段|保留对象|涉及图像|保留/11|
|---|---:|---:|---:|
|base|11|8|100%|
|reference_reliable|11|8|100%|
|reference_gap：reference IoU < .70|4|4|36.36%|
|teacher own quality|4|4|36.36%|
|mapped RGB quality|4|4|36.36%|
|selected（含固定领先>.05）|4|4|36.36%|

|稳定ID末段|历史reference IoU|历史L2 selected|原C selected|
|---|---:|:---:|:---:|
|130231.jpg::native_gt:1|0.748729|否|是|
|130231.jpg::native_gt:2|0.723067|否|是|
|020011.jpg::native_gt:1|0.656922|是|是|
|020011.jpg::native_gt:2|0.732264|否|是|
|130313.jpg::native_gt:2|0.588235|是|是|
|030056.jpg::native_gt:1|0.704113|否|否|
|160573.jpg::native_gt:1|0.728077|否|否|
|160573.jpg::native_gt:2|0.711267|否|是|
|100910.jpg::native_gt:3|0.699322|是|是|
|100171.jpg::native_gt:2|0.741624|否|否|
|180167.jpg::native_gt:2|0.643717|是|是|

7个未通过对象都已有reference_reliable=true，但参考IoU在[0.704113, 0.748729]，不满足原 `<.70` 条件。程序因此未评估它们的teacher候选、映射质量及领先门，不能据记录空值归因“教师未提供质量”或“映射失败”。4个通过reference_gap的对象后门全部通过。候选实际是L2自己的P3/P4、支持域、unique-owner后选出的confidence-first R anchor与独立teacher anchor；native见证只定义本次11对象的外部诊断桶。

## 全80对象与原C的关系

|原生对象桶|对象数|历史L2 selected|原C selected|
|---|---:|---:|---:|
|双方≥.5且双方≥.75|60|0|20|
|双方≥.5、仅T≥.75|11|4|8|
|双方≥.5、仅S≥.75|1|0|0|
|双方≥.5、双方未到.75|5|2|3|
|仅T≥.5|1|1|1|
|双方未到.5|2|0|0|
|全部|80|7|32|

全80对象中C/L2同时selected6个，仅C26个，仅L2 1个，均未selected47个；在11定位候选内，同时selected4个，仅C4个，均未selected3个。summary交叉键的`C0/C1`及`L20/L21`只是selected布尔0/1，**不是方法臂C0/C1或L2变体**。原C覆盖与历史L2覆盖分别来自各自真实记录，不能互换。

这解释了当前固定版本的一处范围差异：`.75`定位诊断会纳入参考已达到`.70`的对象，而原L2把这些对象挡在教师门之前。它没有验证放宽`.70`会带来学习收益，不授权现场改阈值/系数，也不解除L1物理几何合同。旧L2短训结果、原剂量和训练结论全部保留。

## 复跑与后续

程序只用Python标准库和PyYAML，拒绝覆盖已有输出：

```text
python bridge_l2_records.py --calibration ../../2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/llvip --witness ../../2026-09-08_probe_同帧选择覆盖/witness_evidence_1315_final/probe --core ../output_attempt1 --output output_new
```

以上命令从本bridge目录运行。核心与本桥接无新GPU；已交独立核验。根据root决定，不自动扩128图。若继续研究，应先据这些具体真实门和学习位置讨论新协议，不能由本单批4/11直接升级增益或整个训练分布结论。
