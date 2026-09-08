# L3 当前阶段证据

**完整概率目标的CPU接口已执行并独立接受；修复后的v2实际8批校准已完成，7/8批提供有限非零共享参数梯度。首次v1显存失败保留，本页尚无三臂短筛端点或AP结论。** 本页归集固定的 `transport/output_attempt1`、`training_evidence_1703`（v1失败）、`training_evidence_1716/calibration/llvip`（v2校准）、初始参照与CPU回执；v2实际校准成功依据其执行回执，不以源码修复或CPU通过代替GPU事实。

## 已完成的完整概率运输

只复用上一阶段固定32张LLVIP训练图、80个GT和317份真实分布缓存，两种输入事先列定；未新增推理、样本或机会统计。学生学习位置固定为历史R候选，T=1概率由全部实际logits稳定FP64 softmax重算，原FP32/native向量分别保留。任何正质量落出学生0–15 bin支撑都拒绝整个目标，不clamp、裁尾、重归一或改选anchor。

|预列输入|固定GT|双方anchor可用|完整目标合法|越界拒绝|缺失|合法目标中代数恒等|
|---|---:|---:|---:|---:|---:|---:|
|教师同R索引，主接口|80|79|79|0|1|79|
|教师原native匹配anchor，压力接口|80|78|15|63|2|15|

CPU执行耗时 **0.1001483秒**；主接口质量/期望/边坐标闭合误差均为0，压力接口最大边坐标闭合误差为`5.684341886080802e-14`。独立审计从原80×2输入重算全部628条可用边，完整概率和距离一致。作者10项小真值与独立37项检查通过。[原始汇总](transport/output_attempt1/summary.json)；[逐对象结果](transport/output_attempt1/objects.jsonl)；[独立执行审计](independent_review/EXPERIMENT_AUDIT.json)。

这说明主接口在该固定批上可以表达目标，且因标签/anchor相同退化为恒等；它没有补齐物理配准证据，也不证明跨anchor运输普遍可用。63个压力对象的拒绝是真实完整支撑限制，不能删除尾质量来补救。原L1几何BLOCKED保持。

## 已冻结的新短筛及当前实现证据

[SHORT_SCREEN_PROTOCOL.md](SHORT_SCREEN_PROTOCOL.md)定义独立的LLVIP N/L3-DFL/L3-GT三臂：共同visible42成熟初始化、固定2048图、3轮×64个B32 batch、BN运行统计冻结、fresh optimizer/EMA，最终2406图full dev。192 batch不等于192次成功更新；不挑best，不据中期AP改门或λ。

R基础集合和0.70门保持原定义，教师只在同R学习anchor重新判断自身归属、质量及完整概率支撑。DFL先对教师logits除以2再运输，学生亦使用温度2；四边平均KL乘4，再除筛选前R基础分母。GT控制共对象、anchor、全部门、分母和λ，仅改为RGB GT双bin分布的sqrt归一目标。目标FP64→FP32若丢失正质量，整对象拒绝，不缩减基础分母。N同辅助路径且λ=0。新方法是标签定义的对象坐标监督，不解除旧L1、不单独声称形状独立于均值有效。

L3的14项合成CPU检查覆盖原R集合、同门GT、教师不换点、完整支撑、精度下溢、温度顺序、loss比例与梯度隔离。v2实际pinned CPU入口共 **36项通过**（22项wrapper/queue与14项L3），总耗时 **4.519332秒**，CUDA可见设备为空；这仅为CPU接口证据。[loss接口](trainer_release/L3_LOSS_INTERFACE.md)；[v2 pinned CPU回执](PINNED_TRAINING_CPU_ACCEPTED_v2.json)。

## v1实际校准：资源失败，不是方法AP负结果

首次队列的实际总耗时 **35.623915秒**，终态为`OBJECT_DFL_MATRIX_PARTIAL_FAILURE`；数据集已完成臂列表为空。8批校准只进行至第2批即停止，没有完成校准系数、三臂canary或FT3评价。[队列原回执](training_evidence_1703/queue/completion.json)；[校准失败](training_evidence_1703/calibration/llvip/failure.json)。

|校准批次|进程NVML峰 MiB|框架allocated峰 MiB|框架reserved峰 MiB|进程树RSS峰 MiB|显存峰加余量 MiB|固定显存上限 MiB|结果|
|---|---:|---:|---:|---:|---:|---:|---|
|1|5454|4389.493164|4954|17156|5888|8192|通过|
|2|8506|7376.645508|8006|17197|8960|8192|资源门拒绝|

两批RSS加余量均为19456 MiB，低于固定32768 MiB上限；停止由显存预算引发。[逐批资源原值](training_evidence_1703/calibration/llvip/calibration_resource_batches.jsonl)。整卡总占用与本进程峰不同；外部guard没有错误不等于可以豁免候选自己的8192 MiB限制。

代码已定位到v1循环变量保留上一批学生计算图的引用链，v2显式释放相关引用并增加跨批allocated生命周期记录。旧源码和失败attempt保留，预算、损失、流、门和λ规则不变。CPU弱引用真值支持该修复机制，其本身不能证明v2 GPU结果；后续实际结果另列如下。[修复与证据范围](CALIBRATION_LIFETIME_V2.md)。

## v2实际固定8批校准

`OBJECT_DFL_CALIBRATION_COMPLETED`，耗时 **37.565724秒**；8批资源检查全通过，进程NVML峰 **5454 MiB**、框架allocated/reserved峰 **4390.573242 / 4954 MiB**、进程树RSS峰 **17280 MiB**。每批恢复全部初始参数/buffers，optimizer和EMA更新均为0。7/8批的DFL共享P3/P4参数梯度有限且非零，按冻结规则得到 **λ=0.6900524651944485**，N=0，L3-DFL与L3-GT共用该λ；这不表示两臂梯度剂量相等。[实际校准回执](training_evidence_1716/calibration/llvip/calibration_receipt.json)。

只按原8条[逐批JSONL](training_evidence_1716/calibration/llvip/calibration_batches.jsonl)中实际字段归集：

|batch|图数|R基础对象数|最终selected|含selected的唯一图片数|
|---|---:|---:|---:|---:|
|1|32|79|7|6|
|2|32|85|2|2|
|3|32|69|0|0|
|4|32|78|2|2|
|5|32|97|1|1|
|6|32|86|4|3|
|7|32|67|3|3|
|8|32|88|6|5|
|合计|256|649|25|22|

8批共有256个不同图片路径，其中250图出现R基础对象，22图有最终selected。表中DFL/GT的base、normalizer、selected数量及实际对象/anchor身份逐批一致。第3批selected为0，对应DFL/GT单位梯度范数均为0，是记录中的空选择，不补成缺失或失败。**CPU主接口79个可表达目标不是79个通过学习门的对象**：实际首批基础数79，最终selected为7。这里的25次对象选择、22张图片及梯度非零只说明本固定校准流的可执行性，不能当作AP、收益或总体覆盖率。

## 旧短筛结果与成熟初始化双参照

旧LLVIP协议中，L2-box−匹配N为 **−0.023992 pp**，L2-GT−N为 **−0.008525 pp**，L2-box−GT为 **−0.015467 pp**。旧坐标目标未显示优于同掩码GT的短程结果；不能把这些不同算子、掩码和协议的臂当作本轮L3的实际对照。[已完成旧短筛报告](../2026-09-08_probe_快速方向筛选/FINAL_REPORT.md)。

原visible42成熟初始化的完整dev原值为：mAP50–95 **0.3287840510939928**，AP50 **0.7161396006544973**，AP75 **0.24282631410771383**，均为fraction。mAP显示为百分数时是 **32.87840510939928**。其checkpoint路径/stat与先前真实DFL S/R初始化、新三臂配置绑定，full dev为2406图/7879 GT；native配置和有效评价参数与新入口相符。该包是`INITIAL_BASELINE_NATIVE_RECHECK`来源复核，原`accepted_endpoint_claim=false`保留，没有新算AP。[初始来源包](initial_endpoint_reference/README.md)；[独立复核](initial_endpoint_reference/INDEPENDENT_REFERENCE_REVIEW.json)。

旧匹配N微调后mAP为32.171224，相对成熟初始化约 **−0.707181 pp**。因此新L3即使超过本轮N，也可能只减轻微调漂移。新三臂必须重新完成，并核对各自实际初始化、full dev roster及actual profile后，才同时报告DFL−N、DFL−GT与每臂−成熟初始化。不能以旧N代替新N，也不能将微小短程差升级为E200、多seed或四臂结论。

修复后固定校准已有上述实际回执；本页待纳入的是三臂实际canary、各192 batch训练及各自完整dev端点，最终运行状态由根任务更新。下一步顺序、固定停止规则及唯一global lease均按协议；本页不授予新的资源豁免或自动扩展。本文仅归集既有产物，没有SSH、新GPU/训练、权重读取、新hash或修改共享索引。
