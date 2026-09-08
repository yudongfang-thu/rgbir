# 对象坐标完整概率目标：独立执行审计

**PASS（限固定CPU目标接口）**。实际执行完成后，从原始四文件全量重算80 GT × 2路径、628条可用边分布和全部16-bin；输出与独立hat-basis参考一致。未将源码READY当成已执行通过。

|固定输入|全部GT|可用|支持|越界拒绝|缺失|支持且恒等|
|---|---:|---:|---:|---:|---:|---:|
|教师同R索引（主）|80|79|79|0|1|79|
|教师native anchor（压力）|80|78|15|63|2|15|

主路径支持的79对象全部退化为标签和anchor相同下的恒等变换。压力路径63对象因正质量越界而完整拒绝，不能丢尾、截断或挪anchor补齐。通过仅说明本次目标接口的计算与记录可靠，不说明非恒等运输可用、物理配准成立、旧L1解锁或KD有效。

源概率、目标概率和未截断距离的独立重算最大差均为0；边坐标期望最大差1.1368683772161603e-13。实际源定义明确为原logits派生FP64 T=1 softmax，缓存FP32/native原向量完整保存；与缓存FP32最大差2.443613299485392e-07，不冒充逐位相同。生产CPU耗时0.100148300秒；没有新推理、训练、GPU或新增样本。

独立数学小真值37项通过，作者CPU测试10项通过。原四输入、最终四源码及执行结果在结论前重新逐字节核对，路径和GT/角色/anchor/forward身份齐全；按用户本轮范围未计算新hash。源码旧快照和补丁历史均保留。

审阅者为 `/root/object_transport_review`，与执行者 `/root/ap_error` 分离；可观察环境未提供确切模型标识，不宣称跨模型复核。输入数据属于真实模型缓存与标签坐标（real_gt），数学小真值属于synthetic_proxy。全部字段、证据路径和逐对象核对见 [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json)、[recomputation_summary.json](recomputation_summary.json) 与 [recomputation_rows.json](recomputation_rows.json)。
