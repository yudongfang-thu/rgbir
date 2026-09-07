# 固定 anchor 定位目标平移压力诊断（2026-09-07，attempt2修正）

旧attempt1的worst汇总`all_directions_gate_n`被统计分母覆盖，旧README表格误写61/61、47/47；该字段不能作为存活数引用。旧原始attempt与源码/报告快照完整保留于`outputs_attempt1/`和`attempt1_source_snapshot/`。attempt2用`success_n`/`denominator_n`区分成功计数和分母；原逐对象数组、比例与正文4px的34/61、9/47未受影响。修复未改协议、cohort、anchor、阈值或扰动集合。

> **已执行两数据集的固定 25 条件 CPU 诊断，等待独立审阅。LLVIP 的条件定位质量在本压力测试中比 Drone 更稳定；其 dev 固定质量门的 61 对象在四输入像素八方向下仍有 55/61 的教师 DFL CE 始终更低，但 gate 仅 34/61 全方向存活，训练快照只有 23 个门内对象。此结论不能替代原 L1 准入或实际定位训练。**

## 目的与设置

检查同一教师定位目标在人为坐标扰动下的质量和检测头局部监督方向。先冻结 [PROTOCOL.md](PROTOCOL.md)，执行 11 项已知真值检查，然后按 LLVIP→Drone 顺序读取上一阶段 raw DFL/logits 和对象 JSONL；只用 CPU，完整执行约14秒，没有新推理、拟合、GPU、test、训练或旧产物修改。

25 条件为零扰动及 a=1/2/4 时的四轴向与四对角方向。a 是 L∞ 幅度，对角 L2=a√2。原 RGB/IR GT、N42、对象和 anchor 均固定；平移教师期望框，不裁画布、不 clamp、不按扰动后 gate 重新选择。所有主表保留零扰动固定分母。

基础 cohort 与上一阶段定位 readout 的 source_row/object_id/顺序完全一致：非背景、paired GT IoU≥.5、有N参考候选、RGB GT四边未 clamp 距离全在[0,14.99]。quality_gate070 沿用旧质量条件：N/T类正确且conf≥.25，N IoU<.70，T对RGB GT IoU≥.60、对IR GT IoU≥.50、T−N>.05。它只是本共同anchor上的描述性分层，不等于 L1/D2 selected；未把pair≥.8、唯一owner或独立几何证据补造成已通过。

## 固定分母与训练覆盖上下文

|数据集/划分|全部RGB GT|基础cohort|N IoU<.70|quality_gate070|门内唯一图|门内唯一owner=1|
|---|---:|---:|---:|---:|---:|---:|
|LLVIP train|2753|2728|82|23|22|19|
|LLVIP val|643|548|199|61|49|57|
|Drone train|15782|15019|366|87|70|81|
|Drone val|3084|2821|229|47|17|34|

LLVIP 门内 train 23/2728=0.84%基础对象、dev 61/548=11.13%；Drone相应87/15019=0.58%、47/2821=1.67%。这是已训练端点和固定letterbox的静态快照，不能估计真实自然增强64批的有效监督量。LLVIP train门内来自10组且全为stride16，dev来自5组；Drone dev的source_group为`unavailable:val`，不能当作有来源多样性证据。

## 零扰动 dev

CE差定义为 teacher−native，负数表示教师对固定RGB GT的DFL目标更好。框IoU优势为 teacher−native，正数表示教师更好。

|数据集|固定层|n|框IoU优势均值|DFL CE差均值|教师CE更低|KD/GT cosine>0|
|---|---|---:|---:|---:|---:|---:|
|LLVIP|全部基础对象|548|+0.04116|-0.15590|61.13%|79.20%|
|LLVIP|仅N IoU<.70|199|+0.14629|-0.53847|80.40%|88.44%|
|LLVIP|固定quality_gate070|61|+0.26215|-0.94545|100.00%|100.00%|
|Drone|全部基础对象|2821|-0.03431|+0.12561|39.06%|64.37%|
|Drone|仅N IoU<.70|229|+0.13213|-0.40660|80.35%|91.27%|
|Drone|固定quality_gate070|47|+0.22357|-0.67746|97.87%|100.00%|

同anchor框优势与上一阶段“各自候选框”的对象级IoU优势不是同一指标。零扰动CE、KL与cosine逐对象重算与上一阶段定义最大差均为0。

## dev固定质量门：八方向最差情况

每个对象先在给定幅度的八方向取最差，再对原固定集合求均值；“CE始终更低”等比例要求八方向全部满足。gate存活是另报的覆盖量，下面损失/IoU均没有改用幸存者分母。

|数据集|a输入px|固定n|最小IoU优势均值|最大CE差均值|IoU始终更好|CE始终更低|cosine始终>0|gate全方向存活|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|LLVIP|1|61|+0.22546|-0.90463|100.00%|96.72%|100.00%|60/61|
|LLVIP|2|61|+0.18200|-0.85904|98.36%|96.72%|100.00%|51/61|
|LLVIP|4|61|+0.09700|-0.75282|78.69%|90.16%|98.36%|34/61|
|Drone|1|47|+0.16707|-0.59182|97.87%|95.74%|100.00%|44/47|
|Drone|2|47|+0.10067|-0.49204|89.36%|93.62%|95.74%|31/47|
|Drone|4|47|-0.02310|-0.22990|38.30%|72.34%|89.36%|9/47|

这支持继续检查LLVIP的条件定位目标，不能推出整个LLVIP全对象稳健：基础cohort在a=1/2/4的逐对象最差IoU优势均值为+.01471/−.01745/−.08695，最大CE差均值为−.11324/−.06483/+.05158。Drone基础对象零扰动已是平均IoU/CE质量不利；其固定条件层仍有局部机会，但a=4时47对象的最差IoU优势均值转为−.02310、仅9/47全方向保住quality gate。

## DFL近似、支持域与梯度边界

DFL平移是离散概率质量重采样近似：每个原bin质量移动到相邻整数bin，小于0/大于15的部分另记underflow/overflow，不clamp到边界。T=1与T=2概率分别搬运；用于CE/KL/梯度的是保留质量的条件分布。搬运与温度化不交换，这不是可反推的真实新网络logits。零扰动概率逐元素exact，边界流失为0。

a=4八方向下，固定质量门的逐对象最大T1边界流失（四边均值）再平均为LLVIP1.743%、Drone0.998%；均值小不代表所有边都小。作为透明示例，预定(+4,+4)条件的单边最大流失分别19.65%/25.04%，条件分布解码框偏离直接平移框最大3.778/3.654输入像素。因此不能把损失CE在压力下较好等同于精确坐标已稳健。所有25条件与每对象每边质量保存于原始产物，没有仅挑该示例作为主结果。

同样在(+4,+4)时，LLVIP/Drone门内未clamp教师期望距离仍全在[0,14.99]的对象为59/61、46/47；域外对象保留在主表。RGB GT支持域固定，IR GT支持域与owner上下文单列。分布的边界质量流失和期望距离是否越界是两种不同读数。

KD为KL(pT_shift,T=2 || pN,T=2)×4、四边平均；head梯度为2(pN2−pT2)/4，相比GT梯度(pN1−qGT)/4，逐对象64维cosine/dot/norm都保存。这里只读DFL raw logits局部梯度，没有完整native/CIoU/TAL/共享特征梯度或实际学习更新。两数据集全部对象×25条件的CE/KL/cosine非有限数均为0；不能把cosine正读作训练可学性保证。

## 产物、验证与下一步

- [执行脚本](localization_stress.py)、[冻结协议](PROTOCOL.md)、[报告生成脚本](write_readout.py)。
- [精简逐条件原始表]（服务器/本地中间产物：outputs_attempt2/compact_metrics.csv）、[完整分布统计]（服务器/本地中间产物：outputs_attempt2/all_metrics.csv）、[八方向最差原始表]（服务器/本地中间产物：outputs_attempt2/all_worst_case.csv）。
- 每数据集 `cohort.csv` 保存全部源行和固定层；`per_object.npz` 保存全部基础对象×25条件指标，以及T1/T2逐边underflow/overflow、支持域与gate布尔表；`summary.json` 保存过滤链、模型身份路径、源文件stat及零扰动回验。
- [CPU真值回执]（服务器/本地中间产物：outputs_attempt2/known_truth_tests.json）：11项通过；64个KD与GT梯度均以中心有限差分核验，另含质量守恒/无clamp/零扰动exact/严格.70/固定mask等。
- [保存产物自检]（服务器/本地中间产物：saved_output_self_check.json）及[复核脚本](verify_saved.py)：两数据集零扰动逐对象对旧dfl_per_object.csv的CE/KL/cosine最大差0，全部保存表均值/固定mask/gate存活回算最大差0，源文件size/mtime保持不变。这是执行者自检，不能代替独立接受。
- [完成回执]（服务器/本地中间产物：outputs_attempt2/completion_receipt.json）：`COMPLETED_PENDING_INDEPENDENT_REVIEW`。

建议下一步按既有方案核对LLVIP匹配N、自然训练流的覆盖与同mask GT控制，并保留Drone局部条件机会；不得据本结果放宽.70门或宣布L1通过。此处没有置信区间、独立seed干预、真实配准误差测量或AP/KD增益；LLVIP共享标签也不是物理配准真值。
