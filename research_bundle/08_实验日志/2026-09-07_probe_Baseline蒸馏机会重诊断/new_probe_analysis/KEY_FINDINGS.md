# 新baseline信息诊断：已执行结论（2026-09-07）

**本轮支持“Drone的IR对象机会主要是低置信/未检出，LLVIP有更一致的定位目标线索”；不能支持“特征是更优KD载体”。固定ridge对Drone少数类严重欠拟合，IR特征又未稳定打赢独立RGB特征。**

## 数据与验收

两数据集各固定train1024＋dev200图；Drone共23210行（18866前景、4344背景），LLVIP8288行。所有结果只使用新冻结baseline导出，未访问test。完整数组行数、summary数量与roster图数预检通过。模型训练recipe并非全部相同，teacher/native差不能归因于模态唯一因素。

主分类cohort的dev为Drone3794行（3084前景＋710背景）、LLVIP1442行（643前景＋799背景）。GT-ROI辅助cohort跨全部模型P3/P4共同有效：Drone3725行、LLVIP1440行。固定patch避免GT宽高定义窗口，但GT关联anchor和双模态标注排除背景仍带诊断特权。

冻结分析器通过23真值测试，独立审阅另通过5个反例；正式CPU分析BLAS/OMP线程上限4。冻结参数在查看dev输出前确定。之后仅按root授权增加自然head的既有conf=.25直接读出 sanity，未修改已冻结probe、alpha、阈值或结果。该追加由独立代理以等价logit阈值log(1/3)从真实logits重新生成预测，两数据集、全部模型与两cohort的evaluation_views字典逐字段exact，见[独立复算](../independent_audit/supplemental_readout_verification.json)。

## 对象机会不是AP

|量（dev，全RGB GT作分母）|Drone|LLVIP|
|---|---:|---:|
|RGB GT数|3084|643|
|N正确对象|2386|404|
|N低置信|499|130|
|N无粗候选|103|59|
|N仅类别错误（conf和定位正确）|83|0|
|N仅定位错误（conf和类别正确）|11|50|
|T可修复|461（14.948%）|146（22.706%）|
|若以T替换N造成损伤|193（6.258%）|42（6.532%）|
|修复中N低置信/无候选/仅定位|351 / 73 / 8|90 / 29 / 27|
|共同候选T−N RGB坐标IoU均值|−0.019416（2830对象）|+0.075986（577对象）|

Drone另有2个分类与定位同时错误对象；互斥状态先归低置信，因此JSON保留可重叠错误旗标。这里的正确固定为conf≥.25、类别正确和RGB GT IoU≥.5，T修复还要求GT配对IoU≥.5。损伤是替换机会，包括T未检出，并非蒸馏实际损害。

Drone独立RGB N0同样存在自然互补：修复221、损伤192、净29；N0按相同RGB GT身份比较，无IR配对要求。T比较须IR配对（2946/3084），不可忽略两者有效身份集合差异。该观察支持检验IR内容是否额外有用，不等于训练收益保证。

## 特征与类别载体

下表为相同主cohort、相同每层128维投影的Drone**前景macro recall %**；S为N42，T为IR，N0为独立RGB。所有输入都包含N_logits。

|增加的特征|P3|P4|P3＋P4|
|---|---:|---:|---:|
|S特征|35.786|45.780|50.490|
|T特征|37.112|42.573|49.058|
|N0特征|36.534|46.483|51.226|
|打乱T特征|19.947|19.955|19.947|
|S＋T特征|44.266|52.770|57.050|
|S＋N0特征|45.294|53.540|58.091|

在这个固定读出器内，空间关联特征确实可被解码，打乱内容不保留这些读出收益；但IR没有跨层稳定胜过独立RGB，条件于S特征后也未胜过N0内容。不能将“feature比弱logit ridge好”解释成跨模态独有知识或KD载体优越性。

**读出器适用性检查揭示更强限制。** Drone N_logits ridge前景macro只有19.947%，少数类1–4全部0 recall；同anchor原生N head按既有conf=.25直接读出为60.143%。原生head并非没有可读类别信息，而是固定α=1、未加权one-hot ridge/类别比例在此处严重限制了probe。因此教师logits与高维feature的输入维数、参数数和有效正则不同，不能据该对照排序KD载体。

LLVIP是person/background。N_logits、metadata-only的dev accuracy分别为96.186%、95.354%；后者只用log宽、高、stride，提示固定背景窗/GT样本构造的尺度混杂很强。N_logits＋T patch的P3/P4/P3P4 accuracy为96.394/96.533/97.018%，打乱T为96.047/96.255/96.047%。这些是本抽样上的小幅读出差，缺N0同模态端点且不是实际KD。N_logits＋S patch＋T patch联合为97.226%，同样不能越过采样和双模态评价限制。

## DFL定位目标质量与局部方向

|量（dev，同RGB anchor、已配对、有N粗候选、四边支持[0,14.99]）|Drone|LLVIP|
|---|---:|---:|
|对象数 / 全RGB GT|2821 / 3084|548 / 643|
|N−T GT-DFL CE均值（4边均值）|−0.125607|+0.155898|
|T的GT CE更低的对象比例|39.064%|61.131%|
|KD与GT raw-DFL-logit梯度cosine均值|0.208408|0.406313|
|正cosine对象比例|64.374%|79.197%|

Drone教师DFL在RGB坐标下的平均GT CE更差，和其直接框IoU均值更差一致；LLVIP这两类目标质量线索则同向较好。仍有相当比例对象不支持教师更优。GT CE用T=1，KL用T=2乘T²；梯度仅对native raw DFL logits，正cosine不是共享骨干梯度或训练可学性证据。所有统计不满足/替代冻结L1几何准入，GT中心fallback已另表保存。

## 证据入口与边界

- [Drone完整分析](dronevehicle_analysis_v1/README.md)、[完整JSON](dronevehicle_analysis_v1/summary.json)。
- [LLVIP完整分析](llvip_analysis_v1/README.md)、[完整JSON](llvip_analysis_v1/summary.json)。
- [原生head sanity](direct_head_baseline_v1/README.md)，脚本为direct_head_baseline.py。
- compact_real_findings.json为完整结果抽取，保留上述关键原值；各分析目录有逐行预测、训练标准化参数、逐对象DFL。

这些是固定baseline与抽样的探索诊断，未新训KD、未产生新AP、未证明可迁移到RGB-only模型；对象配对不证明像素配准。推荐后续先解决读出器适用性与匹配同模态控制，再预注册可学性检验，不按这批dev结果调参后回写“成功”。
