# 从当前 baseline 出发的蒸馏方向判断

**2026-09-07 实际完成两数据集各1024 train＋200 dev的统一推理：当前证据支持优先核查Drone前景置信度/排序与LLVIP条件定位；尚不支持“换成局部特征就能获得更大的跨模态收益”。分类/定位是任务，logits/DFL/局部特征是载体，应交叉比较。**

## 1. 这次实际做了什么

Drone使用当前同协议weight0 N42、IR seed42教师和独立RGB N0；LLVIP使用现有visible42/IR42，尚没有新的匹配N三seed。两组共2448图、31498条前景/背景区域记录。所有模型同640方形letterbox、FP32、无随机增强，实际读取检测头输入P3/P4、完整类别logits及同anchor四边DFL。模型训练recipe不完全一致，各自args及端点身份保留；不把此分析当成跨模态因果归因。

特征比较包括固定anchor局部patch与GT窗口辅助诊断；加入独立RGB、错配IR、窗口元数据对照。所有读出固定train拟合/dev评价，没有扫λ、改变正式训练或读取test。两次canary及全量导出共用现有lease，单任务NVML约0.6GiB；首次import失败、后续逐对象GPU同步慢的attempt原样保留，最终attempt3采用CPU后处理。两图重测raw logits/DFL完全一致，特征差<5e-7。

## 2. Drone的主要机会是低置信度，不是大量类别混淆

这是pre-NMS、GT辅助一对一对象关联的描述性计数，conf=.25、命中IoU=.5；不是完整AP或训练收益。

|dev对象机会|Drone|LLVIP|
|---|---:|---:|
|RGB GT数|3084|643|
|RGB当前正确|2386|404|
|教师正确、RGB错误|461（14.95% GT）|146（22.71% GT）|
|上述修复中RGB低置信|351|90|
|上述修复中RGB无粗候选|73|29|
|上述修复中仅定位不足|8|27|
|盲目用教师替代时可损伤的RGB正确对象|193|42|

Drone另有28个纯类别错和1个类别/定位均错被教师修复。表中互斥状态首先识别低置信，其可能同时定位差；独立重叠维度保存在完整JSON。粗候选阈值.05，低于该阈值的“无候选”不能当作网络完全没有响应。N0自然互补为修复221、潜在损伤192，说明RGB独立seed自身也有不少差异。

**重要：低置信候选数不等于AP可提升幅度。** 固定框、类别和分数排序时，单纯把分数整体抬高不会提高AP；必须检查真目标与假目标的排序、重复框和质量分数是否改善。这可以解释“阈值下有许多教师独有命中”与“C0只有+.266655pp”并不矛盾，但尚未完成该解释的因果验证。

## 3. 定位机会在现有LLVIP模型组合中更明确

共同对象各自候选框映射到RGB标签坐标，T−N IoU均值为Drone−.01942（2830对象）、LLVIP+.07599（577对象）。它们是对象关联后的条件统计，不是全数据AP，也不证明物理精确配准。

同anchor、具备N粗候选、配对标签且未clamp距离在native支持域内：

|dev同anchor定位量|Drone|LLVIP|
|---|---:|---:|
|对象数|2821|548|
|教师−N的GT-DFL CE均值（越低越好）|+0.12561|−0.15590|
|教师GT CE更低比例|39.06%|61.13%|
|KD与GT头部梯度cosine为正|64.37%|79.20%|

这支持LLVIP定位优先级，反对Drone全对象直接搬运定位分布；不证明Drone的条件定位不存在价值。共同anchor由N预测与GT几何匹配选出，对N有选择优势。局部DFL梯度不是完整native/共享特征梯度；原L1几何合同仍未准入。

固定anchor特征定位读出也已完成。Drone在同一RGB读出上添加IR特征，平均IoU .75738→.76342；添加RGB N0为.76021、错配IR为.75697。LLVIP相应 .69454→.71282，错配为.69330。但同一固定读出中，N+T DFL为Drone.77158/LLVIP.73245，优于特征路径；原生N同anchor期望框为.84845/.72648。这是GT关联条件下的信息线索，不能叫作单模态蒸馏增益或用来挑方法AP。

## 4. 特征载体尚没有得到足够支持

在N logits及自身P3/P4特征上增加教师特征，Drone前景macro recall为57.05%；增加相同维数独立RGB N0特征为58.09%。P3/P4分开比较也没有稳定IR优势。因此当前数据并不支持将更多特征维度本身作为跨模态贡献。

还发现分析工具自身的限制：固定alpha=1、未加权ridge对Drone少数类欠拟合，N logits前景macro只有19.95%；原生检测头同anchor规则为60.14%。特征优于这个弱ridge基线不能证明logits缺少类别信息。冻结结果保留，另加自然head sanity，不看dev后改alpha重报。

LLVIP二分类probe出现IR局部特征增量，但背景窗口偏容易；仅窗口大小/stride元数据已有95.35%准确率，接近N logits的96.19%。因此该前景/背景结果不能证明真实检测假阳性会下降，也不能据此宣称feature KD优于输出KD。

## 5. 下一步建议（不改正在训练的C1）

1. **先解释AP瓶颈。** 复用完整N/C0的post-NMS预测，做分类、定位、漏检、重复框、背景错误的AP贡献分解；与IR在统一推理协议下的条件修复集合相连。置信阈值计数不能替代排序分析。
2. **Drone优先检查前景质量/排序知识。** 对真实高分假阳性和低分真阳性比较IR与N0提供的增量，并检查当前C0选择是否覆盖这些对象、是否反而抬高背景。先有排序改善证据再设计新的选择/内容形式。
3. **定位优先准备LLVIP的有限验证。** 它目前同时有对象框和DFL目标质量证据；仍要完成匹配N、训练覆盖、任务目标可用性及同掩码GT控制，不能从这份probe自动宣布L1通过或切换长训。
4. **局部特征保留为候选载体。** 只有在真实难例上，额外IR特征超过输出信息和独立RGB，并且与RGB可学习的表示有关，才值得启动对应最小训练。当前不扩成大特征损失或C+L联合方法。

本轮回答的是“哪些方向有更强证据值得验证”，还不能从推理直接给出三种蒸馏的最终AP排名。

## 复核入口

- [完整分类/DFL结果与关键发现](new_probe_analysis/KEY_FINDINGS.md)
- [Drone全表](new_probe_analysis/dronevehicle_analysis_v1/README.md) · [LLVIP全表](new_probe_analysis/llvip_analysis_v1/README.md)
- [定位特征读出](feature_review/localization_readout_v1/README.md)
- [旧证据独立审计](independent_audit/EXPERIMENT_AUDIT.md) · [新分析器独立审阅](independent_audit/new_analyzer_review.md)
- [两图CPU后处理对照](postprocessing_move_check.json)
- [Drone采集回执](dronevehicle_collection_receipt.json) · [LLVIP采集回执](llvip_collection_receipt.json)

错误分解参考[TIDE](https://arxiv.org/abs/2008.08115)，任务读出参考[linear probes](https://arxiv.org/abs/1610.01644)。本次没有运行TIDE官方dAP，没有三seed新方法效果或物理点几何认证；统计范围、模型身份、抽样与读出器局限均不能省略。
