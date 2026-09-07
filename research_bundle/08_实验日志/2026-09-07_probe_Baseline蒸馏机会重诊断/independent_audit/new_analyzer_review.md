# 新 baseline 信息分析器：独立代码与小测试审阅

**接受冻结版本用于CPU描述性分析：23项既有真值测试及5项独立反例全部通过，未发现阻止运行的确定错误。接受范围是导出接口、计算语义和分析代码；不替代真实导出文件完整性与模型身份验收，也不支持KD收益结论。**

日期：2026-09-07。审阅者：`/root/baseline_opportunity_audit`。全程未SSH、运行GPU或计算hash，未修改root导出器或分析器代理文件。

## 被接受版本与实际执行

- [冻结分析器](new_analyzer_review_v2/analyze_baseline_probe.py)：**42833 bytes**，源文件身份与精确mtime_ns见 [snapshot_manifest.json](new_analyzer_review_v2/snapshot_manifest.json) 为准。
- [冻结真值测试](new_analyzer_review_v2/test_analyze_baseline_probe.py)：**12559 bytes**。
- `python -m unittest discover` 在冻结副本执行：**23 passed，0.385s**。完整输出：[independent_unittest_output.txt](new_analyzer_review_v2/independent_unittest_output.txt)。
- [独立反例脚本](independent_contract_checks.py) 另执行 **5 passed**，原值见 [independent_contract_checks.json](new_analyzer_review_v2/independent_contract_checks.json)。
- 导出器静态审阅副本：[export_baseline_information.py](new_analyzer_review_v1/export_baseline_information.py)，13295 bytes。没有执行此GPU导出器。

首轮39040-byte分析器另有21项通过记录，保留为v1，已由本轮v2接受替代。独立补充脚本初次完成全部断言后，因NPZ句柄未关闭导致Windows临时目录清理失败；关闭句柄后第二次通过。此为独立测试harness问题，未修改分析器，失败输出原样保留。

## 检查结论

|范围|结论|核查细节|
|---|---|---|
|单位与分母|pass|对象机会和DFL均以全部RGB GT为分母，排除背景；coverage与selected分母分别命名。分类accuracy与present-class macro recall分别计算，前景/背景分开；差值为×100个百分点|
|背景|pass with qualifier|导出静态代码要求固定窗口在两模态共同内容域内、与两侧所有标注无交集；局部环背景排除两侧GT。它是annotation-background代理，不能当作独立验证的真实负样本|
|对象机会 vs 同anchor|pass|`assigned`与`same_anchor`独立计算；教师框对RGB GT评分，不回填IR-own IoU。真实参考候选、GT中心fallback、背景有分组；不能把GT中心位置称作模型已检测候选|
|类别/分数/定位|pass with qualifier|有互斥状态，也保留重叠旗标和factor repair。互斥状态仍优先low_confidence；JSON明确`*_ok`键的计数是条件失败数。未候选单列。修复/替换损伤是诊断替代，不是训练损伤|
|DFL GT目标|pass|按输入像素中心/stride得到l,t,r,b距离，两bin插值；主诊断四边必须在[0,14.99]，不clamp。数学原语允许精确15，但主native分析排除并将目标设N/A|
|DFL CE/KL/梯度|pass|CE用T=1且4边均值；KL为T=2、乘T²。native raw DFL logits的梯度分别为`softmax(z)-q`与`T(softmax(z/T)-softmax(t/T))`；正/负cosine及零范数分开，有限差分测试通过|
|DFL证据边界|qualified|只是冻结N42头部logit梯度；不是共享骨干梯度、完整检测loss、当前学生梯度或几何准入。配对/合法支持/真实候选与fallback集合分别报告|
|train-only拟合|pass|投影seed固定无标签；列均值/SD仅train拟合；ridge目标为mean squared loss+α‖W‖²，α=1固定、截距不惩罚。没有dev选α/层/臂|
|同维度特征对照|pass|每层固定投影128维，N0/T/shuffled T的同组输入维度相同；两层一致。N42 feature与extra T、extra N0、extra shuffle的组合均已定义|
|GT-ROI共同有效性|pass|跨所有导出模型P3/P4共同valid，缺元数据不默认有效。所有辅助ROI臂和N_logits参考使用同一有效train/val集，并报告排除量/类频；与主fixed-anchor集分开|
|shuffle|pass|ROI只在同split共同有效集合置换，anchor在各split全量置换；P3/P4共享对应置换，两套索引分别保存，fixed points明示|
|载体比较|pass with qualifier|教师logits与feature共9个固定同cohort比较，加入N42特征后仍有配套比较，region logits也有T/N0组合。logit和feature输入维数不同，报告直接承认参数数/有效正则差异，不能叫作KD载体因果效果|
|N0机会|pass|N0与N42使用相同RGB GT，不要求IR配对；输出comparison_model与comparison_requires_ir_gt_pair明示。兼容字段`teacher_correct_on_paired`在该表应按comparison eligibility解释|
|输入元数据|pass within local interface|读取metadata.json，缺失时读取exporter实际summary.json；保留model_identity、源路径/字节和图像数，校验声明的xyxy/input_pixels/reg_max/N42，禁止同一完整image同时出现在train和val|

## 独立反例的具体覆盖

1. 36行合成样本中，将无效ROI行特征改为1e9，所有valid ROI预测保持一致；全部辅助臂仍是23 train/11 val。
2. P3/P4/P3+P4各组T、N0和shuffled T的fixed-anchor输入维数完全相同。
3. 两套shuffle索引都不跨train/val，ROI donor还严格留在common valid集合。
4. 主分类读出的annotation-background、真实参考候选和GT-center-fallback三组人数恰好覆盖12个val样本。
5. 将所有held-out特征任意改为约1e8，训练标准化参数与训练预测不变。

## 审阅期间发现并已解决的问题

- 初版忽略ROI valid，失效ROI中的零region evidence/无背景减除特征会进入拟合；最终版已改为共同valid独立cohort及匹配参考。
- 初版只读取metadata.json，与导出器summary.json接口不符；最终版已读取实际输出并保留模型身份。
- ROI和anchor现在使用不同置换集合，最终版保存两套映射，避免用一套映射追溯两种control。
- 初版error flag键名容易把失败数读成成功数；最终版加了明确语义字段。

## 仍需真实导出验收与解释边界

本轮没有读取正式dev结果或其大数组，模型/checkpoint参数身份、实际1224图及所有row/array对应关系仍需collector/执行方核对。分析器支持缺失的可选metadata以方便合成测试，因此真实结果接受不能只看分析器返回COMPLETED；必须确认collection receipt、summary完成状态、frozen roster、model_identity与各NPZ首维。

固定anchor patch降低GT窗口宽高直接注入，但anchor关联和fallback仍读取GT，背景窗也读取两侧GT；metadata-only控制仅暴露宽高/stride混杂，不能消除所有位置/采样混杂。各图内多对象相关，train拟合对象不是独立训练seed。LLVIP没有N0导出，不能补称同模态完整对照；其单类+背景probe是前景识别，不是多类别识别。

教师特征在dev读出阶段是直接可见的，这是双模态诊断。结果最多支持“该固定线性读出函数族能否解码额外信息”；不能直接叫作RGB-only部署性能、KD增益、方法胜负或已经验证的可学性。native和IR训练recipe不等仍按输入identity保留。
