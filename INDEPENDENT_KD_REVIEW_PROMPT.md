# 请独立复核：RGB–IR 独立分类与定位蒸馏

先读 [实施README](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/README.md)、[冻结计划](research_bundle/refine-logs/EXPERIMENT_PLAN.md) 和用户提供的 [Formal Spec](research_bundle/07_研究分析/RGBIR_Independent_Class_Loc_Formal_Spec_20260907.md)。执行身份已从 C+L 联合训练切换为单分支 C1/C1_y/L1/L_GT。C0 为已有 OEv1；历史 native 不是新实验的正式分母。

请优先回答以下问题，并把已证实错误、待验证假设、资源限制分开：

1. [分类实现](research_bundle/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/classification_logit.py) 是否严格对应原C0同选择、同区域、P3/P4有效尺度、教师±16截断、T=2、Bernoulli KL(T‖S)及基础对象分母？C1是否只替换C0而非叠加？非目标类平均的η=.25与单类情况是否正确？
2. 检查[真实兼容验证](research_bundle/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/verify_compatibility.py)、[梯度校准](research_bundle/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/calibrate_independent.py)及已产生的失败/接受回执。6条真实轨迹、64自然批、固定R副本/参数集合和24成功update不能由CPU合成测试替代；关注首次导入RNG修复是否保留正式训练随机流。
3. [旧结果](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/OLD_RESULTS_1407.md)支持何种有限主张？C0−N、C0−random小幅同向并不构成完整跨模态四臂归因。注意汇总recall下降；缺失seed、未完成评价或中途权重不得升级为完整结果。
4. [几何审核](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/PRIMARY_GEOMETRY_VISUAL_24.md)现均UNKNOWN。评估是否存在可验证的物理点位/覆盖证据，而非用GT边或预测框制造真值。定位代码、对象机会、真正可施加的DFL监督机会是不同证据；本轮不得放宽0.70、换自然批或静默改box loss补救。
5. [对象分析器](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/object_error_analysis.py)的修复/损伤分母、背景误检与各组定义是否可解释？这是固定conf=.25/IoU=.5诊断，不是COCO AP匹配或真实昼夜标签。
6. [外部基线准备](research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/EXTERNAL_BASELINE_PREPARATION.md)哪些是作者复现、哪些是协议迁移或partial？BCKD/FGD/LD还没有本轮训练结果，不能按文献算子名字宣称完整复现。

请给出能指向具体代码或原始产物的反馈。不因当前AP小、缺精确外部材料或某一定位证据线阻塞而要求全项目停机；不把未验证几何接受或未来实验结果写成事实。不要建议看完AP后调λ或阈值。最终晋级门、三seed四臂与条件预算见冻结计划。
