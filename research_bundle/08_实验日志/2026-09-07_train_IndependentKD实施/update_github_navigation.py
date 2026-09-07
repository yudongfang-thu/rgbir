"""Update derived GitHub navigation; archive the earlier 03:22 entry unchanged."""
from pathlib import Path

repo=Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
archive=repo/'ARCHIVE_README_20260907_0322.md'
if not archive.exists():archive.write_bytes((repo/'README.md').read_bytes())
log='research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施'
code='research_bundle/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
(repo/'README.md').write_text(f'''# RGB–IR 独立分类与定位蒸馏：证据与复核入口

**最新执行已切换为 C1 分类三 seed 优先、L1 定位条件准入。请先读 [实施状态与实际证据]({log}/README.md)，再读 [本轮复核请求](INDEPENDENT_KD_REVIEW_PROMPT.md)。**

旧 N/C0/random 九份 E200 last/EMA 独立端点已齐。C0−N 为 **+0.266655±0.144373 pp**，C0−random 为 **+0.174939±0.038853 pp**，均三 seed 同向。旧 C0 shuffled42/same-modal42 继续；四臂尚未齐，不能把这两个配对差当成已完成跨模态归因。

**C1 seed42/0/123已正式启动，λ=0.09227393550836771**；六条真实兼容、固定64批校准、两路径24-update canary及单卡双开实测均通过。新独立模块保留原生检测损失，C1用原C0选择与区域，传递P3/P4全类别相对raw logits。当前尚无新C1 AP，旧CL/CGT排程已被替代。

- [旧三 seed 原值、配对差及限制]({log}/OLD_RESULTS_1407.md)
- [冻结执行计划](research_bundle/refine-logs/EXPERIMENT_PLAN.md) · [追踪表](research_bundle/refine-logs/EXPERIMENT_TRACKER.md)
- [独立模块：源码、测试、训练与评价入口]({code}/)
- [首批24对几何诊断及 UNKNOWN 原因]({log}/PRIMARY_GEOMETRY_VISUAL_24.md)
- [实际失败、最小修复与资源证据]({log}/GPU_ADMISSION_FIRST_ATTEMPTS.md)
- [对象修复/损伤规则]({log}/OBJECT_ERROR_ANALYSIS_RULES.md) · [外部基线准备]({log}/EXTERNAL_BASELINE_PREPARATION.md)
- [所有实验日志](research_bundle/08_实验日志/README.md) · [03:22 历史入口](ARCHIVE_README_20260907_0322.md)

L1 当前缺合格物理对应点证据，未进入新长训；0个接受点集不等于0个潜在定位机会或数据集未配准。两个数据集没有换64批、过采样或放宽门控。封存 test 不参与方法选择。归因、内容控制和 LLVIP 增量均分阶段触发，不一次性提交全部预算。

仓库保留正负结果、失败attempt和勘误。小体积原始数值与源码保持字节一致；Markdown链接适配GitHub。导出清单为根目录按时间命名的 `INDEPENDENT_KD_BUNDLE_MANIFEST_*.json`。不含凭据、大权重或数据集原图全集。
''',encoding='utf-8')
(repo/'LATEST_RESULTS.md').write_text(f'''# 最新结果与执行状态

当前请读 [Independent KD 实施证据]({log}/README.md) 与 [旧九份端点收口]({log}/OLD_RESULTS_1407.md)。

N/C0/random 三 seed 完整配对：C0−N **+0.266655±0.144373 pp**，C0−random **+0.174939±0.038853 pp**。C0 汇总 recall 相对 N 三 seed 同降，需固定阈值对象诊断，不声称负迁移已消除。

C1三seed已正式启动，λ=.09227393550836771；L1因几何证据不足尚未准入。不得将旧C0数字或两批资源探针当作新方法结果。实际状态见[正式启动验收]({log}/FORMAL_LAUNCH_ACCEPTANCE_1535.md)。

[旧 Task-Conditional 状态](TASK_CONDITIONAL_STATUS_20260907.md) 是历史快照，已被当前独立分支排程替代。
''',encoding='utf-8')
(repo/'INDEPENDENT_KD_REVIEW_PROMPT.md').write_text(f'''# 请独立复核：RGB–IR 独立分类与定位蒸馏

先读 [实施README]({log}/README.md)、[冻结计划](research_bundle/refine-logs/EXPERIMENT_PLAN.md) 和用户提供的 [Formal Spec](research_bundle/07_研究分析/RGBIR_Independent_Class_Loc_Formal_Spec_20260907.md)。执行身份已从 C+L 联合训练切换为单分支 C1/C1_y/L1/L_GT。C0 为已有 OEv1；历史 native 不是新实验的正式分母。

请优先回答以下问题，并把已证实错误、待验证假设、资源限制分开：

1. [分类实现]({code}/classification_logit.py) 是否严格对应原C0同选择、同区域、P3/P4有效尺度、教师±16截断、T=2、Bernoulli KL(T‖S)及基础对象分母？C1是否只替换C0而非叠加？非目标类平均的η=.25与单类情况是否正确？
2. 检查[真实兼容验证]({code}/verify_compatibility.py)、[梯度校准]({code}/calibrate_independent.py)及已产生的失败/接受回执。6条真实轨迹、64自然批、固定R副本/参数集合和24成功update不能由CPU合成测试替代；关注首次导入RNG修复是否保留正式训练随机流。
3. [旧结果]({log}/OLD_RESULTS_1407.md)支持何种有限主张？C0−N、C0−random小幅同向并不构成完整跨模态四臂归因。注意汇总recall下降；缺失seed、未完成评价或中途权重不得升级为完整结果。
4. [几何审核]({log}/PRIMARY_GEOMETRY_VISUAL_24.md)现均UNKNOWN。评估是否存在可验证的物理点位/覆盖证据，而非用GT边或预测框制造真值。定位代码、对象机会、真正可施加的DFL监督机会是不同证据；本轮不得放宽0.70、换自然批或静默改box loss补救。
5. [对象分析器]({log}/object_error_analysis.py)的修复/损伤分母、背景误检与各组定义是否可解释？这是固定conf=.25/IoU=.5诊断，不是COCO AP匹配或真实昼夜标签。
6. [外部基线准备]({log}/EXTERNAL_BASELINE_PREPARATION.md)哪些是作者复现、哪些是协议迁移或partial？BCKD/FGD/LD还没有本轮训练结果，不能按文献算子名字宣称完整复现。

请给出能指向具体代码或原始产物的反馈。不因当前AP小、缺精确外部材料或某一定位证据线阻塞而要求全项目停机；不把未验证几何接受或未来实验结果写成事实。不要建议看完AP后调λ或阈值。最终晋级门、三seed四臂与条件预算见冻结计划。
''',encoding='utf-8')
print('Derived review/navigation pages updated.')
