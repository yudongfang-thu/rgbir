# 快筛之后做什么（未读取本轮新 AP）

**当前最有训练证据的方向仍是 Drone 的对象级 C0；下一项优先查它是否依赖正确配对内容，不立即扩大 logits、定位或特征载体。** 原匹配 E200 中 C0−N 为 +0.266655±0.144373 pp、3/3 seed 正向，C0−random 为 +0.174939±0.038853 pp；全量 E8 的 N/C0/C1 为43.376542/43.920584/43.408107。它们支持受限开发协议下的小幅 C0 信号，尚无齐备内容归因，不能直接称跨模态收益。来源：[九端点审计](../2026-09-08_audit_项目与94最新全景/README.md)、[E8原结果](../2026-09-08_train_分类快速反馈E8/README.md)。本轮新结果只由已冻结分析器读取，本文件不预填正负。

## 若本轮 C0−N 正向

只说明**一条已知对照在这个2048子集、seed42、E8中方向保留**；不能因此用该筛选器对 L3/F/C1 作一般排名。建议下一个短实验仅增加 **C0-shuffled 一臂**，以当前 N/C0 为已有参照；不重跑这两臂，不另开三类方法矩阵。若原 E200 内容控制在实施前已有接受端点，先读那个结果，已有答案则不重复本短控制。

实施基础是已接受的[内容控制定义](../2026-09-07_train_OEv1内容归因/README.md)，具体模块为 `03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/{content_controls.py,shuffled_pair_data.py}`（94 对应 `artifacts/rgbir_task_conditional_v1_20260907/release_v8/`，实施时核对应版本）。在**新 scope/新 wrapper**中保持当前学生初始化、2048名单、8轮512批、正常BN、SGD日程、λ=.1和原C0。只从这2048个train配对内按已有 seed20260907/形状分层无自配规则生成一次 donor；保持正确 paired IR 的对象、区域、E/q/K及分母，仅将最终内容目标换为错配IR前向。不能直接打乱主IR loader导致选择一起变化，不能使用dev donor。

先按原规则做有上限canary，核共同初始五类state、RGB/正确paired IR前30批全像素双标签，以及 paired 选择链 exact；donor单列。增加一次固定末轮 full-dev 评价，原控制的复用须明确配置/数据/初始state/流投影。**建议本新增臂含canary/eval执行上限20分钟**，以实际成本准入，不改日程凑预算。若 paired C0 比 shuffled 更好，仅支持正确实例/空间内容的依赖线索；shuffled同时破坏实例和空间，并非完整因果隔离，仍缺同模态及多seed归因。若没有优势，不改错排规则或系数求差异，暂不把C0小正差写成跨模态内容收益。

## 若本轮 C0−N 非正向或技术不完整

**结束这个廉价筛选版本；不按它淘汰分类、定位或特征，也不继续换seed、子集、LR或轮数直到阳性。** 非正向说明当前2048重复曝光/短预算未保留既有方向，不能识别是样本分布、优化时程还是随机波动；技术失败更不是科学负结果。

下一步不新增 GPU 训练：由原责任队列收齐既定 C0-shuffled/same-modal、C1端点；本条目仅用已有训练记录和完整dev回执并排报告 N/C0 的学习进度、五类原值及与全量E8的曝光差，不额外前向或再做教师机会普查。**可继续的主线是已有 C0 与其归因；当前没有证据足以推荐另一个新损失替代它。** 若仍要建立另一廉价筛选协议，应另立一次前瞻性校验，不能把本次失败后调出的方案当作原校验通过。

## 当前不优先的三个“看起来有空间”的方向

- **阈值置信度 ≠ AP收益。** Drone的1873个0.25/0.5“仅IR正确”中1441在低阈值下双方已有框；统一抬分可能改变工作阈值召回，却未必改善排序，还需误检代价。因此不因这1873个机会直接加大置信度剂量。
- **定位机会 ≠ 现成可学目标。** Drone高标签一致性的对象确有RGB坐标优势，但教师 own-GT 更准不能直接推出对RGB更准。LLVIP L2目标120边中110同向于GT，L3完整分布FT3仍未胜N/GT（−.010679/−.043234 pp），且三臂均低于成熟起点；本版已结束，不重复放.70门、增λ或再做相同DFL读出。它也没有否定从其他学习阶段做定位的可能性。
- **类别/特征扩展缺少优先训练证据。** car选中88.67%与少数类宏分类oracle贡献94.26%不是梯度份额；已做C2归一未胜C1。F-rel-GM仅比N/C1高.037257/.011926 pp且召回下降；线性probe受读出器与独立RGB控制限制，尚未证明特征带宽收益。不要凭这些计数或非零梯度再开大载体矩阵。

证据：[完整Drone与cross-GT读出](../2026-09-08_probe_Drone教师完整推理/FINAL_REPORT.md)、[目标审计](../2026-09-08_probe_定位学习位置与目标/README.md)、[九次成熟FT3](../2026-09-08_probe_快速方向筛选/FINAL_REPORT.md)、[L3终点](../2026-09-08_probe_对象坐标分布目标/FINAL_REPORT.md)、[综合数据分析](../../07_研究分析/RGBIR数据分析综合报告_20260908.md)。

判断记录：`assessment_status=assessed`；C0受限训练信号 `claim_supported=partial`，一般廉价排序能力 `claim_supported=no`（尚未建立，不是反证）；`confidence=medium`。沿用[既有审计](../2026-09-08_audit_项目与94最新全景/EXPERIMENT_AUDIT.json)的 `integrity_status=warn`：已完成端点可追溯，内容归因/运行中任务未齐；它不覆盖本轮尚未读取的新结果。本页为只读综合建议，不授权或启动新实验；未改源、复算旧结果或新增hash。

## 结果到齐后的实际选择与下一工作

以上两个分支写于新AP之前，原文保留。随后冻结分析器给出 **N/C0 mAP=27.588046/26.459942，C0−N=−1.128105 pp**；实际队列约9分42秒墙钟、9分31秒执行。来源：[本轮接受读出](analysis/output_attempt1/summary.json)。执行**非正向分支**：本版固定子集筛选器结束，不重抽名单、换seed/日程求阳性，不用它筛退C0，也不追加shuffled来修复本次校验。

下一项是**一次固定8批、零更新的按类别共享参数梯度测量**，不是空等旧E200，也不是重做教师输出普查。先完成的只读字段核对如下；同名文件的历史/加速副本不当作独立样本。

|已读取的真实记录|已经回答|仍缺什么|
|---|---|---|
|`2026-09-07_train_IndependentKD实施/remote_admission_1532/formal_C1_gpu5_attempt2/runs/C1_seed42/shared_gradient_observations.jsonl`，本地1行epoch0；加速`evidence_20260908_162401/profile_attempt1/original/`同类文件|P3/P4共享参数整体KD/native，以及目标项/非目标项的范数与夹角|没有按对象GT类别分解；目标/非目标不是car/freight/truck/bus/van|
|同实施目录`remote_admission_1532/c1_calibration64_attempt1/calibration_batches.jsonl`，64行|整体C0/C1、target/off-target范数/余弦，逐类对象计数|没有逐类共享参数梯度、类间Gram或带符号贡献|
|`2026-09-08_probe_快速方向筛选/results_1203_snapshot/calibration/drone/calibration_batches.jsonl`，8行|C1/C2/F各自整体共享梯度范数及native cosine|没有各GT类的梯度方向，也没有各类native分类梯度|
|`2026-09-08_ops_训练吞吐诊断/remote_short_profile_attempt1/profile_C0/gradient_checks.jsonl`，1行|整体raw-score梯度与原生零权重恒等检查|不是共享参数梯度，不能补出类别分解|

**因此不能从已有JSON用CPU还原car的实际梯度份额，更不能据之判定少数类被损伤。** 聚合范数和夹角无法唯一反演被加总的类别向量；特征/DFL/检测输出缓存也不含回到共享参数的Jacobian。当前CPU可做的是上表字段与来源核对，不伪造缺失的梯度。

具体实施任务固定如下（新独立probe，由root统一调度；本次未启动）：

1. 直接复用本轮N canary服务器私有 `flow_reference/{initial_student.pt,batch_01.pt…batch_08.pt}`；原始远端路径见[已完成N的初始化](evidence_final/canaries/N/initialization_check.json)和[完整前缀回执](evidence_final/canaries/N/flow_check.json)。这是真实增强后uint8双模态及双标签，不重新抽图、增强或读取dev。T/R仍当前配置的IR42/RGB42。每批恢复共同五类初始参数和buffers，用原train-mode/AMP；不做optimizer/scaler/EMA更新，最终恢复state。它只观察这个固定初始状态，不能反演后续512批的历史梯度。
2. 复用本条目 `release/train_short_screen.py:private_builder`、`screen_common.py:load_tensors/validate_amp_prior`，以及 pinned `runtime.to_device`、`selection_adapter`、`classification_logit._classification_terms`、`gradient_observation.shared_parameter_set`。沿已接受固定8批校准的共享P3/P4参数集合，记录全部参数名；采用已修复的逐批图生命周期清理，不直接照搬旧calibrator遗留的graph引用。S/T/R每批各前向一次，原C0和原C1读取同一raw；λ仍.1/.09227393550836771，不校准、不修改门。
3. 按**对象GT类别**拆解原KD标量，保留全base分母、实际B、原selected/有效尺度和原target/off-target权重，禁止把某类对象另喂selector或改为类内均值。记录五个 `g_c=∇Θ(Bλ·KD_c)`，及其总和对原整体KD梯度的数值闭合；C1可另保留target/off-target子分解，但不混同GT类别。Θ仅为上述共享参数集合，不冒称整个网络。
4. 输出每类对象支持量、`||g_c||`、`||g_c||/Σ||g_d||`、类间dot/Gram、`dot(g_c,g_all)/||g_all||²`与global native cosine。前一种是范数占比，后一种是可负的有符号投影；向量可抵消，不能把范数占比当更新比例。若要判断少数类局部冲突，必须另截获**同一次原生assigner**的实际target scores，将原native分类BCE按五个输出通道拆出`h_d`，含原负anchor/权重/分母，先核五项标量之和等于原native分类项，再算`dot/cos(g_c,h_d)`。只和整体native夹角不足以判少数类损伤；任一类无支持或零范数写null，不补抽样。
5. 固定8批、零AP/零更新，只保存小范数/Gram/身份与有限性回执，不下载图像或大梯度张量；一次2批资源canary通过后，完整测量执行上限10分钟、原sole lease。结果用于决定是否值得另立“减少某类冲突项”的具体候选；局部负cos仍不是已经造成AP损伤，缺少支持则明确停在缺证，不自动重加权或开新矩阵。

这个工作填补的是**类别对象计数→实际学习方向**之间尚未测过的空白；不重复已经结束的ROI/CKA、同anchor DFL、L2门覆盖或完整dev机会普查，也不再依赖本版失败的快筛来证明新损失优劣。
