# RGBIR independent KD v2 实验跟踪表

> **本阶段完成（2026-09-07 21:55）**：两组真实64批及全量独立复算通过，LLVIP定位待认证候选207（62批/187图/14来源组），Drone602（63批/268图/31组）。LLVIP优先定位目标认证/学习验证，Drone优先少数类与来源归因；不是L1准入。原C1按42/0/123为22/21/22轮、旧控制69/59轮、3卡5训练、RSS141.51GiB，无新E200端点。

> **最新证据（2026-09-07 21:29）**：LLVIP完整2406dev旧RGB/IR mAP32.8784/48.8529与独立AP审阅完成；Drone六端点AP/逐类/对象桥接和两数据集定位压力均已独立接受。C1三seed第20轮（CSV19），旧控制66/58，项目RSS141.50GiB、3卡5训练。自然64批选择诊断正在执行准备；无新E200或定位准入。

最新授权执行：LLVIP与Drone均保留，新增证据工作优先LLVIP定位，Drone并行AP错误/排序；见[本阶段](../08_实验日志/2026-09-07_probe_双数据集证据优先推进/README.md)。20:54只读核对C1三seed均第18轮（CSV完成17轮）、旧控制64/55轮，资源合规、无新端点。旧下方快照保留历史意义。

最新查询2026-09-07 20:09：C1按42/0/123的progress.epoch为16/15/16（CSV均完整15轮，分别刷新），旧shuffled42/same-modal42为63/53；五个原CUDA进程及lease均有效，RSS141.58GiB、3卡、每卡不超过2个项目任务。无新端点或失败回执，无新增训练/自动扩展。详见实施日志 `heartbeat_20260907_2008_runtime/README.md`。

Baseline信息诊断已完成2448图及独立复核：[阶段判断](../08_实验日志/2026-09-07_probe_Baseline蒸馏机会重诊断/STAGE_REPORT.md)。Drone优先核查前景置信度/排序，LLVIP定位信息证据更强；特征未确立优先性。这些诊断不等同KD收益、L1准入或新方法冻结，现有训练继续。

2026-09-07 17:06：显式逐类补评接口获独立接受，六旧端点完整接入；C0 harm为REVIEW_REQUIRED且missing为空。16:53 C1三seed第5轮，旧shuffled42/same-modal42第58/41轮，均继续原训练。λ=.09227393550836771和L几何阻塞不变。最新证据见 `../08_实验日志/2026-09-07_train_IndependentKD实施/heartbeat_20260907_1653/README.md`，执行主计划见 `EXPERIMENT_PLAN.md`。

| ID | 阶段 | 实验/交付 | seeds | 优先级 | 当前状态 | 依赖/说明 |
|---|---|---|---|---|---|---|
|B02|诊断|LLVIP优先、Drone对照定位目标扰动|固定20260907|MUST|RUNNING_CPU|0及1/2/4像素25条件，固定mask/anchor，压力诊断不等于真实配准或L1准入|
|B03|诊断|完整post-NMS AP错误/排序|Drone N/C0三seed；LLVIP旧N/T42|MUST|RUNNING|Drone复用六完整缓存；LLVIP补全dev导出先canary再共享lease，保持旧baseline身份|
|B04|准备|LLVIP新协议baseline/定位配置迁移核对|42/0/123|PREPARE|IN_PROGRESS|查单类退化/实际数据流与评估器Drone硬编码，配置未准入不得入队|
|A00|审阅|MD/ZIP一致性、C/L/矩阵三路审阅|—|MUST|DONE|94最终release工程173项+参考包18项测试通过；独立代码复核完成|
|A01|基线|N/C0六个独立端点绑定|0/42/123|MUST|BOUND|14:07 N/C0/random九份独立端点全部接受；A03兼容及实际补评桥接已接受|
|A02|审计|旧native差异/实际worker流/门控敏感性|—|MUST|DONE|已有CPU原始脚本和回执|
|A03|兼容|新N/C0全链路、多batch/update/EMA比较|0/42/123|MUST|ACCEPTED|六轨迹各30完整batch/24成功update严格相等；首个失败与最小修复保留|
|A04|策略|旧CL/CGT扩展延期登记；在跑任务责任核对|—|MUST|DONE|旧random三seed完成；旧串行worker按预定ownership退役；C0内容控制继续|
|C01|实现|selection adapter、C1/C1_y、单任务trainer|—|MUST|CODE_ACCEPTED|单分支接口/映射/有效尺度/梯度观测已实现与独立CPU验收；真实准入另列|
|C02|校准|新train-mode 64batch剂量校准|20260907诊断种子|MUST|CALIBRATED|64/64非零，λ=.09227393550836771，无裁剪；不是两批临时值|
|C03|短测|C1/C1_y真实canary及峰值|42|MUST|ACCEPTED|各24成功update/6AMP跳步；C1单卡双开实际通过，NVML6510MiB|
|C11|长训|C1 vs N/C0|42|MUST|RUNNING_E200|attempt2 GPU2；15:32已有64成功更新；旧启动因解释器导入失败0batch保留|
|C12|长训|C1 vs N/C0|0|MUST|RUNNING_E200|GPU2双开；15:32已有44成功更新；E200后独立评价责任已绑定|
|C13|长训|C1 vs N/C0|123|MUST|RUNNING_E200|GPU5与旧same-modal共享；15:32已有24成功更新；不因其他seed AP停训|
|C20|判读|三seed升级/保留决定|0/42/123|MUST|WAIT_RESULTS|C1−N及C1−C0全正且mean≥.10pp，损伤复核|
|C21|消融|C1_y|42/0/123|CONDITIONAL|NOT_QUEUED|C1值得升级后；3次，拆非目标项，非剂量等价|
|G00|覆盖|固定自然64批源图/增强清单和审核命中上界|—|MUST|DONE|两数据集固定64批；旧清单Drone命中25批，LLVIP仅11批|
|G01|几何|最多24对审核成本试样+20%复核|—|MUST|DONE_UNKNOWN|24对初审+5对独立复核；0个接受点集，不作配准失败估计|
|G02|几何|有限扩审及verified D2|—|CONDITIONAL|BLOCKED_COVERAGE|依可辨认率、64批命中和一天预算；不凑对象放门|
|L01|实现|L-only adapter/L_GT与支持域检查|—|MUST|IMPLEMENTED|L-only/GT同mask与支持域测试完成；真实路径受几何阻塞|
|L02|校准|L1/L_GT 64batch真实梯度与剂量|20260907诊断种子|CONDITIONAL|BLOCKED_GEOMETRY|≥16非零batch、≥16唯一图、≥2来源；L_GT同λ|
|L03|短测|L1/L_GT真实canary|42|CONDITIONAL|BLOCKED_GEOMETRY|各24成功update；无合成替代|
|L11|长训|L1 vs N|42|CONDITIONAL|NOT_QUEUED|L02/L03通过；与L_GT42同批|
|L12|长训|L_GT早期内容控制|42|CONDITIONAL|NOT_QUEUED|与L1相同E/anchor/mask/λ；不用于改门或停其它L1 seed|
|L13|长训|L1 vs N|0|CONDITIONAL|NOT_QUEUED|L准入后收齐三seed|
|L14|长训|L1 vs N|123|CONDITIONAL|NOT_QUEUED|同上|
|L20|判读|L1独立效用|0/42/123|CONDITIONAL|WAIT_ADMISSION_NOT_STARTED|定位训练尚未准入；未来判断仍为全正mean≥.10pp且定位诊断相容|
|L21|控制|L_GT补齐|0/123|CONDITIONAL|NOT_QUEUED|L1有效后2次，完成教师内容判断|
|P00|分析器|新arm/错误分类/配对统计独立验收|—|MUST|POSTHOC_CLASS_ADAPTER_ACCEPTED|独立17测试及真实六端点重跑通过，每端点五类；C0 harm无缺失且REVIEW_REQUIRED，30份旧关键文件不变。仅接受接口，CLI保持非自动授权；未来C1实际评价源码仍须核验|
|P01|归因|最终类别方法四臂|0/42/123|MUST_IF_CLAIMED|WAIT_WINNER_C0_HARM_REVIEW|C0背景误检三seed同增，其后续自动扩展需专项复核；C1按原门槛，已有控制继续|
|P02|归因|L1自己的四臂|0/42/123|MUST_IF_CLAIMED|WAIT_L_EVIDENCE|额外6次；L_GT不能替代same-modal|
|P03|证据|GitHub阶段同步|—|MUST|PUBLISHED|本阶段证据提交00bfb59已推送并核对远端：589文件、518份raw在Git blob层字节核对，2个大派生数组仅记录路径；根目录DUAL_DATASET_EVIDENCE_INCREMENT_20260907.json及PUBLICATION_DUAL_EVIDENCE_20260907.json为复核入口。此前6ed99b2为上一baseline诊断阶段，原始证据保留|
|B01|诊断|当前baseline分类/定位/局部特征中间结果|诊断20260907|USER_REQUEST|COMPLETED|两数据集各1024train+200dev；固定读出、独立RGB/错配及原生head复核完成；不是新KD AP或训练准入|
|B02|诊断|两数据集完整dev AP与少数类桥接|旧N/C0三seed+LLVIP旧42|USER_REQUEST|ACCEPTED_DESCRIPTIVE|Drone1469dev六端点、LLVIP2406dev两模型；官方TIDE/独立COCO及真值通过，非KD增益|
|B03|诊断|定位固定25条件压力|固定train/dev缓存|USER_REQUEST|ACCEPTED_DESCRIPTIVE|两数据集attempt2独立全量复算通过；旧汇总计数错误保留，非物理配准证明|
|B04|诊断|真实自然64批C/L选择|20260907|USER_REQUEST|ACCEPTED_UNVERIFIED_GEOMETRY|各2048图、原流exact、L207/602；有候选批62/63非非零梯度批；未启动新定位训练|
|X01|外部|BCKD/FGD/LD协议适配和历史CMD/CCLKD审计|—|PREPARE_ONLY|PREPARED|BCKD分类partial、FGD、LD公开资产及冻结适配草案齐；旧CMD/CCLKD workers8独立列|
|X02|数据集|LLVIP独立N/L1/L_GT或类别泛化|—|CONDITIONAL|INFORMATION_DIAGNOSIS_DONE_ADMISSION_BLOCKED|baseline定位信息诊断完成；L1几何/64批校准/canary未通过，新协议N矩阵未启动；v2.2已有条件授权，不能以probe替代准入|
|D01|延期|旧CL/CGT/dynamic router|—|DEFERRED|DEFERRED_BY_INDEPENDENT_CL_V2|不融合、不以联合结果救单支|
|D02|延期|ROI特征/dense C2/关系频率组合|—|DEFERRED|FEATURE_TRAINING_DEFERRED|局部特征信息诊断已完成，未证明优于输出或独立RGB；新feature训练方法仍需另行冻结|
|D03|暂停|OS-SSL、VEDAI|—|PAUSED|PAUSED|维持用户既定方向|

核心预算：最初C1×3；L准入后再L1×3+L_GT42，共7；随后C1_y×3+L_GT余2，共最多12。胜出分支四臂额外计算，两条都闭合最高24；不是一次提交的队列许可。LLVIP替代时明确新增N三seed：核心上限15、完整两线归因27；均为条件预算，不一次入队。
