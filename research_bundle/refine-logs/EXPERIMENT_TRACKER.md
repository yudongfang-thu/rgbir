# RGBIR independent KD v2 实验跟踪表

2026-09-07 17:06：显式逐类补评接口获独立接受，六旧端点完整接入；C0 harm为REVIEW_REQUIRED且missing为空。16:53 C1三seed第5轮，旧shuffled42/same-modal42第58/41轮，均继续原训练。λ=.09227393550836771和L几何阻塞不变。最新证据见 `../08_实验日志/2026-09-07_train_IndependentKD实施/heartbeat_20260907_1653/README.md`，执行主计划见 `EXPERIMENT_PLAN.md`。

| ID | 阶段 | 实验/交付 | seeds | 优先级 | 当前状态 | 依赖/说明 |
|---|---|---|---|---|---|---|
|A00|审阅|MD/ZIP一致性、C/L/矩阵三路审阅|—|MUST|DONE|94最终release工程173项+参考包18项测试通过；独立代码复核完成|
|A01|基线|N/C0六个独立端点绑定|0/42/123|MUST|BOUND|14:07 N/C0/random九份独立端点全部接受；新wrapper复用仍需A03|
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
|L20|判读|L1独立效用|0/42/123|CONDITIONAL|WAIT_RESULTS|全正mean≥.10pp且定位诊断相容|
|L21|控制|L_GT补齐|0/123|CONDITIONAL|NOT_QUEUED|L1有效后2次，完成教师内容判断|
|P00|分析器|新arm/错误分类/配对统计独立验收|—|MUST|POSTHOC_CLASS_ADAPTER_ACCEPTED|独立17测试及真实六端点重跑通过，每端点五类；C0 harm无缺失且REVIEW_REQUIRED，30份旧关键文件不变。仅接受接口，CLI保持非自动授权；未来C1实际评价源码仍须核验|
|P01|归因|最终类别方法四臂|0/42/123|MUST_IF_CLAIMED|WAIT_WINNER_C0_HARM_REVIEW|C0背景误检三seed同增，其后续自动扩展需专项复核；C1按原门槛，已有控制继续|
|P02|归因|L1自己的四臂|0/42/123|MUST_IF_CLAIMED|WAIT_L_EVIDENCE|额外6次；L_GT不能替代same-modal|
|P03|证据|GitHub阶段同步|—|MUST|PUBLISHED|c7bbd6d补充提交已推送原分支并核对远端；7423文件检查通过，含六旧端点补评、观察桥接及对象损伤诊断。C1完整端点尚待训练结束；见实施日志PUBLISH_RECEIPT_c7bbd6d.md|
|X01|外部|BCKD/FGD/LD协议适配和历史CMD/CCLKD审计|—|PREPARE_ONLY|PREPARED|BCKD分类partial、FGD、LD公开资产及冻结适配草案齐；旧CMD/CCLKD workers8独立列|
|X02|数据集|LLVIP独立N/L1/L_GT或类别泛化|—|CONDITIONAL|DIAGNOSIS_FIRST|新数据集N/预算另列；不借Drone端点|
|D01|延期|旧CL/CGT/dynamic router|—|DEFERRED|DEFERRED_BY_INDEPENDENT_CL_V2|不融合、不以联合结果救单支|
|D02|延期|ROI特征/dense C2/关系频率组合|—|DEFERRED|DEFERRED_PAYLOAD_STUDY|先解决C1与C1_y；新方法另冻结|
|D03|暂停|OS-SSL、VEDAI|—|PAUSED|PAUSED|维持用户既定方向|

核心预算：最初C1×3；L准入后再L1×3+L_GT42，共7；随后C1_y×3+L_GT余2，共最多12。胜出分支四臂额外计算，两条都闭合最高24；不是一次提交的队列许可。LLVIP替代时明确新增N三seed：核心上限15、完整两线归因27；均为条件预算，不一次入队。
