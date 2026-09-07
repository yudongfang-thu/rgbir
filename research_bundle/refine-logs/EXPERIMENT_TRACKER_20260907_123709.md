# RGBIR independent KD v2 实验跟踪表

2026-09-07；当前为计划与材料审阅状态，未启动本表中的新训练。执行主计划见 `EXPERIMENT_PLAN.md`。

| ID | 阶段 | 实验/交付 | seeds | 优先级 | 当前状态 | 依赖/说明 |
|---|---|---|---|---|---|---|
|A00|审阅|MD/ZIP一致性、C/L/矩阵三路审阅|—|MUST|DONE|8文件、18提供方测试回执；本地未重跑pytest|
|A01|基线|N/C0六个独立端点绑定|0/42/123|MUST|EVIDENCE_AVAILABLE|11:19原始快照和accepted v2表；新wrapper复用尚需A03|
|A02|审计|旧native差异/实际worker流/门控敏感性|—|MUST|DONE|已有CPU原始脚本和回执|
|A03|兼容|新N/C0全链路、多batch/update/EMA比较|0/42/123|MUST|TODO|保持正式loader；24成功update，30batch流；标准native同recipe另核验|
|A04|策略|旧CL/CGT扩展延期登记；在跑任务责任核对|—|MUST|PLANNED|不改/终止原C0控制，核实训练和独立评估后再释放lease|
|C01|实现|selection adapter、C1/C1_y、单任务trainer|—|MUST|TODO|注意matched→E_C映射；C1非dense；不含局部特征|
|C02|校准|新train-mode 64batch剂量校准|20260907诊断种子|MUST|TODO|BN逐批恢复、至少16有效批；C1_y同λ实际剂量另报|
|C03|短测|C1/C1_y真实canary及峰值|42|MUST|TODO|各24成功update；共用lease|
|C11|长训|C1 vs N/C0|42|MUST|NOT_LAUNCHED|A03/C02/C03通过|
|C12|长训|C1 vs N/C0|0|MUST|NOT_LAUNCHED|技术通过后依序入队，不按C11 AP取消|
|C13|长训|C1 vs N/C0|123|MUST|NOT_LAUNCHED|同上|
|C20|判读|三seed升级/保留决定|0/42/123|MUST|WAIT_RESULTS|C1−N及C1−C0全正且mean≥.10pp，损伤复核|
|C21|消融|C1_y|42/0/123|CONDITIONAL|NOT_QUEUED|C1值得升级后；3次，拆非目标项，非剂量等价|
|G00|覆盖|固定自然64批源图/增强清单和审核命中上界|—|MUST|TODO|先于扩展人工点审；原roster不能直接授权folder|
|G01|几何|最多24对审核成本试样+20%复核|—|MUST|TODO|保留全图象限合同并补目标邻域；图/两侧边界支持定义冻结|
|G02|几何|有限扩审及verified D2|—|CONDITIONAL|BLOCKED_COVERAGE|依可辨认率、64批命中和一天预算；不凑对象放门|
|L01|实现|L-only adapter/L_GT与支持域检查|—|MUST|TODO|复用DFL核；显式关闭类别KD；strict07|
|L02|校准|L1/L_GT 64batch真实梯度与剂量|20260907诊断种子|CONDITIONAL|BLOCKED_GEOMETRY|≥16非零batch、≥16唯一图、≥2来源；L_GT同λ|
|L03|短测|L1/L_GT真实canary|42|CONDITIONAL|BLOCKED_GEOMETRY|各24成功update；无合成替代|
|L11|长训|L1 vs N|42|CONDITIONAL|NOT_QUEUED|L02/L03通过；与L_GT42同批|
|L12|长训|L_GT早期内容控制|42|CONDITIONAL|NOT_QUEUED|与L1相同E/anchor/mask/λ；不用于改门或停其它L1 seed|
|L13|长训|L1 vs N|0|CONDITIONAL|NOT_QUEUED|L准入后收齐三seed|
|L14|长训|L1 vs N|123|CONDITIONAL|NOT_QUEUED|同上|
|L20|判读|L1独立效用|0/42/123|CONDITIONAL|WAIT_RESULTS|全正mean≥.10pp且定位诊断相容|
|L21|控制|L_GT补齐|0/123|CONDITIONAL|NOT_QUEUED|L1有效后2次，完成教师内容判断|
|P00|分析器|新arm/错误分类/配对统计独立验收|—|MUST|TODO|旧accepted不自动覆盖新增分析器|
|P01|归因|最终类别方法四臂|0/42/123|MUST_IF_CLAIMED|WAIT_WINNER|C0胜出补4次；C1胜出新增6次|
|P02|归因|L1自己的四臂|0/42/123|MUST_IF_CLAIMED|WAIT_L_EVIDENCE|额外6次；L_GT不能替代same-modal|
|P03|证据|GitHub阶段同步|—|MUST|PLANNED|沿用已有授权和分支；本次计划尚未发布|
|X01|外部|BCKD/FGD/LD协议适配和历史CMD/CCLKD审计|—|PREPARE_ONLY|TODO|最多三家族；GPU复现另排，不与首批争资源|
|X02|数据集|LLVIP独立N/L1/L_GT或类别泛化|—|CONDITIONAL|DIAGNOSIS_FIRST|新数据集N/预算另列；不借Drone端点|
|D01|延期|旧CL/CGT/dynamic router|—|DEFERRED|DEFERRED_BY_INDEPENDENT_CL_V2|不融合、不以联合结果救单支|
|D02|延期|ROI特征/dense C2/关系频率组合|—|DEFERRED|DEFERRED_PAYLOAD_STUDY|先解决C1与C1_y；新方法另冻结|
|D03|暂停|OS-SSL、VEDAI|—|PAUSED|PAUSED|维持用户既定方向|

核心预算：最初C1×3；L准入后再L1×3+L_GT42，共7；随后C1_y×3+L_GT余2，共最多12。胜出分支四臂额外计算，两条都闭合最高24；不是一次提交的队列许可。旧N/C0复用失败或换LLVIP时的新N另列。
