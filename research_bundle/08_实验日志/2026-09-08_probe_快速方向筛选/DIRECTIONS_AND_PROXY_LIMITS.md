# 当前方向判断与快速筛选的代理限制

> **2026-09-08 后续进展**：本文末尾的同帧探针及原生 NMS 见证现已完成，见[新报告](../2026-09-08_probe_同帧选择覆盖/FINAL_REPORT.md)。本批旧定义9个机会中8个实际已有原生正确检出；原生机会仅1个且入选。本文旧200dev的低置信数字是 assigned 候选状态，不能直接称漏检；不改原数，也不以这批训练图外推dev。下一项改为先复用完整dev预测核对真实错误谱。

**LLVIP 仍有最大的已见 RGB–IR 检测差距，但当前没有证据把这段差距变成某种 KD 的可达收益。下一步最有信息量的是查清“教师能补的错误是否真的被当前选择器选中”，而不是再开一组短训。** 以下核对主六端点、本轮 LLVIP N/C0，并于12:25结果发布后补入已接受的 F-rel-GM；本阶段共9次训练收口，不扩大GPU矩阵。

## 当前已完成读数

下表 AP 为百分数，Δ为相对**同一轮 N** 的百分点。全部 seed42、2048训练图、BN running freeze、lr=1e-4 constant、3轮192批；Drone92、LLVIP91次成功更新。两个 LLVIP N 属独立实际运行，数值相同不把回执互相替换。

|筛选/臂|mAP50–95|AP50|AP75|ΔmAP vs matched N|
|---|---:|---:|---:|---:|
|Drone主 N|54.518217|77.149703|64.041865|—|
|Drone C1|54.543548|77.261131|63.979043|+0.025331|
|Drone C2|54.540724|77.237743|64.011569|+0.022507|
|Drone F-rel-GM（独立修订scope）|54.555474|77.250604|64.149608|+0.037257|
|LLVIP主 N|32.171224|70.593275|23.475438|—|
|LLVIP L2-box|32.147232|70.551017|23.418618|−0.023992|
|LLVIP L2-GT|32.162699|70.581472|23.425215|−0.008525|
|LLVIP置信度 N|32.171224|70.593275|23.475438|—|
|LLVIP C0|32.185721|70.437398|23.417782|+0.014496|

C2−C1 mAP=−.002824pp；L2-box−L2-GT=−.015467pp；C0的AP50/AP75分别−.155876/−.057655pp。它们描述这个成熟起点、有限预算的结果，未形成稳定正方向或正式增益证据；不据此否定分类/定位/特征这类载体，也不据小正值自动延长。来源：[主筛选](analysis_1203_snapshot/README.md)、[置信度筛选](confidence_analysis_1210/README.md)及各目录 summary.json；成功更新/selected读取各自原 completion_receipt。

F-rel-GM相对N/C1的mAP为+.037257/+.011926pp；相对N AP50/AP75为+.100901/+.107743pp，同时回执recall−1.038440pp、precision+.989186pp。相对C1 AP75+.170564pp但AP50−.010527pp，不能概括为各方面更好。P/R不换算成固定阈值下新增/修复的对象数量。该新协议在原8批校准后、任何F AP前修订，固定λ=14.438521129817886，经显式跨scope投影复用主N/C1；原F-rel BLOCKED不撤销。这给出“同选择上的Gram特征内容确实已训练并评价”的证据，但没有检验选择之外的IR漏检覆盖或正式多seed收益。来源：[F-rel-GM已接受读数](feature_gm_analysis_1225/README.md)、[完整原值与投影](feature_gm_analysis_1225/summary.json)。

## 哪种空间仍最大，为什么这些短训可能看不出来

1. **LLVIP 的可见缺口仍最明确。** 旧 matched RGB/IR native mAP=32.878405/48.852941，差15.974536pp；RGB AP50的TIDE Loc/Miss/Bkg分别8.754/7.685/5.534pp，单类Cls=0不能说明前景判别已解决。AP75定位oracle虽大，IR自己也有51.892pp残余。模型 AP 差、oracle和候选替换机会都不是可蒸馏上限。200dev静态对象中，643 GT有146个IR修复候选（90低置信、29无粗候选、27仅定位）及42个潜在损伤，仍支持先检查**漏检/前景响应覆盖**，不能只看定位平均优势。来源：[LLVIP完整AP](../2026-09-07_probe_双数据集证据优先推进/ap_error/README_LLVIP.md)。
2. **本轮 L2 实际验证的对象与剂量很窄。** 三轮累计904次selected是对象出现次数；原8批只有30/649 base通过定位门，主要因成熟R已可靠且IoU≥.7。λ截到1后，活跃批共享参数KD/native范数比中位仅1.262%。teacher与GT目标在这些R框上的输出坐标导数cosine≈.959，不能当共享参数梯度相似性，但提示这次控制的内容差别小。负的小幅AP差不能单独判定位无信息，也不能归因为“训练太短”。来源：[校准与门链](early_llvip_evidence/README.md)。
3. **成熟参考R与当前学生S不是同一概念。** C选择基于冻结R的候选和T−R质量；L2也以R定位不足为门。它没有直接测当前S的错误。LLVIP本次S初始化与R同一checkpoint，之后仍更新S、冻结R；Drone S是新weight0 N42、R是旧native RGB42。从随机通用初始化训练时，成熟R已正确不表示早期S已正确。因此“成熟模型三轮微调无明显提升”不能替代新学生整个E200学习过程的判断；反过来，也不能无证据保证早期训练会获益。
4. **换成F-rel仍沿用C1对象选择，未专门检验IR独有漏检。** 没有冻结R稠密候选、共同匹配或有效区域的对象不在该学习集合；已有粗候选的低置信漏检可能入选，不能笼统说全部IR独有对象都被排除。F仅换了被选对象上的内容载体，不自动扩大对象覆盖；当前不知道实际selected覆盖146类机会中的多少。Drone少数类仍有宏AP分类瓶颈与教师互补线索，但C2相对C1未显示更好mAP；“freight/truck/van占94.26%”仅指分类oracle宏贡献，不是全错误份额。不能将对象数量、共享梯度份额和AP贡献混同。

## 已执行的最小 CPU 诊断与下一项接口

已运行 [cpu_selection_coverage_audit_v1](cpu_selection_coverage_audit_v1/README.md)：复用原对象状态函数，在**同一静态缓存行内**交叉统计，train/dev各自独立，未推理。LLVIP dev146个修复候选的旧ROI代理全有效，旧P3/P4 anchor有117/无29；Drone461个修复候选中旧anchor有385/无76，双ROI有效457，两代理均有382。逐对象 [objects.csv](cpu_selection_coverage_audit_v1/objects.csv) 和逐类/错误桶 [summary.json](cpu_selection_coverage_audit_v1/summary.json) 已落盘。

**这还不能给 actual selected 覆盖率：** 旧anchor字段来自P3/P4一对一空间归属，当前C门为稠密any-candidate；旧ROI也有公共窗口/padding差异。静态 ID 是原图局部GT行，训练 selected 是增强batch全局GT行，缺同forward/frame和原GT映射；Drone静态N42还不同于当前R。实际selected/oracle连接明确为不可识别、数值留null，不能把29/76直接称现行gate排除，也不能用train/dev计数差宣称过拟合。

**下一项已确定的最小信息探针：** 固定已有LLVIP首个32图批次，在一次同帧S/T/R推理中同时输出每个GT的稳定原图ID、原/增强双GT行和框、S/R/T固定错误状态，以及原C选择的pair/valid/ref-candidate/teacher-correct/q/eligible/selected和批内排序分母。按“IR正确且S错误”的无候选/低置信/定位桶统计逐门覆盖，并列T错误、S正确的风险桶；每桶写真实分母，不改阈值、不训练读出、不做optimizer或矩阵。它先回答是否为覆盖问题，不能单凭一批估计总体AP或决定换loss。现有完整低阈值post-NMS缓存能补全局错误图谱，却缺全稠密窗口，无法恢复原selector；目前的准确缺口和提取字段已在CPU目录冻结。根任务已指定该同forward导出为下一工作；本阶段尚未启动该推理，也没有新GPU/hash。
