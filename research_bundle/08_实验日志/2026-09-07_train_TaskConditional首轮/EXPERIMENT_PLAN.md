# 冻结计划：OEv1 + 条件定位，CL/CGT 首轮

冻结依据：用户于2026-09-07完整批准上一轮计划，实施前冻结；已看见C42/N42等历史端点，未看新CL/CGT结果。本文件不追溯声称与原OEv1同时预注册。

## 方法与身份
- 新模块 rgbir_task_conditional_v1，旧 OEv1 C 的代码/recipe/权重不改。
- C保留 foreground−background evidence、P3/P4、T=2、lambda=.1、原选择与分母；不是当前S定位已准的门。
- T=原IR seed42；R=原RGB seed42；S从原generic预训练初始化。训练期额外IR GT权限显式报告，部署只RGB。
- L：GT同类一对一IoU≥.5最大有效匹配，额外pairIoU≥.8；冻结R预选单P3/P4 anchor；R coarse confidence≥.05、RGBGT IoU≥.1；原生全部GT候选几何唯一归属，包括小框扩展、未配对/其他类GT。
- 未裁剪RGB/IR GT距离均在[0,reg_max−1−.01]且inside；几何通过独立证据掩码。基础E形成在T质量门之前。
- gate：R和T argmax GT类、score≥.25；R RGBIoU<.70；T RGBIoU≥.60、IRIoU≥.50；T RGBIoU−R RGBIoU>.05。
- L=四边KL(T||S)，温度2、T²/4；分母max(1,pre-teacher E对象数)。不读取当前S改门。
- CGT保持相同集合/门/分母/系数，目标两bin q软化为 q^(1/T)/sum；精确处理零。不是RGB-only。
- total=native_total.sum()+actual_B*(.1*C+lambda_L*L)。N/C运行L诊断但权重0，不改变旧C。

## 系数与协议
- lambda_L正式值待真实训练批校准，不填伪造值。固定64batch，校准seed20260907，实际增强沿原recipe；R权重学生副本不延续成正式学生。
- 参数集合为检测头P3/P4输入对应的两个特征模块全部可训练参数。
- lambda=min(1,.1*median(norm(g_native)/norm(g_unit_L)))，g_unit_L包含B。仅非零L梯度批计算；全部0返回NO_LOCALIZATION_SIGNAL，不用epsilon救援。记录截断和实际比例。
- 同数据集L/CL/CGT/random_l共用固定lambda，无AP调参或epoch调权。
- Drone：train17990/dev1469，E200、640、batch32/nbs64、workers4、AMP、deterministic、SGD及原lr/增强完整继承；固定last/EMA独立评估。无test读取，无val挑best。
- seeds固定0/42/123；正式部署记录真实环境/配置/模型路径和文件副本，不写哈希。

## 诊断和几何
- 先复用D1，再raw前向D2；分清对象最佳框与实际监督anchor。记录类别/尺度/来源/亮度代理、过滤漏斗、native owner与定位分布误差。
- train-only roster：Drone每来源folder首/中/末≤258对，LLVIP fit每prefix首/中/末42对；独立图像结构点，每图至少6可信点、3象限，固定20%独立复核。
- 记录图/组/region的经验覆盖，不称相机标定或逐像素真值。P95误差≤min(stride/4,.1*短边)，max≤stride/2；增强后传播误差。GT重合或特征峰值不算独立点证据，未覆盖拒绝。
- 先Drone；若其几何/D2未通过而LLVIP通过则移到LLVIP补N/C。均未通过不启动L长训，继续C归因。无box自动fallback。
- 几何工作日预算内记录真实完成范围及未覆盖；不得用空表/占位verified过关。

## 矩阵与推进
第一批既有N/C/C-random三seed收尾，仅新CL42、CGT42；C42等价验收后复用，否则修复或补跑匹配C。
两臂固定E200完成。自动扩展须mAP(CL42)−mAP(C42)≥.3pp且CL42−CGT42>0，全部完整独立receipt和实现校验通过。.3pp为投入门槛，不是显著性标准。
通过后新增16长训：CL0/123、CGT0/123、L0/42/123、C+randomL三seed、最终方法shuffled三seed、same-modal三seed。未通过不修改阈值/seed/lambda补救，也不声称定位KD领域无效。
- randomL从完整preteacher E抽K=paired门通过数，独立CPU RNG，保留分母/系数。
- shuffled：固定train-only无自配错排，所有seed共用；paired选择/门/E/K不变，错配IR同几何增强在线T前向，仅换目标。它同时破坏身份和空间，是压力对照，不是纯配准因果证明；单独canary。
- same-modal：独立RGB weight0 seed0教师、原RGBseed42参考；只读RGB图/GT建集合和门，不继承IR mask；不能T=R归零。
- CMDistill/CCLKD partial先审计后复用PROTOCOL-ADAPTED三seed。FGD/LD外部资产核查在CPU准备线，不占首轮训练优先级。
- LLVIP先诊断；主pilot通过后N/C/CL42再补0/123和最终四臂，机制全矩阵集中主数据集。

## 验收和资源
- CPU必要测试：GT温度/支撑/layout/KL/B/空集、唯一owner、RNG、随机同K、analyzer配对单位ddof1和缺receipt。
- 真实canary每新路径≥24次成功optimizer update，报告AMPskip、初态/首批/样本流、非零KD梯度、冻结T/R、EMA/optimizer隔离、峰值VRAM/RSS。
- 资源由统一guard lease负责：常规3GPU，例外4GPU须占后≥2完全空卡；用户≤3task/card，但现行guard≤2CUDA且现recipe只双开；全卡≥2GBfree、项目VRAM<70%、总RSS≤300GB，保留提前准入门。
- 不自动减batch/修非有限值/按AP早停；技术失败保存attempt，修复新attempt。原run不覆盖。
- 启动/运行使用screen/tmux；输出仅/mnt/dataset/yudongfang项目目录，不写系统盘、不写密码。

## 分析与交付
主指标mAP50–95百分点，辅助AP50/AP75/逐类/召回；逐seed+mean±sampleSD(ddof1)。缺三seed或归因只描述不升级claim。报告C−N、CL−C、CL−L、CL−CGT及random/same/shuffled；伤害与修复对象分开。
先接受独立analyzer，再分析正式结论；test封存直到最终方法/协议冻结。每实验README+脚本+小结果、服务器路径互指、更新08索引。阶段成果推送既有 research/full-evidence-20260906，不上传凭据、大权重/数据全集。
