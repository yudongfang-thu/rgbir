# Baseline 特征证据复核与共样本关联（2026-09-07）

> 已完成 12,357 行对象×层、519 张含共同对象图的 CPU 关联表。旧能量图支持局部响应差异，旧 CKA 支持共享结构；两者均不足以在分类、定位和特征 KD 中判定收益优先级，任务可读出性需要新的 ROI/raw 导出。

## 目的
在相同对象上对齐分类/前景候选、定位误差和已保存特征指标，区分共享结构证据与额外任务知识。

## 设置
仅本地 CPU，无模型推理、无训练、无 SSH/GPU、无新哈希计算。读取 2026-09-06 probe 的 DroneVehicle/LLVIP 各 200 对、VEDAI 121 对及 seed42 baseline 输出。所有新产物只写本子目录，不修改冻结工程与旧证据。

## 结果

完整结果为 [CPU summary](cpu_join_v2/summary.json)、[对象×层表](cpu_join_v2/object_level_join.csv)、[图级关联表](cpu_join_v2/image_level_join.csv)、[响应分组表](cpu_join_v2/energy_group_summary.csv)、[图级相关表](cpu_join_v2/image_descriptive_correlations.csv)。总共同对象数为 4,119；521 张输入图中有 2 张 VEDAI 图没有共同对象，因此对象关联表覆盖 519 张。

同对象前景响应定义为原生格点中心落在各自模态 GT 内的通道 L2 幅值均值，除以 1.75 倍扩框区域、排除该模态全部 GT 后的局部背景环均值。排除 letterbox padding；没有插值补足小目标，前景/环任一为空即无效。以下 T/S 都在同一组双有效对象上统计；不是跨网络原始幅值直接相减。

|数据集|仅教师命中对象|P4 两侧有效对象|教师前景/环比中位数|学生前景/环比中位数|
|---|---:|---:|---:|---:|
|DroneVehicle IR→RGB|360|321|1.552|1.187|
|LLVIP IR→RGB|107|107|1.549|1.189|
|VEDAI RGB→NIR|41|36|2.052|1.840|

这些比值是模型内部的空间响应对比，不是信息量、判别率或可达 KD 增益。对象框标注差异和层的空间格点仍会影响数值。双方命中对象中 Drone P4 比值为 T 1.558/S 1.601，反例说明更强教师不必处处有更高响应。

为了排除旧层级 CKA 比较的有效图集合变化，额外只取 P3/P4/P5 前景 CKA 都有效的同一组图。下表是 **paired−donor ΔCKA**，也是 summary 中 `fg_cka_on_all_layers_valid_images` 字段的实际含义。

|数据集|三层共同有效图|P3 ΔCKA均值|P4 ΔCKA均值|P5 ΔCKA均值|
|---|---:|---:|---:|---:|
|DroneVehicle|179|.6003|.5970|.2799|
|LLVIP|147|.3390|.4708|.0455|
|VEDAI|1|不作层级判断|不作层级判断|不作层级判断|

图级 P4 ΔCKA 与 teacher-only 对象比例的 Spearman 相关为 Drone −.341（198 图）、LLVIP −.442（200 图）；与双方命中对象平均定位优势的相关为 −.235（194 图）、−.349（186 图）。这只是开发集共变关系，含亮度、尺度、拥挤及选择混杂，无显著性或因果主张。它足以反对直接把“高 CKA”解释为“这张图蒸馏机会更大”，不证明低 CKA 越好。

脚本独立重算 teacher-only 中学生已有严格低置信正确类候选，仍为 Drone118、LLVIP25、VEDAI20，与已保存独立候选审计一致；这验证对象、模型方向与预测记录的 join 没有混淆。

## 已有特征证据的边界

实际执行的 v2 脚本为 `../../2026-09-06_probe_RGBIR数据特性与可迁移知识/remote_execution/probe_rgbir_v2.py`：使用 Detect 前置 hook 捕获输入三层，沿空间 token 中心化 CKA，使用原生网格，修复了 v1 的统一20×20池化。`probe_review.md` 中针对 v1 的问题不能原样当成 v2 未修故障。

前景是两侧 GT 的 union；随机 donor 在 recipient 前景位置取另图 token，不要求 donor 是同类同尺度对象。正 ΔCKA混合对象存在、位置、类别和场景结构，不能专门归因于同实例细节。原始 C×H×W tensor 只保留在当时进程内，落盘只有逐图区域 CKA 和通道 L2 能量图；对象 ROI 向量、完整类 logits、raw DFL 缺失，不能逆推。

## 最低成本额外诊断

由主代理统一导出 Drone 的当前 N42、独立 RGB N0、IR42，以及 LLVIP 旧 visible42/IR42；旧六 baseline join 仅作历史描述，不能充当当前 N42 的特征结果。固定 train/dev 和来源分组，所有标准化、降维、线性映射和超参选择只在 train 内完成。

1. **前景与类别分开。** 相同对象 ROI 的类间 logistic/ridge 读出，另做前景与尺度匹配背景窗口读出；LLVIP 单类别只做后者。报告 macro loss/逐类正确率与背景误报，不能把单类高置信叫类间知识。
2. **额外信息与可迁移性分开。** 比较 S、T、S+T、S+独立同模态教师及容量相同的 S+副本，保持降维维数和对象权重一致。S+T 是教师参与推理的诊断上界，不是单模态 KD 效果。另在 train 拟合 S→T 映射，用 held-out 任务读出检查可由学生表示预测的教师分量。
3. **定位采样不可泄露回归目标。** GT 框自适应2×2 ROI可作“已知对象窗口”条件语义诊断；用它回归该 GT 框会混入框形状/采样特权。定位特征应取共同 N anchor 的固定窗口、保存空间格点，回归统一 RGB-GT 的 offset/LTRB，加入仅 anchor/尺度元数据控制；DFL同anchor比较则另列。
4. **对象 null 和同模态控制。** 跨组 donor 在类别/尺度层内固定选择，报告无法匹配的分母和唯一 donor 数。原图随机 donor 的 CKA不能替代该 null。独立模型之间不直接通道余弦；可用 centered CKA、train-fitted Procrustes/ridge、各自训练的同容量任务读出。
5. **保留检测状态。** 对所有共同对象及 teacher-only/student-only/both-hit/neither分层报告；背景单独。若只有GT ROI探针，只声称GT窗口条件下读出性，不能推广为检测器可找到这些对象。任务probe是选择候选方向的证据，不是方法成功或三seed增益结论。

## 产物路径
本目录为主实验下的独立子分析；主实验索引由主代理统一登记。来源为 `../../2026-09-06_probe_RGBIR数据特性与可迁移知识/`。本地执行入口为 [join_saved_features.py](join_saved_features.py)，需 numpy/scipy。运行 `python join_saved_features.py --out <新目录>`，输出目录必须不存在。`cpu_join_v1/ATTEMPT_NOTE.md`保留首轮重复解压导致提前结束的记录；v2 约10秒完成。

可复用工程模块在 `03_现行工程/SpaceNet6_OTD_official_reproduction/`：

- `yolo_osssl/cross_modal_decodability.py`：fixed patch、Procrustes、centered CKA、group bootstrap基础算子。
- `tools/analyze_cross_modal_decodability.py`：PCA/logistic/ridge流程，可参考但不能不改直接引用结果；当前 StandardScaler/PCA 在内部GroupCV前拟合，且 cross_direct 比较独立PCA坐标无自然语义。
- `tools/extract_cross_modal_probe_features.py`：旧SpaceNet/SAR导出入口，图像直接resize、raw dict布局及GT中心几何与本次640 letterbox合同不同。
- `tools/analyze_fm_roi_features.py`：同一冻结网络内的 paired-flip/donor ROI余弦；合法的网络内对比，不能移作独立RGB/IR模型逐通道余弦。
- `tools/visualize_rgbt_baseline_features.py`：显著性可视化辅助，仍不产生任务信息质量。

`shuffled_pair_indices` 当前实现逐接收对象找跨组 donor，允许重复 donor，不是严格双射；新分析需显式报告或换成满足合同的匹配。以上是只读代码观察，未修改这些工程文件。

## 局限与下一步
旧特征仅逐图区域 CKA 和通道 L2 能量。新的统一导出由主代理完成，避免重复推理。本子分析无模型效果增益结论。所有图级 SD/相关都不是训练多seed不确定性；不据此修改冻结分类/定位训练。
