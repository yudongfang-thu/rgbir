# C1 分块池化性能候选（2026-09-08）

**静态主热点是逐对象全类别masked-LME及每批统计同步。独立block16候选已通过CPU数值等价检查；没有改在跑release，也没有准入长训切换。**

## 身份与已做工作

正式依据为 `2026-09-07_train_IndependentKD实施/remote_admission_1532/formal_C1_gpu5_attempt2/runs/C1_seed42/implementation_snapshot`。只读逐字节比较train_independent、runtime、independent_criterion、selection_adapter、classification_logit、gradient_observation以及legacy的object_evidence_loss/train_object_evidence共8文件：当前03工程及seed0/123快照均与seed42一致。未计算hash。

本地仅CPU。`cpu_checks_attempt1/receipt.json`和加强冻结辅助梯度断言后的`cpu_checks_attempt2/receipt.json`均12项通过，torch1.8.0+cu111未初始化CUDA。覆盖M=0/1/2/15/16/17/33/64、C=1/5/8、无效前景/背景、单类、多图/空中间图、完整C0选择映射与C1 loss；S/T/R delta和valid、C0 stats/身份/RNG核对，S score梯度数值比较，S DFL与T/R scores/DFL梯度必须None。

CPU所有前向输出和完整选择器检查exact；若干纯pool学生梯度非bitwise，最大绝对差9.536743e-7。预先固定容差atol=1e-6、rtol=1e-5，逐字段同时报告exact标志，不以allclose冒称完全相同。首个默认Python缺torch的启动命令在导入时失败，未生成实验输出；随后明确使用KGJ_proj CPU环境。

## 具体热点与候选边界

以下链接指向与正式快照逐字节相同的本地源文件；它们未被修改。

1. [selection_adapter.py:162](../../../../03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/selection_adapter.py)：190–194逐对象计算两次masked_fill/logsumexp。260–263每图、每P3/P4分别调用S/T/R。正式快照前三批common_count为450–496，均值474.33，因此平均每批约2846次Python对象循环、5692次全anchor logsumexp（不含旧C0 evidence）。这些循环本身是GPU小kernel/图节点密集，**不是每对象都调用item同步**。本候选仅固定16对象分块，保持原float32、-1e30哨兵、最小token数、invalid可微零和最后anchor轴LME数学定义。
2. 同文件170/180的GPU bool验证，每次pool两次；B32、两层、三模型最多192次pool调用，即最多384次此类同步；264还有最多64次GPU torch.equal。验证不能静默删除，可另候选集中计算标志后单次传CPU，但需要保留失败检查和数值合同。
3. [selection_adapter.py:297](../../../../03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/selection_adapter.py)及[classification_logit.py:121](../../../../03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/classification_logit.py)：完整stats每批都计算，很多float/int/tolist/布尔读取让GPU排队工作等待CPU。每100批只是写日志节流，见[independent_criterion.py:117](../../../../03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/independent_criterion.py)。可把独立标量统计stack后集中传输，同时保留所有原值、last_stats及日志；不能简单删字段或改统计分母。
4. [object_evidence_loss.py:92](../../../../03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/task_conditional_reference/legacy_oev1/object_evidence_loss.py)：每图GT匹配把GPU cost传CPU供SciPy Hungarian；selector253又逐图转两组GT索引；_choose210/222还有计数和q传CPU稳定排序。可保留GPU原cost和同一SciPy调用，把各图cost统一传CPU，再按原图顺序求解并一次回传索引。不能改成贪心或不稳定topk，不能让GT配对、q并列、ceil(rho*eligible)改变。
5. selector237–240原C0 GT-channel evidence与260–263全类relative evidence重复做相似区域计算，且完整R全类delta主要服务stats。第一版仍保留：直接复用新pool的GT通道可能因浮点归约改变q或边界选择，需要比本候选更强的选择等价验收。
6. [object_evidence_loss.py:114](../../../../03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/task_conditional_reference/legacy_oev1/object_evidence_loss.py)：每批S/T/R重复创建固定layout网格。只读按device/shape/stride/input_size缓存网格是较小候选；仍须保留每个raw输入检查，无RNG消费。

教师没有被同一batch重复前向：[independent_criterion.py:38](../../../../03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/independent_criterion.py)是T一次、冻结R一次，S由原trainer一次。不能把R换成正在变化的S，也不能跨随机增强batch缓存T/R输出。当前C1不执行定位分支。固定共享梯度观察仅在epoch0/10/50/100/199各首批一次；canary额外梯度sanity由max_steps控制，不能把它们当每批主要开销。

以上是源码工作量/同步位置，不是耗时占比。正式source前三批旧日志含启动与epoch0观察，不能据此外推持续训练速度；GPU utilization高也不证明矩阵算力饱和。

## 运行接口

CUDA入口强制pinned torch2.10.0+cu128、ultralytics8.4.115及现有单GPU全局lease；本目录不提交资源任务。输出目录必须全新。

```text
python benchmark_candidate.py --reference-dir <release_gpu5> --mode synthetic --device cuda --output <new_output> --iterations 7 --warmup 2
```

synthetic固定两个[M,C,A]案例：[32,5,6400]和[128,5,1600]。先等价核对，旧/新交替计时，CUDA显式前后同步；每次记录allocated/reserved峰值，完成时附lease资源。计时范围是pool forward+raw-score backward，不是完整训练。

```text
python capture_real_batch.py --reference-dir <release_gpu5> --config <formal_C1_seed42.yaml> --student-checkpoint <completed_stable_N42/weights/last.pt> --coverage-dir <coverage_drone_attempt1> --output <new_capture> --batches 2
python benchmark_candidate.py --reference-dir <release_gpu5> --mode bundle --bundle <new_capture/raw_batch_00.pt> --device cuda --output <new_replay_00> --iterations 7 --warmup 2
python benchmark_candidate.py --reference-dir <release_gpu5> --mode bundle --bundle <new_capture/raw_batch_01.pt> --device cuda --output <new_replay_01> --iterations 7 --warmup 2
```

capture只加载稳定已完成N42，不读取正在改写的C1 checkpoint。使用冻结coverage自然loader seed20260907，逐批对原64批trace的图像/增强/双GT完全核对；这是原probe/calibration流，不冒称正式训练sampler。S在私有模型副本train mode，T/R eval且无梯度；使用cfg AMP，无optimizer/EMA更新。分别用CUDA同步墙钟记录loader等待、preprocess、S/T/R forward、native loss、原selection、KD+统计、native+BλKD反向。反向为未GradScaler缩放，不能当真实24次optimizer更新验证。

pool阶段另作同raw独立重放、每次pool同步计时，避免插桩改变原selection计时；这个pool分项不能直接从原pipeline时间相减。stage同步关闭了通常的数据/计算重叠，所有阶段墙钟仅为诊断。首次batch的loader等待还含worker启动，需单列。

bundle保留真实scores、DFL、双GT与原feature逻辑shape/dtype；未使用的feature值用零stride占位存储，并明确feature_values_omitted。C1源码仅用feats的shape、device/source/version身份，不读取feature值；此bundle不能用于feature KD、网络前向或图像重建。完整selector重放保留原C0 oracle、q、rho、稳定排序、全部统计及学生score梯度。微基准不会替代真实模型参数/EMA验收。

## 不能热改的部分与验收顺序

block16的数学式和RNG不变，但backward归约次序可能改变浮点结果。只有raw-batch等价/速度证据仍不足以切换正式三seed训练。后续需独立候选release、冻结真实同流至少24次成功optimizer更新，核查raw/delta/valid/选择身份、native/KD/total、学生与辅助梯度、AMP跳步、optimizer/EMA及所有参数差异，明确容差和是否可恢复继续。不能改batch、workers、sampler、增强、rho、标签、C1系数、AMP或训练长度来伪装等价加速；现有runs/receipt/source不覆盖。

## 文件变更记录

仅新建本目录：pool_block16.py（纯算子候选）；benchmark_candidate.py（CPU／synthetic／raw bundle等价与计时）；capture_real_batch.py（仅待94 lease短测的真实捕获与分阶段计时）；README.md及cpu_checks_attempt1/2回执。未修改任何正式/共享源码、权重、原实验产物；本地未运行GPU或停止任务。
