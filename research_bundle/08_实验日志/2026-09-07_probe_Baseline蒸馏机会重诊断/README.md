# Baseline 分类、定位与特征机会重诊断（2026-09-07）

> **解释范围补充（2026-09-07）**：下文“461中351低置信”仅为200dev对象计数结论。新完整1469dev、三seed AP诊断显示AP50的少数类混淆更突出，freight car/truck/van贡献macro分类oracle的94.26%。不能将旧对象计数解释推广为整个AP瓶颈。原数值保留，见[双数据集新阶段判断](../2026-09-07_probe_双数据集证据优先推进/STAGE_REPORT.md)。

**已完成2448图的统一baseline中间结果导出和分类/定位/局部特征CPU读出。Drone修复机会主要为低置信，LLVIP定位证据更强；尚不支持局部特征比输出蒸馏更有跨模态收益。请先读[阶段判断](STAGE_REPORT.md)。此诊断不产生蒸馏AP增益结论，不修改C1或原L1。**

## 目的
响应用户要求，从数据与已训练baseline出发决定下一步，而非围绕既定定位机制继续补规则。旧图册主要使用历史native，不能直接代表当前weight0/N的剩余错误。

## 冻结设置（导出结果前）
- Drone：当前weight0 N42、IR历史native42、当前weight0 N0；LLVIP：历史visible42与IR42，明确没有新的匹配N矩阵。
- 复用旧D1/D2冻结清单，train2048中按来源比例固定取1024，dev固定200；不选择AP好坏图，不访问test。
- 640 centered letterbox，FP32 eval/no_grad，batch=1，无随机增强。本次推理协议相同，模型训练recipe身份分别保留。
- 输出每RGB GT、同类IoU≥.5的一对一IR标签关联；原始pre-NMS候选按类无关/conf≥.05/IoU≥.1最大数量再最大IoU匹配。共同anchor优先N42的P3/P4候选，缺失时记录GT中心P3诊断位置，不能冒称存在学生候选。
- 保存同anchor全类别raw logits、4×16 DFL、P3/P4共同RGB窗口2×2特征减背景均值、同区域前景背景logit差。特征GT区域关联带特权，不代表实际部署可直接获得。
- 正式导出前审阅补充：另存同公共anchor的固定3×3局部特征（池化2×2），用于降低GT窗口大小对读出的影响；有参考候选与GT中心fallback分别统计。canary以Detect输入hook核验raw feats逐层相等，所有模型的anchor/layout逐项一致。
- 每图最多4个64px固定网格窗口，与RGB和IR全部标注无交集；私有seed=20260907，仅用于标注背景读出诊断，明确此选样读取两侧GT。
- 前景/背景均排除两模态共同有效内容域外的letterbox padding。固定背景窗口与可变GT ROI有尺度差异，因此固定anchor patch为前景背景读出主表，GT ROI仅辅助，增加窗口尺度meta对照并记录有效区域数。
- 类别/定位机会分母是全部RGB GT，同时报告教师可修复与可能损伤，不能用最终KD通过集合筛掉困难对象。
- 特征固定train拟合、dev评价的ridge读出，mean-objective alpha=1，无开发集调参；固定随机投影128维，比较RGB logits/feature、额外IR feature、额外独立RGB feature和错配IR。先分辨类间与前景背景信息，不把特征能量或CKA当可蒸馏收益。
- DFL只在未clamp合法GT距离内分析CE及头部梯度方向；不等于完整检测损失或共享特征梯度校准。

## 资源与产物
94输出：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_baseline_information_20260907/`。所有GPU任务共用原project_resource_guard lease；先2图短测，完整导出按实测峰值预约，GPU动态选择，screen运行，不改变正在运行的训练。

本地保留源码、协议、资源与完成回执、分析小产物。较大中间数组记服务器路径；原始实验产物不移动、不覆盖、不删改。真实结果、统计解释与独立检查已完成，范围及限制见[阶段判断](STAGE_REPORT.md)。

## 执行记录
首次启动因94未使用本地 `repo/experiments` 目录而在import阶段失败，未申请GPU；该attempt原文保留。修复为既有冻结release_v8路径，新增 `attempt2/`。Drone两图canary通过：Detect输入与raw feats一致，NVML峰值586MiB、框架reserved124MiB、进程树RSS1484MiB。完整导出按2048MiB显存/8192MiB主机RSS预约，batch1、无DataLoader，GPU由全局lease选择；较大RSS预留用于最多1224图的ROI数组积累。

attempt2完整导出因逐对象GPU同步开销而技术性停止，25图112.56秒，原始部分行与停止回执保留，没有根据科学结果停止。最终使用独立attempt3，每图一次搬到CPU后处理；两图对象/anchor/raw logits/DFL一致，feature最大差4.77e-7、区域logit差1.91e-6。Drone1224图约297秒、LLVIP1224图约100秒（不含全部序列化/调度），全队列已正常完成。两组数据与源码、任务/资源/完成/采集回执齐；CPU读出及独立复核见阶段报告链接。

本地完整中间数组约574MiB。GitHub阶段包保留压缩逐对象原始记录、raw logits、读出预测/参数/统计、代码和复核；两份较大features.npz仅登记本地与94路径，原始文件保留，不上传数据集原图或权重。

## 分析方法与文献边界
错误分解参考[TIDE](https://arxiv.org/abs/2008.08115)区分类别、定位与漏检的思想；本次对象计数不是TIDE官方dAP，也不是oracle AP。特征读出采用[独立线性probe](https://arxiv.org/abs/1610.01644)的诊断思想，固定ridge实现与其具体实验不同。推理侧额外IR特征可读出性不能直接等同于RGB学生可学的KD增益，需要同模态/错配控制以及之后最小训练验证。
