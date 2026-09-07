# LLVIP full-dev 与 TIDE 最终独立审阅（2026-09-07）

**PASS，限旧LLVIP两模型完整dev评价和描述性TIDE分析：有效attempt2的人口/标签路径/模型身份/资源回执一致，两模型四个TIDE AP与全部oracle复现，独立COCOeval最大AP差1.42e-14pp。此接受不包括新协议N、物理配准、可蒸馏收益或L1准入。**

审阅者 `/root/loc_stress`，使用 experiment-audit。只读取本地实际产物并在本目录运行CPU复算，未启动GPU、SSH、训练或新hash。代码 [recompute_llvip.py](recompute_llvip.py)，实际复算 [verification_receipt.json](verification_receipt.json)，CPU约4.6秒。原始attempt、源码、结果和根验证回执未修改。

## 实际人口、路径与身份

读取当前/远端收集版exporter、campaign、四任务源码快照和实际summary/contract/population/identity/raw capture。当前与执行副本字节相等；复核输入的stat在执行前后未变。

- 两份full均为 **2406唯一dev图、7879 person GT**；两份canary均为 **64图、279 GT**，非零且存在person逐类指标。
- alias保留登记的`grouped_v1/{visible,infrared}/images/dev`路径，原标签清单全部位于各自processed `labels/dev`，不会把raw图片symlink解析路径用于YOLO标签推导。alias→canonical为显式双射，actual loader roster、canonical roster、capture image集和population记录一一闭合。
- 四任务均记录原processed标签GT总数、实际loader GT与capture GT exact；`on_start`在源码中逐图对比loader的`cls`和normalized box float32数组，实际contract的`loader_per_image_labels_exact=true`。这补上了旧attempt1零GT评价可错误通过的问题。
- 两模态按**完整登记根下的相对路径**配对，2406键唯一且相同；每图GT框、GT类别、实际canvas和original_shape全部exact。独立重建pair manifest与根保存文件完全一致，不依赖basename-only猜配。
- 两模型都是旧seed42/E200/训练workers8的固定`last.pt`，本次统一评估workers4/FP32。身份回执为`checkpoint_epoch=-1, ema_present=false, loaded_source=model`，不伪称另有可检查EMA字段，不把N42目录标签视作新协议N。

本地未重新访问94的原始标签文件或物理图片；原标签→loader逐图exact依赖已检查代码和此次实际callback回执，capture→跨模态GT/canvas一致性则由全部本地raw行独立重建。共享标签一致性不是物理配准真值。

## canary与full的验证范围

两份64图canary的native与capture五汇总、逐类metric JSON完全一致；actual kwargs和loader顺序一致。完整2406图**只执行capture一次**，没有额外完整native评价。full的`native_capture_exact=false`表示未执行完整双路径比较，不表示发生指标不一致。

旧外层campaign attempt1因processed symlink canonical化后错误推导标签，全部评价INVALID；其0指标不能用于科学结论。有效路径是`remote_completed_attempt2/{N42,T42}_{canary,full}_attempt1`，内部attempt1只是在第二轮campaign内的任务编号。失败记录、原源码和修复审阅已保留。

## AP、oracle与排序复算

独立从每模型全部raw行构造官方Data/TIDE对象，复跑AP50/AP75；全部main/special dAP、TP/FP/FN及固定score桶TP/FP/Bkg计数与原结果一致。当前alias-aware合成真值也复跑exact，包含显式alias绑定接受、未绑定alias拒绝和GT数量检查。

|模型|TIDE AP50 pp|TIDE AP75 pp|AP50 TP/FP/FN|AP75 TP/FP/FN|
|---|---:|---:|---|---|
|旧visible N42|71.427586|24.537259|6513 / 21893 / 1366|3262 / 25144 / 4617|
|旧infrared T42|92.229627|44.135239|7469 / 5460 / 410|4771 / 8158 / 3108|

另以pycocotools 2.0.7 COCOeval构造独立数据对象，maxDets300、all area、无crowd/ignore，使用默认recall网格及TIDE的`x/100`网格分别保存结果。两模型四个AP在两网格下都与TIDE最大相差 **1.4210854715202004e-14pp**。这是本LLVIP缓存的实际相等，不能推广为默认COCO与TIDE对任意输入逐位一致；Drone曾有recall-grid浮点边界差异。

TIDE与原pinned evaluator的匹配和积分方式仍不同；本次没有替换原pinned mAP50–95（N42 **32.878405%**、T42 **48.852941%**）。两模型AP差属于历史模态模型对比，不是KD增益。

TIDE总表`same_gt_population=false`来自保存的两模态image路径字符串不同。本审阅额外验证登记根下相对键及每图GT完全一致，因此该false不构成两模态GT内容不一致，也不应静默改为true掩盖不同的字段语义。

## 必須保留的定量解释

1. “AP50/AP75定位oracle贡献最大”只针对**六类main error**；AP50 N的special FN15.0440pp大于Loc8.7540pp，不能混称所有oracle之最。
2. 单前景类person使Cls/Both结构性为0，不能称分类任务已解决。前景/背景区分、漏检和分数排序仍可影响Bkg/Miss及AP。
3. N的AP75 Loc **56.373148pp**、T的 **51.892331pp** 是分别修正或抑制定位错误的独立oracle差值，不能相加，也不是IR教师可提供或RGB学生可学出的收益。IR自身的大定位残余也不允许把其目标当完美框。
4. N的AP50仍有Miss **7.684950pp**、Bkg **5.533965pp**；本次支持高IoU定位敏感这一描述，不排除前景质量和覆盖问题。

已回读作者`README_LLVIP.md`，它对旧baseline身份、单类0、main/special和非KD/L1范围的主要限定与证据一致。本审阅另明确64图canary双路径与full capture单评的不同验证范围。

## 资源与执行合规

四任务均在既有全局lease下顺序执行于GPU4，回执记录各任务一个CUDA进程；admission中的项目占卡为[2,4,5]，无第四卡例外。两条full依据各自canary实测NVML峰值1370MiB，预约1882MiB而非未经测量估计。

资源监测回执均COMPLETED/exit0/no monitor errors。最低观测空闲显存 **14662MiB**，高于2GiB余量要求；项目进程树RSS最高 **149020MiB**（约145.5GiB），低于300GiB内存限制。任务NVML峰值均1370MiB。回执声明screen会话`evidence_llvip_eval_s42`，输出位于项目数据盘`/mnt/dataset/yudongfang/.../artifacts/...attempt2`。

该结论基于实际lease/admission/峰值及采样监控记录，不宣称通过本次CPU审阅连续观察了服务器所有用户的每个瞬时进程；没有发现本任务资源违规证据。

## 复核入口与范围

- `llvip_full_eval/export_full_dev.py:15/29/81/96/123/147`：alias、原label检查、子集、canary/full路径、实际loader逐图检查及一致性。
- `llvip_full_eval/remote_completed_attempt2/*/capture_contract.json`、`population.json`、`model_identity.json`和`queue_attempt1/*resource_profile.json`：实际执行证据。
- `ap_error/run_tide_audit.py:69/155/225`：alias-aware合同、合成真值、原path-sensitive同GT字段。
- [verification_receipt.json](verification_receipt.json)：四任务人口/身份/资源、全pair exact、TIDE重算与COCO AP。

接受范围为这两份旧模型完整dev评价及当前描述性分析器路径，不包含新N三seed、四臂归因、定位目标认证、共享特征梯度、自然增强训练覆盖、校准或正式L1/训练准入。
