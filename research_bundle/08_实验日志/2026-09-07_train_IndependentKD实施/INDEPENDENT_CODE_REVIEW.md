# Independent KD 实现的独立源码审阅

**当前结论：数值内核和单分支组合未发现确定的公式/索引错误，可以继续 CPU 核测试与受控真实 canary；正式 E200 准入仍有两类必须修复的问题：readiness 对完整配置/证据绑定不足，以及新 KD 路径在缺双标签 batch 时静默退回原生损失。未满足这些修复前，不签发正式训练接受。**

## 审阅独立性与范围

本审阅者实现了 `protocol.py`、`analyze_independent.py` 及其测试，**没有编写本次被审的八个文件**：`runtime.py`、`independent_criterion.py`、`train_independent.py`、`classification_logit.py`、`selection_adapter.py`、`localization_adapter.py`、`geometry_support.py`、`coverage_probe.py`。本次可作为上述八文件的独立实现审阅，不能代替另一位审阅者对本人的 analyzer/protocol 进行接受。

主线程尝试新增独立审阅 agent 时触发并发/thread limit，故交由未编写上述文件的现有子任务执行。本次只读这些源码及其冻结依赖，未改变被审实现、未运行 GPU/训练、未签发 READY。校准器和兼容工具当时还在编写，**不在本轮已审范围**。

下面行号对应审阅时源码；后续修改需再次核对。被审目录：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/`。

## BLOCKING-1：formal readiness 未完整绑定实际配置和各项准入证据

位置：`train_independent.py:42–64`（`validate_execution`）。

现在只比较 ready 与 cfg 的 dataset、model、teacher、reference、两个 coefficient；对引用 receipt 仅核 status；compatibility_seeds/minimum_successful_updates 取 ready 顶层摘要；`source_files` 只需任意非空列表。

这使以下错误状态仍可通过形式准入：

- 生成 ready 后改变 `augmentation`、optimizer细节、C选择参数、L strict07阈值、数据路径/映射或几何合同内容；顶层身份与系数仍然相同。
- 误引用另一 arm/seed/模型的 `canary_completed`，或真实记录只有少量成功步，却在 ready 顶层写24；代码没有逐项反查。
- L用一份包含合法局部点的非空contract，但没有64批≥16非零、≥16唯一图像、≥2来源的真实训练流证据；`geometry_verified`布尔字段无法替代这些门。
- `source_files`仅列一个文件，未覆盖 criterion、selection、loss、geometry、runtime 和冻结 reference 依赖；实际执行模块变化仍被接受。

**修法：**绑定实际有效配置的规范化完整对象或直接源文件副本，排除仅允许变化的seed/output/readiness路径等运行字段并明确列白名单；不把实际方法/增强/模型/数据/几何规则列入可变项。逐receipt反查 arm/source、模型身份、校准方法/λ、固定64批身份、实际非零/覆盖计数、成功updates、资源峰值和完整性。要求固定的实际执行依赖文件集合存在且bytes一致。L准入需同时绑定geometry/D2/calibration/canary，不只顶层bool。

**验收建议：**先生成有效的合成/受控ready，单独改一个增强参数、L IoU门、teacher路径、canary arm/成功步数、source-copy列表、geometry合同或64批计数，formal入口均应拒绝。拒绝只限制E200；不是要求论文归因齐备才能测数值内核。

## BLOCKING-2：新 KD 路径缺少双标签时静默退回 native

位置：`independent_criterion.py:30–32`。

对C1/C1_y/L1/L_GT，代码先计算native，如果`teacher_batch`缺失直接返回native。若真实loader/预处理集成错误或后续某批元数据丢失，会把该批悄悄变成零KD；最初canary曾出现非零KD不能证明后续所有批都没退化。它违反计划的数据/标签错误必须可解释失败，以及实际每批剂量可追溯要求。

**修法：**在新方法的训练调用中要求`teacher_batch`和`strong_img`及必要标签键完整，否则raise。若确有评估/特殊前向需要原生fallback，用明确的非训练模式或独立入口，不以缺字段推断模式。N/C0兼容路径仍保持原实现，避免借此重定义历史C0。

**验收建议：**C1/L1真实格式batch去掉teacher_batch或一个标签键应抛出异常；纯模型推理/独立验证走其既定接口正常。空合法GT集合与缺少标签字段必须区分：前者应可导零，后者是数据错误。

## Formal 依赖身份的并项修复建议

`runtime.py:10–12`通过修改sys.path后按通用模块名导入legacy；`geometry_support.py:17–23`也使用通用模块名。若同一Python进程先加载另一路同名模块，sys.modules可绕过新的路径优先级。`selection_adapter.py:26–37`已对自己的私有导入做__file__核验，其他入口尚无同等检查。

建议并入BLOCKING-1的执行依赖绑定：正式runtime检查真实`__file__`落在当前独立release的task_conditional_reference，不能仅记录期望路径。CPU核测试允许文档明确的旧目录fallback，但formal必须使用被冻结部署副本。此问题未被观察为当前真实运行错误，是源码上存在的身份绕过路径；不据此否定现有旧实验。

## 已核对、暂未发现错误的核心路径

| 范围 | 源码核对结果与边界 |
|---|---|
| C1 `[M,L,C]` | `selection_adapter.py:288–330`明确matched→pre-teacher E_C映射；selected在base索引重建，分母M为E_C，不是matched/selected数 |
| C0保持与选择 | `selection_adapter.py:237–300`按旧_evidence、共同尺度、reference粗候选、teacher正确性、q与_choose构建；C1非目标内容不进入选择 |
| T/R分离 | teacher/reference在no_grad，返回delta detach；mask/GT元数据不依当前S质量；学生前景和背景池化保留梯度 |
| 相对logit loss | `classification_logit.py:54–74`双方除温度2，教师raw差clip16，Bernoulli KL、目标项1、非目标平均项η，逐尺度损失后有效尺度平均，T²/E_C |
| 空集合 | 空matched通过student scores切片连图；无selected/合法空集合保留可导零，未伪造λ |
| L布局 | 继承`dfl_view`确为[B,4R,A]→[B,A,4,R]；同shape/bin/anchor检查，T温度化分布detach，GT二bin经q^(1/T)归一化 |
| L选择/分母 | `localization_adapter.py:31–57`调用同一个teacher模式选择，L_GT只换目标；未额外加入当前S fg门，normalizer为pre-teacher E_L |
| actual B | `independent_criterion.py:50–51`只在组合层乘实际batch size和系数，L adapter返回未加权scalar，不重复乘B |
| native assignment诊断 | 捕获native实际fg与localGT索引并映射回batch全局GT；仅统计same/different/background，不用于更改选择 |
| T/R生命周期 | 继承旧trainer在student/EMA/optimizer构建后挂plain criterion；旧_setup_train检查state keys与optimizer参数ID。真实canary仍须核验当前集成行为 |
| 几何支持域 | 同时检查RGB/IR框、两侧点凸包共同域和误差边界邻域，未通过返回false mask；全图象限/点数等由原合同保留 |
| 64批覆盖 | 私有generator、worker种子、train-only双标签、无replacement；coverage只报源图命中上界，不伪装成非零L梯度/几何通过 |

## 非阻塞的实施观察

- C1目前每batch重算原C0证据与全类池化，且较多stats会触发CPU同步；性能由真实canary测量，不能据源码维度估算共驻显存/吞吐。没有建议改变科学定义来提速。
- source非paired在当前trainer明确拒绝，属于尚未实施的后续路径，优于默认为同一目标后冒称已实现控制。当前paired核验不因四臂未齐被阻塞。
- L几何/真实信号当前是否已有足够证据，不由本次源码审阅确定。代码实现、几何被接纳、训练正式就绪须分开报告。

## 后续接受条件

修复上述两项formal阻塞并完成针对性拒绝测试，补读新校准器与兼容工具，再依据真实64批、24成功update、资源和数据流回执给出最终接受。仅凭本报告的“未发现公式错误”不能跳过真实验收；也不能把未取得长训成绩当作数值内核失败。
