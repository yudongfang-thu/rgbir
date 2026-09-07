# block16 候选与真实批次捕获独立代码审阅

**池化候选未发现改变 C0 选择门、排序、配额或分母的实现错误；已有回执只支持数值容差内的池化诊断。真实 capture 尚不能证明 KD 自身提供非零共享参数梯度，另有嵌套计时的峰值字段需修正。本文不准入替换在跑 C1、不接受训练吞吐加速或优化器轨迹等价主张。**

2026-09-08，独立审阅者 `/root/loc_stress`。只读源码、CPU/合成 GPU 已执行 JSON；没有运行 GPU、模型前向/反向、CPU 重算或修改候选源码。没有计算 hash。审阅在作者完成真实 capture 期间进行，因此下面区分已执行回执和仅代码路径。

本次最后读取的源码：`pool_block16.py` 2289 bytes，mtime 02:02:26；`benchmark_candidate.py` 15070 bytes，mtime 02:09:06；`capture_real_batch.py` 12959 bytes，mtime 02:08:15。具体以文件随实际 attempt 保存的副本为准，不把之后修改的入口套到旧回执。

## 必须修复或明确收窄的两项

1. **capture 的 finite 总梯度不足以回答 KD 是否有有效梯度。** `capture_real_batch.py:140–145` 只对 `native+B*lambda*KD` 执行 backward，然后要求至少存在一个 grad tensor 且全部有限。全零 tensor 也通过；即便总梯度非零，它也可能完全来自 native。当前只保存 `finite_student_gradients` 和 tensor 数量，没有 KD 独立范数、共享参数非零数量，也没有旧/新候选通过完整 student Jacobian 后的比较。

   若真实 capture 要承担“测到非零 KD 梯度”这一目标，应另行记录 KD 对 raw scores 和所声明学生参数集合的有限性、L2 范数及非零元素/参数数；总/native/KD loss 也应明确检查有限性。额外 autograd 检查要单列仪器开销，不混入原串行 stage 计时。否则当前结果只能称“原 selector 的组合 backward 执行且参数梯度有限”，不能写成非零、有益、足剂量共享梯度或训练可学性验收。现有 bundle benchmark 比较的是 raw score 梯度，无法补上模型 Jacobian 或 optimizer/EMA 证据。

2. **嵌套 replay 的外层 CUDA 峰值字段统计区间不正确。** `capture_real_batch.py:168–174` 的每个 `timed_pool()` 和其外层 `stage('separate_pool_profile_replay_...')` 都使用 `benchmark_candidate.py:182` 的 `timed()`；内层每次 `reset_peak_memory_stats()` 会清掉外层早期峰值。外层 allocated/reserved peak 因而不能解释为整个 replay 峰值。应禁止内层 reset，或不报告这个外层峰值并另立正确区间。外层 wall seconds 与独立 guard/NVML 总峰值不因此失效，不能把这一问题说成已经发生资源越界。

两项均已及时发给 root。最初曾怀疑 iterator 清理不适用于 InfiniteDataLoader；沿真实 call chain 继续检查后已经撤回：本 capture 使用的是 `coverage_probe.make_natural_loader()` 构造的普通 `torch.utils.data.DataLoader`，其 iterator 有 `_shutdown_workers()`，当前 finally 路径适用。没有把错误推断保留成缺陷。

## 选择规则与归一化：源码范围内通过

- block16 仍对每个对象、每个类别、原 foreground/background 做相同 masked log-mean-exp；保留输入验证、`minimum_foreground/background`、空对象、无效区域有限零值和 float32 运算。只把原逐对象循环改成固定 16 对象分块，没有按选中数动态改块，也没有改背景定义、阈值或 temperature。
- `cloned_selector()` 只复制 `build_classification_selection` 的函数 globals 并替换 `pool_relative_logits`；原 module/function 绑定不变。C0 的 `_evidence/_has_candidate/_match_objects/_choose` 仍调用原函数，因此 q、teacher/reference gate、rho、稳定排序及全 batch 配额没有接入新池化输出。
- base 为原 common-valid 与 reference-candidate；选择分母仍为 `max(1, base_count)`，分类有效层平均仍除以各对象的 valid level 数，C1 的 off-target 系数仍 .25。capture 的 `native.sum()+B*lambda*KD` 与实际 `IndependentCriterion` 的公式一致，未再次缩放 native。
- 新旧比较保留 delta、valid_levels、selected、labels、base/matched 映射、quality、eligible、C0 scalar、C0 stats、对象身份、selected_count 和 KD loss/raw 梯度。最新 benchmark 还让 T/R scores/boxes 成为 grad inputs，并断言它们及 student DFL 梯度均 None，避免只因 requires_grad=False 而误称无泄漏。
- GPU reduction grouping 已发生浮点差异；所以“相同公式”不等于逐位 exact。容差 `atol=1e-6, rtol=1e-5` 原样用于两者，不能把结果改称 exact 或完整训练轨迹等价。

## 来源、模式、RNG、清理与计时边界

- `bound_cuda` 要求真实绑定单 GPU lease，以及 torch2.10.0+cu128、Ultralytics8.4.115；reference/runtime/coverage 的实际导入路径均检查。capture 只允许数据盘新输出、固定 B32/workers4/640 与正 C1 系数，保存输入 config、checkpoint args、实际入口及依赖源码副本和 size/mtime。
- capture 对三模型文件做前后 stat；S 是独立载入的 checkpoint 副本，train 模式和 requires_grad=True；T/R eval 且 no_grad。S 的 BN 运行统计会在私有内存里改变，但无 optimizer/scaler/EMA update，也不写回 checkpoint。它不能冒充当前在跑 C1 的 live student 状态。
- 真实流是 seed20260907 的固定 calibration/probe loader，逐批核对 source、增强参数、双标签和 worker 记录与既有 64 批 trace exact；它不是正式 trainer 的 native sampler。未保存的像素字节不作 byte identity 声明。loader 使用私有 generator/worker seed，构造前后恢复 Python/NumPy/Torch CPU RNG。
- benchmark 的 `rng_unchanged` 实测范围为 Torch CPU 和当前单 CUDA 设备，不是全进程所有 RNG。原 paired selector 路径不使用 Python/NumPy 随机采样；block16 本身没有随机操作。capture 是独立新进程，显式初始化 Torch seed，不声称保持形式训练的全局 RNG continuation。
- 真实 capture 的 finally 调用普通 DataLoader iterator 的 `_shutdown_workers()`，然后写 lease 资源记录；正常异常传播交给原资源 runner。本文未读取真实 capture 完成后的进程退出回执，故只接受清理代码路径，不宣称实际残留进程已核完。
- 新旧 benchmark 使用同一 immutable raw bundle，重新克隆 leaf scores/boxes，交替先后顺序，固定 warmup/iterations，并同步 CUDA 前后计时。未使用新 selector 重采样对象。计时包含 selector、KD/statistics、tensor clone 和 raw-score backward；排除了 loader、模型和共享参数更新。
- capture 采用 AMP forward、未使用 GradScaler 的总 backward，且每个 stage 串行同步；这会改变通常的 CPU/GPU overlap，也可能改变小梯度数值。文件已经明确排除生产吞吐/训练准入。单独 pool replay 的时间不能从另一次正常 stage 中直接相减求“生产中 pool 占比”。
- bundle 的 feature values 使用零步长 CPU 占位，保留真实 shape/dtype；原 selector `_layout/_source_tensors` 在该路径只读取这些属性、identity/version，不读取 feature 数值。这支持当前 raw selector 对比，不支持 feature KD 或完整 student backward 的复现。

## 已实际存在的回执

|入口|读到的原值|可以支持的范围|
|---|---|---|
|`cpu_checks_attempt2/receipt.json`|12 项通过；包含空/单类/混合无效/跨图含空图；4 项 full selection 的 T/R 梯度均 None|这些 CPU 合成输入的数值/身份/无梯度泄漏检查；不是 pinned GPU full selector 通过|
|`../remote_pool_probe_attempt1/synthetic_attempt1/receipt.json`|torch2.10 GPU；两 case old median 0.029704458/0.092177238s，block16 0.003975684/0.007908164s|两种合成尺寸的 pool forward+raw-score backward；不能外推完整训练加速|
|同上数值检查|delta 最大差 9.5367e-7 / 1.90735e-6；raw gradient 最大差 7.15256e-7 / 1.43051e-6；均通过固定 allclose，exact=false|数值容差内近似等价，不是逐位等价|
|synthetic resource profile|COMPLETED、exit0、monitor_errors=[]；GPU4、单新增 CUDA PID；进程 NVML 峰值480MiB、RSS1138MiB，整卡最小空闲13314MiB|该已执行合成短测资源记录；不替代真实模型 capture 峰值|

截至本次审阅，真实 capture/bundle replay 的已完成 receipt 未在本地审阅输入中出现。CPU attempt2 是新版 benchmark；GPU synthetic attempt1 使用其实际保存的前一版 benchmark 副本，不可用新版新增的 T/R grad-input 检查追溯升级旧 GPU 回执。

**当前判定：pool/selector 变更的代码语义与已有数值诊断可继续受控验证；真实有效 KD 梯度与嵌套峰值字段需补齐/修正。没有 long-training switch admission，没有 optimizer/EMA/RNG trajectory equivalence，没有 AP/KD 收益结论。**
