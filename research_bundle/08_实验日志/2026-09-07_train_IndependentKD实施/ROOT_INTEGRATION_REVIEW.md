# 根集成代码独立审阅

**结论：C1 校准的主公式、实际 batch 缩放及逐批 train-mode 状态恢复正确；现有代码可以在统一资源 lease 下部署做技术 profile/canary，但正式 readiness 仍有必须修复的证据身份绑定缺口，校准也必须拒绝非有限 FP32 梯度，而不能把坏批次仅当作零信号排除。**

日期：2026-09-07。独立只读审阅根编写的 `calibrate_independent.py`、`admission.py`、`runtime.py`、`train_independent.py`、`independent_criterion.py`；为核对 data seed 与 L 接口，读取 `coverage_probe.py` 和 `localization_adapter.py`。没有修改这些文件、没有运行 GPU、没有生成虚构 ACCEPTED 回执。本报告不把“尚无真实 GPU 实测”当作拒绝部署 canary 的理由。

## 正式准入阻塞 R1：技术回执没有绑定实际执行版本、recipe 和 split 内容

位置：`admission.py` 的 ready.source_files/review.source_files 检查（约 29–34、85–90 行）及 compatibility/canary/calibration 回执检查（约 40–80 行）。

目前逻辑只证明：当前源码等于 ready 所列副本、等于 review 所列副本。它没有证明六份 compatibility、当前 canary、对应 calibration **实际使用的源码**就是这些副本，也没有把这些技术回执的实际有效 recipe / split 内容与即将执行配置绑定。

具体可通过的错误组合：当前 release B 的源码审核与 ready 都匹配，但兼容验收来自 release A；canary 来自另一套 workers/augmentation 配置；只要 dataset/model/teacher/reference 的路径字符串、arm/source、lambda 和 status/计数满足当前检查，就仍可能通过。仅比较 model 路径也不能识别同路径文件被替换。

这会把旧版本/不同训练数据流的技术证据借给新版本，是正式 E200 准入的实质阻塞，**不是要求重新跑历史全部结果**。

建议最小修复：

1. 每份技术回执写入实际执行源码副本清单（或引用有完整副本清单的不可变 release manifest）；admission 逐份验证其 required training source 文件字节与当前 accepted release 相同。至少包括 trainer、criterion、loss、selection、paired loader、native依赖版本，而非只存在一份单独 review 清单。
2. 保存并核对实际有效配置；允许的差异预先列出：compatibility 的 arm N/C0、学生 seed，calibration 的固定诊断 data seed / family / 待确定 lambda，canary 的短运行上限。workers、batch、nbs、优化器/增强、数据 YAML/roster 不得静默排除。
3. split YAML、配对 mapping 和 roster 保存小体积字节副本并逐份绑定；模型身份按工作区禁止哈希政策使用已登记的不可变输入回执/状态证据，不自行引入哈希。
4. 加负向单测：模型路径/lambda 相同但技术 source 或 worker/augmentation/split 变更时必须拒绝；不能仅靠 ready 中人为填写的新 source_files 弥补旧技术回执缺失身份。

## 校准接纳阻塞 R2：缺少执行前校验，非有限梯度可被排除后继续接纳

位置：`calibrate_independent.py:49–64`、`134–157`。

校准直接调用 `build_trainer`，没有调用 `train_independent.validate_execution` 或等价环境/数据/冻结配置校验。L adapter 会检查几何存在，natural loader 会检查 train/dev YAML，但这不能代替 pinned Torch/Ultralytics 与完整 recipe 检查。

更实质的问题是：`valid = all(isfinite(x) and x>0 ...)` 将零范数与 NaN/Inf 范数都变成 ratio=None。假设 64 批中 16 批正常、其余含非有限 FP32 梯度，`len(ratios)>=16` 仍可能输出 `CALIBRATED`。这里没有 AMP scale，不能将非有限真实梯度当作允许忽略的 AMP skip。

建议：

- 执行前检查配置/环境/数据（允许未校准系数为空，但必须满足该 family 的单任务及固定参数定义）。
- loss、native 梯度、KD 梯度及用于成分解释的目标/非目标或 GT 梯度均需有限；任何非有限直接写技术失败回执。只有有限的零信号批次可以排除比值，并单列 zero-native / zero-KD / both-zero。
- 校准回执和 JSONL 禁止输出 NaN/Inf JSON 数字；失败原因必须区分 `NO_SIGNAL` 与 `NONFINITE_GRADIENT`。

## 源码身份阻塞 R3：runtime 只验证了两个入口模块，未覆盖关键 helper 实际来源

位置：`runtime.py` 的导入和 __file__ 检查。

当前验证 `train_object_evidence` 和 `tracked_pair_data` 来自冻结目录是必要的，但 Python module cache 仍可能使它们依赖的 `object_evidence_loss` / `paired_rgbir_data` 来自此前已导入的其它目录。此时入口 __file__ 正确不代表实际函数/基类正确，且 old/new compatibility 可能共同调用错误 helper，形成错误的共同参照。

建议启动时同时校验：`legacy.object_evidence_loss`、`legacy.EvidenceConfig`、ORIGINAL_DATASET 及 tracked loader 原始基类的实际定义文件，另外记录真实 Ultralytics/native criterion 文件来源。来源不符立即拒绝该进程；不通过修改 `sys.modules` 静默替换已加载类。全新部署进程通常不会命中此问题，但它影响本计划要求的实际加载源码身份保证。

## 已核对正确的关键路径

- `state` 从冻结 R 的全 state_dict 克隆；每批 `load_state_dict(... strict=True)` 后 `model.train()`，会恢复所有 state_dict 参数和 BN buffers。没有沿用旧校准器 eval-mode 行为。
- C1 校准在同一个 student forward 图中用 `B*.1*C0` 与 `B*C1_unit` 求同一 P3/P4 输入模块参数梯度；系数是两者范数比的中位数，不重复乘 .1 或 B。C1 超出 (0,1] 不静默 clip；L 系数才按旧约定 min(1,.1*median)。
- `coverage_probe.build_natural_loader` 确实使用独立 generator seed=20260907，同时控制 worker Python/NumPy/Torch seed；正式训练仍保留旧原生 generator，不把校准数据流错误复用为新正式数据流。
- 目标类与非目标类范数和余弦已记录，C1_y 不被宣称必然降低范数。现有 raw norm 与 lambda 足以派生其实际剂量，正式汇总应明确写出配平后的比值。
- L1/L_GT 使用同一逻辑选样并核对 selected anchors/base count；唯一图像和 source groups 在实际 selected 上计数，至少两个真实 source group，不用路径父目录猜组。
- IndependentCriterion 的 C1/C1_y 使用单一额外 KD，actual_B 从真实张量取，分类 KD 不直接到 DFL logits；L 分支不执行分类 KD；native 三项检测损失仍完整保留。
- N/C0 兼容模式走 ORIGINAL_CRITERION 的原路径，没有把 C1 新内核替换进 C0。

## 实施边界

本次仅列影响接纳正确性的阻塞项，未将没有 AP 或没有 CUDA 运行结果当作代码错误。修复后可以进行 94 真实 profile、兼容与 canary；只有收到对应真实回执且证据身份全部绑定后，才能正式放行长训。复审应针对根修复后的实际文件，不覆盖此首次独立审阅记录。
