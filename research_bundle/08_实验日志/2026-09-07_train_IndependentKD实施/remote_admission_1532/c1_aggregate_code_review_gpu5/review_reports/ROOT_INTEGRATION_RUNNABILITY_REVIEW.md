# 根集成复审：技术验证可运行性

**结论：当前 runtime、criterion、trainer、calibrator 未发现阻止 94 真实兼容验证或 C1 技术 profile / 64 批校准启动的代码阻塞。可以由统一资源 lease 调度这些验证；本结论不是正式 E200 准入，也不声称真实 GPU 验收已经通过。**

日期：2026-09-07。对根编写的 `runtime.py`、`independent_criterion.py`、`train_independent.py`、`calibrate_independent.py` 进行只读复核，同时核对冻结 legacy trainer、natural loader、execution binding 和 compatibility 驱动接口。未修改根实现，未使用 SSH 或 GPU，未读取新实验 AP。首次审阅 `ROOT_INTEGRATION_REVIEW.md` 保留不覆盖。

## 前次阻塞项的状态

- **R2 已修复。** calibration 执行前调用 `validate_execution(formal=False)`，因此 pinned Torch/Ultralytics、train/dev YAML、模型文件和单任务配置先检查。每项 FP32 loss 与 autograd 梯度均显式要求有限；范数溢出也失败，不再把非有限梯度简单排除后仍接纳 calibration。有限零梯度仍以 ratio=null 保留。
- **R3 已修复。** runtime 检查冻结 legacy 入口、tracked loader 入口、实际 `object_evidence_loss` 和 `paired_rgbir_data` 模块文件，并验证 tracked 类的直接基类恰为冻结原始 dataset。来源混用即拒绝。历史/新 trainer 仍由已保存的 ORIGINAL_CRITERION / ORIGINAL_DATASET 分派，没有将已替换的类误记为 original。
- **R1 已建立 execution binding 路径，本次不替代完整 admission 审核。** 三类技术阶段均会产出实际配置/加载源码/小输入的 binding；最新 CPU 测试覆盖源文件、recipe、split roster、geometry/D2 内容变更的拒绝。根仍在完成正式 readiness 的逐份绑定，故本报告仅允许运行验证本身。

## 校准协议核查

1. 学生结构在 `_setup_train` 后重载冻结 RGB reference 的完整 state_dict；每个自然 batch 前再次 strict 重载并 `train()`，覆盖模型参数和 BN buffers。教师与 reference 每批 eval / no_grad，正式学生不沿用这个校准副本。
2. 私有自然 loader 真正使用 generator seed=20260907，控制 worker Python / NumPy / Torch 随机状态，shuffle 且无 replacement。固定前 64 个真实增强 batch，要求每批实际 B=32，不抽取“有信号批次”替代自然流。两批 `--profile-only` 只产出 PROFILED。
3. 校准 student forward 未进入 autocast，使用 float32；所选参数为检测头前 P3/P4 的两个实际输入模块。C1 的分子是 `grad(B*0.1*C0)`，分母是 `grad(B*C1_unit)`，两者来自同一学生图、同一对象集合、同一参数集合，系数为有效批范数比中位数；无重复乘 B 或 0.1。
4. C1 超出 (0,1] 不静默 clip；L1 才用 `min(1,0.1*median)`。全部零或不足 16 个有效批不会输出 CALIBRATED。L 还要求实际入选图像数和来源组门槛。
5. C1_y 继续共用 C1 系数。目标项/非目标项范数及夹角已逐批记录，配合最终 lambda 可计算 C1_y 的实际剂量；删去非目标项不被预先等同于梯度范数必然降低。
6. L1 / L_GT 计算同样的 adapter 选择，核对 selected anchors 和 base count，再分别测教师与 GT 目标梯度。当前定位几何是否满足条件仍由真实几何/D2 证据决定，本审阅不替它制造通过状态。

## 兼容验证与真实训练接口

- `build_trainer(... historical=True)` 使用真正冻结原 criterion / dataset，新路径使用 IndependentCriterion / tracked dataset；criterion 的 N/C0 都直接执行原 C0 数学路径，0/.1 剂量未重定义。
- compatibility CLI 固定每 seed/arm 的 30 个完整双标签 loader batch，以及旧/新各 24 次真实成功 optimizer update；旧/新依次构建，结束后恢复 legacy 全局类，避免后一段误继承上一段的替换状态。
- 前向后的 C1 只从 scores 传梯度；L 只从 boxes 传梯度，native 检测三项仍保留。真实 B 从当前输入张量读取，总损失只加一次 `B*lambda*KD`。
- verifier 的 observer 同时核查真实 optimizer.step 调用与原生 AMP 计数，保存完整模型梯度、模型/EMA/optimizer/scaler 状态并逐步直接比较。此次只复核它与根 trainer 的接合；verifier 本身由本审阅者先前编写，不冒称对其进行了独立作者审阅。
- C1 的 draft coefficient 为 null 时可以做 calibration/profile（这些路径直接计算 unit loss），正式 24-update C1 canary 则应在得到合法系数后使用该数值配置。不能把 null 当作训练剂量。

## 已执行 CPU 验证

本机 `D:/Anaconda/envs/KGJ_proj/python.exe`，torch 1.8.0+cu111，仅 CPU：

- `python -m unittest test_admission test_coverage_probe test_protocol -v`：**35/35 通过**。其中 test_admission 测试的是 execution binding 的合成小文件合同，不能单凭这些测试宣称完整 formal admission 已验收。
- `python verify_compatibility.py --self-test`：**4/4 通过**，只覆盖精确 tensor/tree 比较与 RNG observer 的辅助逻辑，不等于真实 30-batch/24-update 验收。

原始输出：`root_integration_rereview_cpu_tests.log`、`root_integration_rereview_verifier_helpers.log`。服务器 pinned 环境的 import/真实数据/API 行为与显存/RSS必须由接下来的受管验证给出，本地没有代做或虚构这部分证据。

## 正式运行前的两个记录补充

这些不阻塞当前兼容验证和 C1 profile，但正式训练证据应补齐：

- 新 criterion 目前的梯度检查限 canary 首批/首个非零 KD。规格 §8.4 要求在预定训练索引或 epoch 记录共享参数的加权 KD/native 范数比与余弦；正式运行需固定这个诊断调度，不能之后按结果挑样。
- 冻结 source-group helper 可能对未登记 Drone stem 返回 `unavailable:…`，LLVIP 也有 prefix fallback。实际 L 校准接纳时应明确排除未知来源占位值，并声明前缀来源的证据身份，不能把未知占位字符串当作第二个已验证来源组。

## 下一步执行判断

可立即进入统一 lease 下的真实 N/C0 兼容验证、C1 两批资源 profile；资源实测通过后完成 64 自然 batch 的校准，再用已定系数做 24-update C1 canary。任何失败使用新 attempt 保留原证据。正式 E200 仍等待 root 的 readiness 完整绑定、真实回执和独立准入检查。
