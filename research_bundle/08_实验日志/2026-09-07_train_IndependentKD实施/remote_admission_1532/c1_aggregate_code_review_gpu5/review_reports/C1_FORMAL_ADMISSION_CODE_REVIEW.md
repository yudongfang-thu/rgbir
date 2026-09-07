# C1 正式准入链代码审阅

**结论：当前 C1 的六兼容/64 批校准/24-update canary 主准入链没有新增代码阻塞。按根授权补充了 LOG 层的同 release CPU 测试证据检查；正式 C1 只能使用最终同 release 实跑的完整证据。本结论是代码范围接受，不是尚未齐备的 GPU 回执接受或正式训练已启动声明。**

日期：2026-09-07。审阅者 `/root/review_c1_spec` 对根实现 `admission.py`、`prepare_c1_stage.py` 原有流程进行独立只读审查，并核对 `evidence_bindings.py`、`protocol.py`、`train_independent.py` 的边界。没有修改 NEW 中任何文件，没有 GPU。唯一授权变更在 LOG/prepare_c1_stage.py 的 `validate_cpu_preflight`，该新增小 helper 与原主链的独立审阅身份分开注明。

## 主准入链核查

- 只有通过实际固定 64 train batch、至少 16 非零批、train-mode 每批恢复参数/BN、seed20260907 的 C1 CALIBRATED 回执，才从实际 lambda_C1 生成数值系数。系数必须在 (0,1]，脚本不另选/重调 lambda、不读取 AP。
- calibration binding 校验实际 family、配置、加载源码、数据 YAML/paired mapping/roster 与当前 release；训练 seed 与待定系数是预定义允许差异，原生 recipe 与 C1 carrier 参数并未被随意忽略。
- C1 canary 配置使用 seed42 和校准所得系数；C1_y 仅把预定义 eta 置 0，沿用同系数。正式输出仅生成 C1 的 seed42/0/123 三份配置，不自动扩到其它核心实验臂。
- 正式 readiness 要求六个实际 `(seed,arm)` 覆盖 0/42/123×N/C0；每份检查模型/数据身份、执行 binding、绑定里的实际 seed/arm、ACCEPTED、trajectory_exact、30 batch 与24成功更新最低范围。旧/新首次导入修复仍保留原精确比较，不能把旧失败 attempt 当作成功。
- canary 必须是当前 C1/paired 的真实 completed canary，至少24成功更新、有非零 KD 梯度、未访问 test；其回执数值系数与实际 binding 中系数均须等于即将执行配置。
- 校准 receipt 的实际 lambda 与正式配置系数再次严格相等检查。准备过程不会仅凭传入 path 相同就接受异配置技术证据。
- readiness 与 implementation review 的全 release `.py` 集合及接受副本逐字节核对；实际技术 binding 另核实际加载文件，避免单独一份新 review 让旧实现的技术回执自动过关。
- 最终 trainer 在进入 GPU/训练前还执行 protocol 的固定单任务与 E200/640/B32/nbs64/workers4、pinned 环境、train/dev YAML 和 FROZEN 检查，再调用 admission；所有资源规则仍由现有统一 lease 完成。
- prepare 将拟生成的 readiness 暂时只在内存交给原 admission 检查，`finally` 恢复读取函数；通过前没有 ACCEPTED 文件。之后使用独占新写，不覆盖已有配置/回执。

## 同 release CPU 证据的最小修复

此前 CPU 简版回执只有 stage 名称/exit_code/time，单独不支持跨 release 复用。根确认每个实际 release 均在 pinned 环境重新执行全部测试，原命令/日志均保留，并要求只补 LOG 层检查，不在冻结 admission 中新增另一套全树绑定。

现在 prepare 要求 preflight 的：

- `release` 为本次实际 release 的绝对路径；
- `environment.torch/ultralytics` 等于 cfg 的 pinned 版本；
- 恰有 operator_tests / reference_package 两个成功 stage；
- 每个实际 command 参数指向该 release，且对应绝对日志文件真实存在。

这不把路径检查冒充全树源码身份校验：源码仍由 readiness/review 和真实技术 bindings 核对。首轮使用根生成的最终 release 新 preflight 字段，不复用旧 `cpu_gpu*.json` 简版成功记录。admission 本身保持不变；这项 CPU 接纳由准备入口与根阶段验收承担。

7 个小 CPU fixture 检查通过：正确同版证据、旧简版拒绝、异 release 拒绝、异 pinned 环境拒绝、旧 release 命令拒绝、缺真实日志拒绝、失败 stage 拒绝。它们是明确合成的控制流测试，不是服务器实跑证明。原始输出见 `c1_formal_admission_code_review_v1/cpu_preflight_fixture.log`。

新增小 helper 由未编写它的 `/root/review_matrix_spec` 独立只读复核，确认覆盖根约定的全部窄范围条件，无实质代码阻塞；该复核没有扩大为全树源码或测试内容验真。

## 接受边界

可以继续生成真实技术验证所需配置并按统一队列推进；只有最终同版六份兼容、实际 64-batch 校准、C1 canary、完整 review/source 集与同版实跑 CPU 证据全部通过时，prepare 才能产出 FORMAL_CONFIGS_READY。真实 GPU4 evaluator_profile 的接口失败属于独立修复，不被本审阅重标成功；它也不使本文件变成训练或 AP 的接受回执。
