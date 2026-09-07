# Attempt 2 显式配置导入复核

**PASS_FOR_REAL_CANARY：5 项导入回归与原 14 项独立 CPU 真值/负例检查全部通过。** 仅支持启动已授权的真实只读 canary；未验收真实 64 batch，不构成 L 训练、几何、校准或 KD 收益准入。

审阅者 `/root/ap_error`；选择诊断作者 `/root/baseline_feature_analysis`；本次最小修复作者 `/root`。使用 experiment-audit 技能。用户禁止新增 hash 的指令优先，采用路径、size、mtime 与源码字节副本核验。本审阅未使用 GPU 或 SSH。

## 失败与修复

真实 attempt 1 在第 0 batch 前失败：`runtime.py` 将 `task_conditional_reference/legacy_oev1`、`task_conditional_reference` 插入 release 之前，裸 `from prepare_configs import configurations` 取到参考实现，缺少 `llvip_C1`。这项启动缺陷不在首轮 CPU 选择真值覆盖范围。首轮源码、测试、审阅及失败输出均保留于 `remote_failed_attempt1/` 和原审阅文件，不将其视为真实自然流证据。

当前 `diagnose_natural_flow.py:153` 使用 `importlib.util.spec_from_file_location` 从 **指定 release/prepare_configs.py** 绑定独立 v2 生成器。独立测试从真实 `runtime.py` AST 执行到重型模型导入前的全部路径设置语句，再实际导入错误生成器；随后执行当前入口的配置绑定 AST。即使错误模块已在 `sys.modules`，新入口仍返回正确的 LLVIP/Drone C1 配置。所有 12 份生成配置与无歧义导入结果相等。

源码字节差异只包含 `import importlib.util` 和显式配置生成器绑定；配置内容、自然流、选择器及阈值未改。`test_attempt2_import.py` 的首次审阅断言没有包含新增 `.util` 导入，4 项行为测试通过、1 项差异白名单测试失败；原 `attempt2_import_test_result.json` 保留。修正测试白名单后，`attempt2_import_test_result_v2.json` 为 5/5 PASS。这不是丢弃执行失败或修改选择算法。

## 核验与范围

- `attempt2_import_test_result_v2.json`：真实路径顺序、错误模块与 C1 KeyError、缓存碰撞隔离、全部配置一致、最小源码差异，共 5 项 PASS。
- `cpu_test_attempt2_result.json`：修复后重跑 14 项原独立测试，全部 PASS。涵盖 C 整批配额反例、L 多图/空图/非连续全局 GT、所有计数与距离/质量/索引拒绝、旧 trace 与 loader 元数据、2/64 入口、有限正资源数及未完成 full 不得写 completion。
- `attempt2_source_stat_manifest.json` 与 `attempt2_reviewed_sources/`：13 份审定源码/测试/协议的 path、size、mtime_ns、直接 byte identity；无新 hash。
- 根执行者另提供 `../remote_import_preflight.json`：94 pinned torch 2.10.0+cu128、Ultralytics 8.4.115 的真实 runtime 导入顺序检查，LLVIP/Drone 12 个配置字段完全一致，nc=1/5，CUDA 未初始化。此项为根执行记录，本审阅未独立连接服务器或执行前向。

本地选择真值测试环境 torch 1.8.0+cu111，CUDA 未初始化。碰撞 fixture 执行实际 runtime 路径设置前缀，没有声称导入完整重型 runtime；根的 pinned CPU preflight 补充了该环境覆盖。任何后续真实 canary/64 batch 失败必须继续保留并独立处理。

机器入口：`ATTEMPT2_IMPORT_REVIEW.json`。真实运行后仍须核验两数据集每个 summary、64 行 trace/selection、每批 B32、固定 seed=20260907、逐字段旧 trace 一致、原模型/配置身份、实际资源、scope false 与 per-image L 重放等价断言。
