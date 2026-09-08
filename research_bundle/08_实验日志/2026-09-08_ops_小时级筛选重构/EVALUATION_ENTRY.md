# 小时级完整 dev 评估入口

**代码已备好，可供根任务检查并接入；尚未运行实际评估。** 源码为 [release/evaluate_hourly.py](release/evaluate_hourly.py)，由已验收 E8 评估入口作局部适配。此处是作者实现说明，不替代独立准入。

CLI：`--reference-dir`、`--run`、`--output`、`--native-profile-binding`、`--native-contract-config` 均必填；`--canary-receipt` 可省略，默认采用训练完成回执中的 canary_receipt。显式 canary 必须与该来源指向同一路径。

输入为新 run 的 `hourly_config.yaml` 与 `hourly_training_receipt.json`。要求 HOURLY_SCREEN_TRAINING_COMPLETED / HOURLY_SCREEN_FT / 3轮 / HOURLY_SCREEN_FT_E3_LAST_EMA，训练身份与配置闭合，last checkpoint stat 一致；对应 canary 为 HOURLY_SCREEN_CANARY_COMPLETED、同 arm、successful_updates≥24，且 config_copy 与本 run 配置字节相同。不读取原 E200 完成/准入状态来代替这些新输入。

原 E8 N 通用初始化/full-data 配置显式作为 native contract config，仅送入旧 `validate_evaluation_profile_binding` 校验原生评估源码、数据与 kwargs。warm-start/subset 训练配置单独保存；其 dev roster 必须逐项等于原完整 1469 图清单，实际 `model.val` 使用原 full RGB YAML。`evaluation_identity_projection.json`、新 contract/receipt 明确记录两种身份，不假装训练配置未变。仍检查实际 kwargs、1469 图、22462 GT、五类有限 [0,1] AP、固定 last stat、单卡 bound lease；不调用 E200 run/load/publish。

输出 `hourly_evaluation_contract.json` / `hourly_evaluation_receipt.json`，失败则 `hourly_evaluation_failure.json`，均使用新身份；另保存训练配置/训练完成/canary/native contract config 副本、完整 roster 和源码来源。新目录须不存在，不改旧源码/旧 run。等待根任务提供的 `hourly_common` 接口：ENDPOINT、read/stat/write_new/copy_sources/data_output/load_config。

[evaluation_cpu_checks.json](evaluation_cpu_checks.json) 的 19 项合成合同检查全部通过，测试源码 [test_evaluate_hourly_cpu.py](test_evaluate_hourly_cpu.py)。测试只注入 I/O helper 替身并检查新纯函数，未导入 Torch、未读真实权重、未启动 GPU、未计算 hash；root common 的真实导入接口尚需集成核对。旧原生指标算法未重写。

后续实际 common 已到位：同19项检查用真实导入重跑全部通过，三份配置及四入口 AST/导入闭合，见 [INTEGRATION_REVIEW.md](INTEGRATION_REVIEW.md) 与 [evaluation_cpu_checks_actual_import.json](evaluation_cpu_checks_actual_import.json)。这仍是 CPU 合同检查，未执行实际评估。
