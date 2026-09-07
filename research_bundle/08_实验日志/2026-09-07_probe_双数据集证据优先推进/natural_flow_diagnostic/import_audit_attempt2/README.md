# attempt2 配置导入之后的 CPU 路径核查

**PASS：显式绑定 v2 prepare_configs 后，没有发现后续同名内部模块、配置键或源码副本缺失。未修改诊断入口，未执行 GPU、SSH 或新哈希。**

`check_follow_on.py` 在新 Python 进程中按入口真实顺序导入 coverage → runtime → legacy/tracked → selection → localization → diagnose。runtime、legacy、配对数据与选择器均执行实际本地源代码；只有无法在本地完整运行的外部 Ultralytics 和 project lease/receipt 接口使用禁止执行的替身。这验证内部路径污染与字段合约，不冒充真实 pinned runtime canary。

成功重现：runtime 的 sys.path 前置使裸 `prepare_configs` 命中 task_conditional_reference 旧生成器，没有 llvip_C1。随后从当前入口提取实际 importlib AST 执行，明确绑定 release/prepare_configs.py 后，LLVIP/Drone 的 v2 C1 配置均存在。所有入口直接访问 cfg 键齐全；expected_nc=1/5、expected_train_images=9619/17990，真实 LocalizationConfig/EvidenceConfig 构造通过，B32/workers4/640、augmentation 与旧 coverage data/mapping 路径一致。

10 个实际内部模块绑定预期位置；入口列出的全部 11 个 source snapshot 文件存在。`follow_on_receipt.json` 保存具体路径和入口大小/mtime。执行时 CUDA 未初始化，未调用 loader、模型、lease 或服务器。远端实际 cfg/权重文件存在性与完整依赖仍由 root 的真实 canary 验证；本核查不解除几何或训练限制。
