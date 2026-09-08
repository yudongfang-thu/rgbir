# 已知首批DFL缓存核对

**CACHE_MISSING_FOR_MATCHED_FIRST32_RAW_DFL：已知首32图两次probe及旧八批校准目录没有持久化本批S/R/T的真实DFL logits或概率。** 框期望、标量损失、置信度和匹配见证不能反推出分布。是否补原批唯一一次无更新前向由root决定，本审计没有运行GPU。

只读94以下三个已知root，深度最多3层，共208个文件；没有搜索其他项目或数据盘：

- `artifacts/rgbir_selection_coverage_20260908/attempt1`
- `artifacts/rgbir_selection_coverage_20260908/witness_attempt1`
- `artifacts/rgbir_direction_screen_20260908/screen_attempt1/calibration/llvip`

完整路径前缀为 `/mnt/dataset/yudongfang/projects/RGBT_campaign/`。[REMOTE_INVENTORY.tsv](output_attempt1/REMOTE_INVENTORY.tsv) 保留路径和字节数，无tensor或压缩数据候选文件。对应本地24个JSON/JSONL已检查字段和数值数组形状，实际4×16/flat64数组为0；source_manifest仅登记代码/输入路径，未登记保存的DFL张量。相关字段命中及文件stat见 [receipt.json](output_attempt1/receipt.json)。

缺失的是“这批图、这次增强/前向、这个模型与anchor”下的原始4×16数值和其坐标合同；以前其他raw导出即使包含DFL，也不能替换不同样本流的本批信息。这个结论不是全盘不存在DFL文件的主张。

复核脚本 [audit_known_cache.py](audit_known_cache.py) 只读SSH与本地JSON，写新输出目录，不计算新hash，不改旧证据、不加载checkpoint、不前向。
