# 补评逐类指标显式适配

实现与真实六端点接入待独立审阅，未自行签署 ACCEPTED；不授权自动扩展。

冻结训练 release 和原分析器均不修改。manifest 必须显式启用 `posthoc_class_metrics`，原端点先通过已接受的分析器；随后仅在内存中补齐同一 checkpoint、seed 和完整开发集上的补评逐类 AP，并增加明确来源字段。旧 JSON 无逐类字段的历史事实保持不变。

接入核对已接受观察桥接、原端点身份、实际补评执行回执、原始副本字节、评价源码、五项汇总指标、唯一完整类别集合及逐类均值。不能将六次汇总指标相等扩充解释为新接口自动接受。补评来源字段包含路径、checkpoint、seed，不能冒充训练当时的记录。

`prepare_actual_manifest.py` 显式关联既有 rect-v2 三 seed 对象分析产物；原分析器已有 checkpoint/seed/roster 检查。本适配器不重算对象匹配、不改 confidence/IoU、不推断昼夜标签，不启动 GPU。

验证命令：`python -m unittest test_posthoc_class_adapter -v`。实际接入：`posthoc_class_adapter.py --manifest actual_manifest.json --analyzer <冻结analyze_independent.py> --output <新文件>`。所有输出均为新文件，不能覆盖既有 attempt。正式接受还需独立代码审阅；C1未来的原生逐类指标直接走原入口，实际评价源码仍须核验。
