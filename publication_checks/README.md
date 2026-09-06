# 发布检查与复算

[独立发布审查](PUBLICATION_REVIEW.md)记录来源保持性、指标口径、内容检查和文档检查结果。本目录的检查只覆盖证据包，不重新认证训练或授予方法收益结论。

在仓库根目录用 Python 标准库复算五组历史比较与 521 对探针摘要：

```sh
python publication_checks/review_publication_numbers.py --repo . --output /tmp/rgbir_numbers_recomputed.json
```

输出可与[独立复算记录](publication_numbers_check.json)核对。脚本读取真实 JSON，不运行 GPU、不需要服务器凭据。

- [内容规则检查](publication_safety_snapshot3.json)：最后新增历史材料纳入检查；统计是该次扫描快照，不包括随后新增的发布说明。
- [最终文档检查](publication_markdown_final.json)：从原始文档恢复后进行链接适配，检查 GitHub 导航和嵌套链接问题。
- [链接适配明细](link_adaptation.json)：记录改写目标和明确未收录的历史路径。
- [完整文件清单](PACKAGE_FILES.json)：相对路径与文件大小，不含 Git 内部对象；清单本身不自列。
- [本地来源清单](../BUNDLE_MANIFEST.json)与[远端快照](../research_bundle/remote_snapshot_20260906/README.md)：用于追溯资料来源、采集时间和压缩/截尾边界。

规则检查未发现凭据候选，不等同于数学意义的秘密不存在证明。旧源码和原始 JSON/CSV 保持来源字节；导出 Markdown 只适配导航及必要的根级阅读说明。
