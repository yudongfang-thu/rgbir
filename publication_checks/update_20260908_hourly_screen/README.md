# 2026-09-08 固定子集微调筛选发布

本次仅发布新阶段必要源码、协议、CPU 验收、2048 清单及统计、最终三臂小回执和 accepted 分析结果。队列 956.625 秒；单 seed FT3 的三臂均未超过微调前 N42，按事前规则不自动扩展。原 block16 失败和旧 E200 证据不改。

[单一逐文件清单](../../HOURLY_SCREEN_INCREMENT_20260908.json)记录源路径、字节数、逐字节验证、Markdown 链接适配与排除项。原始 source 结果不修改，未计算新文件 hash。

最终 `final_evidence_1022` 保留 N/C0/C1 的 train/eval/canary、初始化、资源与队列完成/检查回执，并保留分析器需要逐臂读取的三份相同 dev roster。部署 release_v1 的字节核对失败与未启动 GPU 的身份回执同时保留。`release/` 对应已执行 release_v2；早期 release_v1 仅按回执记身份，不伪作本次执行源码。

重复中间 snapshot 的本次未提交副本从发布包移除，源证据仍在本地/94。完整 17,990 图标签 metadata、重复 30 批 sample_stream、权重、raw tensor、缓存、完整主机 snapshot 与凭据不上传。`canary_checks.json` 和 `postflight.json` 保留已经执行的三臂流一致检查；没有公开的原始大 trace 时不能独立重做这些检查。子集 builder 可在原 94 数据路径重跑。

accepted `analysis/analyze_hourly.py` 的描述性结果复算只需要本包最终小回执和 dev roster。从仓库根运行（输出目录须尚不存在）：

```sh
python research_bundle/08_实验日志/2026-09-08_ops_小时级筛选重构/analysis/analyze_hourly.py --campaign research_bundle/08_实验日志/2026-09-08_ops_小时级筛选重构/final_evidence_1022 --output /tmp/rgbir_hourly_recomputed
```

发布前用 `load_campaign` 纯函数对本包输入执行一次 CPU 可读性/summary 一致检查；不生成重复科学结果，不执行 GPU 或新增评估。
