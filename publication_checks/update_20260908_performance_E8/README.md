# 2026-09-08 性能修复与 E8 启动阶段发布

本次是已授权的阶段同步，没有启动实验、改动本地科学结果或计算新文件哈希。沿既有 `research_bundle` 结构发布 867 份源文件，约 14.61 MB；原始非 Markdown 代码与小产物 826 份逐字节核对一致。Markdown 仅适配仓库导航，对未发布的大产物保留位置说明。

同步范围：

- [吞吐诊断](../../research_bundle/08_实验日志/2026-09-08_ops_训练吞吐诊断/README.md)：block16 数值及真实训练轨迹失败、selected-only 实现、独立审阅、同 raw 回放、两条 fresh 24-update 路径、真实日志频率的 N/C0/C1 吞吐与资源回执。
- [selected24 完整读出](../../research_bundle/08_实验日志/2026-09-08_ops_训练吞吐诊断/SELECTED_UPDATE24_RESULT.md)：两侧 30 批、24 次成功 optimizer 更新、6 次 AMP 跳步、30 次 EMA 更新；学习损失、梯度及模型/optimizer/EMA 状态逐位一致，仅限已执行窗口。总墙钟与扣除审计/完整统计后的时间分别报告。
- [E8 设置与启动](../../research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/README.md)：源配置、源码快照、准入、资源预约与启动证据；截止 03:42，N 已进入第 2/8 轮，首轮 530.879 秒。N/C0/C1 单 seed 队列没有新 AP；训练 CSV 中禁用评价产生的零指标不作为结果。旧 E200 继续。
- [既有数据分析综合报告](../../research_bundle/07_研究分析/RGBIR数据分析综合报告_20260908.md)及其已完成的小体积来源与派生图表；本次不重新分析。
- 最新实验索引与 IndependentKD 实施入口，便于从历史证据继续追溯。

没有上传权重、state/raw `.pt`、原始数据集图片、缓存、凭据、`host_profile` 原 JSON、主机全进程或其他用户命令明细。共 47 个源范围条目明确排除，详见 [发布清单](../../PERFORMANCE_E8_INCREMENT_20260908.json)。仅包含综合报告的派生图表。

[暂存检查](staged_review.json)记录文件范围、逐字节检查与禁传扫描。Git 的普通 whitespace 检查会将保留的 CRLF 及原始文件尾随空格报告为格式提示；本次不为格式检查重写冻结源码或原始回执。正式 Git 提交身份与远端分支核验在提交推送后向请求方返回，未以文件哈希替代来源证据。

Supplement: after the main stage commit `92e3ae402b6b721617de030586bfc16b997ea864`, the three small files in [the linked old-run heartbeat](../../research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/heartbeat_20260908_0335/README.md) were copied byte exactly. See [the supplemental receipt](heartbeat_link_supplement.json). No whole-host process inventory or broader historical link audit was added.
