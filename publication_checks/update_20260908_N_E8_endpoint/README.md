# N E8 端点与状态增量发布

仅同步已完成的 N E8 端点及本次最新入口引用的小目录，没有重跑旧阶段 curation、训练、评价或科学分析。

- [N E8 原值及局限](../../research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/endpoint_N_20260908_043812/README.md)：完整 dev mAP50–95=43.376542，单 seed N 原值，不是蒸馏增益；原训练 CSV 的占位/末行短列不能作为 AP 来源。
- [既有 17 项限定复核](../../research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/endpoint_N_20260908_043812/ENDPOINT_REVIEW.md)：沿用已完成 PASS，不重复科学验收。
- [04:36 队列状态](../../research_bundle/08_实验日志/2026-09-08_train_分类快速反馈E8/heartbeat_20260908_043626/README.md)与[04:35 旧五任务状态](../../research_bundle/08_实验日志/2026-09-07_train_IndependentKD实施/heartbeat_20260908_0435/README.md)，对应 collector、检查与小状态产物一起发布。
- 更新 E8/IndependentKD README、实验索引、端点采集入口，保留上一阶段 GITHUB_PUBLICATION.md。根导航补充已完成 N 端点，历史阶段原值不覆盖。

[清单](../../N_E8_ENDPOINT_INCREMENT_20260908.json)共 37 个源文件、1,135,779 bytes。35 份文件逐字节一致，2 份 Markdown 仅适配仓库链接；未改本地原始科学产物。只比较本次新增/更新范围，未复查上一阶段 867 份文件。

明确排除 `heartbeat_20260908_043626/snapshot.json`：它含主机级快照，原件留本地。发布的 heartbeat README 中该文件名仅是本地来源说明，不能据此认为完整主机快照已上传。没有权重、raw gzip、凭据、主机全进程或缓存；没有新计算文件哈希。

提交前检查记录见 [staged_review.json](staged_review.json)。本次引用的新增目录全部随增量发布。Git 远端与本地一致及 clean 状态在推送后向请求方返回。

Publication check note: the first link check ran before creating its own receipt, so that single self-link was initially reported missing. The initial check is preserved; [the corrected check](staged_review_attempt2.json) confirms PASS after the receipt exists. This was a publication-order issue, with no change to endpoint artifacts or scientific review.
