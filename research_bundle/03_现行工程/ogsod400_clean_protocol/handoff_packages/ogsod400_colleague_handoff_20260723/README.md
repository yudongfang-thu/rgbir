# OGSOD400 RGB→SAR 研究交接包

**交接快照**：2026-07-23 00:48 CST  
**本地工作区**：`/Users/yudongfang/Desktop/光sar/ogsod400_clean_protocol`  
**L20 工程**：`/private/projects/ogsod400_clean_protocol`  
**L20 结果仓**：`/private/results`  
**交接目的**：让没有历史上下文的同事能理解我们做了什么、方法如何实现、哪些结果可信、哪些路线失败，以及下一步应怎样接管。

## 一句话结论

项目已经建立了完整的 RGB 特权信息→SAR-only 检测研究与审计基础，并证明了两个重要但有限的事实：

1. RGB 相对 SAR 存在很大的检测性能上限差距；
2. 在 SiXiang scene-clean **开发集**上，正确配对的 full RGB-CMD 信号相对强 same-modal `H_S`、严格 shuffle 和零剂量控制有稳定正增量。

但截至本快照，**还没有一个新方法达到 paper-ready 标准**。已有方法 full RGB-CMD 不是我们的新方法；`H_F` 只是 source ablation；LCSR、RIF 以及多条离线 proxy 路线没有通过各自的任务门。两个真正的新方法候选为：

- **MM-ARCS R2A**：OGSOD exact400 原始训练已经完成，但最终 analyzer 未被接受，结果仍为 `DEFER/EMBARGOED`；
- **SX-APR-v1**：SiXiang 7 arms × 3 seeds 的 B2 正在 L20 自然运行；00:48 时为 `10 completed / 2 running / 9 pending`。按照预注册规则，本交接没有读取或转述其 AP、loss 或 arm 排名。

总的研究决策是 **`REDESIGN`**，不是宣布成功，也不是否定所有 RGB privileged learning。新证据应先完成现有 B2，再修复/独立复核 R2A analyzer；不要启动第三个大矩阵。

## 推荐阅读顺序

1. [01_PROJECT_PROTOCOL_AND_EVIDENCE.md](01_PROJECT_PROTOCOL_AND_EVIDENCE.md)：问题、数据、固定协议、证据等级和本地/L20 拓扑。
2. [02_METHODS_AND_IMPLEMENTATION.md](02_METHODS_AND_IMPLEMENTATION.md)：主方法、失败路线、离线原型和具体代码入口。
3. [03_EXPERIMENT_CATALOG_AND_RESULTS.md](03_EXPERIMENT_CATALOG_AND_RESULTS.md)：本地与远程全部 campaign、关键数值、完整性与结论。
4. [04_IDEAS_CLAIMS_AND_DECISIONS.md](04_IDEAS_CLAIMS_AND_DECISIONS.md)：idea 演进、已经排除的解释、可说与不可说的 claim。
5. [05_REPRODUCTION_AND_TAKEOVER.md](05_REPRODUCTION_AND_TAKEOVER.md)：如何接管、核验、监控和继续工作。
6. [06_EVIDENCE_INDEX.md](06_EVIDENCE_INDEX.md)：权威源文件与机器清单索引。

`catalog/` 中的 CSV 是机器可读的完整索引：

- `campaigns.csv`：14 个可解释 campaign；
- `claims.csv`：11 个 claim 的允许/禁止措辞；
- `remote_results_manifest.csv`：L20 `/private/results` 的 241 个顶层条目；
- `remote_campaign_artifacts.csv`：campaign 到远程证据的映射。

远程 manifest 中的每个目录并不等于一个独立科学实验。它混合了正式 run、smoke、失败、hung、retry、旧版本和证据目录；正文按 campaign 归并，CSV 保留逐根目录可追溯性。

## 重要边界

- OGSOD 历史 `images/test` 被反复用于 validation/selection，因此所有 OGSOD AP 最多是 **contaminated development evidence**。
- SiXiang 使用 scene-clean 110/14/14 scene split；val 是开发集，test 仍保持 untouched，但 acquisition/近重复审计仍需补齐。
- “paired > shuffled”不等于“paired > 强 SAR anchor”；RGB 归因、净 efficacy 和共享/可达解释必须分开。
- `H_S` 是当前默认强锚点。`full RGB-CMD` 与 `H_F` 不是新方法。
- 入口可执行、训练目录存在、checkpoint 存在，都不等于有可写入表格的结果。没有 canonical analyzer/commit-last artifact，不升级 claim。
- 本包不含数据、权重、`.aris/traces`、私钥或大体积训练输出；通过路径和 SHA/manifest 指向原始证据。
- 本轮未调用新的外部模型评审。历史 GLM/Kimi 记录的范围和状态见 `04`，不得虚构为当前方法共识。

## 快照后的状态变化

L20 训练会继续自然推进。阅读本包时，应先按 `05` 的只读命令重新读取 B2 `queue_state.json`。除非已经满足 `21/21 completed + terminal validator PASS + commit-last analyzer`，仍不得读取 arm outcome。

## 当前最终决策

`DEFER`：等待现有 B2 自然完成并通过终局完整性检查；同时只修复/复核 R2A 的 analyzer 证据链。能够改变决策的新证据是：

1. B2 完整 21/21、validator PASS、六个预注册对比门的 commit-last 结果；
2. R2A/confirmation 的独立复核 analyzer 与不可变 terminal bundle；
3. 通过开发门后，在新 freeze 下加入 matched full RGB-CMD comparator 和独立 group/untouched/external evaluation。
