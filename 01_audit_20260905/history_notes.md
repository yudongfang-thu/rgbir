# 历史方法与证据审计笔记（2026-09-05）

> 历史项目没有形成稳定的新方法主张，但不能把所有分支统称为科学失败：早期主要是预算混杂和机制未识别；后期分为有效性失败、归因失败、实现失败、成本失败及尚未进入效果实验的设计分支。

本笔记由独立 history_audit 子任务形成。只读本地资料，不连接服务器、不跑实验、不改实现；新增本文件用于持久记录。所有历史运行状态均为文档当时切点，不代表 2026-09-05 服务器现状。下文路径以 `E:/SHARE/光sar/` 为根；`L` 后为核对时行号。

## 1. 时间线与机制

| 时期 | 方法变化 | 可接受判断 |
|---|---|---|
| 2026-04-03～04-09 | TSKD 将 RGB teacher 分解 z_t/u_t、SAR student 分解 z_s/r_s；normalized relative reachability、task head、重建、弱 energy 约束；A1/A2/B/C 多阶段；detector 读 raw features | 历史单 seed skip-B 最优 0.54008，相对 baseline 0.52826 为 +0.01182；不是当前已识别机制或论文结论 |
| 4～6 月 | LADD 减少阶段、移除 A2，形成 warmup + detector/KD；Static、Dynamic、冻结 reach probe 的 dynprobe；cap2、BN-freeze 等修正 | 对阶段/优化路径敏感，旧 +1～2 AP 后来被 reload 对照动摇 |
| 6 月语义支线 | CoRe-LADD/Static SemReach：类别级 LLM/人工语义先验、assigner-aligned KD 权重，可选 frozen text embedding projector；仅训练期辅助 | 有实现和静态测试记录；本地未找到 detector results.csv/log/summary，catalog 也无 accepted detector result；不宜说已科学证伪 |
| 6 月 23 日以后 | reload/continued-training、YOLO-init、same-pipeline det-only 对照 | 优化预算本身足以接近/超过旧方法；原 baseline 不能再独立支撑归因 |
| 7 月初 | exact direct400、singleproj、registered rescue、failure isolation | 旧 direct400 弱正且 seed 不稳；rescue 大幅负，但 source/BN/phase 等同时变化 |
| 7 月中旬 | RIF、LCSR、可预测 residual 等；转向 strong H_S anchor + optical increment | proxy 可预测性不等于 detector utility，shared/private 内在分解不再是可防守主张 |
| 7 月 17～28 日 | research reset、预注册、scene-clean、配对/null；SX-APR、MM-ARCS | 形成较强负结果：SX-APR 六门全失败；MM-ARCS SiXiang 前瞻重复 1/2；不少其余分支尚停工程/可构造性层 |

来源：`04_方法演化档案/2026-04~06_LADD主线/LADD_EVOLUTION_AND_EXPERIMENTS.md` L21–43。该文件 L7 明确是叙述性索引，数字应回原始来源。

**术语漂移：** 总览将 NRRL 写作 Noise-Robust Representation Learning，但 TSKD 原文 L169–195 明确定义为 normalized relative reachability learning。总览简化的 B/C 阶段说明也与原文不完全一致：原文 B 为 optional fixed-target distillation，C 为 joint refinement。机制引用优先 `04_方法演化档案/2026-03~04_TSKD探索/TSKD_METHOD_MAINLINE_CN.md` L169–195、L241–346；历史最优数值见 L379–385。

## 2. 定量事实及边界

| 证据桶 | 数字（除注明外均 AP50:95，0–1） | 解释 |
|---|---|---|
| 旧 mosaic100 | baseline 0.54091、LADD 0.56841、det-only reload 0.56520 | reload 解释大部分表观收益 |
| 旧 no-mosaic | baseline 0.55654、LADD 0.57662、reload 0.57982 | 单纯 continuation 已超过 LADD |
| YOLO-init custom800，singleproj，seeds0/1/2 | 对 det-only +0.01756/−0.00136/−0.00404 | 只有 1/3 正；seed0 不代表稳定性 |
| 旧 exact400，seeds0/1/2 | plain +0.00305±0.00605；singleproj +0.00682±0.00738 | 均 2/3 正，SD 与均值同量级 |
| registered rescue，seeds0/42/123 | plain −0.08530±0.00442；singleproj −0.06874±0.00489 | 全三 seed 负，但只否定该 rescue；s42/123 复用 s0 teacher/A1 |
| custom800 机制诊断 | det 0.56022，KD-to-z 0.56416，KD-to-u 0.56717，no-probe 0.57159 | z 可迁移/u 不可学的语义未识别 |
| SiXiang full RGB-CMD，scene-clean direct300 | paired−H_S +0.010507±0.009002；逐 seed +0.01367/+0.00035/+0.01750 | 受限正 premise；已有方法，不是 novelty；n=3 t 区间约 [−0.0119,+0.0329] |
| OGSOD matched exact400，seeds42/123 | RGB-CMD−H_S −0.00262±0.00204 | paired>shuffle 仍可能输 same-modal anchor |
| SX-APR | P−H +0.002960；P−U1 −0.000070；P−F −0.008040（三 seed 全负）；六门全失败 | 不支持净 efficacy、paired attribution、gate specificity |
| MM-ARCS SiXiang | exploratory s0 +0.023670；前瞻 s42 −0.026630、s123 +0.009050 | 前瞻仅 1/2；不能事后剔除 s42 |

精确来源：

- `06_历史工程_只读/LADD_public/docs/experiments/LADD_RELOAD_CONFOUND_20260623_CN.md` L49–58：88.3%/115.9% 是当时部分仍 running 的描述性比值，不宜当普遍精确因果解释率；L7 补述 baseline 从800延至1600、reload继续改善。
- `06_历史工程_只读/LADD_public/docs/reports/LADD_PROGRESS_REPORT_POST_RELOAD_20260712_CN.md` L140–164：LADD 数字和机制；L138 明确不少历史行是 provisional classification。
- `06_历史工程_只读/LADD_public/docs/experiments/DIRECT400_REGISTERED_SEED_PANEL_FINAL_AUDIT_20260706_CN.md` L42：source reuse；L54–71：逐 seed/mean±SD；L75–85：多个协议轴同时变化。
- `04_方法演化档案/2026-07_重审与交接/LADD_0723/research_reassessment_20260723/EVIDENCE_AUDIT.md` L7–24：SiXiang premise/headroom；L30–53：强锚反证与 proxy 不转化。
- `04_方法演化档案/2026-07_重审与交接/AUDIT_CLAUDE_CODE_20260727/03_claim_evidence_audit.md` L74–84：SX-APR 六门；L95–108：MM-ARCS SiXiang。**OGSOD R2A 未 accepted analyzer** 与 **SiXiang M1 异质性结论** 是不同 campaign。

旧 VEDAI/DroneVehicle 也不是尚未尝试：7/12 报告 L119 记 VEDAI200 IR student 0.30690、RGB teacher 0.36069、LADD 0.29790；DroneVehicle RGB student best0.51053、IR teacher0.56964、LADD0.50992。它们是历史协议快照，不等于当前 RGB-T 复现矩阵。

## 3. 反复出现的流程问题

### 优化预算和真实训练身份比方法名更容易决定数字（事实）

SAR800-init+KD400 被一度视为 direct400；800 轨迹第400行与 exact400 的 LR 位置不同；baseline 延长本身明显增益。来源：`06_历史工程_只读/LADD_public/docs/experiments/DIRECT400_FAILURE_ROOT_CAUSE_ANALYSIS_20260707_CN.md` L24–35；7/12 报告 L117、L213。

### 设计、配置、真实代码曾不一致（事实）

- legacy LADD 从通用 YOLO 进入 B，冻结随机分解模块，不能叫有效历史 LADD 复现。
- FGD comparison 删除 global relation，不能叫完整 FGD。
- identity corrected feature 代数恒等；配置 foreground-only、实现 all-token。
- current singleproj 不是 r=f−z，而是 learned branches；detector raw 路径与 KD projector 分离。

来源：`04_方法演化档案/2026-07_重审与交接/AUDIT_CLAUDE_CODE_20260727/06_history_baggage.md` L16–20；7/12 报告 L160–164。

**根因推测须标注：** projector 可能自身吸收KD、高维loss stack可能过约束、BN/source/phase mismatch可能锁定差分布。原报告明确 projector bypass 未经梯度路由实验确认，不能将此推测当实证机制。

### Proxy 成功被反复误当任务成功（事实）

TCPAD 5/5 folds 可预测，但 PTSA/CATR 0/5 task gates；RIF 有 oracle paired signal 而无 downstream utility。后期又出现 coverage/solver exactness/static tests 成功但无 detector efficacy。来源：7/23 `EVIDENCE_AUDIT.md` L46–53。

### 治理复杂性也产生瓶颈（历史事实）

21-cell 物理完成但 analyzer/union 未闭环；POPA 多轮 receipt/verifier 消耗 token 尚无 detector；POTS 曾在零 image/teacher bytes 下通过旧边界。来源：`04_方法演化档案/2026-07_重审与交接/LADD_0723/research_reassessment_20260723/PORTFOLIO_PROGRESS_AND_RESULTS_20260726.md` L194–205、L234–250。不能将历史阻塞直接当2026-09-05当前状态。

### 数据与统计范围（事实）

OGSOD official test 已用于开发选择；SiXiang shipped split 有 scene overlap；teacher headroom 只说明 privileged information 存在，不说明学生自身输入可恢复。来源：7/23 `EVIDENCE_AUDIT.md` L19–24、L78–83。

## 4. CoRe-LADD：实现资产，不是已证伪的效果方法

- `06_历史工程_只读/CoRe-LADD/README.md` L7–19、L114–119 支持机制；L94–110 明确 smoke 不是正式消融，应看最终 B detector metrics。
- `06_历史工程_只读/CoRe-LADD/docs/review_logs/semreach_second_review_status.md` L94–133 只有 compile/static/tests。
- 历史修改意见的 sem/mask phase 耦合与 foreground 梯度稀释，在当前本地源码已有修正：`ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` L2652 分开 sem auxiliary，L748–754 foreground-only；`trainer.py` L956/977/1000/1041 phase scales。不要将旧修改要求当当前bug。

## 5. 值得保留的资产

1. matched det-only/reload/训练预算控制和负结果，可避免下一条方法重犯同一错误。
2. 强 H_S/full-CMD comparator、shuffled/weight0/foreground/random 控制、SiXiang受限正 premise。
3. 冻结协议、逐 seed 终态、原始args/log/checkpoint、真实loader/teacher/source追踪、negative gates。
4. 高价值反例：KD-to-u≥KD-to-z；paired>shuffle但输H_S；predictability有而utility无；GT前景控制更强。
5. CoRe 先验模块、paired loader、分层loss/phase可作基础设施，但不能继承方法claim。

本次 `Import-Csv 04_方法演化档案/hub索引文档/catalog/METHOD_CATALOG.csv` 实际为 **65行，41行 ARCHIVED_OR_NEEDS_AUDIT**。README所写51、历史总览所写55已陈旧；这些行混合方法、控制、版本和工程设施，不是65个独立完成的有效性实验。

## 6. 当前线交叉发现（交由主审计核实）

P2 probe README L7 使用 `native_rgb_s42_b64_e200`；P3昼夜 README L24 明确同名checkpoint实为VEDAI且首轮误用。若P2未重跑，则其 native→dist CKA +0.32 的对比有checkpoint身份混杂。此交叉发现已发给主审计；最终裁决以本次README和remote snapshot核验为准。
