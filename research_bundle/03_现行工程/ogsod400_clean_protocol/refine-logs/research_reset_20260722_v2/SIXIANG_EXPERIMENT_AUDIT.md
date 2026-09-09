# SiXiang 实验审计：按 split、协议和 campaign 分层

## 1. 数据层

源数据为 3,182 组三模态 512×512 chips。shipped split 存在大规模 scene overlap，永久弃用。当前
coarse10 HBB split 由 138 scenes 按 seed42 重建为 train/val/test=110/14/14 scenes，图像
2305/304/573，scene ID 零交叉。

尚未闭合 acquisition/site、near-duplicate 和历史 test-access，因此 val 最高只算 development；test
在本轮方法选择中保持 untouched，但在正式使用前仍需 outcome-free integrity audit。

## 2. 协议分区

| 分区 | epochs/imgsz/batch | 地位 |
|---|---|---|
| shipped split | 任意 | invalid，所有结果禁用 |
| scene-clean direct100 | 100/512/64 | legacy unfrozen/off-protocol exploratory WIP，不与 direct300 合表 |
| scene-clean direct300 | 300/512/64 | 当前 frozen development protocol |
| A1–A7 | direct300 意图，多次 runtime/queue 工程修订 | 工程 provenance，不是七次科学 replication |
| A8+A9 | direct300 runtime-v8 + two-cell repair | 当前唯一 hash-verified full-CMD matched composite |
| old unified u300 | direct300 意图，协议 ID 为 `legacy_unfrozen_sixiang_u300_runtime_unknown` | physical outputs 可发现，但不得与 frozen direct300 按 protocol_id 聚合 |
| SX-APR B1/B2 | direct300，独立 runtime | B1 工程失败；B2 活跃且 outcome embargoed |

## 3. Baselines

direct300 文档和远端 teacher pool 给出：

| arm | s0 | s42 | s123 | mean ± sample SD |
|---|---:|---:|---:|---:|
| SAR | 0.38941 | 0.38721 | 0.39274 | 0.38979 ± 0.00278 |
| RGB | 0.65708 | 0.63626 | 0.63988 | 0.64441 ± 0.01112 |

RGB−SAR 历史差值为 +0.25462 AP50-95。这些远端 roots **提示** SiXiang 可能是高 headroom
开发集，但在 lineage 闭合前该 claim 仍为 `pending/observation`；它更不证明 RGB 信号可被
某个 KD 方法稳定转给 SAR。`registry/runs.csv` 的 7 条 baseline/smoke 行发生字段语义错位，且
缺 args/manifest/code/data/init lineage；因此在补齐 physical rows 前，结果由远端 canonical roots 与
A8 teacher locks 支撑，不能仅以 registry 为 authority。

## 4. Direct100 premise panel

SAR/RGB/H_S/full RGB-CMD/H_F/H_R/H_O ×3 共 21 路完成。主要 endpoint observations：

| arm | AP50-95 mean ± SD | relative to H_S mean ± SD |
|---|---:|---:|
| H_S | 0.37708 ± 0.00798 | — |
| full RGB-CMD | 0.38345 ± 0.00296 | +0.00637 ± 0.01067 |
| H_F | 0.38844 ± 0.01357 | +0.01135 ± 0.00872 |
| H_R | 0.36769 ± 0.00262 | -0.00939 ± 0.00996 |
| H_O | 0.37768 ± 0.00668 | +0.00059 ± 0.00978 |

direct100 欠训练、方差大，且 immutable protocol/provenance 不完整。这些只用于提出 direct300 问题；
不能说 H_F 稳定有效，更不能与 OGSOD direct400/direct50 横拼。

## 5. Old unified u300：物理完整不等于身份有效

L20 顶层共有 73 个 `sixiang_u300_*` root，全部没有 registry 行：

- 33 个主根表面有 301-line results、args、last；
- 4 个主根不完整（LADD-singleproj 的 5/0/0 行副本与 LCSR-s42 0 行）；
- 36 个是 exit/failed/hung/superseded 工程快照；其中有些也可能有完整表或重复行，但永不晋级科学格。

当前只可作 discovery 的 endpoint summary 是：

| arm | mean ± sample SD | delta vs H_S | 身份边界 |
|---|---:|---:|---|
| H_S | 0.403453 ± 0.008412 | — | old runtime，非 A8 provenance |
| full RGB-CMD | 0.413960 ± 0.002567 | +0.010507 ± 0.009002 | launcher intent only |
| H_F | 0.419993 ± 0.030937 | +0.016540 ± 0.035970 | 极不稳定；不是方法 |
| H_R | 0.403103 ± 0.015541 | -0.000350 ± 0.022579 | 近零 |
| H_O | 0.392253 ± 0.009342 | -0.011200 ± 0.017635 | discovery only |
| FGD-focal-only | 0.397890 ± 0.012173 | -0.005563 ± 0.014839 | adaptation/identity unverified |
| LD adaptation | 0.386800 ± 0.005690 | -0.016653 ± 0.010960 | 3/3 negative，identity unverified |
| LADD plain | 0.396777 ± 0.004503 | -0.006677 ± 0.004453 | 3/3 negative |
| LADD singleproj | 0.395783 ± 0.013831 | -0.007670 ± 0.020267 | duplicate/incomplete roots exist |
| MM-ARCS P3/far | 0.405483 ± 0.017741 | +0.002030 ± 0.025874 | signs mixed |

这些值不能进 matched table，原因不是“数值不好”，而是：

1. 36 个逻辑单元零 registry，attempt/quarantine/retry lineage 不完整；
2. checked-in premise trainer 甚至不接受 `direct300` run-mode，说明依赖未冻结的远端手工状态；
3. v2 queue 仅以 results 行数判 done，并曾杀进程/改名/重启；
4. args 缺完整 teacher/KD provenance，旧 runtime 被 A8-v8 文档显式排除；
5. `lcsr_v1` 与 `ladd_plain` 的有效 argv 相同，`reach-input-mode adapter` 已是全局默认，且从未调用
   `method/` 的 LCSR 实现；已有 s0/s123 saved model state 与 LADD plain 相同，不是独立 LCSR；
6. `ladd_identity` 的一版实际是 H_S-like CMD，另一版在 route none 时传 SAR teacher，与 checked-in
   validation contract 冲突；它不是稳定 no-op identity。

因此 old u300 的正确标签是 `completed_unregistered_unvalidated` 或 engineering history，而不是“旧方法
已经完整重做”。

## 6. A8+A9：唯一 fresh validated full RGB-CMD matrix

A8 贡献 10 格；A9 只替换 A8 的 `paired_s0` 与 `weight0_s123` 两个工程失败格。14 个物理 attempt
仍只构成 12 个 logical cells、每 arm 3 个 optimizer seeds；A9 不增加 replication。source lock、
validator 与 COMMIT-last bundle hash 匹配。

A8+A9 的 H_S/full RGB-CMD 三个 endpoint 与 old u300 历史表逐 seed 相同，但不是同一组
`results.csv`：远端六对文件 SHA256 均不同，路径也分属 `sixiang_u300_*` 与 A8/A9
新 root。在排除 time 列后，大部分轨迹一致、少数 epoch 的 validation metrics 有差异；最合理的
解释是相同 seed/config 的确定性 fresh execution，但无法从现有证据唯一确定原因。因此
A8+A9 的价值是 provenance closure，不能和 old u300 叠加成六次独立 replication。

| frozen gate | endpoint deltas s0/s42/s123 | mean ± sample SD | verdict |
|---|---|---:|---|
| paired − H_S | +0.013670 / +0.000350 / +0.017500 | +0.010507 ± 0.009002 | PASS |
| paired − strict shuffle | +0.025890 / +0.071200 / +0.072820 | +0.056637 ± 0.026640 | PASS |
| paired − weight0 | +0.032000 / +0.010780 / +0.006750 | +0.016510 ± 0.013565 | PASS |

支持：SiXiang val direct300 下，full paired RGB-CMD 对 H_S 有窄 development gain，并依赖正确 pairing
与非零 KD dose。注意 seed42 对 H_S 仅 +0.00035，整体伴随 precision/recall trade-off。

不支持：full RGB-CMD 是新方法；该信号可泛化到 OGSOD/test；已识别 shared/private factor；可发论文。

## 7. 哪些旧方法没有在 SiXiang 上完全重做

达到 A8 等级 fresh freeze、hash-bound runtime、matched controls 和 commit-last analysis 的只有 H_S 与
full RGB-CMD 概念，以及 strict shuffle/weight0 controls。以下都**没有** A8-grade end-to-end 重做：

- H_F、H_R、H_O；
- FGD、LD；
- LADD plain、singleproj、identity；
- LCSR-v1（实际连 LCSR 都没运行）；
- MM-ARCS P3/far；
- DCR、PG-CMD、RIF/CFID、TCPAD、PTSA、CATR、RMA-CMD 等其它历史路线。

“没有重做”不等于现在应全部重跑。只有在新 claim ledger 中仍有独立可证伪价值的版本才获得新 freeze；
否则大矩阵只是重复制造不可解释结果。

## 8. SX-APR B1/B2

- B1：首格在科学 outcome 前因 import 工程错误退出，其余未运行；只能写 engineering failure。
- B2：修订后 7 arms×3 seeds。2026-07-23 00:45 CST snapshot 为 10 completed、2 running、9 pending；
  已启动格 attempts 全为 1，两个运行格 results/diagnostics 为 50/54 行，serial gate、
  args/last、进程与错误扫描健康。21/21 前所有指标继续 embargo。
- B2 的 `F` 是 label-only same-mass gate control，不是 legacy `H_F`。

更重要的是：B2 矩阵没有 same-campaign full RGB-CMD arm。即使它通过 P−H/P−A/P−U/P−R/P−F，
也只证明 SX-APR 相对自己的 anchor/controls；不能证明它胜过目前 SiXiang 最强且已验证的 existing method
full RGB-CMD。若 B2 存活，下一次 confirmation 必须把 full RGB-CMD 放入同一 freeze，而不能拿 A8+A9
跨 temporal/runtime 的均值作论文级 superiority comparator。

## 9. SiXiang 当前结论

**支持**：A8+A9 full RGB-CMD 的窄 efficacy/pairing/dose development signal。

**待对账**：scene-clean baseline headroom；历史 roots 数字只允许作 pending observation。

**不支持**：H_F 是方法；旧方法已正式复现；SX-APR 有效；任何稳定新方法或 paper-ready claim。

**决策：`DEFER`**，等待 B2 terminal validator + commit-last bundle。若 B2 通过，下一步仍是带 full
RGB-CMD 的新 confirmation freeze 与 untouched scene-group evaluation；若失败，按冻结版本 KILL/REDESIGN，
不做 rescue sweep。
