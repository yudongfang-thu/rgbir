# LADD 阶段进度汇报：reload 混杂之后的全部实验与分析

> 状态：`current_progress_report`
>
> 时间范围：2026-06-15 至 2026-07-12 CST；对应 2026-06-18 上次进度汇报之后的工作，并向前保留 6 月 15–17 日正在运行、在 6 月 18 日汇报中可能只呈现为 progress 的实验
>
> 事实基线：`PROJECT_EXPERIMENT_AUDIT_20260710_CN.md`（本包未收录：`PROJECT_EXPERIMENT_AUDIT_20260710_CN.md`）、`../experiments/catalog/experiment_family_summary_20260710.csv`（本包未收录：`../experiments/catalog/experiment_family_summary_20260710.csv`）、RIF 显式 run records
>
> 总结边界：`legacy_LADD claim_ready=no`；`RIF-LADD claim_ready=no`

> 配置审计说明：本文中的历史分组仍需服从 `EXECUTED_RUN_CONFIG_AUDIT_20260712_CN.md`（本包未收录：`EXECUTED_RUN_CONFIG_AUDIT_20260712_CN.md`） 的逐 run 审计。run 名称或旧文档与 `args/manifest/checkpoint/code` 冲突时，以执行证据为准；无法证明的结果不得进入 matched comparison。

## 0. 汇报结论

上次汇报截止点之后，项目先完成了 clean A1B、dynamic、dynamic-probe、容量扩展、comparison 和跨数据集等大量实验，随后又逐层拆解并推翻了其中的混杂解释。6 月 15–22 日的结果不能省略，因为 reload 问题正是从这批看似积极的曲线中暴露出来的。

1. detector-only reload 可以解释旧 LADD 表观增益的 `88.3%–115.9%`，因此旧“约 +2 AP”不能归因于 LADD。
2. no-reload、YOLO-init 的旧 direct400 panel 只显示弱正向：plain 平均 `+0.00305`，current singleproj 平均 `+0.00682` AP50-95，且均只有 2/3 seeds 为正、粗略置信区间跨零。
3. registered rescue `0/42/123` 明确失败：plain `-0.08530`、singleproj `-0.06874`；reset/fusion 也没有救活。但这些实验否定的是具体 rescue/implementation 组合，不是对所有可能 LADD 设计的定理性否定。
4. FGD/LD/CMD 的 `0.55147/0.56048/0.56240` 是 SAR800-init 后再训练 400 epoch 的 transferred endpoint，有效学生预算约 1200；它们不是 YOLO-init direct400 的公平 comparison gate。
5. CCLKD YOLO11 controlled seed0 的 LLD+FLD 有 `+0.015269` 信号，但 paper reproduction 仍未通过，且单 seed 不能升级为稳定结论。
6. 外部数据结果方向敏感，DroneVehicle 和 VEDAI 都不能支持泛化 claim。
7. 7 月 11–12 日建立的 RIF-LADD 是独立重构线。单一 file-level pilot 中 RawPaired 相对 Anchor 仅 `+0.000792`，相对 RawShuffled `+0.003134`；K-convergence、response rescue 和 Q_det capacity 均未建立 SAR-only 收益，五项正式 claim 仍全部 `untested`。

因此当前最准确的项目定位是：已经完成一轮系统性的失败定位、协议审计和证据重构，但尚无可用于论文主张的稳定性能结果。

## 1. 时间线与所做实验

| 时间 | 实验阶段 | 主要问题 | 结果 | 证据等级 |
|---|---|---|---|---|
| 06-15 | mosaic100 baseline controls | n/s/m 的 SAR/RGB 对照是否齐备 | n/s/m seed0 baseline 与 checkpoint 逐步补齐；部分 RGB/m 早停 | historical baseline/progress |
| 06-15–06-16 | from-YOLO comparison 与 A2-core | 不加载 SAR trained source 是否可训练 | CMD/LD from-YOLO 与 YOLO-init A1→B(A2-core) 启动；A2-core epoch270 AP `0.49436` | progress/diagnostic |
| 06-16–06-18 | A1-A2-B、skip-A2、clean A1B | 去掉不稳定 A2 后能否提高健康度 | skip-A2/clean A1B 多机并行；旧 A1-A2-B 被归档 | historical diagnostic |
| 06-17–06-18 | static/dynamic/dynprobe | teacher core 动态更新与冻结 probe 是否有用 | dynamic 在 s 中早期明显领先 static；dynprobe 被定为当时主线 | progress；后续稳定性审计改判 |
| 06-18–06-20 | mosaic100/no-mosaic mainline | n/s/m 主线是否可形成 paper candidate | 多条 800-row 结果完成，best 约 n `0.57042`、s `0.63487`、m `0.65405` | historical candidate，后被 reload/budget 降级 |
| 06-19–06-22 | structure/loss ablations | reach、taskL、probe、teacher/student split 是否必要 | static/dynamic/wo-reach/wo-taskL 与结构队列运行；部分仅 progress/中断 | diagnostic only |
| 06-20–06-22 | comparisons | LD/CMD/FGD/HalluciDet 是否优于 LADD | transferred 与 from-YOLO 混合存在；当时表中高 endpoint 后来被确认预算不公平 | disputed then reclassified |
| 06-20–06-22 | VEDAI/DroneVehicle | 是否有跨数据集收益 | VEDAI LADD 低于 student；DroneVehicle main/dynamic 未超过 RGB student | negative diagnostic |
| 06-22 | dynamic stability audit | dynamic 是否可作为主线 | YOLO11s 4090 run best `0.63647` 后掉至 `0.60079`；dynprobe drop 仅 `0.00723` | instability confirmed |
| 06-23 | reload/no-distill controls | 旧提升是否来自方法 | reload 解释旧 gain 的 88.3%–115.9% | 已确认混杂 |
| 06-24–06-27 | YOLO-init no-reload early search | 去掉 trained SAR init 后是否仍有信号 | dynamic、singleproj、no-srec 有少量早期/长程信号；KD-to-u 也可为正 | diagnostic only |
| 06-26–07-06 | 800、row400-of-800、多 seed | seed0 亮点是否稳定，400/800 能否混算 | final800 singleproj mean delta `+0.00405`，仅 1/3 positive；row400 学习率位置不同 | 不稳定；协议必须分桶 |
| 06-30 | stronger baseline audit | 旧 SAR800 是否接近上限 | SAR1600 `0.58973/0.59140`，reload 后 `0.59575` | detector anchor |
| 07-02–07-04 | old exact direct400 audit | 无 reload 的 matched gain | plain `+0.00305±0.00605`；singleproj `+0.00682±0.00738` | weak signal |
| 07-02–07-10 | comparison audit | FGD/LD/CMD 是否是公平对照 | 三者均为 SAR800-init+KD400；缺 same-source no-KD continuation | practical endpoint only |
| 07-06 | registered rescue panel | BN freeze/slow schedule/phase-B 能否救活 | plain `-0.08530`；singleproj `-0.06874` | rescue failure |
| 07-07–07-09 | reset/fusion failure localization | 减少 aux 或融合 LD/CMD 是否修复 | best final `0.50350`，且缺 exact matched det-only | diagnostic only |
| 06-29–07-10 | CCLKD controlled/reproduction | 组件信号与论文复现 | LLD+FLD seed0 `+0.015269`；paper baseline/full 未复现 | single-seed signal / gate failed |
| 06-22–07-10 | DroneVehicle/VEDAI | 外部数据是否稳定泛化 | DroneVehicle 两方向一正一负；VEDAI 尚不完整 | no generalization claim |
| 07-11–07-12 | RIF-LADD G12/K/response/Q_det | pair signal能否转为 SAR-only 收益 | RawPaired 微弱正差；K、response、Q_det gates 未通过 | independent diagnostic line |
| 07-12 | 新统一 protocol | 未来怎样获得公平主表 | 冻结 YOLO11n、YOLO-init、exact400、SGD、batch64、no-mosaic | protocol candidate；尚无新主表 |

## 1A. 6 月 15–22 日详细实验复盘

这一阶段包含大量“正在跑”和“完成但尚未完成 provenance gate”的实验。下面按研究问题汇总，不把同一 `results.csv` 的备份、副本或阶段目录重复计成独立科学实验。

### 1A.1 6 月 15 日：baseline 和 comparison 起点

4090 本地备份中保存了 `baseline_controls/mosaic_baselines_20260615`：YOLO11n 的历史 SAR/RGB mosaic100 checkpoint，以及新训练的 YOLO11s SAR/RGB seed0。90 和 AutoDL 同期还启动了 YOLO11m baseline。它们解决了“不同容量有没有同协议 teacher/student 起点”的工程问题，但当时并没有解决训练总预算和 reload 公平性。

同日还启动 CMDistill/LD 等 from-YOLO comparison。后续同时出现另一批从 SAR baseline checkpoint 开始的 transferred comparison；当时的汇总文档未始终把这两类初始化分开，是后来 comparison 数字被误读的重要来源。

### 1A.2 6 月 16 日：A2-core、skip-A2 与旧 A1-A2-B

旧三阶段链路为 `A1→A2→B`。A2 曾出现 NaN、checkpoint selection 和阶段损伤问题，因此实验转向 `A1→B(A2-core)`、skip-A2 和 clean A1B。

AutoDL 的 YOLO-init `A1→B(A2-core)` 在 B epoch270 达到 AP `0.49436`，高于当时 YOLO-init det-only B800sched 在 epoch332 的 `0.45155`，也高于旧 YOLO-init+A2 decomp 在 epoch360 的 `0.46670`。但 SAR-baseline-best continuation 已达到 `0.57521` best，A2-best continuation best `0.55681`。

当时观察：短链路看起来恢复更健康。

事后解释：这些行在初始化历史和 effective epoch 上不匹配，只能说明 A1/A2-core 路径可优化，不能证明方法 gain。

### 1A.3 6 月 17–18 日：static、dynamic 与 dynamic-probe

三条 clean profile 被并行比较：

- `static`：B 阶段 teacher-side core 不动态更新；
- `dynamic`：teacher-side core 与 probe 路径继续变化；
- `dynamic_probe`：动态 core，但冻结/截断 probe 路径，后来被选为当时主线。

早期 matched-B-epoch 诊断显示，YOLO11s dynamic 相对 static 的 AP 差在 epoch50/100/150/200/376 分别为 `+0.06162/+0.05604/+0.06143/+0.04815/+0.04179`；dynamic KD loss 也显著更低。YOLO11n 的差值则在 `-0.00394` 到 `+0.00529` 附近，优势很小。

这说明 dynamic 的早期优势具有容量依赖性：在 s 上明显、在 n 上接近零。它不能被概括为普遍机制收益。

后续完整性审计又发现 YOLO11s dynamic 4090 partial 在 epoch656 达到 best `0.63647`，到 epoch712 降为 `0.60079`，drop `0.03568`；对应 KD loss 从 `0.05968` 升至 `0.17844`。dynprobe 完整 800 epoch best `0.63487`、last `0.62764`，drop 仅 `0.00723`。因此当时选择 dynprobe 的合理部分是稳定性，而不是已经证明 frozen probe 的因果贡献。

### 1A.4 6 月 18–20 日：当时的主线候选

当时完成或汇总的 800-row 主线候选如下：

| protocol | model | mode | best AP50-95 | 当时定位 | 当前定位 |
|---|---|---|---:|---|---|
| mosaic100 | n | dynamic_probe | `0.57042` | main candidate | historical/reload-confounded candidate |
| mosaic100 | s | dynamic_probe | `0.63487` | main candidate | historical candidate；单 seed |
| mosaic100 | m | dynamic_probe | `0.65405` | main candidate | historical candidate；容量/稳定性风险 |
| mosaic100 | n | dynamic | `0.57544` | ablation proxy | diagnostic；late stability unresolved |
| mosaic100 | n | static | `0.57113` | ablation | diagnostic |
| no-mosaic | n | dynamic_probe | `0.57433` | robustness appendix | continuation/reload context，非 matched gain |
| no-mosaic | s | dynamic_probe | `0.64073` | robustness appendix | historical single-seed diagnostic |
| no-mosaic | m | historical LADD | `0.66982` | robustness appendix | later observed severe best/last instability in related m line |

这些数字解释了为什么 6 月 18 日附近会形成“方法似乎有效”的印象；但它们跨越 mosaic/no-mosaic、容量、服务器和初始化来源，并且缺少同 pipeline detector-only reload control。6 月 23 日之后的审计将其整体降级为历史或诊断证据。

### 1A.5 6 月 19–22 日：结构和 loss 消融

这一阶段启动或收集了 `static`、`dynamic`、`wo_reach`、`wo_taskL`、`student_single_proj`、`pure_feature_kd`、`teacher_single_proj`、`no_teacher_split`、`no_probe_q` 等队列。很多 run 当时仍在中途，例如 no-probe-q 一度为 366/800、AP `0.50330`；因此不能把所有队列项都当作 final ablation。

可以保留的分析是：

1. dynamic 在 YOLO11s 早中期改善明显，但存在 late collapse；
2. dynprobe 更稳定，但没有 strict matched causal ablation 证明“冻结 probe”本身是唯一原因；
3. reach-rank loss 长期约 `0.153`，而 reach-match 很小，说明 rank 项可能处在近似平台；
4. 后续 KD-to-u、no-probe 和 shuffled-teacher 结果进一步表明，当前 z/u 和 reach/probe 语义并未被唯一识别。

### 1A.6 6 月 20–22 日：comparison 和跨数据集

当时汇总中的 OGSOD comparison 包含 LD `0.56733`、CMD-style `0.56941`、HalluciDet `0.45899`，以及尚在运行的 FGD/from-YOLO lines。这些数字来自不同初始化、进度或实现桶，不能组成公平排名。后续审计确认常被引用的 FGD/LD/CMD `0.551/0.560/0.562` 属于 SAR800-init+KD400 transferred line。

VEDAI 200 epoch 结果为 IR student `0.30690`、RGB teacher `0.36069`、LADD `0.29790`：LADD 低于 student。DroneVehicle IR→RGB 中，RGB student best `0.51053`、IR teacher `0.56964`、LADD main `0.50992`、dynamic partial `0.50545`、no-reach partial `0.48553`。内部 loss 没有爆炸，更接近无效迁移或负迁移，而非纯数值故障。

## 1B. 已关机服务器的本地证据清点

| 原服务器/来源 | 本地路径 | results.csv 数 | 体积 | 主要内容 |
|---|---|---:|---:|---|
| 4090 关机快照 | `_remote_backups/4090_20260618` | 171 | 2.3 GB | 6月18日前 clean/A1A2B、baseline、CCLKD、日志与权重 |
| AutoDL/4090 6月22取证 | `tmp/remote_evidence_20260622` | 58 | 457 MB | 6月17–22日 clean mainline、ablation、baseline provenance；基本不含权重 |
| 3090 最终迁移包 | `/Users/yudongfang/Desktop/光sar/LADD_3090_migration_20260710_224458` | 159 | 9.8 GB | direct400、custom800、rescue/fusion 及运行证据 |
| 汇总 artifact 备份 | `/Users/yudongfang/Desktop/光sar/LADD_experiment_artifacts_20260704_153834` | 452 | 11 GB | 多服务器聚合的 results/args/log/manifest；含重复 alias |
| 后续 AutoDL snapshot | `debug/remote_snapshots/` | 单独管理 | — | no-reload 等关机前快照 |
| 90 服务器证据副本 | `docs/experiments/archive_legacy_ladd_20260618/.../raw/90`、`tmp/remote_queue_90_*` 及 artifact backup | 分散 | — | 旧 A1/A2/B、mosaic baseline、comparison queue |

上述计数是物理 `results.csv` 文件数，不是独立实验数。相同曲线可能同时存在于服务器快照、证据包、文档 raw 目录和迁移包中；后续若建立 run-level 总表，必须按结果 hash 加 provenance alias 去重，不能直接相加。

## 2. Legacy LADD 原始结果表

以下 AP 均为 AP50-95；不同协议之间不得计算 matched gain。

本表是重点结果的人工摘要，不代表其中每一行已经通过完整 provenance gate。逐 run 机器审计已发现大量 missing args、partial、resume 和名称/实际初始化冲突；未完成 checkpoint/code 审计的行均应理解为 provisional historical classification。

| 协议/实验 | 方法 | seeds | 初始化与有效预算 | 结果/增量 | 判断 |
|---|---|---|---|---:|---|
| historical mosaic100 | LADD vs baseline | 0 | mixed/SAR source，预算不清 | `0.56841 vs 0.54091` | 被 reload 混杂 |
| historical mosaic100 | det-only reload | 0 | trained SAR continuation | `0.56520` | 解释 88.3% 表观 gain |
| historical no-mosaic | LADD vs baseline | 0 | mixed/SAR source，预算不清 | `0.57662 vs 0.55654` | 被 reload 混杂 |
| historical no-mosaic | det-only reload | 0 | trained SAR continuation | `0.57982` | reload 高于 LADD |
| YOLO-init 800 | det-only | 0 | source0 + run800 | `0.54944` | 长程 control |
| YOLO-init 800 | singleproj | 0/1/2 | source0 + run800 | deltas `+0.01756/-0.00136/-0.00404` | mean `+0.00405`，1/3 positive |
| legacy YOLO-init exact400 | det-only | 0/1/2 | source0 + run400 | mean `0.51558` | matched control |
| legacy YOLO-init exact400 | plain | 0/1/2 | source0 + run400 | mean `0.51863`; delta `+0.00305±0.00605` | weak, unstable |
| legacy YOLO-init exact400 | current singleproj | 0/1/2 | source0 + run400 | mean `0.52240`; delta `+0.00682±0.00738` | weak, unstable；非 identity residual |
| registered rescue400 | det-only | 0/42/123 | YOLO init，但 control 不完全同 BN/source | mean `0.50737` | rescue control |
| registered rescue400 | plain | 0/42/123 | phase source + rescue wrapper | mean `0.42207`; delta `-0.08530` | failed |
| registered rescue400 | singleproj | 0/42/123 | phase source + rescue wrapper | mean `0.43863`; delta `-0.06874` | failed |
| reset/fusion400 | best: single+LD replace | 0 | phase source + run400 | final `0.50350` | 缺 exact matched control |

统计解释：旧 direct400 的正均值规模很小，种子间方差与均值同量级，不能支持“稳定提升”；registered rescue 的大幅负值则从早期就出现，更像具体加载、BN、schedule、loss stack 和 phase source 的组合失败。

## 3. 机制诊断

custom800 seed0 诊断中，det-only 为 `0.56022`：KD-to-z `0.56416`，KD-to-u `0.56717`，shuffled teacher `0.55402`，student-z/no-probe `0.57159`。另一些 alpha/EMA 设置下 z/u 排名还会反转。

观察：正确配对可能含有信号，因为 shuffled teacher 更差。解释：KD-to-u 有时不弱于 KD-to-z，说明当前 learned dual-branch decomposition 没有识别出“z 可迁移、u 不可学习”的预期语义。含义：不能把 singleproj 或 KD-to-z 的结果写成 identity residual 或语义分解验证。

当前 student 实际实例化 `StudentMimicResidualBlock`，z/r 是两个独立可学习分支并带 learned reconstruction；`single_proj` 只是使用 z branch 做 KD 并默认关闭 student reconstruction，不是 `r=f-z`。默认 detector 读取 raw SAR neck feature，KD 主要约束 projector，因此存在 projector bypass 的强推断，但尚未通过梯度路由实验成为已确认根因。

## 4. Baseline、comparison 与 CCLKD

| 家族 | 方法 | 初始化/预算 | AP50-95 | 正确用途 |
|---|---|---|---:|---|
| stronger detector | SAR1600/reload | YOLO→1600 或 trained continuation | `0.58973–0.59575` | 证明旧 SAR800 不是上限 |
| transferred comparison | FGD | SAR800→KD400，effective≈1200 | `0.55147` | practical endpoint |
| transferred comparison | LD | SAR800→KD400，effective≈1200 | seed0 `0.56048` | practical endpoint；provisional source delta |
| transferred comparison | CMD | SAR800→KD400，effective≈1200 | seed0 `0.56240` | practical endpoint；provisional source delta |
| CCLKD controlled400 | LLD+FLD | YOLO-init400，legacy auto/AMP | `0.530729`，delta `+0.015269` | seed0 component signal |
| CCLKD paper400 | full | YOLOv5x paper-style | approx `0.447` | reproduction gate failed |

FGD/LD/CMD 相对 source checkpoint 的 provisional delta 分别约为 `-0.00507`、`+0.00373±0.00308`、`+0.00578±0.00152`。由于缺少同 source、同 schedule 的 detector-only continuation，这些差值仍不能完全归因于 KD。

## 5. 外部数据实验

| 数据/方向 | baseline | 方法 | delta | 结论 |
|---|---:|---:|---:|---|
| DroneVehicle IR→RGB | 0.51007 | 0.50664 | -0.00343 | 负向 |
| DroneVehicle RGB→IR | 0.56889 | 0.57114 | +0.00225 | 微弱正向 |
| DroneVehicle IR→RGB sub2k | 0.35385 | 0.35244 | -0.00141 | 负向 |
| VEDAI CMD native | RGB baseline mAP50 0.6919 | all-KD partial 0.7292 | 不计算正式 delta | split/final checkpoint 未统一 |

方向改变后符号改变，且证据不完整，因此不能声称跨数据集泛化。

## 6. RIF-LADD 独立实验线

RIF-LADD 不是 legacy LADD 的 patch，也不能继承其结果。以下为 7 月 11–12 日按显式 record 产生的独立诊断。

| 实验组 | 原始结果 | 观察 | 含义 |
|---|---:|---|---|
| G12 Anchor | `0.5309706644` | SAR-only anchor | file-level pilot control |
| G12 RawPaired | `0.5317630478` | vs Anchor `+0.0007923835` | 信号过小，未支持 C1 |
| G12 RawShuffled | `0.5286290995` | Paired-Shuffled `+0.0031339484` | 单 fold/seed 配对信号，未支持 C2 |
| K16 vs K32 | feature relative RMSE `0.1721` | 阈值要求 ≤0.1 | target convergence fail |
| independent K32 vs primary K32 | feature relative RMSE `0.2429` | bank-sensitive | target convergence fail |
| response rescue | `0.530495–0.530957` | 均未超过 Anchor | response loss/联合训练未救活 |
| selected optimizer | Full `0.530547`; Centered `0.526938` | surrogate loss 改善未转成 AP | centered 更差 |
| Q_det seeds 0/42/123 | `0.530836/0.530190/0.530523` | mean delta `-0.000454` | `inconclusive_or_sub_mwe` |

当前 RIF 使用 file-level candidate split，仍缺可信 scene/acquisition ID、registration provenance 和跨 fold duplicate/overlap 审计。RIF-C1 至 RIF-C5 仍全部 `untested`；已观察到的 pilot 数字不得自动升级 claim。

登记层面还有一个单独问题：结果汇总记录推导为 48 records/39 verified，而 `project_state.json` 仍是 45/33，派生 registry 尚未同步。它影响导航一致性，不产生新的科学结论。

## 7. 当前能说与不能说

可以说：旧 reload-based gain 已被量化为严重混杂；legacy direct400 只有弱且不稳定的正信号；registered rescue 和 reset/fusion 没有成功；两个不同诊断提示正确 RGB/SAR 配对可能含有信号，但尚未稳定转化为 SAR-only 检测收益；当前实现与 identity-residual 设计不一致。

不能说：LADD 稳定提升约 2 AP；LADD 已公平优于或劣于 FGD/LD/CMD；`0.551/0.560/0.562` 是 direct400 公平 floor；z/u 语义已验证；CCLKD paper reproduction 成功；外部数据证明泛化；RIF-LADD 已提升 AP、成功恢复 pair interaction 或优于 Q_det。

## 8. 下一阶段建议

传统 comparison 主线已经在 `../experiments/OGSOD_YOLO11N_NOMOSAIC400_SGD_B64_PROTOCOL_20260712_CN.md`（本包未收录：`../experiments/OGSOD_YOLO11N_NOMOSAIC400_SGD_B64_PROTOCOL_20260712_CN.md`） 冻结为 YOLO11n、YOLO init、exact400、SGD、batch64、no-mosaic。当前尚没有按该新协议完成的公平主表。

如果继续 legacy LADD，优先统一真实 runtime source、实现真正 identity residual 或重命名 current singleproj、完整保存 provenance，并先做 det/KD 梯度路由和 exact config diff。若继续 RIF，应先同步 registry，并解决 scene/group、registration、duplicate/overlap 和 target convergence；这些门未通过前不应扩成大规模性能实验。

## 9. 给老师的口头汇报版本

> 上次汇报后，我们首先验证了加载训练好权重带来的虚假提升。加入 detector-only reload control 后发现，reload 本身可以解释旧 LADD 提升的 88% 到 116%，所以旧的约 +2 AP 结论不能保留。
>
> 随后我们把实验严格拆成 YOLO 冷启动 400、800、800 中途第 400 行、1600/reload 和 SAR800 continuation 等协议。最干净的旧 direct400 三种子中，LADD 只有约 +0.3 到 +0.7 AP 的弱平均信号，且不稳定；后来注册的 rescue 三种子则明显失败。FGD、LD、CMD 的较高数值其实使用了 SAR800 权重再训练 400 epoch，也不能和 direct400 公平比较。
>
> 我们还检查了分解语义、配对打乱、CCLKD、外部数据和新的 RIF-LADD 方案。两条不同诊断都提示正确 RGB/SAR 配对可能有一点信号，但目前没有证据表明它能稳定转化为 SAR-only 检测收益。当前成果主要是完成了混杂排除、协议重建和失败定位，性能 claim 仍然暂停。

## 10. 证据入口

- 全项目冻结审计：`PROJECT_EXPERIMENT_AUDIT_20260710_CN.md`（本包未收录：`PROJECT_EXPERIMENT_AUDIT_20260710_CN.md`）
- reload 混杂：`../experiments/LADD_RELOAD_CONFOUND_20260623_CN.md`（本包未收录：`../experiments/LADD_RELOAD_CONFOUND_20260623_CN.md`）
- 400/800 分桶：`../experiments/LADD_400_800_PLAIN_SINGLEPROJ_BASELINE_ANALYSIS_20260702_CN.md`（本包未收录：`../experiments/LADD_400_800_PLAIN_SINGLEPROJ_BASELINE_ANALYSIS_20260702_CN.md`）
- direct400 matched audit：`../experiments/DIRECT400_FINAL_FACT_MATCHED_GAIN_ANALYSIS_20260704_CN.md`（本包未收录：`../experiments/DIRECT400_FINAL_FACT_MATCHED_GAIN_ANALYSIS_20260704_CN.md`）
- registered rescue：[`../experiments/DIRECT400_REGISTERED_SEED_PANEL_FINAL_AUDIT_20260706_CN.md`](../experiments/DIRECT400_REGISTERED_SEED_PANEL_FINAL_AUDIT_20260706_CN.md)
- 新统一协议：`../experiments/OGSOD_YOLO11N_NOMOSAIC400_SGD_B64_PROTOCOL_20260712_CN.md`（本包未收录：`../experiments/OGSOD_YOLO11N_NOMOSAIC400_SGD_B64_PROTOCOL_20260712_CN.md`）
- 旧 RIF 结果汇总（已归档）：`../../archive/rif_ladd_legacy_rgb800_20260712/source/docs/RIF_LADD_EXPERIMENT_RESULTS_20260712_CN.md`（本包未收录：`../../archive/rif_ladd_legacy_rgb800_20260712/source/docs/RIF_LADD_EXPERIMENT_RESULTS_20260712_CN.md`）
- 新 clean RIF 入口：`/Users/yudongfang/Desktop/光sar/ogsod400_clean_protocol/methods/rif_interaction/README_CN.md`（本包未收录：`/Users/yudongfang/Desktop/光sar/ogsod400_clean_protocol/methods/rif_interaction/README_CN.md`）
- 已执行实验真实配置审计：`EXECUTED_RUN_CONFIG_AUDIT_20260712_CN.md`（本包未收录：`EXECUTED_RUN_CONFIG_AUDIT_20260712_CN.md`）
- 逐 run 配置表：`../experiments/catalog/executed_run_config_audit_20260712.csv`（本包未收录：`../experiments/catalog/executed_run_config_audit_20260712.csv`）
