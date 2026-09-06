# 3. 实验总表与结果

## 3.1 如何理解“所有实验”

本项目有三个粒度：

- **physical run**：一次训练/评估 attempt，失败、smoke、OOM、retry 都保留；登记在 `registry/runs.csv` 或远程目录。
- **campaign**：一组能共同解释的矩阵；本章以 14 个 campaign 为主线。
- **claim**：可写入报告的句子；一个 campaign 完整不等于 claim 成立。

L20 `/private/results` 在 reset-v2 时有 241 个顶层条目，其中包含 30 个 MM-ARCS confirmation 条目、24 个 R2A 条目、73 个 SiXiang u300 历史根、多批 smoke/失败/证据目录。逐条清单在 `catalog/remote_results_manifest.csv`；不要按目录数统计科学实验数。

## 3.2 Campaign 总览

| ID | 数据/协议 | 内容 | 状态/完整性 | 结果访问 | 当前决策 |
|---|---|---|---|---|---|
| OG-BL-400 | OGSOD exact400 | SAR/RGB baselines seeds42/123 | complete/PASS | unlocked | DEFER，开发基线 |
| OG-OLD-400 | OGSOD 多历史协议 | legacy LADD/LCSR/LD/CMD 等 | excluded/WARN | unlocked | KILL/历史观察 |
| OG-CMD-MATCHED-400 | OGSOD exact400 w2 | full RGB-CMD vs H_S matched panel | complete/PASS | unlocked | REDESIGN |
| OG-HFHS-50 | OGSOD direct50 | H_F vs H_S source swap | complete/PASS | unlocked | KILL H_F claim |
| OG-DIAGNOSTIC-100 | OGSOD 多 direct100 | DCR/PG/TCPAD/PTSA/CATR/RIF | excluded/WARN | unlocked | KILL/diagnostic only |
| OG-R2A-5 | OGSOD exact400 w2 | MM-ARCS 5-arm seed42 | complete/unvalidated | embargoed | DEFER |
| OG-R2A-CONF | OGSOD exact400 w2 | MM-ARCS H/A/P/U confirmation | complete/unvalidated | embargoed | DEFER |
| SX-SHIPPED | SiXiang shipped split | split audit | FAIL | N/A | KILL |
| SX-BL-300 | SiXiang scene-clean direct300 | SAR/RGB baselines 3 seeds | complete/WARN | unlocked | DEFER lineage |
| SX-PREMISE-100 | SiXiang direct100 unknown | teacher-source premise panel | complete/WARN | unlocked | DEFER |
| SX-U300-OLD | SiXiang historical u300 | 多方法 sweep，73 roots | excluded/WARN | unlocked | REDESIGN |
| SX-A8A9 | SiXiang direct300 | full RGB-CMD attribution 4×3 | complete/PASS | unlocked | EXPLORATORY_PILOT premise |
| SX-APR-B1 | SiXiang direct300 | SX-APR first attempt | engineering fail | N/A | DEFER_ENGINEERING |
| SX-APR-B2 | SiXiang direct300 | SX-APR 7×3 recovery | running/PENDING | embargoed | EXPLORATORY_PILOT active |

CSV 的结果状态与本章有冲突时，以更高层 primary artifact/validator 为准；不要手工“修正”原始 evidence。

## 3.3 OGSOD baseline 与 RGB headroom

formal seeds 仅 42/123：

| 模态 | exact400 AP50-95 mean ± sample SD | 解释 |
|---|---:|---|
| SAR | 0.488250 ± 0.000212 | 部署 baseline |
| RGB | 0.587555 ± 0.000177 | privileged teacher upper reference |
| RGB−SAR | +0.099305 | 大 headroom，但不等于可蒸馏 headroom |

历史 seed0：SAR 0.49003、RGB 0.58105，只作上下文，不进入当前 formal aggregate。

**完整性**：结果文件与 current ledger 对齐；但 OGSOD test 反复作为 val，证据上限为 contaminated development。

**支持**：RGB detector 在该开发设置明显强于 SAR。  
**不支持**：任意 RGB distillation 都能把这 9.93 AP points 转给 SAR student。

## 3.4 OGSOD matched CMD teacher-source panel

exact400 matched-v2，workers=2，full RGB-CMD 与 H_S 三 seed描述性汇总；严格 same-seed formal 对比按 current seeds42/123：

| 对比/arm | AP50-95 或 delta | 证据边界 |
|---|---:|---|
| full RGB-CMD | 0.50478 ± 0.00027，n=3 descriptive | existing method |
| H_S | 0.50867 ± 0.00259，n=3 descriptive | strong anchor |
| full RGB-CMD − H_S | −0.00262 ± 0.00204，n=2 formal | paired 未胜强锚点 |
| paired − strict shuffle | +0.03053，seed42 only | 正确配对重要，但不能证明净 efficacy |
| RGB feature-only − SAR source | −0.01206 | raw RGB feature source有害 |
| RGB relation-only − SAR source | +0.00138 | 很小，诊断性 |
| RGB output-only − SAR source | +0.00189 | 很小，诊断性 |

**关键结论**：`paired > shuffled` 与 `paired < H_S` 可以同时成立。前者说明错配 RGB 有害/正确配对重要，后者说明在 matched strong anchor 下没有净 RGB 增益。因此 `C-OG-RGBCMD-BEATS-HS` 为 unsupported。

## 3.5 H_F vs H_S direct50 source-swap

formal seeds42/123：

| seed | H_F−H_S endpoint AP50-95 |
|---:|---:|
| 42 | −0.00885 |
| 123 | −0.00978 |
| mean ± SD | −0.009315 ± 0.000658 |
| late10 mean ± SD | −0.008422 ± 0.000824 |

历史 seed0 为 −0.01359，只作上下文。

**支持**：在 direct50 CMD 条件下，直接把 P3/P5 feature source 从 SAR 换成 paired RGB 对两个 formal seed 都有害。  
**不支持**：所有 RGB feature distillation 都有害；不能跨 schedule/机制推广。  
**决策**：`KILL` 把 H_F 当方法的路线。

## 3.6 OGSOD 历史 exact400 与比较方法

以下主要是 seed42 历史记录，协议/身份/CRC 并不完全齐，不能组成新的 matched main table：

| 方法 | AP50-95 | 相对 SAR seed42 | 完整性/结论 |
|---|---:|---:|---|
| SAR baseline | 0.48840 左右（历史同期） | — | 对应历史行；正式 current aggregate 见 3.3 |
| legacy LADD diagnostic | 0.48872 | +0.00032 | identity 有问题，不能称有效 LADD |
| LADD identity | 0.48851 | +0.00011 | control |
| LCSR v1 | 0.49469 | +0.00629 | 小于 +0.010 gate，KILL v1 |
| LD adaptation | 0.49715 | +0.00875 | checkpoint CRC fail，历史观察 |
| CMD feature-only | 0.49645 | +0.00805 | component observation |
| CMD output/logit-only | 0.49275 | +0.00435 | component observation |
| old full CMD | 0.51024 | +0.02184 | checkpoint CRC fail；已被 matched panel supersede |
| CCLKD seed0 | 0.501666 | +0.01164 vs seed0 | online teacher、seed/protocol 不同，不可池化 |

不能因为 old full CMD 数值较高而覆盖后来的健康 matched panel。证据优先级要求使用可验证的 current matched result。

### LCSR gap-detach 短门历史

为定位 LCSR v1 的 gradient conflict，曾将 gap loss 对 backbone detach。以下是不同 budget/schedule 的短门，不能与 exact400 主表直接比较：

| probe | 当时 AP50-95 | 关键同期差值 | 结论 |
|---|---:|---|---|
| seed42 direct50 e50 | 0.29253 | vs SAR +0.02359；vs CMD +0.00100；vs shuffle +0.00695 | 单点接近 CMD，但 pair attribution 不足 0.01 gate |
| seed42 direct100 e36 | 0.24189 | vs SAR +0.01609；vs CMD −0.00674；vs shuffle +0.00604 | 扩预算后落后 CMD |
| seed123 direct50 e36 | 0.25145 | vs SAR +0.02383；vs shuffle +0.00187 | 无 seed123 matched CMD |

这些 probe 揭示 gap-to-backbone conflict，但 raw target 含 `-rho*z_sar` shortcut，且 shuffled 分支也能成为 SAR-only adapter。因此它们只用于失败定位，不能升级为 LCSR efficacy。

## 3.7 RIF、DCR、PG-CMD 与 task diagnostics

### RIF

P4 oracle：

| gate | seed42 | seed123 | 结论 |
|---|---:|---:|---|
| G1 paired oracle − anchor | +0.012504 | +0.006091 | 通过局部 headroom |
| G2 paired oracle − shuffle | +0.009838 | +0.003317 | 通过 pair signal |

下游 seed42：

| gate | delta | verdict |
|---|---:|---|
| G3 Center − anchor | +0.000685 | FAIL |
| G4 Center − FullResidual | −0.000030 | FAIL |
| G5 Center − Qdet | −0.007775 | FAIL |

RIF 证明“oracle 中存在 paired signal”但没有证明 SAR-only projector 能利用它。当前版本 KILL。

### 其他 diagnostics

| 路线 | 结果 | 结论 |
|---|---|---|
| DCR P5 | whole-level main effect −0.003335 | KILL |
| PG-CMD | 0.36233 vs vanilla 0.36541 | reliability gate 未改善，KILL |
| TCPAD | 5/5 OOF folds；paired-over-prior improvement 7.535%–8.099%，familywise CI lower 6.919%–7.456%；每折胜 19/19 shuffle | 只支持 SAR-context predictability |
| PTSA | 0/5 folds pass | task-safe auxiliary claim失败 |
| CATR/task replay | 0/5 pass | predictor不能升级为 task utility |
| TCI | paired eligible 0/16 | 版本未建立可用 route |
| RMA | reachability/mediation部分成立，task advantage失败 | 不能叫可迁移知识 |
| SAFR | implementation/review failure | 无 detector科学结果 |

## 3.8 MM-ARCS R2A 与 confirmation

### 已知工程事实

- R2A 五臂 seed42 exact400 物理训练完成；
- H/A/P/U confirmation 共 12 个 physical runs 完成；
- 当前 formal inference 只允许 seeds42/123，seed0 为历史上下文；
- input/runtime/campaign/shuffle locks、q0/no-op、full-loader smoke 等工程链存在；
- R2A epoch50 选择阶段曾有 P3/far：endpoint +0.00597、late10 +0.005163，是当时唯一 exploratory candidate。

R1 版本预设 `rho>=0.99`，被真实 augmentation 后 exposure 否定；R2A 只修订该工程 estimand 为自然 `beta=0.025*rho` 与 `rho>=0.80`，没有改 q、atom mass 或四原子定义。后续多次 recovery/dispatch/analyzer repair 目录是工程 provenance，不是额外科学 replication；尤其 analyzer recovery 不得在看 outcome 后反向修改 estimand。

### 不能报告的内容

accepted exact400 analyzer 缺失/历史 analyzer 被 invalidated，因此：

- 不报告 exact400 五臂排名；
- 不报告 H/A/P/U efficacy 或 RGB attribution；
- 不从 raw `results.csv` 手算一个“临时结论”；
- 不因为物理训练完成而升级 claim。

**决策**：`DEFER`。最小下一步是 fresh review analyzer + immutable terminal bundle，而不是重跑或挑一个看起来最好的 arm。

## 3.9 SiXiang split 与 baselines

### split audit

- shipped split 有 scene overlap：`SX-SHIPPED = KILL`；
- scene-clean coarse10：110/14/14 scenes，2,305/304/573 images；
- val 为 development，test untouched。

### direct300 baseline

| 模态 | AP50-95 mean ± SD，3 seeds | 状态 |
|---|---:|---|
| SAR | 0.38979 ± 0.00278 | remote roots存在，registry lineage未完全 reconcile |
| RGB | 0.64441 ± 0.01112 | 同上 |
| gap | +0.25462 | 高 headroom，但只是 observation |

该 gap 说明 SiXiang 是有潜力的开发 testbed，不说明增量可被蒸馏。`SX-BL-300` 为 WARN/DEFER，接管后应先对 canonical baseline roots/args/checkpoints 做 lineage reconciliation。

## 3.10 SiXiang direct100 premise panel

这组协议短、runtime/provenance 不完整，只能 exploratory：

| arm | AP50-95 mean ± SD | 相对 H_S |
|---|---:|---:|
| H_S | 0.37708 ± 0.00798 | — |
| full RGB-CMD | 0.38345 ± 0.00296 | +0.00637 ± 0.01067 |
| H_F | 0.38844 ± 0.01357 | +0.01135 ± 0.00872 |
| H_R | 0.36769 ± 0.00262 | −0.00939 ± 0.00996 |
| H_O | 0.37768 ± 0.00668 | +0.00059 ± 0.00978 |

不能用这里的 H_F 均值把它升级为方法：schedule 短、波动大、与 OGSOD direct50 的负结果不一致，而且 H_F 本身只是 ablation。

## 3.11 SiXiang 历史 u300 sweep

L20 有 73 个相关顶层根：33 个 surface-complete、4 个 incomplete、36 个 failure/hung/superseded。它们没有完整登记与一致 provenance，值仅用于 discovery：

| arm/family | AP50-95 mean ± SD | 相对 H_S/说明 |
|---|---:|---|
| H_S | 0.403453 ± 0.008412 | anchor |
| full RGB-CMD | 0.413960 ± 0.002567 | +0.010507 |
| H_F | 0.419993 ± 0.030937 | +0.016540，极不稳定 |
| H_R | 0.403103 ± 0.015541 | 近零 |
| H_O | 0.392253 ± 0.009342 | 负向 |
| FGD adaptation | 0.397890 ± 0.012173 | 低于 H_S |
| LD adaptation | 0.386800 ± 0.005690 | 低于 H_S |
| LADD plain | 0.396777 ± 0.004503 | 低于 H_S |
| singleproj | 0.395783 ± 0.013831 | 低于 H_S |
| MM-ARCS P3/far | 0.405483 ± 0.017741 | 高波动，非 current R2A formal evidence |

身份异常：历史 `lcsr_v1` roots 的 argv 与 `ladd_plain` 相同，不是真正 LCSR；`ladd_identity` 也不稳定/无效。该批不能进入 matched table，campaign 决策 `REDESIGN`。

## 3.12 SiXiang A8+A9 full RGB-CMD attribution

这是目前最完整的正面开发证据。direct300、4 arms × 3 seeds 的 composite 通过完整性验证：

| gate | endpoint delta mean ± SD | 正向 seeds | late10 delta mean ± SD | verdict |
|---|---:|---:|---:|---|
| paired − H_S | +0.010507 ± 0.009002 | 3/3 | +0.011481 ± 0.008690 | PASS |
| paired − strict shuffle | +0.056637 ± 0.026640 | 3/3 | +0.056511 ± 0.025822 | PASS |
| paired − weight0 | +0.016510 ± 0.013565 | 3/3 | +0.017516 ± 0.015207 | PASS |

arm means：H_S 0.403453、paired 0.413960、shuffle 0.357323、weight0 0.397450。

完整性：A8 10 cells + A9 2-cell temporal repair；source map、206-value args/checkpoint contract、commit-last analyzer PASS。A9 不是新的 replication；部分 source campaign 与时间/设备 block 混杂；只有一张 shuffle mapping；per-class/scale 仍需冻结 evaluator。

A1–A7 是 direct300 意图下的 runtime、queue、strict-shuffle、serialization 和 first-cell gate 迭代；它们是工程修订史，不是七次实验复现。A8 最终有 12 个 logical cells 中的 10 格可用，A9 仅重做 `paired_s0` 与 `weight0_s123` 两个工程失败格；合计 14 个 physical attempts 仍只构成 12 个 logical cells。远程 A1–A9 roots 均保留在 manifest，不能把重复/repair root 合并增加 n。

**支持**：在 SiXiang scene-clean val/direct300/full-CMD bundle 中，正确配对 RGB 相对 H_S、shuffle 和 zero-dose 有稳定正增量。  
**不支持**：这是我们的新方法、已经 paper-ready、所有 operating point 都一致改善、机制已因果识别。paired 的 precision 平均上升但 recall 下降，不能说全面支配。

## 3.13 SX-APR B1/B2

### B1

首格启动后 validator import path 出错。已生成 `B1_ENGINEERING_FAILURE_RECEIPT.json`；没有科学 outcome，不能算方法失败。

### B2 当前盲态快照

L20 于 2026-07-23 00:48 CST：

| 状态 | 数量 |
|---|---:|
| completed | 10 |
| running | 2 |
| pending | 9 |
| total | 21 |

running cells：`U1_s42` 与 `U2_s123`；两张 L20 GPU 正常有利用率，队列 manager 持续运行。快照只读了 status/进程/GPU，没有读取 AP、loss、曲线或 arm 排名。

终局前必须同时满足：

```text
21/21 terminal
AND campaign validator PASS
AND source/args/checkpoint/runtime/gate/dose integrity PASS
AND commit-last analyzer
```

此后才允许打开预注册六个对比。B2 即使全过，也没有 same-campaign full RGB-CMD comparator，因此下一步仍是新 confirmation freeze，而不是直接称 paper method。

## 3.14 DroneVehicle 与 M4-SAR 基础设施

### DroneVehicle

- schema-v5 audit/materialization PASS；共 113,579 files，tree/hash 证据存在；
- 它是 RGB-thermal，不是 SAR；
- B0 native loader 多次工程尝试失败，包括 bytecode hash、archive xattr/quoting、`dd` flag、duplicate label row/config state；
- r4n2 仅完成 local dry design pass，没有 detector outcome。

**结论**：可作 secondary mechanism dataset，但当前没有科学结果，license/group lineage 仍需解决。

### M4-SAR

- 已完成 acquisition/archive/readiness 文档与工具；
- 当前没有安全下载、解包和审计后的真实 archives；
- 没有 detector outcome。

**结论**：它是外部 optical-SAR evaluation 的优先基础设施方向，不是当前方法证据。

## 3.15 本地证据与远程根目录对应

本地主要保存：

```text
registry/{runs,campaigns,claims}.csv
refine-logs/research_reset_20260722_v2/
results/sixiang_rgb_attr_v1_20260722_a8_a9_composite/
results/diagnostics/
artifacts/l20_20260712/
tools/analyze_*.py / tools/validate_*.py
```

远程主要保存：

```text
/private/projects/ogsod400_clean_protocol/runs/...   # 一些 baseline 与 canonical run
/private/projects/ogsod400_clean_protocol/logs/...   # queue state/manager logs
/private/results/mm_arcs_r2a_...                     # R2A roots 与 analysis attempts
/private/results/mm_arcs_confirmation_...            # confirmation roots
/private/results/sixiang_u300_...                     # 历史 SiXiang sweep
/private/results/sixiang_rgb_attr_...                 # A8/A9 attribution
/private/results/sixiang_anchor_residual_...          # SX-APR B1/B2
/private/results/dronevehicle_...                     # 数据/loader工程尝试
```

精确 root→campaign 映射见 `catalog/remote_campaign_artifacts.csv`。未登记远程根应保留原地，不移动、不改名、不用目录名猜方法身份。
