# LADD 版本演变与实验记录总览

> 排查时间：2026-08-02。综合 5 处来源：GitHub `yudongfang-thu/LADD`、
> GitHub `yudongfang-thu/LADD_public`、本地 `光sar/LADD`、本地 `光sar/LADD_public`、
> 本地 `光sar/LADD_0723`（7/23 交接 + 重评估）、`光sar/LADD_3090_migration_20260710_224458`、
> `光sar/00_RESEARCH_HUB`、90 服务器 `/mnt/dataY/ydf/projects/LADD_public`（在线解析）。
> 本文件是**叙述性索引**，引用具体数值时以各来源原始文档为准。

---

## 0. 一句话结论

LADD 经历了 **TSKD（Teacher-Student Decomposition KD）→ A1-A2-B → clean A1B →
direct400 matched → 伪收敛/reload 混淆揭示 → RIF-LADD 重构 → research reset → REDESIGN**
的完整生命周期；截至 2026-07-26，**不存在 paper-ready 的新方法**，LADD 旧正结果被
reload（延长的第二个完整 lr schedule）混淆覆盖，当前第一研究方向已转向 CF-ARPD（离线
kill-test 阶段），而同一时刻 L20 上正在用 300ep 严格协议重做 premise 诊断（见 §8）。

---

## 1. 时间线速览

| 时间 | 阶段 | 地点/载体 | 关键事件 |
|---|---|---|---|
| 04-03 ~ 04-09 | TSKD v0→v7 | 90 服务器 D2AD_obb，GitHub LADD | 最早 LADD 形态；skip-B 0.54008 最强 |
| 04 中 ~ 05 | 4-stage → skip-B → minimal | 本地 LADD/ | A2 反转为带检测；minimal role split |
| 05 底 ~ 06 初 | OGSOD 转换 | 90/117 服务器 | cap2 反坍缩、MuSGD lr0=0.001、BN-freeze；n 三 seed +1.1~+1.8 AP |
| 06-04 ~ 06-23 | LADD_public 开源清理 | GitHub LADD_public（143 commits） | A1-A2-B → clean A1B（f59abfa 06-16）；LADD 命名（22229a9 06-19）；cap2 fix |
| 06-14 | checkpoint shutdown | 备份包 | 服务器停摆，冷备份 |
| 06-16 | ladd_mosaic_s | 90 服务器 | sar 0.61972 / rgb 0.66029 / skipA2 B 0.64409@703；A2 全崩 |
| 06-23 | reload_controls | 90 服务器 | laddtrainer 0.5802@ep300（best=last）、yolotrainer 0.5743；init 0.55288 → **reload 混淆起点** |
| 06-25 ~ 06-29 | 机制探索 | 3090 服务器 | clean_a1b_dynprobe；learnability audit；singleproj 系列（scale_seed/decomp_repair/ablation_matrix/matched_detonly） |
| 07-02 ~ 07-07 | direct400 matched 系列 | 3090 服务器 | det-only 0.51558 / plain 0.51863 / singleproj 0.52240；registered rescue 失败 |
| 07-10 | 3090 迁移 | 备份包 | SSH+tar 36 chunks，2,977 文件 / 10.09 GB，校验 0 缺失 |
| 07-12 | RIF-LADD 重构 | 本地 LADD_public | paired interaction projector；oracle 有信号但 projector 无 task utility |
| 07-17 / 07-22 | research reset / reset-v2 | L20 | 双盲治理、预注册门、H_S 锚制度化 |
| 07-19 | 300ep 协议冻结 | 本地仓库 | sixiang_yolo11n_nomosaic_direct300_sgd_b64_v1 |
| 07-21 | 4-arm premise panel (e100) | L20 | sar/rgb/hs/rgbcmd 各 3 seed |
| 07-23 | handoff 六文档 + reassessment 七文档 | 光sar/LADD_0723 | 总决策 REDESIGN；SiXiang paired−H_S = +0.010507 为唯一解盲正面证据 |
| 07-26 | SX-APR 解盲 | L20 | 六预注册门全 FAIL → REDESIGN（非 KILL） |
| 07-28 | MM-ARCS seed42 FAIL / seed123 PASS | L20 | 1/2 → DEFER_PORTABILITY_HETEROGENEITY；RESEARCH_HUB 整理完成 |
| 07-29 | clean_protocol 冻结 → 迁移 | 光sar→GitHub 私有仓库 | 旧目录只读溯源，新工作全部进入 ogsod400-research-core |
| 07-31 ~ 08-02 | reload 3-seed 诊断（进行中） | L20 | n×3 至 ep68/300，s0 至 ep31/300；协议严格保持 300ep |

---

## 2. 方法版本演变（核心）

### 2.1 时代零：TSKD 原始形态（2026-04-03 ~ 04-09，D2AD_obb / GitHub LADD）

- 最早形态是 **TSKD（Teacher-Student Decomposition KD）**：teacher feature 分解为
  task-relevant common 部分与 private/unlearnable 部分，student 只学 common 部分的 KD。
- GitHub `LADD` 仓库是 04-09 的一次性快照（1 commit，原始工作区 `D2AD_obb` 在 90 服务器）。
- 版本迭代 **v0 → v7**：
  - NRRL（Noise-Robust Representation Learning）→ `teacher_u_aux`（u 分支辅助）命名演进；
  - **skip-B**：证明原 4 阶段 B（分解）可以跳过，直接 A1→B 路径，v7 时 skip-B 的
    obb 结果 **0.54008** 是全时代最强；
  - cap2（rank_d_neg_cap=2.0）反坍缩机制在此时代后期引入。
- 术语对应：这里的 "A1/A2/B" 与后来的 A1-A2-B 是同一条演化线。

### 2.2 时代一：4-stage → skip-B → minimal → OGSOD（04 中 ~ 06 初，本地 LADD/）

- **4-stage（A1→A2→B→C）**：A1 基础训练 → A2 分解/解耦 → B 学生蒸馏 → C 推理合并。
- **skip-B**：C 阶段证明 B 可跳过；随后 minimal role split（每个阶段职责最小化）。
- **A2 反转为带检测**：A2 从纯特征学转为带检测头的完整训练。
- **转换到 OGSOD HBB**：数据集从 D2AD_obb 转到 OGSOD HBB。
- **修正链**：cap2 反坍缩 → MuSGD（lr0=0.001）→ BN-freeze。
- **主结论**：LADD n 三 seed **+1.1 ~ +1.8 AP**，是早期最稳定 claim（后受 reload 混淆影响）。
- **DetOnly400 ≈ LADD B400**：400ep 时 det-only 与 LADD 收益接近，质疑 400ep 收益来自方法。
- 服务器演化：117 → 90 → 4090D/5090D。

### 2.3 时代二：LADD_public 开源与协议清理（06-04 ~ 06-23，GitHub 143 commits）

- **06-16 f59abfa**：clean A1B protocol 确立（A1-B 两阶段，A2 移出主线只作诊断）。
- **06-19 22229a9**：正式以 "LADD"（Learnability-Aware Decomposition Distillation）命名，
  区别于 TSKD 时代的旧名。
- **A2 移除证据**：commit 级记录显示 A2 在 06 月中旬从主线移除。
- **cap2 anticollapse fix**：cap2 反坍缩的 bug 修复 commit。
- **6/14 checkpoint shutdown → 6/19 public 精简**：PACKAGE_AUDIT_CN.md 记录只保留论文主线
  代码/协议/摘要，移出证据包、权重、服务器连接文档。
- **`docs/ladd_method_definition.md`** 是最权威的方法定义文档（与 `docs/paper/PAPER_PROTOCOL_CN.md`
  共同构成 3090 时代的 paper 协议）。

### 2.4 时代三：机制探索与投影器变体（06-25 ~ 06-29，3090 服务器）

- 主协议 **clean_a1b_dynprobe**：OGSOD HBB **mosaic100**（imgsz=256、800ep、mosaic=1.0、
  close_mosaic=700）；no-mosaic 只作 fallback。
- `capr_gatedkd_*` 6 组：gated KD 机制探索。
- `yoloinit_probe / mainline / dynamic_*`：从 YOLO init 开始的机制验证。
- `singleproj_scale_seed_20260626/28/29` + `singleproj_decomp_repair_20260627` +
  `singleproj_ablation_matrix_20260626` + `singleproj_matched_detonly_20260629`：
  单投影器变体的规模/seed/分解/消融/匹配检测头系列。
- `learnability_audit_epoch400_20260625`：400ep 学习性审计。
- 制品规模（迁移包统计）：108 个 `*.cmd.sh`、128 个 `*.log`、69 个 PDF、691 张 PNG、
  147 份 `args.yaml`、**125 个 best.pt、119 个 last.pt**、159 份 results.csv。

### 2.5 时代四：direct400 matched 系列（07-02 ~ 07-07，3090 服务器）

- 目录族：`direct400_matched / realloc / dense / failure_isolation / registered_rescue /
  ladd_implementation_fusion / ladd_over_comparison_reset`。
- **关键数值（matched det-only 基线，OGSOD exact400）**：
  - det-only **0.51558**、plain **0.51863**、singleproj **0.52240**；
  - direct400 的弱正信号：+0.003 / +0.007（不足以跨越预注册门）。
- **registered rescue 失败**：预注册的 rescue 方案未兑现收益。

### 2.6 时代五：伪收敛与 reload 混淆揭示（06/23 controls + 07 月审计）

- **90 服务器 reload_controls（06-23）**：
  - laddtrainer reload：**0.5802@ep300，best=last**（真收敛特征）；
  - yolotrainer reload：**0.5743**；init 起点 **0.55288**。
- **机制解释**：从 SAR best 重新训练 = 完整新 lr schedule（峰值 0.0297，线性衰减到
  0.0004）→ 纯优化收益约 **+0.0272**，解释了表观 LADD 收益的 **88.3% ~ 115.9%**。
- **伪收敛 vs 真收敛**：极小 lr 下 SGD "原地抖动"（噪声带 ±0.004），best 是噪声尖峰；
  真收敛时 best=last；reload 运行的 best 落在 ep300。
- **oldscheme**：P1_c 0.58258 / P2_c 0.58049（与 reload 数值同档，进一步支持混淆解释）。
- **ladd_mosaic_s（06-16）**：sar 0.61972 / rgb 0.66029 / skipA2 B s0 **0.64409@703**，
  A2 全崩（A2 是负贡献的来源）。
- 90 服务器早前 baselines（04-19，obb）：sar 0.5530@239 / rgb 0.6733@185。

### 2.7 时代六：RIF-LADD 独立重构（07-12）

- 用 paired interaction projector 独立重构 LADD；结论：**oracle 配对信息有信号
  （G1/G2 PASS），但 projector 无 task utility（G3-G5 FAIL）**——特征层面可预测不等于
  检测收益。

### 2.8 时代七：research reset → handoff → REDESIGN（07-17 ~ 07-26）

- **7/17 research reset → 7/22 reset-v2**：双盲治理、预注册 gates、H_S 锚制度化。
- **7/23 handoff 六文档 + reassessment 七文档**（光sar/LADD_0723）：
  - 总决策 **REDESIGN**：当前不存在 paper-ready 新方法；
  - 研究主张收窄为操作性定义：**强 SAR anchor + RGB-over-SAR 增量 + 严格 attribution controls**；
  - **唯一已合法解盲的正面证据**：SiXiang scene-clean direct300 上 full RGB-CMD
    paired − H_S = **+0.010507 ± 0.009002**（3/3 正向）、paired − strict shuffle
    **+0.056637**、paired − weight0 **+0.016510**。这是 development premise，不是 novelty；
  - **关键反证**：OGSOD matched exact400 上 RGB-CMD − H_S = **−0.00262 ± 0.00204**
    （证明 paired > shuffled 不足以保证跨模态净价值）；raw RGB feature source 有害
    （H_F − H_S = **−0.009315 ± 0.000658**，2/2 adverse）；
  - TCPAD/PTSA/CATR/RMA/RIF 证明 predictability/reachability/CKA 均不能替代 detector efficacy。
- **7/26 18:00 SX-APR 解盲**（B2 21/21 物理完成；validator 曾因 ultralytics
  strip_optimizer 剥离自定义契约键系统性 INVALID——工程问题，无需重训）：
  - P−H +0.002960、P−U1 −0.000070、P−U2 +0.005830（但两 seed 为负）、P−F 三 seed 全负
    −0.008040 → **六门全 FAIL → REDESIGN（非 KILL）**。
- **CF-ARPD 提案**（Cross-Fitted Anchor-Relative Privileged Distillation）：fold-excluded
  teachers + scene-group cross-fitting 的低容量风险校准器；v0 因 q=0 构造保证 + 逐样本 q
  携带标签被内部对抗审稿否决；当前仅 OFFLINE_KILL_TEST_ONLY。
- **7/28 MM-ARCS R2A**：seed42 FAIL（endpoint −0.026630）/ seed123 PASS（+0.009050）→
  1/2 → DEFER_PORTABILITY_HETEROGENEITY；另发现 seed 治理冲突（freeze 0/123 vs registry 42/123）。
- **7/29 clean_protocol 冻结**（MIGRATION_POINTER.md）：旧目录只读溯源，新工作全部迁入
  GitHub 私有仓库 ogsod400-research-core（迁移终态 9dddae8490f8，329 受管文件，验证通过）。
- L20 从历史双卡变为当前仅 1 张可见（GPU-ed5c0525…）；POTS-DIAG A7 双卡 packet 失鲜。

### 2.9 时代八（进行中）：L20 300ep 协议时代（07-19 ~ 今）

- **07-19 协议冻结**：sixiang_yolo11n_nomosaic_direct300_sgd_b64_v1（epochs 300、
  patience 300、imgsz 512、batch 64 strict、SGD lr0=0.01/lrf=0.01/momentum 0.937/
  wd 0.0005/warmup 3、mosaic 0、seeds [0,42,123]）。
- **07-21 4-arm premise panel（e100）**：sar / rgb / hs(H_S) / rgbcmd，各 3 seed，
  YOLO init 等预算（无 reload 混淆）。数值见 §3.3。
- **07-31 ~ 08-02 S 模型 premise（e100）** 与 **reload 3-seed 诊断（e300）** 并行进行。
- 判定规则（SX-PREMISE-RELOAD-v1）：reload_best > sar_best + 0.01 且 best 接近 ep300
  ⇒ 300ep 是伪收敛；reload_best ≈ sar_best ⇒ 300ep 足够。

---

## 3. 关键实验记录与数值

### 3.1 90 服务器（D2AD_obb / OGSOD / SiXiang 早期）

| 实验 | 日期 | 数值 |
|---|---|---|
| baselines（obb） | 04-19 | sar 0.5530@239、rgb 0.6733@185 |
| ladd_mosaic_s | 06-16 | sar 0.61972、rgb 0.66029、skipA2 B s0 0.64409@703、A2 全崩 |
| reload_controls | 06-23 | laddtrainer 0.5802@ep300（best=last）、yolotrainer 0.5743、init 0.55288 |
| oldscheme | 06 前后 | P1_c 0.58258、P2_c 0.58049 |

### 3.2 3090 服务器（OGSOD HBB）

| 实验 | 数值 |
|---|---|
| formal_nomosaic_20260528 800ep SAR YOLO11n | 0.55859 ± 0.00244（三 seed） |
| formal_nomosaic_20260528 800ep RGB YOLO11n | 0.63018 / 0.62664 / 0.62933 |
| **n 级 RGB−SAR gap** | **约 0.074（最大）**；YOLO11s 后缩到 0.02~0.03 |
| direct400 matched（exact400） | det-only 0.51558、plain 0.51863、singleproj 0.52240 |
| OGSOD RGB−SAR headroom | +0.099305（0.5876 vs 0.4883） |
| OGSOD matched exact400 RGB-CMD − H_S | **−0.00262 ± 0.00204**（反证） |
| raw RGB feature H_F − H_S | **−0.009315 ± 0.000658**（2/2 adverse） |

### 3.3 SiXiang（L20，scene-clean direct300 / e100 premise）

| 实验 | 数值 |
|---|---|
| SiXiang RGB−SAR headroom（direct300） | +0.25462（0.64441 vs 0.38979） |
| full RGB-CMD paired − H_S（direct300，解盲） | **+0.010507 ± 0.009002（3/3 正向）** |
| paired − strict shuffle | +0.056637 |
| paired − weight0 | +0.016510 |
| 300ep sar best（协议内，07-20 训练） | n：0.4046@158 / 0.3959@227 / 0.3970@234；s：0.4365 / 0.4445 / 0.4341 |
| 4-arm premise e100（07-21） | sar 0.3686/0.3868/0.3783、rgb 0.6412/0.6225/0.6142、hs 0.3923/0.3759/0.3746、rgbcmd 0.3869/0.3887/0.3888 |
| premiseS e100（07-31~08-02） | hs 0.4295/0.4444/0.4624、rgbcmd 0.4261/0.4448/0.4219 |

### 3.4 L20 reload 诊断（08-01 ~ 进行中）

- 目录：`/private/projects/ogsod400_clean_protocol/comparison/ablations/sixiang_premise_reload_v1/`
- 输入：协议内 300ep SAR best.pt（n×3 + s×3），完整新 lr schedule，e300 重训，fail-closed。
- 进度（08-02 02:05）：n×3 至 ep68/300（约 24s/ep，预计 03:30 前完成）；s0 至 ep31/300
  （约 70s/ep，预计 07:00 前后）；s42/s123 排队自动启动。
- 输出：`/private/results/sixiang_premise_reload_v1/`。

---

## 4. 已证伪的探索（负知识，禁复活）

| 方法/机制 | 结局 |
|---|---|
| LCSR（feature split） | shortcut / 可重参数化 |
| RIF（paired interaction） | oracle 有信号、projector 无 task utility（G3-G5 FAIL） |
| DCR / PG-CMD / CDLS / PIROD / POKS | 多轮已成立负知识 |
| TCPAD | 5/5 predictability 但 PTSA/CATR 0/5——可预测≠收益 |
| CKA / reachability（RMA） | 不能替代 detector efficacy |
| ACBR / GSPAD-v0.2 / PSRMD-v0.8 / SWBC / LROG | 禁复活清单 |
| SX-APR（B2） | 六预注册门全 FAIL（7/26 解盲）→ REDESIGN |
| MM-ARCS（R2A） | 1/2 seed PASS → DEFER_PORTABILITY_HETEROGENEITY |
| M-PSRMD-v1.1 | 十项 outcome-bearing 定义缺口 → DEFER_SPEC_AND_IMPLEMENTATION |
| reload/continued-training/budget | **混淆源**：旧 LADD 正结果 88.3~115.9% 由此解释 |

## 5. 仍成立的正知识

1. **H_S（same-modal 自蒸馏锚）**：永远是强锚点；H_F/H_R/H_O 仅消融，不作主对比。
2. **SiXiang scene-clean direct300：full RGB-CMD paired − H_S = +0.010507 ± 0.009002**，
   3/3 seed 正向——当前唯一已解盲的跨模态增量证据（development premise）。
3. **OGSOD vs SiXiang 对比**：同一方法在 OGSOD 上 paired − H_S 为负、在 SiXiang 上为正
   ⇒ 收益是数据集特定的（scene-clean 的 SiXiang 更有利），不具跨数据集可迁移性。
4. **headroom 显著**：SiXiang RGB−SAR gap 0.25462，n 级 gap 比 s 级更大（3090 时代 0.074 vs
   0.02~0.03）——选择 n 模型做 premise 是合理的。
5. **工程治理遗产**：双盲、预注册门、H_S 锚、evidence immutable、strict protocol freeze
   本身成立，是 7/23 后所有工作的方法论基础。

---

## 6. 关键文档路径索引（光sar/）

| 文档 | 角色 |
|---|---|
| `00_RESEARCH_HUB/README.md` | 研究总枢纽与当前总判断 |
| `00_RESEARCH_HUB/catalog/METHOD_CATALOG.csv` | 55 个方法族的状态/结论/继续决策 |
| `00_RESEARCH_HUB/catalog/EXPERIMENT_CATALOG.md` | 当前可引用科学结果入口 |
| `LADD_0723/research_reassessment_20260723/PORTFOLIO_PROGRESS_AND_RESULTS_20260726.md` | 最新总审计（含 7/26 解盲终局） |
| `LADD_0723/research_reassessment_20260723/DECISION.md` | 当前总决策 REDESIGN 与保留/放弃清单 |
| `LADD_0723/ogsod400_colleague_handoff_20260723/01~06_*.md` | 协议/方法/实验/想法/复现/证据六文档 |
| `ogsod400_clean_protocol/handoff.md` | canonical handoff（328KB，最新 0V27 快照） |
| `LADD_public/docs/ladd_method_definition.md` | LADD 最权威方法定义（3090 时代） |
| `LADD_public/docs/paper/PAPER_PROTOCOL_CN.md` | paper 协议 |
| `LADD_3090_migration_20260710_224458/` | 3090 全部实验/权重迁移快照（校验 0 缺失） |
| `LADD/` | 早期（4-stage→OGSOD）演变工作区 |
| `LADD_experiment_artifacts_20260704_153834/` | 7/4 迁移包（3090/90/local_context） |

---

## 7. 术语与代码库映射

| 名称 | 含义 | 位置 |
|---|---|---|
| TSKD | Teacher-Student Decomposition KD（最早形态） | GitHub LADD / 90 服务器 D2AD_obb |
| LADD | Learnability-Aware Decomposition Distillation（06-19 正式命名） | GitHub LADD_public / 3090 |
| A1-A2-B | 早期三阶段（A2 后移出主线） | 本地 LADD/、GitHub LADD_public |
| clean A1B | A2 移除后的主线协议（06-16 f59abfa） | LADD_public |
| dynprobe | 机制探索协议（mosaic100、800ep） | 3090 |
| direct400 matched | 07 月初 matched 系列（exact400、no-mosaic） | 3090 |
| RIF-LADD | paired interaction 重构（07-12） | LADD_public |
| SX-APR / MM-ARCS / CF-ARPD | 7 月后方法族（后者为当前第一候选） | L20 / clean_protocol / 仓库 |
| reload | 从 best 重新训练（混淆源） | 90 controls 06-23 |

---

## 8. 对当前工作（L20 300ep 协议）的含义

1. **L20 4-arm premise 没有 reload 混淆**（所有 arm 均 YOLO init、等预算）——但 e100 的
   premise 与冻结的 e300 协议不一致；**300ep 是否伪收敛**正是当前 reload 3-seed 诊断要回答的问题。
2. 若 reload_best ≈ sar_best ⇒ 300ep 足够，e100 premise 的结论需在 e300 重验；
   若 reload_best > sar_best + 0.01 且 best≈ep300 ⇒ 300ep 仍欠训练，premise 数字需按
   reload 后基准重估。
3. H_S 与 reload 是不同渠道（机制 vs 预算），不可相互替代：H_S 仍是对照锚，
   reload 是预算诊断。
4. 所有被接受的实验按用户决定统一 300ep（frozen protocol），s 模型排队（先开一个排两个）。
