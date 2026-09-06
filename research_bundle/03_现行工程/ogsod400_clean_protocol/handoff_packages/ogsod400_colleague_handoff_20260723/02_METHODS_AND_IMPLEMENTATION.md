# 2. 方法设计与实现

本章按“基础锚点 → 历史主方法 → 当前候选 → 离线/诊断原型”介绍。一个目录存在并不代表方法训练完成；每节都给出当前证据状态。

## 2.1 基础锚点：CMDistill 与 hybrid teacher panel

### 设计

当前 CMDistill 适配将蒸馏分成三块：

1. **PCC feature**：P3、P5 上标准化 feature 的 Pearson-correlation 风格误差；
2. **SLRD relation**：P5 token relation 矩阵差异；
3. **IBCLD output**：Detect 输出中的分类/框分布蒸馏。

hybrid panel 通过改变每一块的 teacher source，分离“RGB feature、relation、output”对最终 SAR student 的贡献。代码中固定路由为：

| route | feature | relation | output | 科学角色 |
|---|---|---|---|---|
| `none` | RGB | RGB | RGB | full RGB-CMD，已有方法适配 |
| `h_s` | SAR | SAR | SAR | 强 same-modal anchor |
| `h_f` | RGB | SAR | SAR | feature-source swap |
| `h_r` | SAR | RGB | SAR | relation-source diagnostic |
| `h_o` | SAR | SAR | RGB | output-source diagnostic |
| `h_ro` | SAR | RGB | RGB | relation+output diagnostic |
| `h_r_shuffled` | SAR | shuffled RGB | SAR | relation pairing null |

### 实现

核心路径：

```text
comparison/runtime/mm_arcs_v2_r2a_hbb/src/teacher_student_decomposition_kd_hbb/
  hybrid_teacher.py   # 固定 route、strict relation shuffle、hash/resume contract
  loss.py             # PCC/SLRD/IBCLD 与 routed teacher forward
  trainer.py          # paired RGB/SAR batch 和训练集成
  model.py            # teacher/student wrapper
comparison/ablations/cmdistill_hybrid_teacher/
  run_hybrid_teacher.sh
  PROTOCOL_AMENDMENT_*.md
```

`H_S` 不是 plain SAR baseline：它仍保留 CMD 的 feature/relation/output loss，只是所有 teacher source 都是 SAR，因此是更难击败、也更公平的容量/训练形式锚点。

### 状态

full RGB-CMD 是当前最重要的已有方法 comparator，但不是我们的 novelty。`H_F/H_R/H_O` 是 ablation，不得包装成主方法。

## 2.2 LCSR v1：显式公共/私有子空间与 SAR residual predictor

### 设计

LCSR 试图为 RGB/SAR feature 建立共同坐标，并把 SAR 私有部分与可由 RGB 提供的公共增量分开。简化形式：

```text
c_r = W_r f_r
c_s = W_s f_s
p_s = f_s - W_s^T c_s
f_oracle = p_s + W_s^T c_r
delta_hat = T_s(f_s)
f_plus = f_s + W_s^T delta_hat
```

其中 `W_r/W_s` 采用 row-orthogonal projection；oracle 用真实 RGB 公共坐标替换 SAR 公共坐标，部署路径则只由 SAR predictor `T_s` 预测修正量。新增路径零初始化，使初始 detector 尽量保持原状态。

### 实现

```text
method/src/lcsr/
  modules.py   # projection、predictor、orthogonality/Barlow 类组件
  model.py     # SAR-only corrected feature 注入 Detect head
  loss.py      # common/private/gap 与 detection loss
  trainer.py   # paired teacher/student 训练
method/src/lcsr_detach/  # 后续 gap-to-backbone detach 变体
methods/lcsr/             # 四方法隔离矩阵版本
methods/lcsr_gap_detach/  # 失败定位版本
```

关键实现约束是 Detect head 必须真正消费 corrected features，且零权重/no-op 要与基线一致。

### 为什么失败

- shared/private 线性分解在没有识别假设时可重参数化，不能由正交和相关性唯一确定；
- raw target `rho(z_rgb-z_sar)` 含可从 predictor 输入直接得到的 `-rho z_sar` shortcut；
- shuffled 条件下新增模块仍可成为 SAR-only residual adapter；
- batch-level common loss 缺少逐样本配对身份约束；
- exact400 seed42 虽有弱正增量，但未过预定门，也未胜过 CMDistill。

当前决策：**`KILL LCSR-v1`**；保留代码和结果用于失败机理说明，不扩 formal seed。

## 2.3 RIF Interaction：配对 RGB 相对负样本中心的交互 residual

### 设计

RIF 不再直接把 feature 命名为 shared/private，而是构造 paired interaction 相对 hard negatives 的中心化响应：

```text
T_full   = G(X_sar, V_rgb+)
T_center = G(X_sar, V_rgb+) - mean_k G(X_sar, V_rgb_k-)
```

先在 P4 oracle 上验证：正确配对是否优于 anchor 与 shuffle；再训练 SAR-only projector 去预测 `T_center`；最后用 detector outcome 分离以下解释：

- G1：paired oracle > strong anchor；
- G2：paired oracle > shuffled oracle；
- G3：Center predictor > anchor；
- G4：Center > FullResidual；
- G5：Center > `Q_det`/capacity control。

### 实现

```text
methods/rif_interaction/
  src/rif_interaction/   # interaction、cache、projector、detector adapter
  configs/               # P4/P5 freeze
  tools/                 # cache/build/train/analyze/validator
  tests/                 # algebra、cache identity、provenance、dry-run
results/diagnostics/rif_*
artifacts/l20_20260712/
```

### 结果与结论

P4 oracle 的 G1/G2 在 seed42/123 上通过，说明正确配对局部 signal 存在；但下游 seed42 的 G3/G4/G5 都失败。结论不是“RGB 无信息”，而是“当前中心化 interaction 无法被 SAR-only predictor 转成净任务收益”。当前版本 **`KILL`**。

## 2.4 LCSR-v2 / CPRR：task-response bottleneck

### 设计

为避免 raw feature residual shortcut，后续把目标移到冻结 SAR Detect P4 的 task response：

```text
target = response(paired RGB) - mean response(8 hard negatives)
```

response 由 raw DFL/class 输出构成；先用 `psi` 去除 DFL side mean，再做 PCA/whitening（8 或 16 维），形成低维 conditional paired response residual（CPRR）。核心问题从“能否重建 RGB feature”改为“能否预测 paired RGB 相对 prior 的 task-space 增量”。

### 实现

```text
methods/lcsr_v2_taskp4/  # P4 response target、PCA/whitening、cache
methods/cprr_detector/   # detector 集成版本
  src/.../               # FrozenP4CrossHead、低秩 adapter、CPRRLoss
methods/c2md_p0/         # grouped-OOF predictability screen
methods/task_gradient_gate/
methods/task_safe_advantage/
```

`cprr_detector` 的 teacher decoder 权重转为 buffer；student Detect 中插入零初始化低秩 adapter；loss 读取与 checkpoint/hash 绑定的 cache，支持 paired/prior/shuffle/raw/noRGB controls，并区分 task-anchored 与 auxiliary 路线。

### 结果与结论

strict TCPAD 的 5/5 group-heldout folds 证明 SAR context 能预测一部分 paired-over-prior response advantage；但后续 task gate/PTSA/CATR 没有证明该可预测量能改善 detector。**可预测性不等于任务 utility**，因此只保留窄 diagnostic claim，方法不升级。

## 2.5 RMA-CMD：reachable mediation analysis

### 设计

RMA-CMD 避免把 `RGB-SAR` 直接作为 target，改为仅从 SAR 预测归一化 RGB response，再用 ridge/eigenspace 分解测试：

1. paired RGB response 是否在 SAR 条件下可达；
2. 可达分量是否介导 task response；
3. 介导分量是否在 native detector replay 中胜过 null/capacity control。

### 实现

```text
methods/rma_cmd/
  PROTOCOL.md
  src/rma_cmd/           # cache、ridge/eigen、full-graph replay
  tools/                 # fit/analyze/validator
comparison/runtime/mm_arcs_v2_r2a_hbb/src/.../rma.py
```

### 状态

reachability/mediation 的部分门可通过，但 task advantage 失败。不能把 reachability 叫作可迁移知识，当前 **`KILL/DEFER`**。

## 2.6 MM-ARCS R2A：强锚点上的局部 RGB source substitution

### 设计

MM-ARCS 不学习新的 projector，而是在 `H_S` CMD feature loss 中做小剂量、可审计的 RGB-for-SAR target substitution。候选原子为：

```text
P3/object_support, P3/far_background,
P5/object_support, P5/far_background
```

对 routing-eligible image，原子质量严格为该 level `H×W` 的 10%；`q=0.5`。如果 `e_RGB/e_SAR` 是原生 PCC element error：

```text
L_A = L_HS + q/N * sum_i a_i (e_RGB_i - e_SAR_i)
```

P3/P5 等权平均，因此 nominal RGB coefficient ceiling 为：

```text
beta_eligible = 0.5(level) × 0.1(mass) × 0.5(q) = 0.025
beta_realized = 0.025 × rho
```

`rho` 是 augmentation 后含有效 GT candidate 的图像比例；R2A 不为了填满质量而除以 `rho`。`rho>=0.80` 只是工程 non-degeneracy gate。

object/far 由 augmentation 后 GT box 中心的 normalized distance 排序；stable flat-index tie breaking，最多一个 fractional boundary cell。relation/output 仍为 SAR source。因为 PCC normalization 是 batch-wide，不能声称 parameter gradient 严格 mask-local。

### 五臂选择与四臂确认

R2A 五臂：`H_S` 加四个原子。epoch50 开发选择曾指向 P3/far，但 exact400 outcome 目前因 analyzer 未接受而禁读。

confirmation 用固定选中原子：

```text
H = L_HS
A = L_HS - q A(e_SAR)
U = L_HS + q A(e_RGB_strict_shuffle - e_SAR)
P = L_HS + q A(e_RGB_paired - e_SAR)
```

主要比较：`P-H` 净 efficacy、`P-A` RGB beyond attenuation、`P-U` pair-specific value。A/U/P 的 removed-SAR coefficient dose 匹配；U/P 的 added-RGB nominal dose 匹配，但不是 gradient-norm matching。

### 实现

```text
comparison/runtime/mm_arcs_v2_r2a_hbb/
  MM_ARCS_R2A_RUNTIME.md
  src/.../mm_arcs.py
  src/.../loss.py
  tests/test_mm_arcs.py
comparison/ablations/mm_arcs_v2_r2a/
  launch_seed42_wave.sh
  run_mm_arcs_paired_job.sh
  *_LOCK.sha256
comparison/ablations/mm_arcs_r2a_confirmation_v1/
  freeze_confirmation_campaign.py
  launch_confirmation_wave.sh
  run_confirmation_job.sh
tools/analyze_mm_arcs_*.py
tools/validate_mm_arcs_r2a_exact400_analysis_lock.py
```

实现包括 q=0 native H_S 直接分支、GT geometry validation、occupancy/beta/degenerate telemetry、输入与源码 SHA locks、strict shuffle replay、epoch50 immutable checkpoint receipt 和 exact400 analyzer。

### 状态

五臂 seed42 exact400 和 H/A/P/U 三 seed物理训练均完成；当前正式推断只用 seeds42/123，seed0 为上下文。由于 accepted analyzer 缺失，**不得报告 exact400 arm 排名或 efficacy**。决策 `DEFER`，下一步是独立复核 analyzer，不是重跑训练或挑选结果。

## 2.7 SX-APR-v1：anchor-protected detached RGB−SAR residual

### 设计动机

SiXiang A8+A9 表明 full RGB-CMD 的 paired signal 在开发集上确实胜过 `H_S`、shuffle 和 weight0，但 full replacement 仍可能带来跨模态负迁移。SX-APR 因而完整保留 `H_S`，只添加小范围 detached RGB-over-SAR residual。

对 P3/P5，`N` 为逐 image、逐 channel、跨空间的 zero-mean/unit-variance normalization：

```text
D_s = N(F_student) - stopgrad(N(F_SAR))
D_t = stopgrad(N(F_RGB)) - stopgrad(N(F_SAR))
e   = 0.5 * (D_s - q * D_t)^2
L_total = L_native_HS_CMD + lambda_res * L_res
```

gate `g` 只读取 augmentation 后 train GT 和 frozen SAR teacher。用独立 TaskAlignedAssigner 得到 teacher foreground，uncertainty=`1-max assigned score`，在 P3/P5 foreground 中取 exact 10% 质量；稳定 tie-break、fractional boundary、无前景则零 gate。gate 不得读取 student、RGB、gradient、val/test 或 detector outcome。

### 七臂矩阵

| arm | λ | q | source/gate | 解释角色 |
|---|---:|---:|---|---|
| H | 0 | 0 | native early branch | fresh `H_S` |
| A | 1 | 0 | P gate，不读 RGB residual | extra dose/capacity control |
| P | 1 | .5 | paired RGB + uncertainty gate | treatment |
| U1 | 1 | .5 | strict donor map 1 + P gate | pairing null 1 |
| U2 | 1 | .5 | strict donor map 2 + P gate | pairing null 2 |
| R | 1 | .5 | paired RGB + random same-mass SAR-foreground gate | ranking control |
| F | 1 | .5 | paired RGB + GT-only same-mass gate | label-only explanation control |

U1/U2 是两张无 fixed point、无 same scene/content group 的 2,305-row bijection；mapping seed 不算统计 replication。P/A/U 必须复用 bit-identical gate coordinates/weights，U 只更换 RGB donor。

预注册 hard gates：

- `P-H`、`P-A`：三 seed mean `> +0.005`，每个 seed endpoint 与 late10 均 `>= +0.002`；
- `P-U1/U2/R/F`：各自三 seed mean `> +0.003`，每 seed endpoint 与 late10 均严格 `>0`。

### 实现

```text
comparison/runtime/sixiang_anchor_residual_hbb_v1/
  src/teacher_student_decomposition_kd_hbb/anchor_residual.py
  src/teacher_student_decomposition_kd_hbb/loss.py
  src/teacher_student_decomposition_kd_hbb/trainer.py
  tests/test_anchor_residual.py
refine-logs/sixiang_anchor_residual_20260722/
  EXPERIMENT_FREEZE.md
  SCIENTIFIC_LOCK.json
  B2_*CONTRACT*.json
tools/freeze_sixiang_anchor_residual_contracts_b2.py
tools/sixiang_anchor_residual_queue_b2.py
tools/validate_sixiang_anchor_residual_campaign_b2.py
tools/analyze_sixiang_anchor_residual_b2.py
```

`anchor_residual.py` 实现 config plan、normalization、独立 assigner、exact mass、coordinate digest、strict donor、R/F gates 和 diagnostics。H 分支在读取 RGB/gate/residual 前直接调用 native H_S；A 的 q=0 分支不读取 RGB residual；teacher tensor 全部 detached。args/checkpoint 绑定 206 个 custom fields，队列采用固定双卡 round 顺序，禁止 outcome-based rescue。

### 状态

B1 在首格因 validator import path 失败，是工程失败，无科学结论。B2 是新 recovery campaign；本快照为 10/21 completed、2 running、9 pending，仍 outcome embargoed。即使六门全过，B2 也只支持 SiXiang val 的 exploratory development claim；它还没有同 campaign full RGB-CMD comparator，下一阶段必须新 freeze 后补。

## 2.8 SAFR-CMD v2：条件 RGB innovation 注入

### 设计

保留 `H_S`，只在 P3/P5 PCC target 中用条件 RGB innovation 替换一部分 SAR target，并要求 `a=0` 与 H_S exact identity。目标是避免 full RGB feature replacement 和 raw gap shortcut。

### 实现/状态

```text
methods/safr_cmd/
```

设计包括 fit-isolated predictor、paired/prior/shuffle/null、source/hash contract，但当前存在实现/独立审查 blocker，没有形成可信 detector experiment。状态 **`DEFER_ENGINEERING`**，不能写方法结果。

## 2.9 Registration transport：局部软配准

### 设计

用局部 5×5 soft transport 对齐 RGB/SAR task response，尝试处理配准误差，而不是强制同坐标蒸馏。

### 状态

```text
methods/registration_transport_p0/
```

离线门中不如 zero-shift/shuffle/prior，说明当前 transport 容易吸收无关匹配。`KILL` 当前版本；不能把配准当作主要失败解释继续扫参。

## 2.10 PG-CMD、DCR、TCPAD、PTSA、CATR

这些是为区分“统计可预测、梯度对齐、实际任务收益”而建立的诊断链：

- **PG-CMD**：按 teacher reliability/gate 选择 CMD 信号；结果 0.36233 vs vanilla 0.36541，`KILL`。
- **DCR P5**：whole-level main effect 为 −0.003335，`KILL`。
- **TCPAD**：group-heldout OOF 显示 response advantage 可预测；只支持 narrow predictability claim。
- **PTSA/task_safe_advantage**：将 auxiliary gradient 安全投影到 task gradient，比较 paired/prior/shuffle/GT/self；5 folds 全未通过 task gate。
- **CATR/task_gradient_gate**：counterfactual task-response replay；0/5 通过。

相关目录：

```text
methods/task_gradient_gate/
methods/task_safe_advantage/
methods/c2md_p0/
tools/analyze_*tcpad* 及 refine-logs 中对应冻结/结果
```

共同结论：相关性、重建 R²、OOF predictability 或单步 gradient alignment 都不能单独升级为 detector claim。

## 2.11 SW-ARCS 与 QR-ARCS：离线 source-risk replay

### SW-ARCS v1–v5

在 P3 的 Haar LL 空间做 A/P/U/C source substitution，并逐版加固数值稳定、matched panel 和 receipt。v5 已是较完整的离线实现，但没有 native trainer/evaluator 接口。

```text
methods/sw_arcs_p0_v1/ ... methods/sw_arcs_p0_v5/
```

### Query-Risk ARCS v1–v5

在同一进程重算 raw object response，严格约束 optimizer step/gradient accumulation，比较 A/P/U/C 的 loss delta 与 gradient replay。

```text
methods/query_risk_arcs_p0/ ... methods/query_risk_arcs_p0_v5/
```

两条线均是 **offline kill-test infrastructure**，不等于 native detector outcome；不能把合成/离线 PASS 写成 AP 改善。

## 2.12 EC-AFS v0.1–v0.7：complementary-error atom selection

### 设计

用 teacher complementary error 选择 object-level RGB source，逐版加入 role/group isolation、calibration、max-T correction、construction receipts 和 fail-closed validation。

```text
methods/ec_afs_v0_1_p0/ ... methods/ec_afs_v0_7_p0/
```

v0.7 仍是 construction-only：不能从 artifact 自身认证 teacher/source/cache，也未完成 mandatory replay。状态 `DEFER_POWER_MODEL`，用于说明可审计选择器如何设计，而非结果。

## 2.13 OCUOT-TE：object conditional unbalanced OT

### 设计

以 class KL + decoded HBB SmoothL1 组成 object task target，通过带 dustbin 的 semi-unbalanced OT 匹配 source/recipient；包含 P/A/U/C/R 与 `G_S/I/N` controls，以及 10-step H references。

```text
methods/ocuot_te_p0_v1/
```

当前仅 synthetic/algebra/protocol 实现，没有真实 native training outcome。状态 `OFFLINE_KILL_TEST_ONLY`。

## 2.14 SAR Router：SAR-only recipient selector

### 设计

在 P3/far exact-10%-mass 域中，从 8 个 SAR-only feature 训练 logistic selector；标签来自 H/A/P/U risk 的 contextual comparison。提供 P/G/L/R/F controls 和 two-snapshot proxy。

```text
methods/sar_router_p0_v1/
```

当前只有合成测试，未证明 selector 能带来 AP。状态 `OFFLINE_KILL_TEST_ONLY`。

## 2.15 PARS-P5

用 grouped-OOF CCA/PLS 在 P5 预测条件 RGB innovation，再要求 task replay 证明 utility。

```text
methods/pars_p5_p0/
```

设计未进入正式 detector campaign；不能沿 predictability 结果直接推广。

## 2.16 RCFD-RS 与 replay-ready H_S shadow

- `methods/rcfd_rs_v1/`：generic offline replay infrastructure，包含 exact H_S atoms、controls、optimizer replay 和 reachable predictor；目前 synthetic-only。
- `methods/replay_ready_hs_shadow_v1/`：安全、非 pickle 的 student/EMA/optimizer/RNG/sampler 快照合同，为 exact replay 准备；尚未集成正式 runtime。

二者是证据基础设施，不是候选方法。

## 2.17 CFID-v0：fusion inverse idea

### 原 idea

设想通过 cross-attention 融合 RGB/SAR，利用 reconstruction uncertainty 与 hallucination residual 识别 RGB 中可能可迁移的分量。

### 实际实现与审计

```text
methods/cfid_v0/
```

审计发现当前代码是 concat-conv，而非报告所述 cross-attention；缺 Stage C，controls 不完整/不匹配。SiXiang premise 文档可作历史线索，不能证明 CFID。当前 **`REDESIGN/DEFER`**。

## 2.18 legacy/comparison 方法的身份边界

| 名称 | 当前准确说法 |
|---|---|
| FGD | 工作区历史实现不能无条件称 paper-exact FGD；旧 audit 判定名称/实现不匹配 |
| LD | clean-protocol adaptation，可作为对比但需带 adaptation 限定；历史 checkpoint CRC 有问题 |
| CMDistill | clean-protocol adaptation，不是逐字 paper reproduction |
| CCLKD | clean-protocol adaptation，online teacher，seed0；不可与当前 formal panel直接池化 |
| legacy LADD | 旧 launcher/identity 有缺陷，不能作为有效新方法 |
| ladd_identity | identity/control；部分 SiXiang 根不稳定/不可验证 |
| SiXiang `lcsr_v1` roots | argv 与 `ladd_plain` 相同，不是真正 LCSR replication |

目录：

```text
comparison/runtime/
comparison/code/legacy/       # 仅追溯，禁止用于正式主表启动
methods/legacy_ladd/
methods/ladd_identity/
methods/parallel/
```

## 2.19 方法谱系的总判断

我们从“显式 shared/private feature decomposition”逐步转向“强锚点相对、任务空间、反事实 control、低剂量局部 intervention”：

```text
LCSR feature split
  → RIF paired-vs-negative interaction
  → CPRR/RMA task response 与 reachability
  → task replay / safe-gradient kill tests
  → MM-ARCS 局部 source substitution
  → SX-APR anchor-protected detached residual + gate controls
```

这一演进的核心不是方法越来越复杂，而是 claim 越来越可证伪：每个新版本都必须回答净 efficacy、pair attribution、容量/剂量、gate specificity 和 provenance。当前唯一 active 的新方法线是 MM-ARCS R2A 与 SX-APR；其余应作为失败证据、工具或备选原型保留，不能并行扩成新的大实验矩阵。
