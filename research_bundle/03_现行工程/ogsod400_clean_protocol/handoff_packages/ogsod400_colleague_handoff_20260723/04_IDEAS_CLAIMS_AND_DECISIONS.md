# 4. Idea 演进、Claims 与决策

## 4.1 我们对问题的认识如何变化

### 阶段 A：把 RGB/SAR 分成 shared/private

早期 LCSR 假设正交 projection、相关性/Barlow 和 reconstruction 能发现共同分量，再由 SAR 预测 RGB 公共增量。这个想法工程上可实现，但理论上 shared/private 可重参数化、不可自动识别；实践中 raw gap target 还含 SAR shortcut。

**保留的洞见**：需要限制 RGB signal，避免 full replacement。  
**放弃的 claim**：不能把学到的分支直接叫 intrinsic shared/private。

### 阶段 B：paired 相对 negative/prior 的 conditional increment

RIF/CPRR 把对象改成：正确配对 RGB 相对 hard negatives 或 prior 的 task response 增量。P4 oracle 和 TCPAD 表明这类 signal 存在、且在 SAR context 下有一定可预测性。

**保留的洞见**：应研究 `RGB over SAR/prior`，而不是 RGB absolute feature。  
**放弃的 claim**：predictability、reconstruction、CKA、mutual information 不等于 task utility。

### 阶段 C：强 H_S anchor 与 native task replay

RMA、PTSA、CATR 等要求 predictor/auxiliary signal 在 native optimizer/detector 上通过 task gate。多条路线在这里失败。

**保留的洞见**：所有方法必须相对 matched H_S，并做 outcome-bearing replay/pilot。  
**放弃的 claim**：局部 gradient 正号或 mediator 存在就足以称可迁移。

### 阶段 D：低剂量、局部、可归因的 source intervention

MM-ARCS 把问题缩成四个固定 atom，直接在 H_S loss 中替换很小的 source；SX-APR 则保留 H_S 原 loss，只加 detached RGB−SAR residual，并加入两张 shuffle、dose、random gate、label-only controls。

当前新方法的共同原则：

```text
强 H_S anchor 不削弱
+ RGB-over-SAR 而非 absolute RGB
+ exact mass/dose
+ pairing/capacity/gate controls
+ no-op/provenance/commit-last
```

## 4.2 历史 idea 池

### 仍有理论价值但未获准扩成 campaign

1. **CF-BAR-CMD / counterfactual full-bundle advantage router**  
   对 paired/self/zero/shuffle/GT-only action 做 held-out group risk 比较，学习何时使用哪种 source。优点是 estimand 直接；风险是标签构造昂贵且容易 overfit。未形成 current implementation。

2. **RGB privileged dense geometry + full CMD**  
   用 RGB 提供几何/边界而不是 raw feature。需要明确 label-only 与 detector-teacher controls，否则无法区分“RGB知识”和更强几何监督。

3. **structural teacher ladder + clean SAR tail**  
   先用 privileged teacher 预训练结构，再用 SAR-only tail 清除跨模态依赖。可能更稳，但 training phase/compute 与 attribution 复杂。

4. **Task-signature partial OT**  
   先在 function/task-signature 空间做 partial matching，再局部蒸馏。可解决坐标错配，但现有 registration transport 失败，先验低；只有当前主线通过/明确失败后才值得小型离线 kill test。

5. **Frequency/region SAR-native protected distillation**  
   只在 SAR 不擅长、RGB 可能补充的频段/区域加信号。风险是频率分解基依赖，且“频率模块”本身 novelty 弱。

### 已实现为离线 P0、但未证明 detector efficacy

- SW-ARCS v1–v5；
- Query-Risk ARCS v1–v5；
- EC-AFS v0.1–v0.7；
- OCUOT-TE；
- SAR Router；
- PARS-P5；
- RCFD-RS；
- replay-ready H_S shadow。

这些原型的价值主要是完善 matched controls、group isolation、task replay 和 receipts。没有 native outcome 时，不能把它们列为“实验方法已验证”。

## 4.3 已明确不应继续扩展的路线

- 再做一套高维 orthogonal shared/private decomposition；
- 只凭 paired-shuffle gradient difference 或 GNS/gradient span 选方法；
- 把频率分解本身当 novelty；
- 只优化 reconstruction/CKA/correlation/predictability；
- 不相对 H_S 的 prototype/uncertainty router；
- 在现有 MM-ARCS/SX-APR 未闭环前启动第三个大矩阵；
- 继续扫 q、lambda、mass、transport window 来“救”已失败版本；
- 把 deterministic duplicate device/epoch 当统计 replication。

原因不是这些想法永远不可能有效，而是当前证据已显示它们最便宜的必要门未过，或无法回答项目的核心 claim。

## 4.4 当前 claim ledger 的可说/不可说

| Claim | 状态 | 允许的说法 | 禁止的说法 |
|---|---|---|---|
| OGSOD RGB-SAR headroom | supported, development | OGSOD contaminated development split 有显著 RGB-SAR detector gap | gap 可由任意方法转移 |
| OGSOD H_F source | supported negative | direct50 两个 formal seed 中 raw RGB feature source 均劣于 H_S | 所有 RGB feature distillation 有害 |
| OGSOD full RGB-CMD > H_S | unsupported | paired 胜 shuffle，但未胜 H_S | paired-shuffle 证明净 RGB 价值 |
| SiXiang headroom | pending | 历史 remote roots 暗示大 gap | baseline lineage 已完全验证 |
| SiXiang full CMD efficacy | supported development | direct300 scene-clean val 上 paired 相对 H_S 三 seed 正向 | 是我们的新方法或 paper-ready |
| SiXiang full CMD attribution | supported development | 增量依赖正确 pairing 与非零 KD dose | 因果 shared/private 已识别 |
| H_F 是新方法 | unsupported | H_F 是 component-source ablation | 可发表主方法 |
| 旧 OGSOD 方法均正式 rerun SiXiang | unsupported | 仅部分有探索性 roots | 已有正式跨数据复现 |
| SX-APR efficacy | pending/embargoed | 无 outcome claim | promising、胜 full CMD、paper-ready |
| MM-ARCS efficacy | pending/embargoed | 无 exact400 outcome claim | positive/negative/backup winner |
| 当前有 paper-ready 新方法 | unsupported | 当前没有 paper-ready method | full CMD 或 H_F 是论文 novelty |

机器版见 `catalog/claims.csv`。

## 4.5 当前支持的科学结论

### Fact

1. OGSOD 和 SiXiang 都有较大 RGB-SAR detector headroom；SiXiang baseline lineage仍需 reconcile。
2. OGSOD matched exact400 中 full RGB-CMD 没有胜 H_S。
3. OGSOD direct50 的 H_F 明确劣于 H_S。
4. SiXiang A8+A9 中 full RGB-CMD paired 相对 H_S、shuffle、weight0 均为三 seed 正向开发信号。
5. RIF oracle/TCPAD 表明 paired task signal 或其条件 predictability 可以存在；下游 task gates 表明这不足以保证 detector gain。
6. 多个“看起来合理”的分解/路由方法在强 control 下失败，这是正当的负结果。

### Interpretation

- RGB 对 SAR 的价值是条件性、局部性和训练状态依赖的；full replacement 可能混合有用增量与跨模态负迁移。
- 最可辩护的 intervention object 是相对强 H_S 的 `RGB-over-SAR` task contribution，而不是不可识别的 semantic shared/private latent。
- SiXiang 正面 premise 为 SX-APR 提供了立项理由，但不能预告其成功。

### Hypothesis

- MM-ARCS：P3/far 等特定 atom 的 source substitution 可能提供小而稳定的净增量。
- SX-APR：SAR-teacher uncertainty foreground 上的 detached paired residual 可能同时胜 H、A、两张 U、R、F。

两者都尚未形成可打开的 terminal claim。

## 4.6 当前不支持的结论

- 已找到稳定优于强 baseline 的论文主方法；
- 已证明 RGB 可迁移知识是某个共享/私有子空间；
- paired > shuffled 足以证明 paired > H_S；
- H_F 的 SiXiang 高均值能覆盖 OGSOD 负结果；
- R2A exact400 或 SX-APR B2 “看起来有希望/失败”；
- OGSOD 数值是 untouched-test 结果；
- 一组 optimizer seeds 等于跨数据泛化；
- 任何历史外部评审看过并认可当前两个候选。

## 4.7 论文故事与 novelty 判断

当前最诚实的论文骨架不是“我们已经发明最终方法”，而是：

1. 强 same-modal anchor 揭示传统 cross-modal KD 的 attribution 错觉；
2. paired signal 可以存在，但 raw feature/source replacement 可能负迁移；
3. 用 anchor-relative、low-dose、control-complete intervention 识别可用增量；
4. 只有在 MM-ARCS 或 SX-APR 通过严格开发/独立确认后，才能把其中一个升级为方法贡献。

novelty 审计认为 SX-APR 与 MM-ARCS 都与 selective/region KD、residual distillation、uncertainty selection 有较高邻近性。潜在 novelty 只能来自完整组合及其 estimand/control design，不能来自“用了 residual/gate/uncertainty”本身。当前 `preliminary_overlap_high/unaudited`，尚不足以定稿。

目标 venue 为 IEEE JSTARS 多模态迁移专题（截止 2026-08-31），但证据门不因 deadline 降低；不够则转 general track。

## 4.8 外部与内部评审状态

- 2026-07-17 reset 进行了内部对抗式审计和文献机制重构。
- 历史存在一次真实的 GLM 评审，但时间早于当前 SX-APR/MM-ARCS terminal 状态，不能称它评审了当前方法。
- 历史 Kimi 有一次脱敏、有效的单轮 review；另一次 full-project 尝试 timeout、无 substantive response。
- reset-v2 的 `INTERNAL_AUDIT.md` 明确是内部审计；本交接没有新 external/cross-model call。

因此对外表述只能是：**historical advisory review exists; current candidates have not received a new verified external adjudication**。

## 4.9 下一步决策树

### SX-APR B2

- integrity + 六门全过：`EXPLORATORY_PILOT` positive，创建新 confirmation freeze，加入 full RGB-CMD comparator；仍不读 test。
- `P-H` 或 `P-A` 明确失败：`KILL SX-APR-v1`，只杀版本，不否定所有 privileged learning。
- efficacy 正但 attribution/gate 门失败：`REDESIGN`，准确说明是 regularization、pairing 或 gate specificity 哪一层没证实。
- provenance/dose/missing cell/validator 失败：`DEFER_ENGINEERING`，不得作科学解释。

### MM-ARCS R2A

- analyzer/terminal bundle 通过后，严格按预注册 H/A/P/U 门判断；
- 不能读取 raw outcome 来设计 analyzer；
- 不能把 seed0 与 42/123 混成 formal replication；
- 若失败，`KILL/REDESIGN` 当前版本；若过，仍需独立 group/dataset confirmation。

### 研究总决策

**`DEFER`**。改变决策所需的最小新证据是现有两个 active line 的完整、不可变、control-complete terminal analysis；在此之前不启动新昂贵 campaign。
