# Canonical 工程独立审计笔记（2026-09-05）

> **结论：本地 7 月 26—28 日的 accepted 产物已推进并部分否定 7 月 23 日交接包的状态。SiXiang full CMD 有三 seed 正向开发证据；SX-APR 六门失败；MM-ARCS SiXiang 前瞻两 seed 方向分裂；OGSOD MM-ARCS 仍无 accepted 最终分析。尚无稳定、可归因、独立推广的自研论文主方法。**

## 范围与复核方式

- 执行日期：2026-09-05。只读审计 canonical 工程文档、ledger、已公开的本地分析产物与定向代码；最后仅新增本文件。
- 未连接服务器、未运行训练/评估/analyzer、未读取 embargoed raw outcome、未读取凭据文件。
- 本文“最新”指**本次检查到的本地已保存证据**，不是服务器实时状态。主工程 `handoff.md` 顶部切点为 2026-07-28；7 月 23 日交接包不代表 9 月实时进度。
- 下文路径前缀 **CROOT** 均为绝对目录 `E:/SHARE/光sar/03_现行工程/ogsod400_clean_protocol/`。`CROOT/相对路径:行号` 可准确定位真实文件；行号按本次读取的 UTF-8 文本。
- 状态依据优先级：本地不可变 primary 分析/终态产物 → 当前 campaign/claim ledger → 最新 handoff 段 → 历史综合报告。原始远端权重/数据/训练 CSV 不在此次完整验证范围内。

## 1. 可信正负证据总表

下表数字统一为 **AP50-95 百分点**，即把原文件 0—1 数值乘 100；“±”为跨 seed sample SD，除另有说明。不同数据集、schedule、runtime 和分析集合不能合并。

| 实验 | 核心结果 | 支持与边界 | 主要来源 |
|---|---|---|---|
| OGSOD baseline exact400 | SAR 48.825±0.021；RGB 58.756±0.018，正式 n=2 | RGB detector 有约 9.93 点 headroom；不等于可蒸馏 headroom；长期反复使用 test 作 val | `CROOT/refine-logs/research_reset_20260722_v2/OGSOD_EXPERIMENT_AUDIT.md:5,30` |
| OGSOD matched full RGB-CMD − H_S | −0.262±0.204，严格 same-seed n=2 | full RGB bundle 未胜同模态强锚点 | 同文件 `:41,47`；原始 canonical report 仅远端 |
| OGSOD paired CMD − shuffle | +3.053，只有 seed42 | 正确 pairing 重要；不能证明相对 H_S 的净 efficacy | 同文件 `:50` |
| OGSOD H_F − H_S，direct50 | s42/s123 为 −0.885/−0.978；均值 −0.932±0.066 | 固定 SAR relation/output，仅交换 RGB feature source，在该短协议下有害 | 同文件 `:62`；`CROOT/refine-logs/HF_HS_THREE_SEED_ANALYSIS_20260717.json` |
| SiXiang A8+A9 full CMD − H_S | s0/s42/s123 为 +1.367/+0.035/+1.750；+1.051±0.900 | 三 seed 正向 scene-clean val 开发信号；已有 CMDistill 适配，非自研 novelty | `CROOT/results/sixiang_rgb_attr_v1_20260722_a8_a9_composite/sixiang_rgb_attr_a8_a9_final.md:33` |
| SiXiang A8+A9 paired − shuffle | +5.664±2.664，3/3 正向 | 冻结 donor null 下的 pairing 依赖；只有一张 donor map | 同文件 `:34` |
| SiXiang A8+A9 paired − weight0 | +1.651±1.357，3/3 正向 | 非零 KD dose 有增量 | 同文件 `:35` |
| SX-APR B2+B2E union | 21 logical cells；完整性 PASS；六个预注册门全 FAIL | 当前 residual+uncertainty 版本未建立净 efficacy、pairing 与 gate specificity；REDESIGN | `CROOT/results/sixiang_anchor_residual_v1_20260722_union_b2_b2e_analysis/sixiang_anchor_residual_union_final.md:3,39` |
| MM-ARCS OGSOD M0 | 17 个原训练 physical runs，最终 analyzer 未通过；production calls 为 0 | 无可报告 exact400 efficacy 或 attribution；证据工程失败，不是算法负结果 | `CROOT/refine-logs/mm_arcs_m0_exact400_20260727_050437/TERMINAL_RECEIPT.json:5` |
| MM-ARCS SiXiang M1 | s0 exploratory +2.367；前瞻 s42 −2.663、s123 +0.905；前瞻均值 −0.879 | 前瞻 gate 仅 1/2 PASS，seed 敏感；当前 portability 版本终止 | `CROOT/handoff.md:1219,1227,1238`；下文列 primary JSON |
| PSRMD KT1 R1 S1 | 约 1.39 GPUh 后 KILL_MODE_EXISTENCE | 工程完成，冻结双 block mode-existence 必要门失败；不是 AP 失败 | `CROOT/handoff.md:78` 起；`CROOT/refine-logs/psrmd_v1_2_kt1_r1_s1_execution_20260728/FORMAL_TERMINAL_ACCEPTANCE.json` |
| H-SCARD R2.2 | 共同 member-roster 大小同时要求 m≥2 与 m≤1 | 冻结数据契约不可实例化；无 CUDA、无 outcome | `CROOT/refine-logs/hscard_v0_1_kt1_cleanroom_r2_2_controller_20260728/DATA_BINDING_SCIENTIFIC_BLOCKER.md:16,39` |

## 2. OGSOD 的均值差和配对差为何不一致

历史综合表列：full RGB-CMD `0.50478±0.00027`，H_S `0.50867±0.00259`。直接相减为 **−0.00389，即 −0.389 点**，但正式 contrast 是 **−0.00262±0.00204，即 −0.262±0.204 点**。

这不是同一集合的算术错误：

1. 两个 arm mean 是包含 seed0 的 n=3 descriptive context；H_S seed0 来自 cross-root。
2. cross-root H_S seed0 被 analyzer 标为 `contrast_eligible=0`。
3. 正式 delta 只使用同源、同 seed 的 42/123 两组。

直接来源：`CROOT/refine-logs/research_reset_20260722_v2/OGSOD_EXPERIMENT_AUDIT.md:41-49`。论文表必须让 arm means 和 delta 使用同一 eligible seed 集合，或明确将 n=3 描述表与 n=2 配对表分开，不能让读者以为 Δ 等于旁边两个均值之差。

原始 canonical matched report 的准确远端路径：

`/private/results/mm_arcs_r2a_handoff_lock71835_scannerfix_b48f8a_20260718/final_analysis/exact400_matched_v2_final.md`

保存 SHA256：`a708ed79783390db3078a273dff306cc8f5a86f4ad3388cc086ce229cad89c49`。

该路径和 SHA 来自 `CROOT/registry/campaigns.csv:4` 与 `CROOT/refine-logs/research_reset_20260722_v2/REMOTE_CAMPAIGN_ARTIFACTS.csv:6`；本次没有访问远端文件，因此 OGSOD matched 的数值属于**有本地审计和保存 hash 支持的转述**，不是此次从完整原始训练 CSV 重算。

其他口径边界：

- OGSOD workers8 主 exact400 与 workers2 matched CMD 必须区分 protocol ID；“都是 400 epoch”不足以池化。见 OGSOD 审计 `:14-24`。
- 7 月旧治理把 OGSOD 正式推断限定 seeds42/123，seed0 仅上下文。9 月工作区规范要求增益至少 3 seeds；历史 n=2 证据可按原边界引用，不应包装成当前三 seed 稳定增益，也不能根据结果事后恢复 seed0。
- best checkpoint 与 epoch400 endpoint 不同。早期 RGB s123 的 best `0.58780@399` 与 final `0.58768` 曾混写，见 `CROOT/EXPERIMENT_AUDIT.md:70`。
- epoch 和 late10 内的 epoch 不构成独立 replication。

## 3. SiXiang A8+A9：正面证据是什么级别

本地 accepted bundle 的准确文件：

- `CROOT/results/sixiang_rgb_attr_v1_20260722_a8_a9_composite/sixiang_rgb_attr_a8_a9_final_COMMIT.json`
- 同目录 `sixiang_rgb_attr_a8_a9_final.json`
- 同目录 `sixiang_rgb_attr_a8_a9_final_runs.csv`
- 同目录 `sixiang_rgb_attr_a8_a9_final_contrasts.csv`
- 同目录 `sixiang_rgb_attr_a8_a9_final.md`

状态为 `Integrity=PASS`、`production_composite_subprocess_pre_post`、`formal_report_eligible=true`、决策 `EXPLORATORY_PILOT`。这里的 formal-report eligible 说明可以按冻结开发实验报告，**不等于 confirmatory final-test 或 paper-ready**。

A8 贡献 10 格，A9 仅替换 `paired_s0` 与 `weight0_s123`；12 logical cells、每臂仍只有 3 optimizer seeds，A9 新增独立 replication 为 0。两格来自后来的时间/设备 block，须披露。见最终 Markdown `:3-10`；COMMIT `:2,37,54`。

逐 seed epoch300 AP50-95 原值：

| arm | s0 | s42 | s123 | 均值 |
|---|---:|---:|---:|---:|
| H_S | 0.397990 | 0.413140 | 0.399230 | 0.403453 |
| paired | 0.411660 | 0.413490 | 0.416730 | 0.413960 |
| shuffled | 0.385770 | 0.342290 | 0.343910 | 0.357323 |
| weight0 | 0.379660 | 0.402710 | 0.409980 | 0.397450 |

直接来源：最终 Markdown `:16-27`。

G1 的三 seed 差值为 `+0.013670/+0.000350/+0.017500`。seed42 只有 **+0.035 点**，不能表述为“每个 seed 均有明显增益”。其冻结 gate 是 endpoint mean >0.005、endpoint 3/3 >0、late10 3/3 >0；不是统计显著性检验，也不是每 seed >0.005。已定向读取实现：`CROOT/tools/analyze_sixiang_rgb_attr_a8.py:826-878`。

跨 seed delta 的 sample SD 与每 run late10 内 population SD 不是同一不确定性。A8+A9 analyzer 明确把 late10 定义为 epochs291—300 算术平均，统计单位是一格一个 optimizer-seed run：`CROOT/tools/analyze_sixiang_rgb_attr_a8_a9.py:545-548`。

可支持的表述：在冻结 SiXiang scene-clean val/direct300/full-CMD bundle 内，正确配对 RGB 相对 H_S、固定 shuffle 和 zero-dose 有三 seed 正向开发信号。

不支持：自研新方法、paper-ready、intrinsic shared/private 或因果机制、所有 operating points 全面改善、外部泛化。paired 的 precision/recall 也并非全部一致改善；P/R 是描述性指标。

## 4. SX-APR：7 月 26 日已解盲，不能继续描述成运行中

7/23 交接包 `10 completed/2 running/9 pending` 是旧快照。当前 registry 及本地正式 union 已记录：B2 19 格 + B2E 修复 2 格，共 21 格，完整性 PASS，六门 FAIL，`REDESIGN`。工程修复不新增统计 n。

Primary bundle：

- `CROOT/results/sixiang_anchor_residual_v1_20260722_union_b2_b2e_analysis/sixiang_anchor_residual_union_final_COMMIT.json`
- 同目录 `sixiang_anchor_residual_union_final.json`、`sixiang_anchor_residual_union_final_runs.csv`、`sixiang_anchor_residual_union_final_contrasts.csv`、`sixiang_anchor_residual_union_final.md`

Endpoint 为 exact epoch300，late10 为 exact epochs291—300，见最终 Markdown `:6-7`。

| P 对比控制 | s0/s42/s123，百分点 | mean±sample SD，百分点 | 具体失败 |
|---|---|---:|---|
| H：native H_S | +1.865/−1.295/+0.318 | +0.296±1.580 | 均值不足 +0.5；s42 负；s123 late10 未过每 seed +0.2 |
| A：额外 SAR dose | +0.444/+1.100/+0.010 | +0.518±0.549 | s123 近零，未过每 seed +0.2 |
| U1：shuffle map1 | +0.257/+0.379/−0.657 | −0.007±0.566 | 对该 donor null 无均值优势 |
| U2：shuffle map2 | +2.498/−0.194/−0.555 | +0.583±1.668 | 2/3 负，均值由 s0 驱动 |
| R：同质量随机 gate | +0.754/+1.175/−0.906 | +0.341±1.100 | uncertainty ranking 优势不稳定 |
| F：GT-only gate | −0.808/−1.168/−0.436 | −0.804±0.366 | 3/3 都劣于 GT-only gate |

逐 seed 值见最终 Markdown `:39-44`；sample SD 来自同目录 JSON 的 `gates`。**六门失败不等于六个均值全为负**，应准确报告门失败原因。

最有解释力的结果是 F 在三 seed 都高于 P，以及 P 对 U1 几乎零均值优势。当前 uncertainty gate 的独特贡献与稳定配对收益没有成立。不是仅因任意设定门太高：存在实质负向控制差值。

### 实现含义：residual 名称没有自动消除 target 冲突

已直接读 `CROOT/comparison/runtime/sixiang_anchor_residual_hbb_v1/src/teacher_student_decomposition_kd_hbb/anchor_residual.py:698-714`：

```text
error = N(S)-N(T_sar)-q[N(T_rgb)-N(T_sar)]
      = N(S)-[(1-q)N(T_sar)+qN(T_rgb)]
loss = image mean of sum(gate * 0.5 * error²)/(channels * gate_mass)
```

该项代数上相当于**选区内拟合混合教师 target**。gate 只有 10% 质量，不意味着总 loss 自动被稀释为 10%，因为除以 gate_mass。保留 H_S 再加 λ=1 的归一化损失，仍可能有实质目标冲突。此为代码推导出的解释，**不是经因果实验认定的失败原因**。

与 MM-ARCS `q/N * sum(mask * (e_RGB-e_SAR))` 的总质量限制不同，不能把 MM 的 `beta=0.025*rho` 自动用于 SX-APR。MM 公式见 7/23 方法包 `02_METHODS_AND_IMPLEMENTATION.md:189-204`。

H 的 λ=0 early no-op、A 的 q=0 分支、教师 detach 则正确实现：上述代码 `:681-707`。应区分“真实机制失败”和“没有正确接入”这两种情况。

## 5. MM-ARCS：OGSOD 分析阻塞与 SiXiang 异质性是两件事

### 5.1 OGSOD exact400 M0

7/27 M0 终态 `DEFER_ANALYZER_INTEGRITY`，R2A 与 H/A/P/U production analyzer attempts 都是 0，没有 accepted bundle。Primary：

`CROOT/refine-logs/mm_arcs_m0_exact400_20260727_050437/TERMINAL_RECEIPT.json`

该 JSON 记录阻塞点包括执行期 source 身份未绑定、失败时不能证明回收全部后代、validated inputs 重开导致 provenance 漂移风险、epoch50 提前发布/失败残留。不能把这些证据工程问题当算法负结论，也不能手算 embargoed CSV 得到替代结论。

旧 R2A 另有 7/19 04:39 非计划中间 outcome access，已披露没有据此停止/重选/改训练。不能称完全 operator-blind：`CROOT/handoff.md:2991`。

### 5.2 SiXiang M1：后续确实产生合法 committed outcome

准确 primary 路径：

- seed0：`CROOT/refine-logs/mm_arcs_sixiang_p3far_seed0_recovery_v2_20260727T110706/evidence_mirror_seed0_v16a1/RECOVERY_ANALYSIS_BUNDLE.json`
- seed42/123：`CROOT/refine-logs/mm_arcs_sixiang_p3far_m1_joint_analyzer_recovery_v21_20260727T235200/committed_evidence_mirror/JOINT_ANALYSIS_BUNDLE.json`
- joint 同目录 `COMMIT_MANIFEST.json` 与 `COMMIT`
- 异质性：`CROOT/refine-logs/mm_arcs_sixiang_p3far_m1_heterogeneity_audit_20260728T004331/README.md`、`AUDIT_MATRIX.json`、`HETEROGENEITY_AUDIT_TERMINAL.json`

| seed | H endpoint | P endpoint | Δ endpoint | Δ late10 | 角色 |
|---|---:|---:|---:|---:|---|
| 0 | 0.397990 | 0.421660 | +0.023670 | +0.021189 | analyzer amendment 后 exploratory recovery |
| 42 | 0.413140 | 0.386510 | −0.026630 | −0.025758 | 前瞻训练与 2-of-2 gate |
| 123 | 0.399230 | 0.408280 | +0.009050 | +0.012051 | 前瞻训练与 2-of-2 gate |

joint JSON `:4-16` 明确前瞻两 seed 的均值 endpoint Δ=−0.008790、late10 Δ=−0.0068535，pass_count=1，verdict=`DEFER_PORTABILITY_HETEROGENEITY`，seed0 不计入（`:75`）。不得合成“2/3 成功”替代预注册 gate，也不得剔除 seed42。

两条训练和 gate 前瞻冻结，但 checkpoint-contract amendment 发生在训练完成后，因此分析标 `development_exploratory`，非 confirmatory。见 `CROOT/handoff.md:1227-1238`。

7/28 异质性终态：MATCH=3，EXPECTED_SEED_DIFFERENCE=4，UNOBSERVABLE=5，DRIFT=0。只说明冻结白名单未建立非 seed 漂移，**不解释 seed 分歧，不证明“纯随机性”**。五个不可观测项为 pairing roster hash、组件 RNG 传播、实际 optimizer transition 数、atom eligibility 数、checkpoint save-order trace。见异质性 README `:13-30`。

当前版本停止稳定提升/portability 主张；仅能说某些开发 seeds 可正向。不能声称 RGB attribution，因为这轮只做 P-H，没有完整 A/U 对照闭环。

## 6. 早期实现—命名—科学含义错位

以下分为本次直接代码核对和历史审计转述，避免把审计叙述冒充此次执行验证。

### 已直接核对代码

1. **LADD identity 的 corrected 恒等于 raw feature。** `shared=decode(encode(x))`，`private=x-shared`，`corrected=private+shared=x`。实际训练作用是投影空间 KD，不是 detector feature correction。
   - `CROOT/methods/ladd_identity/src/ladd_identity/modules.py:36-46`
2. **foreground-only 未落实。** 全空间 `F.smooth_l1_loss(student_z, teacher_z.detach())`，没有 foreground mask。
   - `CROOT/methods/ladd_identity/src/ladd_identity/loss.py:37-44`
3. **LCSR rho 是逐通道相关性 EMA，不是 SAR 可预测比例或 predictive R²。** gap target 含 `−rho*z_sar`，该分量可从 SAR predictor 输入访问，存在 shortcut 解释。
   - `CROOT/method/src/lcsr/modules.py:116-129`
4. **LCSR 有真实 SAR-only 修正路径且零初始化。** predictor 末层为零；corrected=`sar_feature+decode(gap)`。不能把全部 LCSR 代码当 identity。
   - 同文件 `:54-61,102-106`

### 历史审计明确记录，本次未运行复现

- legacy LADD phase B 未载入 phase A checkpoint，又冻结新增随机分解模块，只能作 random-decomposition diagnostic：`CROOT/EXPERIMENT_AUDIT.md:101-110`。
- FGD 移除 global relation，只能称 focal-only adaptation；旧 FGD/LD/CMD 的 checkpoint CRC 都失败，较新审计补了旧 FGD 漏标：`CROOT/refine-logs/research_reset_20260722_v2/OGSOD_EXPERIMENT_AUDIT.md:73-80`。
- LCSR v1 seed42 epoch400 `0.49469`，比 plain SAR `+0.00629`，未过既定门且未胜 CMD；只能保留失败机制证据：同文件 `:75-76`。
- RIF P4 oracle G1/G2 两 seed 正，部署 projector 的 G3/G4/G5 都失败；TCPAD 5/5 OOF 可预测，PTSA/CATR task gates 0/5：`CROOT/handoff_packages/ogsod400_colleague_handoff_20260723/03_EXPERIMENT_CATALOG_AND_RESULTS.md:118-145`。

因此不能把“教师信息存在”“特征更像教师”“可预测”“单步梯度看似安全”直接升级为最终 detector utility。

## 7. 本次实际完整性复核与不可声称的范围

此次已重新计算：

| 对象 | 复核结果 |
|---|---|
| A8+A9 COMMIT 绑定的 JSON/runs CSV/contrasts CSV/Markdown | 4/4 SHA256 与字节数精确匹配 |
| SX-APR union COMMIT 绑定的四文件 | 4/4 SHA256 与字节数精确匹配 |
| A8+A9 当前 analyzer 源文件 | SHA `edba21a078989e99709347680e5aed355f960df0ca92b091eb47dfff7b98110d`，与 COMMIT 一致 |
| SX-APR union 当前 analyzer 源文件 | SHA `15ca8f5da53b53540f3613007a0a53ed75cc81717c4197a3ea0899ac62d6c610`，与 COMMIT 一致 |
| MM SiXiang joint bundle | SHA `ec74967f81ef0ba2d49d5258efcb3d853f9955acccc790e5450332de73165fb3`，与 COMMIT_MANIFEST 一致 |

已读取 A8/SX 的本地逐 seed 分析表和 JSON，核对 difference/SD/gate 口径；定向审查 identity、LCSR rho/gap、SX residual、A8 gates 代码。

这些只证明本地镜像与保存的 manifest 一致，**不等于此次重新运行远端 analyzer、核验每个真实 checkpoint、完整原始 CSV/data/runtime 链或独立复现训练**。A8/SX CSV 是 accepted analyzer 输出的逐 run 摘要，不是完整 300 epoch 原始 CSV。

尤其多个 campaign 使用相同 H_S 时不得累加独立样本：MM joint H42/H123 的原 CSV SHA 分别为 `64ab63ec...` 和 `d2144393...`，与 A8 H 相同；A8、SX H 的 endpoint 也相同。共享强锚点有助解释比较，但不是更多独立 baseline replication。

## 8. 对论文与项目复盘的判断

已经形成三条有研究价值的证据：

1. 强 RGB teacher 的检测 headroom 不等于 SAR student 的可转移 headroom。
2. paired>shuffled 与 paired<H_S 可以同时成立；跨模态归因和净 efficacy 是两个问题。
3. residual、uncertainty gate、可预测性和相关性都不足以保证最终检测增益；版本与 seed 依赖明显。

这些足以作为问题动机、系统对照和负结果分析的基础；现有 canonical 证据仍不足以支撑“自研新方法稳定胜强基线”的主方法稿。缺项不只是补 seeds，而是自研净 efficacy、完整 attribution、独立测试/外部数据和 novelty。OGSOD test 已长期作为开发集；SiXiang scene-clean val 是开发集，不能用多 optimizer seeds 冒充独立 acquisition/group 泛化。

治理问题需要在总报告中明确：顶层 README 指向旧审计/交接包，catalog 镜像是 7/23，registry 的某些 pending 又未覆盖后来 SiXiang M1，handoff 顶部则到 7/28。必须按**日期+方法版本+primary artifact**理解状态，不能因旧标题存在就恢复已失败路线，也不能把旧 pending 当成尚无后续。

## 9. 转交给最新实验审计者的跨范围核对线索

启动必读日志时发现：9/5 P2 特征 probe README 使用 `native_rgb_s42_b64_e200` 作 native 对照；同日 P3 昼夜日志明确勘误该同名 checkpoint 实为 VEDAI 模型、首轮指标 0.09，随后换正确 DroneVehicle checkpoint。P2 README 未见相同勘误。

这只是此次发现并已转交父任务的**待核对线索**，不能仅凭同名路径就终局认定 P2 的所有结果无效。应由当前实验审计根据脚本/summary 中准确 checkpoint 绝对路径与身份确认。若确认误用，P2 的 CKA 0.56→0.88、“模态差距集中层位”等归因需要重评，不能直接当后续预注册机制依据。

来源在 canonical 根目录外：

- `E:/SHARE/光sar/08_实验日志/2026-09-05_P2探针_DroneVehicle特征图/README.md`
- `E:/SHARE/光sar/08_实验日志/2026-09-05_probe_DroneVehicle昼夜分桶/README.md`

