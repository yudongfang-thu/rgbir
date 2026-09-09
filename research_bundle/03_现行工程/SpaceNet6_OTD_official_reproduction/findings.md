# Research Findings

## 2026-09-02 — Hnewa-inspired stage-1 adjudication (T0/B0/H1/H2/H3, 26/26)

- Intended claim: paired fusion-MSE feature distillation from a 6ch fusion teacher improves the weak-modality student, the gain is attributable to true pairing, and it exceeds the same-modal KD control.
- Evidence: 26 training endpoints with frozen last.pt evaluation (`rgbt_hnewa_cmkd_mse_inspired_v1/eval_records/`, 26 metrics_records; `llvip_h3_s0.json` completed 2026-09-02). Protocol E200/B32/SGD, grouped dev (LLVIP) / official val (Drone), 3 paired seeds per arm.
- Integrity status: pass. Arms and decision rule were frozen before stage-1 results; official test never accessed; all deltas are same-seed paired against the B0 native arms.
- Verdict: `pass (scoped)`. ① H1>native: +0.48 pp (3/3) LLVIP, +0.18 pp (3/3) Drone — first 6/6 positive paired result in the project. ② H1>shuffled: +0.35 (2/3) LLVIP, +0.34 (3/3) Drone. ③ H1>same-modal: +0.09 (2/3) LLVIP, +0.21 (2/3) Drone — both weak passes by the frozen mean-plus-2/3 rule.

What the results support:

- Feature-level privileged distillation is effective and pairing-attributable on both datasets; the published CCLKD/CMDistill effect magnitude (+1.5/+2.6) remains an open gap.
- The privileged residual (H1−H3) is real but small; most of the H1 gain is generic-KD regularization. The larger oracle gap (LLVIP 15.97 pp) did not produce a larger privileged residual.
- Single-seed LLVIP noise reaches ±1.7 pp (H2 arm), so all future adjudication must use 3-seed paired means.

Constraints for future work:

- The calibration-KD enhancement arm must target the confidence channel (LLVIP AP75 flat while AP50 moves), and its formula must be frozen before the CCLKD-protocol baseline numbers are read (ALIGN plan §3).
- Do not quote stage-1 deltas as the paper's headline; the main table is the CCLKD-protocol alignment group.
- LLVIP official split stays sealed until the frozen-method final evaluation.

Primary synthesis: `refine-logs/rgbt_literature_v1/STAGE1_RESULTS_20260902.md`.

## 2026-08-26 — P3 paired optical DFL full-622 result-to-claim

- Intended claim: P3 paired optical DFL improves full-data SAR detection and retains that advantage under deployable INT8.
- Evidence: nine full-622 detector endpoints (`native/p3/p3_same_modal × seeds 0/42/123`), one frozen 200-image locked-test evaluation, nine fixed640 FP/fake-W8/ORT-QDQ chains, and nine QCS6490 prepare-only AOT contexts.
- Integrity status: provisional; no separate `EXPERIMENT_AUDIT.json` exists. The method was frozen before the locked test was read, all nine final-epoch checkpoints completed, and the locked test was not used for further tuning.
- Verdict: `partial`, routed to `DEFER` rather than method redesign.

What the results support:

- Standard locked-test P3−native is `-0.003/+0.325/+0.117 pp` AP50/AP75/mAP on average, with AP75/mAP positive for 2/3 seeds.
- On the matched fixed640 deployment chain, P3−native is `+0.449/+0.284/+0.337 pp` under fake-W8 and `+0.347/-0.027/+0.226 pp` under ORT-QDQ.
- P3 has a smaller fixedFP→fake-W8 mean mAP drop (`-0.200 pp`) than native (`-0.555 pp`).
- Optical teacher quality is much higher than SAR teacher quality, but student transfer remains weak and seed-sensitive.

What failed:

- The strong development claim did not reproduce on the locked test: FP AP50 is unchanged and seed0 is negative on all three main metrics.
- Collect-equal macro diagnostics do not show a robust positive effect; intervals cross zero and the seed-averaged direction is negative.
- Fixed640 FP and standard rect FP disagree on the AP75 direction, so the small high-IoU effect is protocol-sensitive.
- Earlier dynamic gate, method-zoo, PEMT combination, quantization-counterfactual selector and W4-QAT routes did not produce a stronger frozen method.

Constraints for future work:

- Do not tune the method after observing the locked test.
- Do not add more selector thresholds, loss coefficients or replacement seeds to rescue this result.
- Do not run QAT for the full-622 W8 chain: all mean mAP drops are below 1 pp.
- Do not claim QNN/QCS6490 retention from AOT compilation; require a real board run.
- If the board run agrees with ORT, use a narrow SpaceNet6 case-study statement. If it reverses the direction, remove the hardware-retention statement.

Primary synthesis: `docs/P3_FULL622_LOCKED_RESULTS.md`.

## 2026-08-28 — OGSOD–SiXiang class–geometry router result-to-claim

- Intended claim: paired optical class contrast and paired optical DFL are
  separately useful to SAR detection, enabling a target-level four-state
  `{class, DFL, both, none}` router that beats uniform and null controls.
- Evidence: both clean seed42 seven-arm development panels; OGSOD D1/D2
  confidence-versus-centered-contrast decomposition; fixed-GT roster metrics;
  final seed42 and multiseed decision files.
- Integrity status: provisional because no separate `EXPERIMENT_AUDIT.json`
  exists. All formal endpoints used physical B64 and final `last.pt`; no
  sealed-confirm split was read.
- Verdict: `no`, confidence high. The result-to-claim MCP was unavailable, so
  this verdict was independently checked against the frozen pipeline logic and
  raw result files.

What the results support:

- OGSOD same-class class-score KD improves seed42 mAP by `+0.657 pp`, but exact
  paired class-score KD improves only `+0.391 pp`; precise instance pairing is
  not the source of the strongest classification result.
- OGSOD confidence-only D1 improves AP50/AP75/mAP by
  `+0.410/+0.325/+0.601 pp` over native.
- Centered-contrast D2 reduces true-class NLL from `0.015964` to `0.010908`,
  showing that class information can become more decodable without producing a
  corresponding detector improvement.
- SiXiang same-class KD has a limited scale/calibration signal (`+0.548` mAP pp
  and `+2.141` AP50 pp), but its AP75 is `0.321 pp` below native and its
  per-class effects are uneven.

What failed:

- OGSOD D2 is `0.458 mAP pp` below confidence-only D1 and `0.032 pp` below the
  wrong-class control; AP75 is `0.859 pp` below native. The centered class
  contrast claim therefore fails despite lower true-class NLL.
- OGSOD paired DFL is `0.065 mAP pp` below shuffled DFL, so its positive delta
  over native is not attributable to paired optical geometry.
- SiXiang paired class-score KD, paired DFL and shuffled DFL all reduce mAP;
  neither frozen M0 component gate passes.
- No dataset has both valid classification contrast and paired geometry.
  Consequently D3/D4, R0--R5, seeds 0/123 and sealed-confirm evaluation were
  correctly not run.

Constraints for future work:

- Do not describe the current result as semantic or paired geometric knowledge
  transfer, and do not rescue the four-state router by tuning thresholds or
  loss weights.
- If this evidence motivates a future study, formulate it as a new,
  classification-only confidence/prototype-calibration question and validate
  it independently across seeds. It is not a continuation of the failed router
  claim.

## 2026-08-29 — OGSOD + SpaceNet6 OS-SSL concept-axis Q0a

- Intended claim: exact paired OS-SSL uniquely retains a target-related EO
  object-versus-local-context direction in SAR representations more strongly
  than Native, SAR-only SSL, and shuffled-pair SSL on both public datasets.
- Evidence: eight seed42 module-10 feature extractions over the frozen D0
  property-fit/property-confirm roles and a single frozen 10,000-draw analysis.
- Integrity status: pass. All jobs completed on the first attempt, features were
  finite, four-arm rosters were identical, context coverage exceeded 95%, and
  no method-dev, sealed-confirm, or official-test data were accessed.
- Verdict: `no`, confidence high; route `KILL_CONCEPT_AXIS_RETENTION_V1`.

What the results support:

- The paired arm has a readable EO object-context axis on both datasets:
  OGSOD `0.7156 [0.7024, 0.7285]`, SpaceNet6
  `0.7598 [0.7050, 0.8145]`.
- That EO-derived direction is also readable in SAR coordinates:
  OGSOD `0.6918 [0.6783, 0.7052]`, SpaceNet6
  `0.6939 [0.6330, 0.7535]`.
- Retention ratios are positive and stable: OGSOD
  `0.8895 [0.8374, 0.9426]`, SpaceNet6
  `0.7466 [0.5810, 0.8783]`.
- OGSOD satisfies the frozen class check, and all SpaceNet6
  leave-one-acquisition effects remain positive.

What failed:

- Paired minus the replicate-wise best control is only
  `+0.0041 [-0.0151, +0.0164]` on OGSOD and
  `+0.0168 [-0.0305, +0.0517]` on SpaceNet6. Both miss the frozen `0.03`
  threshold and both lower bounds cross zero.
- Only `2/5` deterministic folds are positive on each dataset, below the
  required `4/5`.
- The best point-estimate control is Native on both datasets. Absolute axis
  readability therefore does not establish a paired-specific mechanism.

Constraints for future work:

- Do not expand this qualification to seeds 0/123 or detector AP.
- Do not rescue it with another layer, ROI, sign, threshold, fold, or null.
- Property-confirm is held out from fitting the EO axis but is not guaranteed
  unseen by the already-trained 10k SSL checkpoints; do not call it independent
  pretraining generalization.
- A future method experiment requires a separately measured, newly frozen data
  property. It cannot be presented as continuation of concept-axis retention
  v1.

Primary synthesis: `runs/osssl_concept_axis_q0_20260829/Q0A_REPORT.md`.

## 2026-08-30 — OGSOD capability v1 grouping preflight

- Intended claim: 冻结的 D4 分组能否支撑固定 SAR 候选库能力实验。
- Evidence: `runs/ogsod_capability_v1_20260830_attempt1_reviewfix_20260830_203254/`（decision.json、M0 evidence）。
- Verdict: `DEFER_GROUPING`，stop 在 `M0_GROUPING_PREFLIGHT`，`cap_d_results_read=false`、`formal_training_started=false`。分组依据不足，不作为科学失败，交由直接元数据审计重建。
- 后续：直接元数据审计（`runs/ogsod_capability_metadata_20260830/`）发现 7,668 个内容家族，产出 7,168/3,072/2,120 的家族隔离重分区；独立审计终态 WARN，限定为 diagnostic-only。

## 2026-08-31 — OGSOD capability v2 formal CAP-D

- Intended claim: 固定 SAR proposals + class-specific candidate bank 上存在有实用规模的 paired RGB 决策增量（C1）及 SAR-reachable 分量（C2）。
- Evidence: `runs/ogsod_capability_v2_clean_20260831/gates/cap_d.json`（10k bootstrap CI）与 `refine-logs/ogsod_capability_v2_formal/EXPERIMENT_TRACKER.md`。
- Verdict: `KILL_NO_PRACTICAL_FIXED_BANK_HEADROOM`（TERMINAL，2026-08-31 05:27 CST）。joint mAP50-95 headroom `+1.345 pp [0.699, 2.179]` 未达 `+2.0 pp` 门；class `+0.866 pp`、veto `+0.924 pp` 均未达 `+1.0 pp` 门。CI 下界均为正——存在统计显著但低于实用阈值的小效应；这不是"无效应"，而是"效应不够实用"。
- Gate 后果：RGB teacher/SAR donor（seed3407）被 gate 中止弃用，aux cache 与 CAP-A/B/C 未运行。禁止以调阈值或换 null 复活。

## 2026-08-31 — RGBT-P3-CAUSAL-v1 LLVIP core gate

- Intended claim: IR→visible 方向上，paired P3 DFL 的收益严格超过 native、same-modal、shuffled、random-dose 全部对照（15 个比较全正）。
- Evidence: `/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/`（metrics_record.json × 5 臂 + queue_state.json）。
- Verdict: `KILL_RGBT-P3-CAUSAL-v1`。seed42 dev 上 paired−对照（pp）：native `+2.539/−0.281/+0.626`，random_dose `+1.506/−0.212/+0.206`，same_modal `−2.185/−2.085/−1.712`，shuffled `−3.653/−1.102/−1.888`（AP50/AP75/mAP）。paired 输给两个 null 对照的全部指标，AP75 对 native 也为负；KILL 无争议。
- 注意：该 gate 是 v2 时代单方法因果门，只关闭"这一方法在这一数据集的 v1"；按 `plans/RGBT_CAMPAIGN_PLAN_V3.md`，跨模态蒸馏方向由 Hnewa-inspired feature-MSE 统一栈重新裁决。

## 2026-09-01 — RGBT-P3-CAUSAL-v1 DroneVehicle core gate 与战役终结

- Intended claim: 同上，DroneVehicle IR→rgb 方向。
- Evidence: 同一 run root；全 track 37/37 作业 COMPLETED。
- Verdict: `KILL_RGBT-P3-CAUSAL-v1`。seed42 val 上 paired−对照（pp）：native `+0.061/−0.264/−0.294`，random_dose `−0.537/−1.215/−0.855`，same_modal `−0.359/−1.173/−0.552`，shuffled `+0.141/−1.025/−0.428`。mAP 输给全部四个对照，AP75 全负；KILL 无争议。两个数据集的 anchor 方向门均 PASS（LLVIP IR>visible、DroneVehicle IR>rgb，native 差 +5.8 mAP pp），说明失败在方法而非方向选择。
- 运维记录：队列驱动 PID 272411 曾被发现处于 SIGSTOP 约 15 小时（4 个已完成训练成僵尸）；2026-09-01 按 v3"自然完成"指令恢复，驱动自动回收并跑完全部剩余作业。
