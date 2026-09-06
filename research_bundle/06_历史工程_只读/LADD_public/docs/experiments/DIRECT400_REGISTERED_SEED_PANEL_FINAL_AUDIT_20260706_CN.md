# Direct-400 Registered Seed Panel Final Audit 20260706

状态：`registered_seed_panel_final_audit_v1__nine_rows_completed_final_audited__rescue_failed__no_claim_upgrade`。

本文件收口审计 `Registered 0/42/123 direct-400 rescue seed panel dense launch v1` 的 3090 YOLO11n pure direct-400 rescue panel。范围只包含 det-only / plain / singleproj x seeds `0/42/123` 共 9 行。所有事实均来自只读核查；没有启动、停止、重启、清理、复制大文件、改训练代码、改结果数值或升级 claim。

## Audit Inputs

- Remote summary: `/tmp/ladd_direct400_rescue_audit_20260706/remote_summary.json`
- Row-level final audit CSV: `figures/jstars_v1/tables/T-DIRECT400_REGISTERED_SEED_PANEL_FINAL_AUDIT_20260706.csv`
- Curve CSV: `figures/jstars_v1/tables/T-DIRECT400_REGISTERED_SEED_PANEL_CURVES_20260706.csv`
- Diagnostic plot: `figures/jstars_v1/plots/P-DIRECT400_REGISTERED_SEED_PANEL_AP_CURVES_20260706.png`
- Launch status source: `docs/experiments/DIRECT400_REGISTERED_SEED_PANEL_LAUNCH_STATUS_20260706_CN.md`
- Rescue plan source: `docs/experiments/DIRECT400_LADD_RESCUE_EXPERIMENT_PLAN_20260706_CN.md`

## Final Fact Audit

所有 9 行均为 `completed_final_audited`：

- `args.yaml` 存在，核心 direct-400 字段为 `epochs=400`, `imgsz=256`, `batch=64`, `mosaic=0.0`, `close_mosaic=0`, `cos_lr=true`, `deterministic=true`, `save_period=-1`, `optimizer=MuSGD`, `lr0=0.001`, `lrf=0.01`, `warmup_epochs=0.0`。
- LADD plain / singleproj 的 method identity、BN freeze、phase-B source、teacher/source 参数由 command file 和 outer log 共同确认；部分自定义字段未回写进 `args.yaml`，因此不能只靠 `args.yaml` 判断 LADD wrapper。
- `results.csv` 全部存在，均为 exactly 400 rows，latest epoch 均为 400。
- `weights/best.pt` 和 `weights/last.pt` 全部存在；`epoch*.pt` count 均为 0，符合 `SAVE_PERIOD=-1`。
- 严格 scoped process match 均为 0，训练已自然退出。
- outer log fatal / OOM / NaN / Traceback scan 均为 0 hit；`results.csv` NaN value count 均为 0。
- AP 字段已确认读取自 `metrics/mAP50(B)` 和 `metrics/mAP50-95(B)`，未混用其它 header。

| Method | Seed | GPU | Rows | Final epoch | Final AP50 | Final AP50-95 | Best AP50 | Best AP50-95 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| det-only | 0 | 0 | 400 | 400 | 0.77340 | 0.50871 | 0.77368 | 0.50939 | `completed_final_audited` |
| det-only | 42 | 1 | 400 | 400 | 0.77195 | 0.50648 | 0.77313 | 0.50682 | `completed_final_audited` |
| det-only | 123 | 0 | 400 | 400 | 0.77882 | 0.50692 | 0.78026 | 0.50851 | `completed_final_audited` |
| plain | 0 | 1 | 400 | 400 | 0.67980 | 0.42015 | 0.68228 | 0.42273 | `completed_final_audited` |
| plain | 42 | 0 | 400 | 400 | 0.67433 | 0.41942 | 0.67655 | 0.42078 | `completed_final_audited` |
| plain | 123 | 1 | 400 | 400 | 0.68412 | 0.42665 | 0.68724 | 0.42832 | `completed_final_audited` |
| singleproj | 0 | 0 | 400 | 400 | 0.69442 | 0.44358 | 0.69543 | 0.44430 | `completed_final_audited` |
| singleproj | 42 | 1 | 400 | 400 | 0.69704 | 0.43970 | 0.69769 | 0.44048 | `completed_final_audited` |
| singleproj | 123 | 0 | 400 | 400 | 0.68732 | 0.43262 | 0.68740 | 0.43396 | `completed_final_audited` |

Best epoch, full run path, `args.yaml`, `results.csv`, outer log path, AP header names, checkpoint presence, PID match count, and scan counts are recorded in `figures/jstars_v1/tables/T-DIRECT400_REGISTERED_SEED_PANEL_FINAL_AUDIT_20260706.csv`.

Seed42 / seed123 LADD rows use seed0 RGB teacher / A1 source reuse. They are therefore labeled `seed0_source_reuse__not_strict_same_seed_source` and must not be presented as strict same-seed-source evidence.

## Matched Result Table

| Seed | det AP50 | det AP50-95 | plain AP50 | plain AP50-95 | singleproj AP50 | singleproj AP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.77340 | 0.50871 | 0.67980 | 0.42015 | 0.69442 | 0.44358 |
| 42 | 0.77195 | 0.50648 | 0.67433 | 0.41942 | 0.69704 | 0.43970 |
| 123 | 0.77882 | 0.50692 | 0.68412 | 0.42665 | 0.68732 | 0.43262 |

Same-seed deltas:

| Seed | plain-det AP50 | plain-det AP50-95 | single-det AP50 | single-det AP50-95 | single-plain AP50 | single-plain AP50-95 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.09360 | -0.08856 | -0.07898 | -0.06513 | +0.01462 | +0.02343 |
| 42 | -0.09762 | -0.08706 | -0.07491 | -0.06678 | +0.02271 | +0.02028 |
| 123 | -0.09470 | -0.08027 | -0.09150 | -0.07430 | +0.00320 | +0.00597 |

Mean / sample SD over seeds `0/42/123`:

| Delta | Mean | Sample SD |
|---|---:|---:|
| plain - det AP50 | -0.09531 | 0.00208 |
| plain - det AP50-95 | -0.08530 | 0.00442 |
| singleproj - det AP50 | -0.08180 | 0.00865 |
| singleproj - det AP50-95 | -0.06874 | 0.00489 |
| singleproj - plain AP50 | +0.01351 | 0.00980 |
| singleproj - plain AP50-95 | +0.01656 | 0.00931 |

Reading: singleproj is consistently above plain in this rescue panel, but both LADD variants are far below the matched det-only control. This panel is negative rescue evidence, not a LADD recovery.

## Failure Triage

Compared with prior evidence, this rescue panel changed several protocol axes at once:

| Axis | Current rescue panel | Prior direct-400 / row400 / 800 context | Interpretation |
|---|---|---|---|
| Schedule | exact 400, `MuSGD`, `lr0=0.001`, `lrf=0.01`, `warmup=0`, slow cosine tail | old direct-400 used the earlier exact-400 command family; row400 / 800 evidence came from longer training dynamics | det-only is only mildly lower than old direct-400, but LADD collapses much more, so schedule alone is unlikely to explain all damage |
| BN policy | LADD plain / singleproj use `--freeze-bn-stats --freeze-bn-after-epoch -1`; logs confirm `bn_stats_mode=always_freeze` | old direct-400 final facts did not use this exact BN-freeze rescue wrapper | BN freeze did not rescue LADD and may have frozen a poor B-stage feature distribution |
| Phase family | LADD rows are phase-B rescue rows from existing A1/decomp source | prior full-chain / row400-from-800 positives may depend on A1/A2/B trajectory and longer optimization | phase-B-only rescue is not equivalent to full-chain evidence |
| Source policy | seed42/123 LADD rows reuse seed0 RGB/A1 source | strict same-seed-source was not available for these rescue LADD rows | seed42/123 are speed-first rescue diagnostics, not strict source-matched LADD evidence |
| Student detector source | LADD rows use `yolo11n.pt` as detector source in the B wrapper | comparison/KD and some historical flows used different detector or stage sources | source mismatch can make B-stage LADD start from a weaker detector state |
| Loss balance | LADD loss stack remains active with KD / decomposition / aux terms; early logs show warmup scaling | previous positives do not isolate which term was beneficial | collapse may be loss-balance or aux/KD over-constraint, not only schedule/BN |
| Save policy | `SAVE_PERIOD=-1`; no intermediate checkpoints | not a learning protocol change | storage fix worked; it does not explain AP collapse |

Most likely failure causes:

1. BN freeze plus phase-B-only loading likely locked in a weak or mismatched feature distribution instead of preventing collapse.
2. Slow-cosine / LR-tail schedule with zero warmup did not help LADD optimization; det-only degradation is small, but LADD degradation is severe.
3. The rescue wrapper is not a matched full-chain run. Seed0 uses an existing A1/decomp source, while seed42/123 reuse seed0 source, so source mismatch is a real interpretation risk.
4. LADD loss weights or auxiliary constraints may still over-constrain detector learning under exact-400, especially with frozen BN.
5. Historical positive signals may be tied to longer 800ep trajectory or row400-from-800 dynamics rather than an exact-400 rescue setting.

No hard protocol bug was found in this audit: no wrong AP header, no short row count, no live scoped process, no missing `best.pt` / `last.pt`, no fatal/OOM/Traceback/NaN hit, no accidental `epoch*.pt` save burst, and no obvious command mismatch against the intended rescue wrapper. The main blockers are interpretation and method failure, not audit invalidation.

## Curve And Diagnostic Artifacts

The curve package was generated only for diagnosis:

- Final audit CSV: `figures/jstars_v1/tables/T-DIRECT400_REGISTERED_SEED_PANEL_FINAL_AUDIT_20260706.csv`
- Curve CSV: `figures/jstars_v1/tables/T-DIRECT400_REGISTERED_SEED_PANEL_CURVES_20260706.csv`
- AP curve plot: `figures/jstars_v1/plots/P-DIRECT400_REGISTERED_SEED_PANEL_AP_CURVES_20260706.png`

The AP curves show det-only staying well above both LADD variants after early training. singleproj is usually above plain, but never approaches det-only.

For historical comparison, use these already documented sources instead of mixing claims here:

- `docs/experiments/DIRECT400_FINAL_FACT_MATCHED_GAIN_ANALYSIS_20260704_CN.md` for old pure direct-400 final facts.
- `docs/experiments/DIRECT400_LADD_RESCUE_EXPERIMENT_PLAN_20260706_CN.md` for row400-from-800 and pure800 context used to motivate rescue.
- `docs/experiments/DIRECT400_COMPLETED_ROW_FINAL_FACT_AUDIT_20260703_CN.md` for old per-row final fact audit provenance.

## Recommendation

Must check first, without launching new training:

1. Compare the rescue command family with the old direct-400 command family and isolate phase-B-only / BN-freeze / LR-tail differences.
2. Audit `b_split_load_manifest.json`, source paths, teacher weights, and detector source compatibility for the rescue LADD rows.
3. Inspect existing logs for loss-term magnitudes, BN-freeze mode, and whether detection loss or KD/decomp terms dominate late training.

Small diagnostics that may be considered only after user confirmation:

1. One seed0 minimal ablation that removes exactly one suspected factor, such as BN freeze or LR-tail, while keeping det-only / plain / singleproj matched.
2. One seed0 positive-control feature-KD or reduced-auxiliary-loss diagnostic if the log audit suggests LADD losses are over-constraining.
3. One strict full-chain seed0 reproduction if source/phase-B-only mismatch is the leading hypothesis.

Do not do now:

1. Do not extend this same rescue configuration to more seeds.
2. Do not hide old `0/1/2` results or present only seed0.
3. Do not expand CCLKD while its online seed0 remains blocked by NaN risk.
4. Do not launch source-policy-gated YOLO11s / YOLO11m LADD rows until policy is confirmed.
5. Do not claim LADD recovery or write paper text from this panel.

## Boundary

This audit establishes that the 9 rescue rows are final audited facts, and that the registered rescue panel failed its purpose. It does not upgrade any claim, does not compute cross-server gain, and does not convert diagnostics into paper-ready evidence.
