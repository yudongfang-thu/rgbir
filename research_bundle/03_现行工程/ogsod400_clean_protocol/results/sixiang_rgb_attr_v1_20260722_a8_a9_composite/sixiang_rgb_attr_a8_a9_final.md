# SiXiang RGB attribution A8+A9 temporal-block repair analysis

- Integrity: **PASS**
- Evidence class: **production_composite_subprocess_pre_post**
- Formal-report eligible: **True**
- Decision: **EXPLORATORY_PILOT**
- Composition: A8 contributes 10 cells; A9 replaces paired_s0 and weight0_s123.
- Replication: still three optimizer seeds per arm; A9 adds zero independent replications.
- Limitation: two cells are from a later temporal/device execution block.
- Scope: SiXiang scene-clean val development evidence from an A8+A9 temporal-block repair; no extra replication, OGSOD/public-dataset generalization, final-test, identifiable shared/private, or causal-mechanism claim.

## Raw selected-run table

| Arm | Seed | Source | e300 P | e300 R | e300 F1 | e300 AP50 | e300 AP50-95 | late10 AP50-95 | best epoch | late10 SD | KD +epochs | KD 0epochs |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hs | 0 | sixiang_rgb_attr_v1_20260722_a8 | 0.655830 | 0.622230 | 0.638588 | 0.650840 | 0.397990 | 0.396519 | 246 | 0.000832 | 300 | 0 |
| hs | 42 | sixiang_rgb_attr_v1_20260722_a8 | 0.762930 | 0.620310 | 0.684268 | 0.668290 | 0.413140 | 0.411268 | 262 | 0.001359 | 300 | 0 |
| hs | 123 | sixiang_rgb_attr_v1_20260722_a8 | 0.702390 | 0.615140 | 0.655876 | 0.655830 | 0.399230 | 0.399267 | 271 | 0.000407 | 300 | 0 |
| paired | 0 | sixiang_rgb_attr_v1_20260722_a9 | 0.820890 | 0.570270 | 0.673005 | 0.685650 | 0.411660 | 0.413073 | 292 | 0.001979 | 300 | 0 |
| paired | 42 | sixiang_rgb_attr_v1_20260722_a8 | 0.638500 | 0.647250 | 0.642845 | 0.670180 | 0.413490 | 0.412715 | 225 | 0.001506 | 300 | 0 |
| paired | 123 | sixiang_rgb_attr_v1_20260722_a8 | 0.799380 | 0.576320 | 0.669766 | 0.667860 | 0.416730 | 0.415708 | 298 | 0.001087 | 300 | 0 |
| shuffled | 0 | sixiang_rgb_attr_v1_20260722_a8 | 0.653100 | 0.607380 | 0.629411 | 0.650320 | 0.385770 | 0.386345 | 199 | 0.000608 | 300 | 0 |
| shuffled | 42 | sixiang_rgb_attr_v1_20260722_a8 | 0.782230 | 0.519700 | 0.624496 | 0.569330 | 0.342290 | 0.342545 | 132 | 0.000454 | 300 | 0 |
| shuffled | 123 | sixiang_rgb_attr_v1_20260722_a8 | 0.618820 | 0.554430 | 0.584858 | 0.579100 | 0.343910 | 0.343073 | 146 | 0.000653 | 300 | 0 |
| weight0 | 0 | sixiang_rgb_attr_v1_20260722_a8 | 0.777190 | 0.571570 | 0.658706 | 0.637770 | 0.379660 | 0.378152 | 199 | 0.000970 | 0 | 300 |
| weight0 | 42 | sixiang_rgb_attr_v1_20260722_a8 | 0.808870 | 0.576470 | 0.673177 | 0.641440 | 0.402710 | 0.401888 | 225 | 0.001243 | 0 | 300 |
| weight0 | 123 | sixiang_rgb_attr_v1_20260722_a9 | 0.791570 | 0.570470 | 0.663074 | 0.649850 | 0.409980 | 0.408908 | 246 | 0.000993 | 0 | 300 |

## Preregistered gates

| Gate | Control | endpoint deltas s0/s42/s123 | mean +/- sample SD | late10 deltas s0/s42/s123 | Gate |
|---|---|---|---|---|---|
| G1_paired_vs_hs | hs | +0.013670/+0.000350/+0.017500 | +0.010507 +/- 0.009002 | +0.016554/+0.001447/+0.016441 | PASS |
| G2_paired_vs_shuffled | shuffled | +0.025890/+0.071200/+0.072820 | +0.056637 +/- 0.026640 | +0.026728/+0.070170/+0.072635 | PASS |
| G3_paired_vs_weight0 | weight0 | +0.032000/+0.010780/+0.006750 | +0.016510 +/- 0.013565 | +0.034921/+0.010827/+0.006800 | PASS |

## Interpretation and boundary

All frozen SiXiang-val efficacy, pairing, and effective-dose gates passed.

The three hard gates are unchanged from A8. AP50, P/R/F1, losses, best epoch, diagnostics, and stability remain descriptive.
The repaired composite is development evidence and must be reported with its temporal-block limitation.
