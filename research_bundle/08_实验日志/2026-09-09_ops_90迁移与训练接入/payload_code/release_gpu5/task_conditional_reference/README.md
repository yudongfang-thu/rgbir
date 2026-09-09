# RGBIR Task-Conditional v1

Independent experiment module. The vendored `legacy_oev1/` is the actual running release snapshot; its C loss and native recipe remain unchanged.

Entry points:

- `diagnose_opportunities.py`: train/dev D1 and actual raw-anchor D2, explicit unverified geometry status.
- `geometry_audit_tools.py`, `geometry_contract.py`: frozen sampling and accepted physical-point/region coverage.
- `localization_loss.py`: fixed-reference DFL L, same-mask GT, equal-K random and zero controls.
- `calibrate_gradient_scale.py`: frozen 64-batch, R-state gradient calibration; no formal coefficient if all L gradients are zero.
- `train_task_conditional.py`: N/C/L/CL/CGT/random-L; C-shuffled and C-same-modal attribution fallbacks. All require a bound existing project lease. Formal localization has additional geometry/D2/calibration/canary gates.
- `shuffled_pair_data.py`, `content_controls.py`: train-only derangement and paired-selection/content separation. CL shuffled/same-modal helpers are preparation assets; their complete formal trainer integration is not claimed as accepted.
- `verify_loader_equivalence.py`, `verify_criterion_equivalence.py`: real-data/dataflow and real-model exact C/N checks.
- `evaluate_task_conditional.py`, `analyze_results.py`: fixed last/EMA development evaluation, per-class metrics, receipt/recipe-bound seed pairing and sample SD.
- `resource_dispatch.py`: screen stage worker over the existing shared resource guard; no independent GPU pool, no automatic launched-attempt retry or batch change.

Current protocol and evidence: [execution review](../../../../08_实验日志/2026-09-07_train_TaskConditional首轮/EXECUTION_REVIEW.md), [frozen plan](../../../../08_实验日志/2026-09-07_train_TaskConditional首轮/EXPERIMENT_PLAN.md). Actual L geometry is not sufficient for long training; draft configs deliberately cannot authorize it. No test-set data are accessed.
