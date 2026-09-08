# Receipt-only prospective N/C0 readout

Frozen before any new AP read. Scope DRONE_SUBSET2048_PRETRAIN_E8_CHECK; endpoint SUBSET2048_PRETRAIN_E8_LAST_EMA; DroneVehicle seed42, two arms N/C0, coefficients0/.1, normal BN training, common generic pretrained initialization, fixed2048 subset, independent8-epoch schedule and full1469-image/22462-GT/5-class dev.

Source is each collected evaluations/{N,C0}/short_evaluation_receipt.json (SUBSET_SCREEN_EVALUATION_COMPLETED). Failure file is short_evaluation_failure.json (SUBSET_SCREEN_EVALUATION_FAILED). Both files imply CONFLICT, neither MISSING; neither case is scored zero or selected as a best attempt. Invalid completed identity/metrics rejects the readout. No checkpoint or raw predictions are read.

Validate overall/per-class fractions and class-ID macro means; fixed arm coefficient, BN mode, initialization stat, train subset mapping paths, training config/completion stat, full population and independent horizon. N/C0 must share all three initialization identities and three subset paths. Actual training/queue receipts separately establish pixel/initial-state execution equality; this analyzer does not re-prove it from a filename.

Only C0 minus N, in pp=(fraction_C0-fraction_N)*100; display percent=fraction*100. Per-class alignment uses class_id, not row order. Precision/recall are native operating-point summaries, not fixed-.25 miss counts. n_seeds=1, SD=null. Primary direction is descriptive only: known-positive-control screener validation, no new method causal claim, no general ranking validation or automatic E200 admission. Missing/failure/conflict deltas remain null.

CLI python analyze_subset_e8.py --campaign COLLECTED_CAMPAIGN --output NEW. Copy source into output. Only synthetic CPU truths run before source acceptance; no new AP, GPU, SSH, hashes, trainer or queue modifications.
