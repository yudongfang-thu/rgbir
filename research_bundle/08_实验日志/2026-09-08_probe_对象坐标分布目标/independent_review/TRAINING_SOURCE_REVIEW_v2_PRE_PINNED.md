# Training source v2: calibration lifetime correction

**READY_FOR_PINNED_CPU**, only for the new v2 source snapshot. Of all 18 running/source/config/AMP entries, only calibrate_object_dfl.py changed; 17 remain byte-identical to the previously accepted version. V1 history and attempt1 failure are retained.

The source diff removes the final loop `loss`, `stats` and explicit `native_items` references together with the original graph cleanup and adds synchronized batch allocation diagnostics. It does not change losses, gates, fixed8 flow, dose calibration or resource limits. Independent saved-tensor-context CPU evidence verifies release before next forward with identical gradient norms. The affected 15 wrapper tests pass. This is a technical correction, not an outcome-based retuning.

Exact files are in training_reviewed_source_v2/ and TRAINING_SOURCE_REVIEW_v2.json. The separate v1 snapshot remains unchanged. Next is exact new release_v2 deployment and pinned affected CPU checks, then a new readiness record for attempt2. Only the actual same fixed8 replay can verify whether production memory growth is resolved; canary and short-training gates remain required afterwards.
