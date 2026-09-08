# Pinned CPU acceptance addendum

**READY_FOR_GPU_PREFLIGHT** for the exact deployed release_v1. All 18 reviewed operational Python/config/AMP input files still match `training_reviewed_source/` directly byte for byte and are included in the deployment receipt as byte-exact. No new hashes were computed.

The actual pinned Python completed 36 CPU checks: L3 14, wrappers/criterion 15, queue 7. The first discovery attempt passed 22 tests and failed only to import the L3 test module without its required CLI arguments; the original failure remains preserved. The unchanged L3 source then passed its 14 tests through the declared CLI. Both original logs/receipts were read independently. This is not a hidden production-source repair.

Readiness permits starting only the queue's encoded sequence: fixed8 calibration with actual per-batch resource/dose gates, all three actual canaries, then guarded fresh three-arm FT3/evaluation. It does not say those CUDA stages have already run or passed. Failed runtime gates retain their attempt and stop the candidate according to the frozen protocol. No physical-registration or KD-effect claim follows from source/CPU acceptance.

Authoritative current status and exact paths are in `TRAINING_SOURCE_REVIEW.json`; the earlier pending-pinned record is retained as `TRAINING_SOURCE_REVIEW_PRE_PINNED.json`.
