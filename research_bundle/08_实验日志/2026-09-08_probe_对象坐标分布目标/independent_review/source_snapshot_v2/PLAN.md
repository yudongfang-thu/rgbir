# Fixed CPU implementation

Implements ../2026-09-08_probe_DFL真实信息读出/NEXT_TARGET_INTERFACE_PLAN.md without changing its estimand. Only the accepted evidence_1602_final/probe cache: 80 GT objects, 317 unique distributions, one current_forward_id. No new forward, matching, AP, CE-based selection, training, hash or expanded sample.

Two fixed interfaces: S/historical_R_candidate destination with T/same_historical_R_index (primary), and the same S destination with T/own_native_iou50 (pressure only). Missing roles remain missing. All four sides must be supported for an object to have a target.

T=1 probabilities use stable Python float64 softmax from all sixteen saved logits. Finite logits imply strictly positive mathematical mass: floating exp underflow is an explicit rejection, never implicit tail deletion. The only normalization is the defining source softmax; transported targets are not renormalized.

Geometry uses exact rational arithmetic on the saved finite binary scalar values. The object affine map reduces to dS(j)=a*j+b. Identity is exactly a=1,b=0; support tests are exact rational comparisons to [0,15], without epsilon. Each supported source mass is split across floor(dS) and ceil(dS); integer/last-bin cases use one bin. A positive source mass outside support rejects the entire target. Preserve source vectors, mapped distances, rejected bins/mass, affine coefficients, input/target mass, expectations and edge-coordinate closure. No clamp, truncation, anchor movement or probability repair.

Source/tests are reviewed before the actual cache run. CLI: python run_cached_transport.py --input ACCEPTED_PROBE --output NEW_DIRECTORY. Output existence is rejected. summary.json distinguishes execution completion from ALL_AVAILABLE_SUPPORTED/PARTIAL_SUPPORT/NO_SUPPORTED_TARGET; there is no automatic training admission or new coverage cutoff. Per-object paths and original role/anchor/GT identities remain explicit. Independent review is written by /root/object_transport_review outside this directory.
