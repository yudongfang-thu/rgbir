# Fixed-sample registration illustrations

CPU-only generation from the completed v2 baseline probe. Each dataset uses the exact sample from feature_case_1 (RGB luminance quantile 0.10). The local window comes from the lower median RGB GT area, with index tie breaking; selection does not use registration, predictions, or feature similarity.

Cyan shows RGB gradients/GT and magenta shows IR or NIR gradients/GT. Shared labels are not independent registration evidence. Gradient differences include modality response differences. No estimated alignment is applied. Original source hashes were checked, and source images/labels remain unchanged.

The per-dataset JSON files contain selected IDs, ROI rules, source hashes, display details and output hashes. The source script and completion receipt are preserved in this directory.
