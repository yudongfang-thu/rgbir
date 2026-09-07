#!/usr/bin/env python3
"""CPU-only analysis of a frozen, GT-associated baseline export. Never computes AP.

Inputs: objects.jsonl, features.npz, logits.npz, optional metadata.json.
Outputs are new files under a separate output directory; source files are read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

VERSION = "baseline-opportunity-probe-v1"
SEED = 20260907
ALPHA = 1.0
CONFIDENCE = 0.25
HIT_IOU = 0.50
PAIR_IOU = 0.50
PROJECTION_DIM = 128
TEMPERATURE = 2.0
NATIVE_DFL_UPPER = 14.99


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not rows:
        raise ValueError("objects.jsonl contains no rows")
    ids = [r["object_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("object_id must be globally unique within dataset")
    if any(r.get("split") not in ("train", "val", "dev") for r in rows):
        raise ValueError("split must be train, val or dev")
    for r in rows:
        r["split"] = "val" if r["split"] == "dev" else r["split"]
    return rows


def load_arrays(path, n):
    with np.load(path, allow_pickle=False) as f:
        result = {k: f[k] for k in f.files}
    for k, arr in result.items():
        if arr.ndim < 1 or len(arr) != n:
            raise ValueError(f"{path.name}:{k}: first axis must equal {n}")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{path.name}:{k}: numeric arrays required")
    return result


def description(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "sd": None, "q10": None, "q90": None}
    return {"n": len(a), "mean": a.mean(), "median": np.median(a),
            "sd": a.std(ddof=1) if len(a) > 1 else None,
            "q10": np.quantile(a, .1), "q90": np.quantile(a, .9)}


def ratio(n, d):
    return float(n / d) if d else None


def group_columns(rows, train):
    luma = np.array([r.get("mean_rgb_luma", np.nan) for r in rows], dtype=float)
    good = train & np.isfinite(luma)
    cuts = np.quantile(luma[good], [1 / 3, 2 / 3]) if good.any() else np.array([np.nan, np.nan])
    luma_bin = np.array(["missing" if not np.isfinite(x) else
                         ("low" if x < cuts[0] else "middle" if x < cuts[1] else "high") for x in luma])
    groups = {
        "source_group": np.array([str(r.get("source_group", "missing")) for r in rows]),
        "scale_bin": np.array([str(r.get("scale_bin", "missing")) for r in rows]),
        "luma_train_tercile": luma_bin,
        "anchor_reference": np.array(["annotation_background" if r.get("is_background", False) else
                                       "native_candidate" if r.get("anchor_has_reference_candidate", False) else "gt_center_fallback" for r in rows]),
        "object_pairing": np.array(["annotation_background" if r.get("is_background", False) else
                                     "paired_gt" if r.get("paired_gt_iou") is not None and r["paired_gt_iou"] >= PAIR_IOU else "unpaired_gt" for r in rows]),
    }
    return groups, {"cuts": cuts, "fitted_rows": int(good.sum()),
                    "method": "Train-row quantiles, labels/outcomes unused; repeated objects weight the image repeatedly."}


def object_view(row, model, same_anchor=False):
    m = row.get(model) or {}
    a = m.get("same_anchor") if same_anchor else m.get("assigned")
    if not a:
        return {"candidate": False, "confidence_ok": False, "class_ok": False,
                "localization_ok": False, "correct": False, "iou": np.nan,
                "state": "no_candidate"}
    iou = a.get("iou_to_rgb_gt") if same_anchor else m.get("iou_to_rgb_gt", a.get("iou_to_rgb_gt"))
    if iou is None:
        # assigned.iou can refer to IR GT and is deliberately never substituted.
        iou = np.nan
    confidence = float(a.get("confidence", np.nan))
    cls = a.get("pred_class", a.get("class"))
    c, y, b = confidence >= CONFIDENCE, cls == row["class"], float(iou) >= HIT_IOU
    if not c:
        state = "low_confidence"
    elif not y and not b:
        state = "class_and_localization"
    elif not y:
        state = "class_only"
    elif not b:
        state = "localization_only"
    else:
        state = "correct"
    return {"candidate": True, "confidence_ok": c, "class_ok": y,
            "localization_ok": b, "correct": c and y and b,
            "iou": float(iou), "state": state}


def opportunity_summary(rows, mask, same_anchor=False, teacher_model="T42", require_pair=True):
    selected = [r for r, use in zip(rows, mask) if use and not r.get("is_background", False)]
    n = len(selected)
    out = {"all_rgb_gt": n, "view": "same_rgb_reference_anchor" if same_anchor else "same_object_assigned",
           "reference_model": "N42", "comparison_model": teacher_model,
           "comparison_requires_ir_gt_pair": require_pair}
    if not n:
        return out
    native = [object_view(r, "N42", same_anchor) for r in selected]
    teacher = [object_view(r, teacher_model, same_anchor) for r in selected]
    matched_pair = np.array([r.get("paired_gt_iou") is not None and r["paired_gt_iou"] >= PAIR_IOU for r in selected])
    pair = matched_pair if require_pair else np.ones(n, bool)
    nc = np.array([v["correct"] for v in native])
    tc = np.array([v["correct"] for v in teacher])
    counts = Counter(v["state"] for v in native)
    out["native_states"] = {k: {"n": v, "fraction_all_gt": ratio(v, n)} for k, v in sorted(counts.items())}
    out["native_overlapping_error_flags"] = {
        k: {"n": sum(v["candidate"] and not v[k] for v in native),
            "fraction_all_gt": ratio(sum(v["candidate"] and not v[k] for v in native), n)}
        for k in ("confidence_ok", "class_ok", "localization_ok")}
    out["native_overlapping_error_flags_semantics"] = "Each *_ok key names a condition; n counts candidate AND NOT that condition (errors), not successes. Missing candidates are separate."
    out["paired_gt"] = {"n": int(matched_pair.sum()), "fraction_all_gt": ratio(matched_pair.sum(), n)}
    out["comparison_eligible_gt"] = {"n": int(pair.sum()), "fraction_all_gt": ratio(pair.sum(), n),
                                     "rule": "IR paired GT" if require_pair else "same RGB GT identity; no IR pairing required"}
    repair, harm = pair & ~nc & tc, pair & nc & ~tc
    out["teacher_correct_on_paired"] = int((pair & tc).sum())
    out["repair"] = {"n": int(repair.sum()), "fraction_all_gt": ratio(repair.sum(), n),
                     "fraction_native_errors": ratio(repair.sum(), (~nc).sum())}
    out["harm_if_teacher_replaces_native"] = {"n": int(harm.sum()), "fraction_all_gt": ratio(harm.sum(), n),
                                             "includes_teacher_absence": True}
    out["net_repair_minus_harm"] = {"n": int(repair.sum() - harm.sum()),
                                     "fraction_all_gt": ratio(repair.sum() - harm.sum(), n)}
    out["repairs_by_native_state"] = dict(Counter(native[i]["state"] for i in np.flatnonzero(repair)))
    out["harms_by_teacher_state"] = dict(Counter(teacher[i]["state"] for i in np.flatnonzero(harm)))
    ni = np.array([v["iou"] for v in native])
    ti = np.array([v["iou"] for v in teacher])
    common = pair & np.isfinite(ni) & np.isfinite(ti)
    out["iou_teacher_minus_native_common_candidate"] = description((ti - ni)[common])
    out["teacher_iou_better"] = {"n": int((common & (ti > ni)).sum()),
                                   "fraction_all_gt": ratio((common & (ti > ni)).sum(), n)}
    out["teacher_iou_worse"] = {"n": int((common & (ti < ni)).sum()),
                                  "fraction_all_gt": ratio((common & (ti < ni)).sum(), n)}
    # These factor-specific counts overlap and do not use a mutually exclusive error priority.
    factor_masks = {
        "class_fix_at_localized_native": [v["candidate"] and v["localization_ok"] and not v["class_ok"] for v in native],
        "low_confidence_fix_at_correct_class_localization": [v["candidate"] and v["class_ok"] and v["localization_ok"] and not v["confidence_ok"] for v in native],
        "localization_fix_at_native_confident_correct_class": [v["candidate"] and v["class_ok"] and v["confidence_ok"] and not v["localization_ok"] for v in native],
    }
    out["factor_specific_repair"] = {k: {"n": int((np.array(v) & pair & tc).sum()),
                                            "fraction_all_gt": ratio((np.array(v) & pair & tc).sum(), n)}
                                       for k, v in factor_masks.items()}
    return out


def opportunity_analysis(rows, groups):
    splits = np.array([r["split"] for r in rows])
    result = {}
    for split in ("train", "val"):
        base = splits == split
        result[split] = {}
        for same in (False, True):
            name = "same_anchor" if same else "assigned"
            entry = opportunity_summary(rows, base, same)
            entry["by_class"] = {str(c): opportunity_summary(rows, base & np.array([r["class"] == c for r in rows]), same)
                                 for c in sorted({r["class"] for r in rows if not r.get("is_background", False)})}
            entry["by_group"] = {key: {v: opportunity_summary(rows, base & (values == v), same)
                                      for v in np.unique(values[base])} for key, values in groups.items()}
            result[split][name] = entry
        if any("N0" in r for r in rows):
            result[split]["same_modal_N0_vs_N42"] = opportunity_summary(rows, base, teacher_model="N0", require_pair=False)
    return result


def standardize_fit(x, train):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Probe features/logits must be finite; no silent deletion or imputation")
    mean = x[train].mean(axis=0)
    std = x[train].std(axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std, mean, std


def fixed_projection(x, dimension=PROJECTION_DIM, seed=SEED):
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Each feature array must be finite N x D")
    # Same input dimension implies the same matrix across models, with no label access.
    p = np.random.default_rng(seed).standard_normal((x.shape[1], dimension)) / np.sqrt(dimension)
    return np.asarray(x, dtype=np.float64) @ p


def shuffled_within_split(x, train, dev, seed=SEED):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    for mask in (train, dev):
        ix = np.flatnonzero(mask)
        indices[ix] = rng.permutation(ix)
    return x[indices], indices


def ridge_fit_predict(x, y, train, classes, alpha=ALPHA):
    z, mean, std = standardize_fit(x, train)
    target = np.eye(classes, dtype=np.float64)[y[train]]
    intercept = target.mean(axis=0)
    a = z[train]
    # Objective: mean_i ||a_i W + b - onehot(y_i)||_2^2 + alpha ||W||_F^2.
    if a.shape[1] <= a.shape[0]:
        gram = a.T @ a / len(a)
        gram.flat[::len(gram) + 1] += alpha
        w = np.linalg.solve(gram, a.T @ (target - intercept) / len(a))
    else:
        gram = a @ a.T
        gram.flat[::len(gram) + 1] += len(a) * alpha
        w = a.T @ np.linalg.solve(gram, target - intercept)
    score = z @ w + intercept
    return score.argmax(axis=1), {"input_dim": x.shape[1], "constant_train_columns": int((np.std(x[train], axis=0) < 1e-12).sum()),
                                 "train_mean": mean, "train_std": std}


def classification_metrics(y, pred, classes):
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(np.diag(cm), support, out=np.full(classes, np.nan), where=support > 0)
    precision = np.divide(np.diag(cm), predicted, out=np.full(classes, np.nan), where=predicted > 0)
    present = support > 0
    macro = float(recall[present].mean()) if present.any() else None
    return {"n": len(y), "accuracy": ratio(np.trace(cm), len(y)), "macro_recall_present_classes": macro,
            "balanced_accuracy": macro, "classes_with_support": np.flatnonzero(present), "confusion_matrix": cm,
            "per_class": {str(c): {"support": support[c], "predicted": predicted[c], "correct": cm[c, c],
                                    "recall": recall[c], "precision": precision[c]} for c in range(classes)}}


def evaluation_views(y, pred, base, classes, background, groups):
    out = {"all": classification_metrics(y[base], pred[base], classes),
           "foreground": classification_metrics(y[base & ~background], pred[base & ~background], classes),
           "background": classification_metrics(y[base & background], pred[base & background], classes)}
    out["by_group"] = {key: {v: {"all": classification_metrics(y[base & (values == v)], pred[base & (values == v)], classes),
                                "foreground": classification_metrics(y[base & (values == v) & ~background], pred[base & (values == v) & ~background], classes)}
                            for v in np.unique(values[base])} for key, values in groups.items()}
    return out


def probe_analysis(rows, features, logits, groups, output):
    train = np.array([r["split"] == "train" for r in rows])
    dev = ~train
    if not train.any() or not dev.any():
        raise ValueError("Frozen train and held-out val rows are both required")
    nc = logits["N42_cls"].shape[1]
    background = np.array([r.get("is_background", False) for r in rows])
    y = np.array([r["class"] for r in rows], dtype=int)
    classes = nc + int(background.any())
    if (y < 0).any() or (y >= classes).any() or (y[background] != nc).any() or (y[~background] >= nc).any():
        raise ValueError("GT labels must be 0..nc-1; background label must equal nc")
    models_with_features = [m for m in ("N42", "T42", "N0") if m + "_P3" in features]
    roi_valid = np.array([all((r.get(m, {}).get("regions", {}).get(level, {}).get("valid", False))
                             for m in models_with_features for level in ("P3", "P4")) for r in rows])
    have_roi_metadata = bool(models_with_features) and all("regions" in r.get(m, {}) for r in rows for m in models_with_features)
    if not have_roi_metadata:
        # Never silently accept rows whose ROI validity is unknown.
        roi_valid[:] = False
    boxes = np.array([r["gt_box_input"] for r in rows], dtype=float)
    wh = boxes[:, 2:] - boxes[:, :2]
    stride = np.array([r["anchor_stride"] for r in rows], dtype=float)
    if (wh <= 0).any() or (stride <= 0).any():
        raise ValueError("Metadata-only probe requires positive ROI width/height and stride")
    blocks = {"N_logits": np.asarray(logits["N42_cls"], dtype=np.float64),
              "metadata": np.log(np.column_stack([wh, stride]))}
    for model in ("T42", "N0"):
        if model + "_cls" in logits:
            blocks[model + "_logits"] = np.asarray(logits[model + "_cls"], dtype=np.float64)
    for key, arr in features.items():
        if key.rsplit("_", 1)[-1] in ("P3", "P4"):
            blocks[key] = fixed_projection(arr)
    shuffle_indices, anchor_shuffle_indices = None, None
    for level in ("P3", "P4"):
        key = "T42_" + level
        if key in blocks:
            blocks[key + "_shuffled"], ix = shuffled_within_split(blocks[key], train & roi_valid, dev & roi_valid)
            if shuffle_indices is not None and not np.array_equal(ix, shuffle_indices):
                raise AssertionError("Shuffle alignment changed across levels")
            shuffle_indices = ix
        key = "T42_anchor_" + level
        if key in blocks:
            blocks[key + "_shuffled"], ix = shuffled_within_split(blocks[key], train, dev)
            if anchor_shuffle_indices is not None and not np.array_equal(ix, anchor_shuffle_indices):
                raise AssertionError("Anchor shuffles differ across levels")
            anchor_shuffle_indices = ix
    plans = {"N_logits": ["N_logits"], "metadata_only": ["metadata"], "N_logits+metadata": ["N_logits", "metadata"]}
    roi_plans = set()
    if (train & roi_valid).any() and (dev & roi_valid).any():
        plans["N_logits_ROI_valid"] = ["N_logits"]
        plans["metadata_only_ROI_valid"] = ["metadata"]
        roi_plans.update(("N_logits_ROI_valid", "metadata_only_ROI_valid"))
    for donor in ("T42", "N0"):
        if donor + "_logits" in blocks:
            plans["N_logits+" + ("T" if donor == "T42" else donor) + "_logits"] = ["N_logits", donor + "_logits"]
    for levels in (("P3",), ("P4",), ("P3", "P4")):
        plans_before = set(plans)
        name = "_".join(levels)
        native = ["N42_" + v for v in levels]
        teacher = ["T42_" + v for v in levels]
        if not all(k in blocks for k in native + teacher) or not (train & roi_valid).any() or not (dev & roi_valid).any():
            continue
        plans["N_feature_" + name] = native
        plans["N_logits+N_feature_" + name] = ["N_logits"] + native
        plans["N_logits+T_feature_" + name] = ["N_logits"] + teacher
        plans["N_logits+shuffled_T_feature_" + name] = ["N_logits"] + [k + "_shuffled" for k in teacher]
        # Conditional information beyond both native logits and native features.
        plans["N_logits+N_feature+T_feature_" + name] = ["N_logits"] + native + teacher
        plans["N_logits+N_feature+shuffled_T_feature_" + name] = ["N_logits"] + native + [k + "_shuffled" for k in teacher]
        same = ["N0_" + v for v in levels]
        if all(k in blocks for k in same):
            plans["N_logits+N0_feature_" + name] = ["N_logits"] + same
            plans["N_logits+N_feature+N0_feature_" + name] = ["N_logits"] + native + same
        region_key = "N42_region_cls"
        if region_key in logits:
            region = np.asarray(logits[region_key])
            if region.ndim != 3 or region.shape[1] != 2 or region.shape[2] != nc:
                raise ValueError("N42_region_cls must have shape N x 2 x nc")
            rname = "N_region_logits_" + name
            blocks[rname] = np.concatenate([region[:, int(v == "P4"), :] for v in levels], axis=1)
            plans[rname] = [rname]
            plans[rname + "+N_feature"] = [rname] + native
            plans[rname + "+T_feature"] = [rname] + teacher
            for donor in ("T42", "N0"):
                key = donor + "_region_cls"
                if key not in logits:
                    continue
                region_donor = np.asarray(logits[key])
                if region_donor.shape != region.shape:
                    raise ValueError("All region_cls arrays must share N x 2 x nc shape")
                dname = ("T" if donor == "T42" else donor) + "_region_logits_" + name
                blocks[dname] = np.concatenate([region_donor[:, int(v == "P4"), :] for v in levels], axis=1)
                plans[rname + "+" + ("T" if donor == "T42" else donor) + "_region_logits"] = [rname, dname]
        roi_plans.update(set(plans) - plans_before)
    # Independent of GT box width/height, but centers still follow the GT-associated
    # reference-candidate contract and background annotation exclusion.
    for levels in (("P3",), ("P4",), ("P3", "P4")):
        name = "_".join(levels)
        native = ["N42_anchor_" + v for v in levels]
        teacher = ["T42_anchor_" + v for v in levels]
        if not all(k in blocks for k in native + teacher):
            continue
        plans["N_anchor_feature_" + name] = native
        plans["N_logits+N_anchor_feature_" + name] = ["N_logits"] + native
        plans["N_logits+T_anchor_feature_" + name] = ["N_logits"] + teacher
        plans["N_logits+shuffled_T_anchor_feature_" + name] = ["N_logits"] + [k + "_shuffled" for k in teacher]
        plans["N_logits+N_anchor_feature+T_anchor_feature_" + name] = ["N_logits"] + native + teacher
        plans["N_logits+N_anchor_feature+shuffled_T_anchor_feature_" + name] = ["N_logits"] + native + [k + "_shuffled" for k in teacher]
        for donor in ("T42", "N0"):
            if donor + "_logits" in blocks:
                plans["N_logits+N_anchor_feature+" + ("T" if donor == "T42" else donor) + "_logits_" + name] = ["N_logits"] + native + [donor + "_logits"]
        same = ["N0_anchor_" + v for v in levels]
        if all(k in blocks for k in same):
            plans["N_logits+N0_anchor_feature_" + name] = ["N_logits"] + same
            plans["N_logits+N_anchor_feature+N0_anchor_feature_" + name] = ["N_logits"] + native + same
    result = {"status": "DEGENERATE_SINGLE_CLASS" if len(np.unique(y[train])) < 2 else "COMPLETED",
              "classes": classes, "detector_classes": nc, "background_class": nc if background.any() else None,
              "train_rows": train.sum(), "val_rows": dev.sum(), "train_class_counts": np.bincount(y[train], minlength=classes),
              "val_class_counts": np.bincount(y[dev], minlength=classes),
              "missing_train_classes": sorted(set(range(classes)) - set(y[train].tolist())),
              "projection": {"dimension_per_level": PROJECTION_DIM, "seed": SEED, "distribution": "Gaussian N(0,1/d)",
                             "before_projection_standardization": False, "after_projection_standardization": "train only, per column"},
              "shuffle": {"seed": SEED, "within_split": True, "conditional_on_class": False,
                          "same_permutation_across_levels": True,
                          "roi_fixed_points_within_common_valid": int(((shuffle_indices == np.arange(len(rows))) & roi_valid).sum()) if shuffle_indices is not None else None,
                          "anchor_fixed_points": int((anchor_shuffle_indices == np.arange(len(rows))).sum()) if anchor_shuffle_indices is not None else None},
              "alpha": ALPHA, "objective": "mean squared one-hot error + alpha * Frobenius squared weights; intercept unpenalized",
              "cohorts": {"main_fixed_anchor": {"train": int(train.sum()), "val": int(dev.sum())},
                          "auxiliary_gt_roi_common_valid": {"validity_metadata_present": have_roi_metadata,
                              "models": models_with_features, "levels": ["P3", "P4"],
                              "train": int((train & roi_valid).sum()), "val": int((dev & roi_valid).sum()),
                              "excluded_train": int((train & ~roi_valid).sum()), "excluded_val": int((dev & ~roi_valid).sum()),
                              "train_class_counts": np.bincount(y[train & roi_valid], minlength=classes),
                              "val_class_counts": np.bincount(y[dev & roi_valid], minlength=classes)}},
              "selection": "All declared arms/levels reported; no hyperparameter or level selection on val.",
              "carrier_comparison_scope": "Compare teacher logits and teacher features on exactly the same cohort. Feature dimension, parameter count and effective regularization differ despite fixed alpha, so differences are linear readout results, not causal effects of KD bandwidth/carrier or deployable KD gains.",
              "arms": {}}
    predictions, transforms = {}, {}
    for name, keys in plans.items():
        print("Ridge arm:", name, flush=True)
        x = np.concatenate([blocks[k] for k in keys], axis=1)
        eligible = roi_valid if name in roi_plans else np.ones(len(rows), bool)
        pred, fit = ridge_fit_predict(x, y, train & eligible, classes)
        predictions[name] = pred
        transforms[name + "__mean"] = fit.pop("train_mean")
        transforms[name + "__std"] = fit.pop("train_std")
        result["arms"][name] = {"blocks": keys, "fit": fit,
            "cohort": "auxiliary_gt_roi_common_valid" if name in roi_plans else "main_fixed_anchor",
            "train_n": int((train & eligible).sum()), "val_n": int((dev & eligible).sum()),
            "val": evaluation_views(y, pred, dev & eligible, classes, background, groups)}
    for arm in result["arms"].values():
        base_name = "N_logits_ROI_valid" if arm["cohort"] == "auxiliary_gt_roi_common_valid" else "N_logits"
        base = result["arms"][base_name]["val"]
        arm["delta_reference_arm"] = base_name
        arm["delta_vs_N_logits_percentage_points"] = {}
        for subset in ("all", "foreground", "background"):
            arm["delta_vs_N_logits_percentage_points"][subset] = {
                k: 100 * (arm["val"][subset][k] - base[subset][k])
                if arm["val"][subset][k] is not None and base[subset][k] is not None else None
                for k in ("accuracy", "balanced_accuracy")}
    result["carrier_comparisons"] = []
    for levels in (("P3",), ("P4",), ("P3", "P4")):
        level = "_".join(levels)
        pairs = [("N_logits+T_logits", "N_logits+T_anchor_feature_" + level),
                 ("N_logits+N_anchor_feature+T_logits_" + level, "N_logits+N_anchor_feature+T_anchor_feature_" + level),
                 ("N_region_logits_" + level + "+T_region_logits", "N_region_logits_" + level + "+T_feature")]
        for logit_arm, feature_arm in pairs:
            if logit_arm not in result["arms"] or feature_arm not in result["arms"]:
                continue
            left, right = result["arms"][logit_arm], result["arms"][feature_arm]
            if left["cohort"] != right["cohort"] or left["val_n"] != right["val_n"]:
                raise AssertionError("Carrier comparison cohort mismatch")
            result["carrier_comparisons"].append({"teacher_logit_arm": logit_arm, "teacher_feature_arm": feature_arm,
                "cohort": left["cohort"], "val_n": left["val_n"],
                "logit_arm_input_dim": left["fit"]["input_dim"], "feature_arm_input_dim": right["fit"]["input_dim"],
                "feature_minus_logit_pp": {subset: {k: 100 * (right["val"][subset][k] - left["val"][subset][k])
                    if right["val"][subset][k] is not None and left["val"][subset][k] is not None else None
                    for k in ("accuracy", "balanced_accuracy")} for subset in ("all", "foreground", "background")}})
    predictions.update(y=y, is_train=train, is_background=background, roi_common_valid=roi_valid)
    if shuffle_indices is not None:
        predictions["roi_shuffle_source_row"] = shuffle_indices
    if anchor_shuffle_indices is not None:
        predictions["anchor_shuffle_source_row"] = anchor_shuffle_indices
    np.savez_compressed(output / "probe_predictions.npz", **predictions)
    np.savez_compressed(output / "probe_train_transforms.npz", **transforms)
    return result


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    q = z - z.max(axis=-1, keepdims=True)
    return np.exp(q) / np.exp(q).sum(axis=-1, keepdims=True)


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    q = z - z.max(axis=-1, keepdims=True)
    return q - np.log(np.exp(q).sum(axis=-1, keepdims=True))


def dfl_targets(distances, bins=16):
    d = np.asarray(distances, dtype=np.float64)
    valid = np.isfinite(d).all(axis=-1) & (d >= 0).all(axis=-1) & (d <= bins - 1).all(axis=-1)
    q = np.full(d.shape + (bins,), np.nan)
    flat = d[valid]
    lo = np.floor(flat).astype(int)
    hi = np.minimum(lo + 1, bins - 1)
    fraction = flat - lo
    target = np.zeros(flat.shape + (bins,), dtype=np.float64)
    ii, jj = np.indices(flat.shape)
    np.add.at(target, (ii, jj, lo), 1 - fraction)
    np.add.at(target, (ii, jj, hi), fraction)
    q[valid] = target
    return q, valid


def gradient_cosine(a, b):
    a = np.asarray(a).reshape(len(a), -1)
    b = np.asarray(b).reshape(len(b), -1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide((a * b).sum(axis=1), denom, out=np.full(len(a), np.nan), where=denom > 1e-14)


def dfl_values(native, teacher, target):
    ce_n = -(target * log_softmax(native)).sum(axis=-1)
    ce_t = -(target * log_softmax(teacher)).sum(axis=-1)
    pn, pt = softmax(native / TEMPERATURE), softmax(teacher / TEMPERATURE)
    kl = (pt * (log_softmax(teacher / TEMPERATURE) - log_softmax(native / TEMPERATURE))).sum(axis=-1) * TEMPERATURE ** 2
    # Derivatives w.r.t. native raw logits. A common 1/4 factor cancels in cosine.
    kd_gradient = TEMPERATURE * (pn - pt)
    gt_gradient = softmax(native) - target
    return ce_n, ce_t, kl, gradient_cosine(kd_gradient, gt_gradient)


def dfl_analysis(rows, logits, groups, output):
    if "N42_dfl" not in logits or "T42_dfl" not in logits:
        return {"status": "UNAVAILABLE", "reason": "Missing native or teacher raw DFL"}
    zn, zt = logits["N42_dfl"], logits["T42_dfl"]
    if zn.shape != zt.shape or zn.ndim != 3 or zn.shape[1:] != (4, 16):
        raise ValueError("DFL arrays must both have shape N x 4 x 16")
    if not np.isfinite(zn).all() or not np.isfinite(zt).all():
        raise ValueError("Raw DFL must be finite")
    foreground = np.array([not r.get("is_background", False) for r in rows])
    distances = np.full((len(rows), 4), np.nan)
    for i, r in enumerate(rows):
        if not foreground[i]:
            continue
        x1, y1, x2, y2 = r["gt_box_input"]
        cx, cy = r["anchor_center"]
        stride = float(r["anchor_stride"])
        if stride <= 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("Positive stride and xyxy boxes required")
        distances[i] = np.array([cx - x1, cy - y1, x2 - cx, y2 - cy]) / stride
    target, support = dfl_targets(distances)
    support &= foreground & (distances <= NATIVE_DFL_UPPER).all(axis=1)
    target[~support] = np.nan
    ce_n, ce_t, kl, cosine = dfl_values(zn, zt, target)
    pair = np.array([r.get("paired_gt_iou") is not None and r["paired_gt_iou"] >= PAIR_IOU for r in rows])
    reference = np.array([r.get("anchor_has_reference_candidate", False) for r in rows])
    split = np.array([r["split"] for r in rows])
    masks = {"all_supported_gt": support,
             "paired_supported_gt": support & pair,
             "paired_supported_gt_with_native_reference_candidate": support & pair & reference,
             "paired_supported_gt_with_gt_center_fallback": support & pair & ~reference}

    def summarize(mask, denominator):
        delta = ce_n.mean(axis=1) - ce_t.mean(axis=1)
        finite_cos = mask & np.isfinite(cosine)
        return {"n": int(mask.sum()), "all_rgb_gt_denominator": int(denominator), "coverage_all_gt": ratio(mask.sum(), denominator),
                "native_gt_ce_mean4": description(ce_n[mask].mean(axis=1)),
                "teacher_gt_ce_mean4": description(ce_t[mask].mean(axis=1)),
                "native_minus_teacher_gt_ce_mean4": description(delta[mask]),
                "teacher_lower_gt_ce_n": int((mask & (delta > 0)).sum()),
                "teacher_lower_gt_ce_fraction_selected": ratio((mask & (delta > 0)).sum(), mask.sum()),
                "teacher_lower_gt_ce_fraction_all_gt": ratio((mask & (delta > 0)).sum(), denominator),
                "kl_teacher_to_native_T2_mean4": description(kl[mask].mean(axis=1)),
                "kd_vs_gt_native_logit_gradient_cosine": description(cosine[mask]),
                "positive_cosine_n": int((finite_cos & (cosine > 0)).sum()),
                "negative_cosine_n": int((finite_cos & (cosine < 0)).sum()),
                "undefined_zero_norm_cosine_n": int((mask & ~np.isfinite(cosine)).sum()),
                "positive_cosine_fraction_defined": ratio((finite_cos & (cosine > 0)).sum(), finite_cos.sum()),
                "native_minus_teacher_gt_ce_by_side": {side: description((ce_n - ce_t)[mask, j]) for j, side in enumerate(("left", "top", "right", "bottom"))}}

    result = {"status": "COMPLETED", "bins": 16, "temperature": TEMPERATURE,
              "support": "All four GT distances in [0,14.99], no clamping, matching pinned native reg_max-1-0.01 support. The mathematical target primitive separately permits exact 15 but native analysis excludes it.",
              "native_support_upper": NATIVE_DFL_UPPER,
              "coordinate_assumption": "Both raw DFL tensors describe same RGB reference anchor/stride on common input-pixel canvas; identity/geometry is not independently certified.",
              "gradient_scope": "Raw native detection-head DFL logits only; no shared-feature/backbone parameter gradients or KD training effect.",
              "splits": {}}
    for s in ("train", "val"):
        base = foreground & (split == s)
        out = {"all_rgb_gt": int(base.sum()), "out_of_support_gt": int((base & ~support).sum()),
               "reference_candidate_gt": int((base & reference).sum())}
        for name, m in masks.items():
            out[name] = summarize(base & m, base.sum())
        primary = masks["paired_supported_gt_with_native_reference_candidate"]
        out["primary_by_group"] = {key: {v: summarize(base & primary & (values == v), (base & (values == v)).sum())
                                         for v in np.unique(values[base])} for key, values in groups.items()}
        result["splits"][s] = out
    with (output / "dfl_per_object.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["row", "object_id", "split", "class", "source_group", "support", "paired", "reference_candidate", "native_gt_ce", "teacher_gt_ce", "native_minus_teacher_gt_ce", "kl_T2", "gradient_cosine"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(rows):
            if not foreground[i]:
                continue
            writer.writerow(dict(row=i, object_id=r["object_id"], split=r["split"], **{"class": r["class"]},
                                 source_group=r.get("source_group"), support=bool(support[i]), paired=bool(pair[i]), reference_candidate=bool(reference[i]),
                                 native_gt_ce=plain(ce_n[i].mean()), teacher_gt_ce=plain(ce_t[i].mean()),
                                 native_minus_teacher_gt_ce=plain((ce_n[i] - ce_t[i]).mean()), kl_T2=plain(kl[i].mean()), gradient_cosine=plain(cosine[i])))
    return result


def report_markdown(result):
    p, d, o = result["classification_probe"], result["dfl"], result["object_opportunity"]
    lines = ["# Baseline 额外信息诊断", "", "本报告是固定 train 拟合、dev 评价的探索性信息诊断，不是 AP、蒸馏训练收益或几何准入证明。", "",
             "GT 共同窗口/anchor 与两模态无标注目标背景选择带有诊断特权；推理时无法直接取得这些关联。主分类表包含已配对和未配对GT位置；只有object_pairing=paired_gt子表可讨论标签关联对象，未配对位置的读出不能称同对象IR信息。", "",
             "## 对象机会（全部 RGB GT 为分母）", "", "|split|GT|N正确|T可修复|替换损伤|净修复|", "|---|---:|---:|---:|---:|---:|"]
    for s in ("train", "val"):
        q = o[s]["assigned"]
        if q["all_rgb_gt"]:
            lines.append(f"|{s}|{q['all_rgb_gt']}|{q['native_states'].get('correct', {}).get('n', 0)}|{q['repair']['n']}|{q['harm_if_teacher_replaces_native']['n']}|{q['net_repair_minus_harm']['n']}|")
    lines += ["", "修复需 paired GT IoU≥0.5，教师预测在 RGB 标签坐标下类别正确、conf≥0.25、IoU≥0.5。损伤包括替换为未检出教师；不是实际蒸馏造成的伤害。互斥错误表将低置信排在分类/定位前，JSON 另保留可重叠错误旗标。", "",
              "## 分类线性 probe（dev）", "", "固定 α=1 的均值平方误差 ridge，未用 dev 选参数；各层固定随机投影128维后，仅 train 标准化。背景类别为 nc，macro recall 仅对当前子集有 GT 支持的类别求均值，与 balanced accuracy 同义。", "",
              "固定 anchor patch 是前景/背景的主读出；GT-ROI仅为辅助且要求所有模型P3/P4区域共同有效。metadata_only只含log宽、高、stride，用来暴露抽样/ROI尺寸混杂，不是视觉分类优势。各臂注明cohort；不同cohort之间不直接作差。", "",
              "|输入|cohort|dev N|All accuracy %|All balanced %|前景 accuracy %|前景 macro recall %|背景 recall %|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    def percent(x):
        return "NA" if x is None else f"{x * 100:.3f}"
    for name, arm in sorted(p["arms"].items(), key=lambda kv: (kv[1]["cohort"] != "main_fixed_anchor", kv[0])):
        v = arm["val"]
        nums = [v["all"]["accuracy"], v["all"]["balanced_accuracy"], v["foreground"]["accuracy"], v["foreground"]["balanced_accuracy"], v["background"]["accuracy"]]
        lines.append("|" + name + "|" + arm["cohort"] + "|" + str(arm["val_n"]) + "|" + "|".join(percent(x) for x in nums) + "|")
    lines += ["", "以上 T 特征输入是在 probe 评价阶段使用教师的双模态诊断；它衡量该线性函数族是否可解码额外信息，不能叫作 RGB-only 推理提升或 KD 增益。shuffled 在各 split 内按固定 seed 全局置换，不条件化于标签；同模态 N0 仅在已导出时报告。N_region_logits 是共同区域内容代理，不是冻结 C1 算子的复刻。教师logits与feature载体对照固定相同样本；输入维数、参数数目及有效正则仍不同，不能把差异解释为KD带宽/载体的因果效果。", "",
              "## 同 anchor 定位分布（dev）", ""]
    if d["status"] == "COMPLETED":
        q = d["splits"]["val"]["paired_supported_gt_with_native_reference_candidate"]
        lines += [f"正式列示的头部代理子集：{q['n']}/{q['all_rgb_gt_denominator']} RGB GT，需配对、四边GT距离在支持域且存在N42粗候选。", "",
                  "|量|值|", "|---|---:|",
                  f"|N−T GT-DFL CE（4边均值）的均值|{q['native_minus_teacher_gt_ce_mean4']['mean']}|",
                  f"|教师GT CE更低的对象数|{q['teacher_lower_gt_ce_n']}|",
                  f"|KD(T=2) 与 GT 的 native DFL-logit 梯度 cosine 均值|{q['kd_vs_gt_native_logit_gradient_cosine']['mean']}|",
                  f"|cosine正 / 负 / 无定义|{q['positive_cosine_n']} / {q['negative_cosine_n']} / {q['undefined_zero_norm_cosine_n']}|", "",
                  "四边目标不截断，支持域外单列；GT CE 用 T=1，KL 用 T=2 且乘T²，梯度均对native raw DFL logits。正cosine只说明该头部局部代理方向一致，不是共享特征参数梯度或完整训练可学性。"]
    else:
        lines.append(d["reason"])
    lines += ["", "## 产物与局限", "", "完整逐类、来源组、尺度及 train-only 亮度分桶统计见 summary.json；probe_predictions.npz 保存逐行预测，probe_train_transforms.npz 保存仅训练拟合的标准化参数，dfl_per_object.csv 保存GT逐对象头部量。", "",
              "这是一个固定抽样/固定模型组合的探索性诊断，没有多seed蒸馏训练、四臂归因或完整AP；所有层与臂同时列示，不按dev结果选择最佳配置。标签对象关联、同画布坐标与背景排除均不构成独立像素配准证据。", ""]
    return "\n".join(lines)


def analyze(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir == input_dir:
        raise ValueError("Use a separate output directory to preserve raw export")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = read_rows(input_dir / "objects.jsonl")
    features = load_arrays(input_dir / "features.npz", len(rows))
    logits = load_arrays(input_dir / "logits.npz", len(rows))
    if "N42_cls" not in logits or logits["N42_cls"].ndim != 2:
        raise ValueError("N42_cls N x nc required")
    train = np.array([r["split"] == "train" for r in rows])
    groups, luma = group_columns(rows, train)
    metadata_file = input_dir / "metadata.json"
    if not metadata_file.exists():
        metadata_file = input_dir / "summary.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8-sig")) if metadata_file.exists() else {}
    for key, expected in (("input_box_format", "xyxy"), ("anchor_center_units", "input_pixels"), ("reg_max", 16), ("reference_model", "N42")):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"Unsupported metadata {key}={metadata[key]!r}; expected {expected!r}")
    identity_file = input_dir / "model_identity.json"
    identity = json.loads(identity_file.read_text(encoding="utf-8-sig")) if identity_file.exists() else {}
    images = {s: {r.get("image") for r in rows if r["split"] == s and r.get("image")} for s in ("train", "val")}
    if images["train"] & images["val"]:
        raise ValueError("The same image appears in train and val")
    result = {"version": VERSION, "input_directory": str(input_dir), "output_directory": str(output_dir),
              "input_metadata": metadata, "input_model_identity": identity,
              "source_files": {p.name: {"path": str(p), "bytes": p.stat().st_size} for p in
                               [input_dir / n for n in ("objects.jsonl", "features.npz", "logits.npz", "summary.json", "metadata.json", "model_identity.json", "frozen_roster.json")] if p.exists()},
              "distinct_images": {k: len(v) for k, v in images.items()},
              "rows": len(rows), "foreground_rows": sum(not r.get("is_background", False) for r in rows),
              "background_rows": sum(r.get("is_background", False) for r in rows), "luma_grouping": luma,
              "protocol": {"confidence": CONFIDENCE, "hit_iou": HIT_IOU, "paired_gt_iou": PAIR_IOU, "random_seed": SEED},
              "object_opportunity": opportunity_analysis(rows, groups)}
    result["classification_probe"] = probe_analysis(rows, features, logits, groups, output_dir)
    result["dfl"] = dfl_analysis(rows, logits, groups, output_dir)
    result = plain(result)
    write_json(output_dir / "summary.json", result)
    (output_dir / "README.md").write_text(report_markdown(result), encoding="utf-8")
    print("Completed:", output_dir, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.input, args.output)
