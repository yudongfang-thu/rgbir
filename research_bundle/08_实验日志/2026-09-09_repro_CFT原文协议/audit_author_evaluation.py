"""CPU-only audit of saved CFT evaluation arrays and source snapshots.

No SSH, model import/forward, training, input mutation, or digest calculation.
Executes only the inspected author's two NumPy AP functions with plot=False.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import re
import sys
import time
import warnings

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stat_record(path):
    s = path.stat()
    return (s.st_size, s.st_mtime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    execution = root / "server_execution"
    run = execution / "official_test_attempt1"
    source = run / "source_snapshot"
    prediction_files = sorted((run / "author_output" / "labels").glob("*.txt"))
    fixed = [run / name for name in (
        "receipt.json", "evaluated_roster.json", "author_metric_inputs.npz",
        "checkpoint_compatibility.json", "effective_data.yaml", "executed_wrapper.py", "plan.json")]
    fixed += [execution / name for name in (
        "official_test.log", "official_test.exit", "data_receipt.json", "annotation_audit.json",
        "author_protocol.json", "legacy_annotations_download.json", "pinned_cpu_recompute.json")]
    fixed += [source / name for name in ("test.py", "utils/metrics.py", "utils/general.py", "utils/datasets.py")]
    inputs = fixed + prediction_files
    before = {str(p.relative_to(root)): stat_record(p) for p in inputs}
    start = time.perf_counter()
    checks = []

    def check(name, value, detail, *, fatal=True):
        checks.append({"id": name, "status": "pass" if bool(value) else ("fail" if fatal else "warn"), "details": detail})
        if not value and fatal:
            raise AssertionError(f"{name}: {detail}")

    receipt = read_json(run / "receipt.json")
    roster = read_json(run / "evaluated_roster.json")
    annotation = read_json(execution / "annotation_audit.json")
    protocol = read_json(execution / "author_protocol.json")
    compatibility = read_json(run / "checkpoint_compatibility.json")
    pinned_cpu = read_json(execution / "pinned_cpu_recompute.json")
    log = (execution / "official_test.log").read_text(encoding="utf-8")
    test_source = (source / "test.py").read_text(encoding="utf-8")
    wrapper = (run / "executed_wrapper.py").read_text(encoding="utf-8")
    general = (source / "utils/general.py").read_text(encoding="utf-8")
    dataset_source = (source / "utils/datasets.py").read_text(encoding="utf-8")
    stems = roster["stems"]
    with np.load(run / "author_metric_inputs.npz", allow_pickle=False) as z:
        values = {key: z[key] for key in z.files}
    tp, conf, pred_cls, target_cls = (values[k] for k in ("tp", "conf", "pred_cls", "target_cls"))
    check("array_schema", set(values) == {"tp", "conf", "pred_cls", "target_cls"}
          and tp.shape == (len(conf), 10) and pred_cls.shape == conf.shape
          and target_cls.ndim == 1 and tp.dtype == np.bool_, "Captured arguments of the author AP function")
    check("array_finite", all(np.isfinite(v).all() for v in values.values()), "All stored arrays finite")
    check("single_class", np.array_equal(np.unique(target_cls), [0.0])
          and np.array_equal(np.unique(pred_cls), [0.0]), "person class id=0")
    check("tp_threshold_monotonicity", np.all(tp[:, 1:] <= tp[:, :-1]), "Per-prediction TP flags never rise with IoU")
    check("gt_denominator", len(target_cls) == annotation["previous"]["splits"]["test"]["objects"] == 7931,
          "GT argument count agrees with previous-annotation audit")
    check("tp_bound", bool(np.all(tp.sum(0) <= len(target_cls))), "No threshold has more TPs than GT")
    check("prediction_confidence", bool(np.all((conf > .001) & (conf <= 1))), "Original strict confidence cutoff retained")
    check("roster", len(stems) == len(set(stems)) == roster["images"] == 3463
          and roster["pair_order_identical"], "3463 unique loader-ordered RGB/IR pairs")
    check("exit_and_receipt", (execution / "official_test.exit").read_text().strip() == "0"
          and receipt["status"] == "COMPLETED" and receipt["ap_computed"], "Successful original full evaluation")
    final_lines = re.findall(r"^\s*all\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", log, re.M)
    check("seen_and_gt_log", len(final_lines) == 1 and final_lines[0][:2] == ("3463", "7931"),
          "Author evaluator increments seen per output image; final counts cover full roster")
    check("all_batches_completed", "55/55" in log and "100%" in log, "ceil(3463/64)=55 batches complete")
    check("no_nms_timeout", "NMS time limit" not in log, "No author NMS cutoff warning in combined run log")
    check("timeout_fail_closed", "NMS time limit" in wrapper and "raise RuntimeError" in wrapper
          and "contextlib.redirect_stdout(tee)" in wrapper and "print(f'WARNING: NMS time limit" in general,
          "Wrapper captures the original stdout warning and rejects timeout before COMPLETED")
    check("capture_delegates_metric", "return metric(tp,conf,pred_cls,target_cls,*args,**kwargs)" in wrapper,
          "Saved arrays are forwarded to the original AP function")
    check("no_hybrid_gt_predictions", "save_hybrid=False" in wrapper,
          "Ground truth is not injected into prediction NMS")
    check("no_prediction_preserves_gt", "seen += 1" in test_source
          and "if len(pred) == 0:" in test_source
          and "torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls" in test_source,
          "Empty-prediction images count as seen and their GT remains in stats")

    def read_predictions(path):
        lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return path.stem, np.empty((0, 6)), []
        if any(len(line) != 6 for line in lines):
            raise AssertionError(f"Malformed prediction: {path}")
        return path.stem, np.asarray(lines, dtype=np.float64), [line[5] for line in lines]

    with ThreadPoolExecutor(max_workers=8) as pool:
        text_predictions = {stem: (rows, confidence_tokens)
                            for stem, rows, confidence_tokens in pool.map(read_predictions, prediction_files)}
    extra = sorted(set(text_predictions) - set(stems))
    missing = sorted(set(stems) - set(text_predictions))
    empty_txt = sorted(stem for stem, (rows, _) in text_predictions.items() if len(rows) == 0)
    check("prediction_file_scope", not extra and len(prediction_files) == 3461,
          "Only official roster predictions; 3461 TXT files")
    check("empty_prediction_scope", missing == ["240113", "240165"] and not empty_txt,
          "Two seen images have no detections and therefore no TXT")
    txt_rows = np.concatenate([text_predictions[s][0] for s in stems if s in text_predictions], axis=0)
    tokens = [t for s in stems if s in text_predictions for t in text_predictions[s][1]]
    check("prediction_count", len(txt_rows) == len(conf) == 18275,
          "TXT rows in evaluated roster order equal captured AP predictions")
    check("prediction_txt_finite", np.isfinite(txt_rows).all() and np.all(txt_rows[:, 0] == 0),
          "Every TXT contains finite one-class six-column predictions")
    check("prediction_txt_confidence_order", tokens == [format(float(x), "g") for x in conf],
          "Confidence tokens exactly match author %g serialization of saved conf, in loader order")
    check("prediction_txt_class_order", np.array_equal(txt_rows[:, 0], pred_cls),
          "TXT class sequence exactly matches saved AP inputs")

    metric_path = source / "utils/metrics.py"
    metric_ast = ast.parse(metric_path.read_text(encoding="utf-8"))
    funcs = [n for n in metric_ast.body if isinstance(n, ast.FunctionDef)
             and n.name in {"ap_per_class", "compute_ap"}]
    check("author_metric_functions", len(funcs) == 2, "Only inspected NumPy functions compiled; module imports skipped")
    namespace = {"np": np, "Path": Path}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(metric_path), "exec"), namespace)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        precision, recall, ap, f1, classes = namespace["ap_per_class"](
            tp, conf, pred_cls, target_cls, plot=False)
    recomputed = {"precision": float(precision.mean()), "recall": float(recall.mean()),
                  "AP50": float(ap[:, 0].mean()), "AP75": float(ap[:, 5].mean()),
                  "AP50_95": float(ap.mean())}
    absolute_differences = {key: abs(value - receipt["metrics"][key]) for key, value in recomputed.items()}
    check("recomputed_equals_receipt", max(absolute_differences.values()) <= 1e-12,
          {"max_absolute_difference": max(absolute_differences.values()), "tolerance": 1e-12,
           "receipt": receipt["metrics"], "recomputed": recomputed,
           "note": "No tolerance relaxation: cross-runtime mismatch is retained as a warning"}, fatal=False)
    check("per_class_map_equals_receipt", np.allclose(ap.mean(1), receipt["maps"], rtol=0, atol=1e-12),
          "Single-class map array compared to receipt at original 1e-12 tolerance", fatal=False)
    _, confidence_counts = np.unique(conf, return_counts=True)
    tie_diagnostics = {"distinct_confidences": int(len(confidence_counts)),
                       "tied_confidence_groups": int(np.sum(confidence_counts > 1)),
                       "predictions_in_tied_groups": int(confidence_counts[confidence_counts > 1].sum()),
                       "largest_tied_group": int(confidence_counts.max()),
                       "sort": "np.argsort(-conf), default quicksort, not stable",
                       "default_equals_stable_order": bool(np.array_equal(np.argsort(-conf), np.argsort(-conf, kind="stable"))),
                       "diagnosis": "Ties and unstable ordering are verified; NumPy/platform tie-order effects are a plausible cause, not yet isolated"}
    check("compatibility_preserves_objects", compatibility["parameter_and_buffer_objects_unchanged"]
          and not compatibility["random_parameters_created"] and len(compatibility["restored_blocks"]) == 24,
          "Runtime receipt records 24 stored CFT blocks rebound, no random parameters")
    check("pinned_cpu_receipt_agreement", all(pinned_cpu["values"][k] == receipt["metrics"][k]
          and pinned_cpu["difference_to_execution"][k] == 0 for k in ("AP50", "AP75", "AP50_95")),
          "Executor supplied separate NumPy 2.2.6 CPU recomputation receipt: all three APs exactly equal")
    check("pinned_cpu_scope", pinned_cpu["numpy_version"] == "2.2.6" and pinned_cpu["cuda_initialized"] is False
          and pinned_cpu["predictions"] == len(conf) and pinned_cpu["gt_labels"] == len(target_cls)
          and pinned_cpu["unique_confidences"] == len(confidence_counts),
          "Pinned CPU receipt matches array scope; this reviewer did not execute on server")
    after = {str(p.relative_to(root)): stat_record(p) for p in inputs}
    check("input_stats_unchanged", before == after, "All fixed inputs and 3461 TXT files retain size/mtime_ns; not a content digest")
    metric_names = ("AP50", "AP75", "AP50_95")
    percentages = {key: recomputed[key] * 100 for key in metric_names}
    targets = protocol["paper"]["target_percent"]
    comparisons = {key: {"recomputed_fraction": recomputed[key], "recomputed_percent": percentages[key],
                         "paper_percent": targets[key], "delta_percentage_points": percentages[key] - targets[key],
                         "one_decimal_percent": f"{percentages[key]:.1f}",
                         "paper_one_decimal": f"{targets[key]:.1f}",
                         "matches_paper_display": f"{percentages[key]:.1f}" == f"{targets[key]:.1f}"}
                   for key in metric_names}
    execution_comparisons = {key: {"execution_fraction": receipt["metrics"][key],
                            "execution_percent": receipt["metrics"][key] * 100,
                            "paper_percent": targets[key],
                            "delta_percentage_points": receipt["metrics"][key] * 100 - targets[key],
                            "one_decimal_percent": f"{receipt['metrics'][key] * 100:.1f}",
                            "matches_paper_display": f"{receipt['metrics'][key] * 100:.1f}" == f"{targets[key]:.1f}"}
                            for key in metric_names}
    output = {
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "auditor": {"agent": "/root/port90_code_inventory", "role": "reused reviewer independent of execution",
                    "fresh_context": False, "cross_model_review": False, "model_identity": "not independently exposed"},
        "execution_integrity": "pass", "recomputation_integrity": "pass" if max(absolute_differences.values()) <= 1e-12 else "warn", "checks": checks,
        "numpy_version": np.__version__, "python_version": sys.version,
        "metric_source": str(metric_path.relative_to(root)), "plot": False,
        "recomputed": recomputed, "receipt_absolute_differences": absolute_differences,
        "receipt_metrics": receipt["metrics"], "confidence_ties": tie_diagnostics,
        "pinned_cpu_recompute": pinned_cpu,
        "pinned_cpu_provenance": "Executor supplied server receipt, read and checked here; reviewer performed only the local NumPy 2.3.5 recomputation",
        "ap_at_each_iou": ap.tolist(), "tp_at_each_iou": tp.sum(0).tolist(),
        "iou_thresholds": [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95],
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in values.items()},
        "images": len(stems), "gt_objects": len(target_cls), "predictions": len(conf),
        "prediction_txt_files": len(prediction_files), "no_prediction_stems": missing,
        "empty_prediction_txt": empty_txt, "txt_conf_max_rounding_error": float(np.max(np.abs(txt_rows[:, 5] - conf))),
        "paper_comparison": comparisons, "paper_url": protocol["paper"]["url"],
        "execution_paper_comparison": execution_comparisons,
        "annotation_version": "official previous annotations", "current_archive_test_gt": 8302,
        "annotation_count_difference": 8302 - len(target_cls),
        "eval_type": "real_gt", "identity": "PAPER-RECONSTRUCTED checkpoint reevaluation",
        "new_model_forward": False, "training_started": False, "ssh_used": False,
        "new_digest_calculated": False, "audited_input_hashes": [],
        "hash_policy_override": "Explicit user instruction forbids new hash; bytes/mtime_ns checked instead, weaker than content identity",
        "declared_input_set": [str(p.relative_to(root)) for p in fixed]
                              + [{"glob": "server_execution/official_test_attempt1/author_output/labels/*.txt",
                                  "actual_files": len(prediction_files), "all_input_stats_checked": True}],
        "input_stats_unchanged": True,
        "unreviewed": ["No checkpoint/model forward replay", "No raw-image or full-XML GT reread",
                       "No independent bounding-box IoU matching replay; saved arrays contain TP flags, not GT box coordinates",
                       "Exact unavailable checkpoint training source/history", "No three-seed or from-scratch training audit"],
        "elapsed_seconds": time.perf_counter() - start,
    }
    (root / "EVALUATION_RECOMPUTE.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks_passed": sum(c["status"] == "pass" for c in checks), "warnings": [c for c in checks if c["status"] == "warn"], "recomputed": recomputed, "paper_comparison": comparisons,
                      "confidence_ties": tie_diagnostics,
                      "images": len(stems), "gt": len(target_cls), "predictions": len(conf),
                      "no_prediction_stems": missing, "elapsed_seconds": output["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
