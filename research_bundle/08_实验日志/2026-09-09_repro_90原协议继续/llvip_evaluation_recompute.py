"""Bounded local CPU review: saved LLVIP author arrays, TXT, receipts and source.
No model import/forward, SSH, new hashes, raw artifact edits, or new evaluation.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import re
import sys
import warnings
import numpy as np


ROOT = Path(__file__).resolve().parent
EXECUTION = ROOT / "llvip_execution"
NAMES = ("AP50", "AP75", "AP50_95")
UPDATED = {"visible": (90.8, 56.4, 52.7), "infrared": (96.5, 76.4, 67.0)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(paths):
    return {str(p.relative_to(ROOT)): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def audit(modality):
    run = EXECUTION / f"llvip_{modality}_full_attempt1"
    source = run / "source_snapshot"
    fixed = list(source.rglob("*.py")) + [run / x for x in
        ("receipt.json", "frozen_protocol.json", "evaluated_roster.json", "author_metric_inputs.npz", "executed_wrapper.py", "effective_data.yaml")]
    log_path = EXECUTION / f"llvip_author_{modality}_evaluate_20260909.log"
    exit_path = EXECUTION / f"llvip_author_{modality}_evaluate_20260909_exit.json"
    fixed += [log_path, exit_path, EXECUTION / "llvip_data_view_receipt.json"]
    txts = list((run / "author_output/labels").glob("*.txt"))
    before = stats(fixed + txts)
    receipt = read(run / "receipt.json")
    protocol = read(run / "frozen_protocol.json")
    roster = read(run / "evaluated_roster.json")
    log = log_path.read_text(encoding="utf-8")
    wrapper = (run / "executed_wrapper.py").read_text(encoding="utf-8")
    val = (source / "val.py").read_text(encoding="utf-8")
    general = (source / "utils/general.py").read_text(encoding="utf-8")
    checks = []
    def check(name, condition, detail, fatal=True):
        checks.append({"id": name, "status": "pass" if bool(condition) else ("fail" if fatal else "warn"), "details": detail})
        if not condition and fatal:
            raise AssertionError((modality, name, detail))
    with np.load(run / "author_metric_inputs.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    tp, conf, pc, tc = (arrays[k] for k in ("tp", "conf", "pred_cls", "target_cls"))
    check("arrays", tp.shape == (len(conf), 10) and tp.dtype == np.bool_ and pc.shape == conf.shape
          and len(tc) == 7931 and all(np.isfinite(x).all() for x in arrays.values())
          and np.all(pc == 0) and np.all(tc == 0), "Finite person-only captured arguments and 7931 GT")
    check("tp_invariants", np.all(tp[:, 1:] <= tp[:, :-1]) and np.all(tp.sum(0) <= len(tc)), "TP flags monotone across IoU and below GT count")
    check("confidence", np.all((conf > .001) & (conf <= 1)), "Original strict confidence cutoff")
    stems = roster["stems"]
    check("roster", len(stems) == len(set(stems)) == roster["images"] == 3463 and roster["gt"] == 7931, "Full unique loader roster")
    check("completed", receipt["status"] == "COMPLETED" and receipt["ap_computed"] and read(exit_path)["exit_code"] == 0,
          "Successful recorded execution and exit")
    check("log_counts", bool(re.search(r"all\s+3463\s+7931\s", log)) and "109/109" in log, "109 batches and final seen/GT counts")
    check("no_nms_cutoff", "NMS time limit" not in log and "NMS time limit" in wrapper
          and 'raise RuntimeError("Original NMS timeout' in wrapper and "WARNING: NMS time limit" in general,
          "No cutoff warning; executed wrapper rejects author NMS cutoff")
    check("empty_prediction_gt", "seen += 1" in val and "if len(pred) == 0:" in val
          and "torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls" in val,
          "Seen count and empty-prediction GT retained by original val loop")
    check("original_ap_delegation", "result = original_metric(tp, conf, pred_cls, target_cls" in wrapper,
          "Captured arrays delegated to author AP function")
    check("frozen_protocol", receipt["protocol"] == protocol and protocol["modality"] == modality
          and protocol["imgsz"] == 1280 and protocol["batch_size"] == 32 and protocol["half"] is False
          and protocol["iou_thres"] == .6 and protocol["conf_thres"] == .001
          and protocol["augment"] is False and protocol["save_hybrid"] is False
          and protocol["label_version"] == "previous", "Disclosed single-modality original public protocol preserved")

    def read_txt(path):
        rows = [x.split() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if any(len(x) != 6 for x in rows):
            raise AssertionError(f"Bad prediction TXT {path}")
        return path.stem, rows
    with ThreadPoolExecutor(max_workers=8) as pool:
        predictions = dict(pool.map(read_txt, txts))
    missing = sorted(set(stems) - set(predictions))
    empty = sorted(k for k, rows in predictions.items() if not rows)
    check("txt_scope", not (set(predictions) - set(stems)) and not empty, "No extra or empty TXT; missing files are no-detection cases")
    rows = [row for s in stems if s in predictions for row in predictions[s]]
    numbers = np.asarray(rows, dtype=np.float64)
    check("txt_count", len(rows) == len(conf) == receipt["metrics"]["predictions"], "All TXT predictions accounted for")
    check("txt_values", np.isfinite(numbers).all() and np.array_equal(numbers[:, 0], pc), "Finite normalized boxes and correct class sequence")
    check("txt_conf_order", [x[5] for x in rows] == [format(float(x), "g") for x in conf], "Exact author %g confidence serialization in loader order")
    metric = source / "utils/metrics.py"
    module = ast.parse(metric.read_text(encoding="utf-8"))
    selected = [x for x in module.body if isinstance(x, ast.FunctionDef) and x.name in ("ap_per_class", "compute_ap")]
    check("metric_source", len(selected) == 2, "Only two inspected NumPy functions executed via AST; no model imports")
    namespace = {"np": np, "Path": Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(metric), "exec"), namespace)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p, r, ap, f1, cls = namespace["ap_per_class"](tp, conf, pc, tc, plot=False, names={0: "person"})
    computed = dict(zip(NAMES, (float(ap[:, 0].mean()), float(ap[:, 5].mean()), float(ap.mean()))))
    diffs = {k: computed[k] - receipt["metrics"][k] for k in NAMES}
    check("ap_receipt_exact", max(abs(x) for x in diffs.values()) <= 1e-12,
          {"original_tolerance": 1e-12, "difference_fraction": diffs}, fatal=False)
    check("ap10_receipt_exact", np.allclose(ap, receipt["metrics"]["ap_at_each_iou"], rtol=0, atol=1e-12), "All ten IoU APs compared at 1e-12", fatal=False)
    check("actual_wrapper", (run / "executed_wrapper.py").stat().st_size == 16888
          and 'device=protocol["device"]' in wrapper and 'device=""' not in wrapper,
          "Actual old executed snapshot retained; not silently replaced by later device-default variant")
    check("cuda_init_order", wrapper.index("torch.cuda.set_per_process_memory_fraction(.68, 0)")
          < wrapper.index("result = author_val.run"), "Allocator initialization precedes author device selection")
    check("physical_gpu", receipt["lease"]["gpus"] == receipt["observed_lease_resources"]["gpu_ids"] == [3]
          and receipt["observed_lease_resources"]["cuda_pid_counts"] == {"3": 1}, "Lease and recorded NVML process scope both physical GPU3")
    check("no_hash_override", "author_datasets.get_hash = path_size_cache_key" in wrapper
          and '"plain_path_size_v1"' in wrapper and "hashlib.md5(" not in wrapper,
          "Executed wrapper replaces author metadata MD5 before loader with plain path-size tuple")
    check("inputs_preserved", stats(fixed + txts) == before, "All reviewed raw input sizes/mtime_ns unchanged; not a content digest")
    references = {"frozen_v1": protocol["paper_reference_percent"], "v2_v4_additional": dict(zip(NAMES, UPDATED[modality]))}
    comparison = {version: {k: {"paper_percent": target[k], "execution_percent": receipt["metrics"][k] * 100,
                   "delta_pp": receipt["metrics"][k] * 100 - target[k],
                   "one_decimal_matches": f"{receipt['metrics'][k]*100:.1f}" == f"{target[k]:.1f}"} for k in NAMES}
                   for version, target in references.items()}
    _, counts = np.unique(conf, return_counts=True)
    return {"modality": modality, "run": str(run.relative_to(ROOT)), "checks": checks,
            "images": len(stems), "gt": len(tc), "predictions": len(conf), "prediction_txts": len(txts),
            "no_prediction_stems": missing, "empty_txt": empty,
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
            "tp_by_iou": tp.sum(0).tolist(), "recomputed": computed, "recomputed_ap10": ap.tolist(),
            "execution_metrics": {k: receipt["metrics"][k] for k in NAMES}, "signed_difference_fraction": diffs,
            "confidence_ties": {"unique": len(counts), "tied_predictions": int(counts[counts>1].sum())},
            "comparisons": comparison, "execution_numpy": receipt["numpy"], "review_numpy": np.__version__,
            "elapsed_seconds_recorded": receipt["elapsed_seconds"], "exit_wall_seconds_recorded": read(exit_path)["wall_seconds"],
            "physical_gpu_recorded": receipt["observed_lease_resources"],
            "executed_wrapper_bytes": (run / "executed_wrapper.py").stat().st_size,
            "declared_inputs": [str(p.relative_to(ROOT)) for p in fixed] + [{"glob": str((run/'author_output/labels/*.txt').relative_to(ROOT)), "count": len(txts)}],
            "inputs_size_mtime_unchanged": True, "audited_input_hashes": []}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = [audit(x) for x in ("visible", "infrared")]
    output = {"date": dt.datetime.now(dt.timezone.utc).isoformat(), "reviewer": "/root/port90_code_inventory",
              "reused_reviewer": True, "fresh_context": False, "cross_model_review": False,
              "model_forward": False, "ssh": False, "new_hash": False, "results": results,
              "limits": ["No independent box/GT IoU rematching", "No model inference replay or XML/image reread",
                         "Reviewer also prepared initial wrapper; separate from executor but not a blind independent code-author review",
                         "Source/receipt sizes and timestamps only, no content identity guarantee",
                         "Later paper reference added after execution for version explanation; frozen v1 retained; no rerun"]}
    (ROOT / "LLVIP_EVALUATION_RECOMPUTE.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("modality", "images", "gt", "predictions", "prediction_txts", "no_prediction_stems", "recomputed", "signed_difference_fraction")}
                      for r in results], ensure_ascii=False))
