"""Independent CPU checks of copied fixed endpoints; no checkpoint/hash access."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = read(args.snapshot / "oev1_endpoints/summary.json")
    metrics = ("mAP50_95", "AP50", "AP75", "precision", "recall")
    endpoints = []
    rosters = []
    for cell in summary["cells"]:
        if cell["status"] != "completed":
            continue
        run = args.snapshot / "raw_runs" / Path(cell["run"]).name
        evaluation = read(run / "evaluation_val.json")
        completion = read(run / "completion_receipt.json")
        launch = read(run / "launch_manifest.json")
        checks = {
            "identities_equal": all(record["seed"] == cell["seed"] and record["arm"] == cell["arm"] and record["method_id"] == "RGBIR-OBJECT-EVIDENCE-v1" for record in (evaluation, completion, launch)),
            "fixed_last_val_endpoint": evaluation["endpoint"] == "fixed_budget_last_ema" and evaluation["split"] == "val" and evaluation["checkpoint"].endswith("/weights/last.pt"),
            "complete_e200": completion["last_epoch"] == completion["epochs_configured"] == 200 and completion["status"] == "training_completed",
            "same_checkpoint": evaluation["checkpoint"] == completion["checkpoint"] == cell["checkpoint"],
            "fraction_finite_metrics": evaluation["metric_units"] == "fraction_0_to_1" and all(isinstance(evaluation[m], (int, float)) and not isinstance(evaluation[m], bool) and math.isfinite(evaluation[m]) and 0 <= evaluation[m] <= 1 for m in metrics),
            "collector_metrics_exact": all(evaluation[m] == cell["metrics"][m] for m in metrics),
            "no_test_or_confirmatory_upgrade": evaluation["official_test_accessed"] is False and completion["official_test_accessed"] is False and evaluation["single_seed_exploratory"] is True,
        }
        for label, folder, raw in (("train", "run_evidence", completion), ("eval", "eval_evidence", evaluation)):
            evidence = run / folder
            receipt = read(evidence / "run_receipt.json")
            checks[label + "_receipt_identity_complete"] = receipt["terminal_status"] == "COMPLETED" and receipt["run_kind"] == label and receipt["dataset"].lower() == "dronevehicle" and receipt["seed"] == cell["seed"] and receipt["inputs"]["arm"] == cell["arm"]
            checks[label + "_metric_snapshot_equal"] = any(read(evidence / p) == raw for p in receipt["metric_snapshots"])
            checks[label + "_references_exist"] = all((evidence / p).is_file() for group in receipt["source_snapshots"].values() for p in group)
            if label == "eval":
                roster = (evidence / receipt["source_snapshots"]["split_roster"][0]).read_text(encoding="utf-8-sig").splitlines()
                checks["val_roster_1469_unique"] = len(roster) == len(set(roster)) == 1469
                rosters.append(roster)
        with (run / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        historical = read(args.historical / f'N_llvip_seed{cell["seed"]}_metrics_record.json')
        checks["historical_reference_identity"] = historical["dataset"] == "dronevehicle" and historical["seed"] == cell["seed"] and historical["split"] == "val"
        endpoints.append({"seed": cell["seed"], "arm": cell["arm"], "checks": checks,
            "metrics_percent": {m: 100 * evaluation[m] for m in metrics},
            "historical_native_difference_pp_descriptive_only": {m: 100 * (evaluation[m] - historical["metrics"][m]) for m in historical["metrics"]},
            "csv_format": {"header_columns": len(rows[0]), "last_columns": len(rows[-1]), "last_epoch": rows[-1][0]},
            "optimizer_updates": {k: completion[k] for k in ("optimizer_updates", "optimizer_update_attempts", "amp_skipped_updates", "ema_updates")}})
    comparisons = []
    for seed in (0, 42, 123):
        pair = {r["arm"]: r for r in endpoints if r["seed"] == seed}
        if set(pair) == {"paired", "weight0"}:
            comparisons.append({"seed": seed, "difference_pp": {m: pair["paired"]["metrics_percent"][m] - pair["weight0"]["metrics_percent"][m] for m in metrics}})
    checks = {
        "complete_count_matches": len(endpoints) == summary["complete_endpoints"],
        "same_seed_pair_count_matches": len(comparisons) == summary["complete_seed_pairs"],
        "only_complete_three_pairs_aggregate": (summary["three_seed_summary"] is not None) == (len(comparisons) == 3),
        "completed_rosters_identical": all(roster == rosters[0] for roster in rosters),
        "all_endpoint_checks_pass": all(all(row["checks"].values()) for row in endpoints),
    }
    if len(comparisons) == 3:
        values = [row["difference_pp"]["mAP50_95"] for row in comparisons]
        observed = summary["three_seed_summary"]["mAP50_95"]["paired_minus_weight0"]
        checks["three_seed_primary_mean_matches"] = math.isclose(statistics.mean(values), observed["mean_pp"], abs_tol=1e-10)
        checks["three_seed_primary_sample_sd_matches"] = math.isclose(statistics.stdev(values), observed["sample_sd_pp"], abs_tol=1e-10)
    report = {"schema": "oev1-independent-night-endpoint-review-v1", "snapshot_captured_at_utc": summary["captured_at_utc"], "checks": checks, "endpoints": endpoints, "same_seed_comparisons": comparisons,
        "interpretation": "Historical native differences are background only. No cross-seed P-minus-N estimand is computed. Only complete fixed last/EMA endpoints count; CSV is inspected for format only. No accepted/confirmatory or cross-modal-specific claim follows.",
        "scope": "Copied local receipt and source evidence, source roster contents, raw metrics and arithmetic. Server checkpoint file existence is established by original collector, not rechecked here; no weights loaded, no new hashes generated."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": all(checks.values()), "complete_endpoints": len(endpoints), "complete_seed_pairs": len(comparisons), "endpoints": [{"seed": row["seed"], "arm": row["arm"], "mAP_pct": row["metrics_percent"]["mAP50_95"], "historical_native_difference_pp": row["historical_native_difference_pp_descriptive_only"]["mAP50_95"]} for row in endpoints]}, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
