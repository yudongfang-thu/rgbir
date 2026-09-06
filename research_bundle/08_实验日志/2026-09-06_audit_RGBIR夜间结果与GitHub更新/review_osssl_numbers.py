"""Independent Decimal reconstruction of saved E200 CSV contrasts; no inference."""
import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osssl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.osssl / "summary.json").read_text(encoding="utf-8-sig"))
    metrics = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")
    runs = {}
    checks = {}
    for arm, seed in (("paired", 123), ("shuffled", 123), ("sar_only", 42), ("native", 123), ("native", 42)):
        root = args.osssl / "raw/runs" / ("cgkd_w1" if arm == "native" else "osssl_ir_20260906") / f"{arm}_rgb_s{seed}_e200"
        with (root / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
        config = yaml.safe_load((root / "args.yaml").read_text(encoding="utf-8-sig"))
        checks[f"{arm}{seed}_contiguous_e200"] = len(rows) == 200 and [int(row["epoch"]) for row in rows] == list(range(1, 201))
        checks[f"{arm}{seed}_seed_matches"] = config["seed"] == seed
        runs[(arm, seed)] = {"metrics": {m: Decimal(rows[-1][m]) * 100 for m in metrics}, "args": config}
    reconstructed = []
    for contrast in summary["contrasts_vs_historical_native"]:
        left, right = runs[(contrast["arm"], contrast["seed"])], runs[("native", contrast["seed"])]
        for metric in metrics:
            delta = left["metrics"][metric] - right["metrics"][metric]
            observed = contrast["metrics"][metric]
            checks[f'{contrast["arm"]}{contrast["seed"]}_{metric}_historical_contrast_matches'] = delta == Decimal(str(observed["delta_pp"])) and left["metrics"][metric] == Decimal(str(observed["left_pp"])) and right["metrics"][metric] == Decimal(str(observed["right_pp"]))
    internal = summary["internal_ssl_contrasts"][0]
    left, right = runs[("paired", 123)], runs[("shuffled", 123)]
    differences = {}
    for metric in metrics:
        delta = left["metrics"][metric] - right["metrics"][metric]
        differences[metric] = str(delta)
        checks[metric + "_internal_contrast_matches"] = delta == Decimal(str(internal["metrics"][metric]["delta_pp"]))
    args_differences = [key for key in set(left["args"]) | set(right["args"]) if left["args"].get(key) != right["args"].get(key)]
    checks["internal_args_only_model_and_names_differ"] = set(args_differences) == {"model", "name", "save_dir"}
    checks["csv_not_independent_final"] = summary["independent_osssl_metrics_records"] == 0 and internal["independent_last_endpoint"] is False
    checks["no_three_seed_claim"] = internal["three_seed_gate_evaluable"] is False and summary["conclusions"]["positive_ssl_claim_supported"] is False and summary["conclusions"]["paired_attribution_supported"] is False
    checks["initialization_and_rgb_self_control_limits_retained"] = summary["conclusions"]["ssl_vs_native_initialization_confounded"] is True and summary["conclusions"]["target_rgb_only_ssl_control_missing"] is True
    checks["raw_csv_ap50_difference_below_0p5"] = Decimal(differences[metrics[0]]) < Decimal("0.5")
    report = {"schema": "osssl-independent-night-numbers-review-v1", "checks": checks, "paired123_minus_shuffled123_pp_exact_saved_csv": differences,
        "args_differences": sorted(args_differences), "limitations": ["Decimal arithmetic uses saved, already rounded training CSV values; it does not recover full precision evaluator metrics.", "Single finetuning seed comparison, a single pretrained checkpoint per arm, no independent last evaluation, no three-seed gate decision.", "Shared nonbackbone state is from prior CPU evidence, not reloaded here; native initialization remains confounded and RGB-only SSL absent."]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": all(checks.values()), "paired123_minus_shuffled123_pp": differences}, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
