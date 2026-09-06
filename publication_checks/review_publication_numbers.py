"""CPU-only recheck of numbers advertised in the external review guide."""
import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo
    rows = []
    for dataset in ("drone", "llvip"):
        for control in ("h2", "h3"):
            differences = []
            inputs = []
            for seed in (0, 42, 123):
                files = [repo / "04_hnewa_eval_records" / f"{dataset}_{arm}_s{seed}.json" for arm in ("h1", control)]
                p, n = [read(f) for f in files]
                assert p["seed"] == n["seed"] == seed
                assert p["dataset"] == n["dataset"]
                assert p["split"] == n["split"] == "val"
                assert p["data_yaml"] == n["data_yaml"]
                assert p["checkpoint"].endswith("weights/last.pt") and n["checkpoint"].endswith("weights/last.pt")
                differences.append(100 * (p["metrics"]["mAP50_95"] - n["metrics"]["mAP50_95"]))
                inputs.extend(f.relative_to(repo).as_posix() for f in files)
            rows.append({"dataset": dataset, "comparison": "paired-minus-" + control, "seeds": [0, 42, 123], "differences_pp": differences, "mean_pp": statistics.mean(differences), "sample_sd_pp": statistics.stdev(differences), "source_files": inputs})
    differences = []
    names = []
    for seed in (0, 42, 123):
        files = [repo / "02_raw_results_dronevehicle" / name for name in (f"L_dronevehicle_seed{seed}_metrics_record.json", f"N_llvip_seed{seed}_metrics_record.json")]
        p, n = [read(f) for f in files]
        assert p["dataset"] == n["dataset"] == "dronevehicle"
        assert p["seed"] == n["seed"] == seed
        assert p["split"] == n["split"] == "val"
        assert p["data_yaml"] == n["data_yaml"]
        assert p["checkpoint"].endswith("weights/last.pt") and n["checkpoint"].endswith("weights/last.pt")
        differences.append(100 * (p["metrics"]["mAP50_95"] - n["metrics"]["mAP50_95"]))
        names.extend(f.relative_to(repo).as_posix() for f in files)
    rows.append({"dataset": "dronevehicle", "comparison": "CMDistill-minus-W1-native", "seeds": [0, 42, 123], "differences_pp": differences, "mean_pp": statistics.mean(differences), "sample_sd_pp": statistics.stdev(differences), "source_files": names, "filename_caveat": "N_llvip filenames are historical misnames; JSON identifies DroneVehicle."})
    probe = read(repo / "research_bundle/08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识/probe_analysis.json")
    probe_rows = []
    for name, value in probe.items():
        if not isinstance(value, dict) or "n_images" not in value:
            continue
        assert value["n_common_objects"] == sum(value[k] for k in ("teacher_only", "student_only", "both", "neither"))
        probe_rows.append({k: value[k] for k in ("dataset", "teacher", "student", "n_images", "n_common_objects", "teacher_only", "student_only", "both", "neither", "both_hit_teacher_minus_student_iou")})
    assert sum(row["n_images"] for row in probe_rows) == 521
    result = {"scope": "Publication wording numerical crosscheck only; no new evaluation or accepted-analyzer claim", "metric": "mAP50_95", "units": "percentage points", "sample_sd_ddof": 1, "historical_comparisons": rows, "probe_summary_arithmetic": probe_rows, "limitations": "Does not certify full recipe identity, independently rerun detection, infer gain for OEv1, or establish causal attribution."}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
