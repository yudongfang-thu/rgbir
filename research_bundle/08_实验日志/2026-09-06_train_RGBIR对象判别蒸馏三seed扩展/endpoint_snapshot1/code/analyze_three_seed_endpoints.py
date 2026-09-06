"""Fixed six-cell OEv1 endpoint collector. CPU-only; never reads training CSV.

Only existing terminal receipts and fixed last/EMA development metrics count.
Output must be a new snapshot directory; original runs are strictly read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import yaml

SEEDS = (0, 42, 123)
ARMS = ("paired", "weight0")
METRICS = ("mAP50_95", "AP50", "AP75", "precision", "recall")
METHOD = "RGBIR-OBJECT-EVIDENCE-v1"
ENDPOINT = "fixed_budget_last_ema"
LIMITATIONS = [
    "Three student seeds share the same frozen seed42 IR teacher and RGB reference; uncertainty does not cover teacher/reference training variation.",
    "Paired minus weight0 estimates the whole intervention effect. Missing shuffled/same-modal controls prevent a cross-modal-specific knowledge or negative-transfer-avoidance claim.",
    "Development val only, fixed E200 last/EMA endpoint. Single-run exploratory flags remain preserved; aggregate reporting does not turn these into confirmatory test results.",
    "No p-value is computed from three seeds; report individual directions and sample SD, without treating images/batches as independent training replicates.",
    "No accepted analyzer or four-arm evidence upgrade is implied by this descriptive collector.",
    "Original files are read separately while queues may progress. Incomplete evidence remains pending; rerun into a fresh snapshot directory to collect later endpoints."
]


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def same_path(value, expected):
    return isinstance(value, str) and Path(value).resolve() == expected.resolve()


def flag(argv, key):
    if key not in argv:
        return None
    index = argv.index(key)
    require(index+1 < len(argv), f"Missing command value: {key}")
    return argv[index+1]


def run_path(base, seed, arm):
    campaign = "rgbir_object_evidence_v1_20260906" if seed == 42 else "rgbir_object_evidence_expand_20260906"
    return base / "runs" / campaign / f"full_{arm}_s{seed}_attempt1"


def check_identity(record, seed, arm, label):
    require(record.get("seed") == seed, f"{label}: seed mismatch")
    require(record.get("arm") == arm, f"{label}: arm mismatch")
    require(record.get("method_id") == METHOD, f"{label}: method mismatch")


def check_receipt(path, record, seed, arm, kind, metric):
    require(record.get("schema") == "jstars-run-receipt-v1", f"{kind}: receipt schema")
    require(record.get("terminal_status") == "COMPLETED", f"{kind}: terminal status")
    require(record.get("seed") == seed and record.get("run_kind") == kind, f"{kind}: identity")
    require(record.get("dataset", "").lower() == "dronevehicle", f"{kind}: dataset")
    require(record.get("method_identity") == "PROTOCOL-ADAPTED", f"{kind}: method identity")
    role = "development_train" if kind == "train" else "development_val"
    require(record.get("data_role") == role, f"{kind}: data role")
    require(bool(record.get("job_id")), f"{kind}: job id missing")
    inputs = record.get("inputs", {})
    require(inputs.get("method_id") == METHOD and inputs.get("arm") == arm, f"{kind}: input identity")
    command = record.get("command", {}).get("argv")
    require(isinstance(command, list) and command and all(isinstance(v, str) and v for v in command), f"{kind}: command missing")
    require(isinstance(record.get("environment"), dict), f"{kind}: environment missing")
    resources = record.get("resources", {})
    require(isinstance(resources, dict) and {"gpu_ids", "cuda_pid_counts", "per_gpu_peak_vram_mib", "peak_rss_mib"} <= resources.keys(), f"{kind}: resource evidence missing")
    exposure = record.get("test_exposure", {})
    require(isinstance(exposure, dict) and exposure.get("dataset", "").lower()=="dronevehicle" and {"test_status", "confirmatory"} <= exposure.keys(), f"{kind}: exposure evidence missing")
    snapshots = record.get("source_snapshots", {})
    groups = ("trainer", "config", "split_roster", "loss") if kind=="train" else ("trainer", "config", "split_roster")
    require(isinstance(snapshots, dict) and all(isinstance(snapshots.get(g),list) and snapshots[g] for g in groups), f"{kind}: source snapshot groups missing")
    metric_paths = record.get("metric_snapshots")
    require(isinstance(metric_paths, list) and metric_paths, f"{kind}: metric snapshots missing")
    referenced = [p for group in snapshots.values() for p in group] + metric_paths
    require(all(isinstance(p, str) and (path.parent/p).is_file() for p in referenced), f"{kind}: referenced evidence file missing")
    # Compare parsed snapshot contents directly, without hashes or checkpoint loading.
    require(any(load_json(path.parent/p) == metric for p in metric_paths), f"{kind}: metric snapshot differs from endpoint")
    return command


def collect_cell(base, seed, arm):
    run = run_path(base, seed, arm)
    result = {"seed": seed, "arm": arm, "run": str(run), "status": "pending_not_started", "metrics": None, "errors": []}
    if not run.is_dir():
        return result
    result["status"] = "pending_training"
    paths = {name: run/name for name in ("launch_manifest.json", "completion_receipt.json", "evaluation_val.json", "run_evidence/run_receipt.json", "eval_evidence/run_receipt.json")}
    result["evidence_paths"] = {k: str(v) for k,v in paths.items()}
    result["present_files"] = [k for k,v in paths.items() if v.is_file()]
    try:
        if (run/"failure_receipt.json").is_file():
            result.update(status="failed_training", failure=load_json(run/"failure_receipt.json"))
            return result
        launch = load_json(paths["launch_manifest.json"])
        actual_args = yaml.safe_load((run/"args.yaml").read_text()) if (run/"args.yaml").is_file() else None
        if launch is not None:
            check_identity(launch, seed, arm, "launch")
            require(launch.get("canary_max_updates") is None, "Canary run is not a full endpoint")
            require(launch.get("test_accessed") is False, "Launch test exposure mismatch")
            argv = launch.get("command", [])
            require(flag(argv,"--arm") == arm and same_path(flag(argv,"--output"),run), "Launch command arm/output mismatch")
            override = flag(argv,"--seed")
            require((seed==42 and override is None) or override==str(seed), "Launch CLI seed mismatch")
        if actual_args is not None:
            require(actual_args.get("seed")==seed and actual_args.get("epochs")==200, "Actual args seed/epoch mismatch")
            require(actual_args.get("batch")==32 and actual_args.get("imgsz")==640, "Actual args batch/image size mismatch")
        completion = load_json(paths["completion_receipt.json"])
        if completion is None:
            result["note"] = "Training terminal receipt absent; CSV metrics are deliberately not read."
            return result
        check_identity(completion, seed, arm, "completion")
        require(completion.get("status")=="training_completed", "Completion is not full training")
        require(completion.get("epochs_configured")==200 and completion.get("last_epoch")==200, "E200 endpoint incomplete")
        checkpoint = run/"weights/last.pt"
        require(same_path(completion.get("checkpoint"), checkpoint) and checkpoint.is_file(), "Completion checkpoint is not this run's existing last.pt")
        require(completion.get("official_test_accessed") is False, "Completion official test exposure mismatch")
        require(completion.get("single_seed_exploratory") is True, "Per-run exploratory status must be preserved")
        require(launch is not None and actual_args is not None, "Completed run lacks actual args/launch evidence")
        train_receipt = load_json(paths["run_evidence/run_receipt.json"])
        if train_receipt is None:
            result["status"]="pending_training_evidence"
            return result
        train_command = check_receipt(paths["run_evidence/run_receipt.json"],train_receipt,seed,arm,"train",completion)
        require(train_command == launch["command"], "Training receipt command differs from launch")
        require(train_receipt["inputs"].get("canary") is False, "Training receipt marks canary")
        for input_key, launch_key in (("initial_weights","model"),("teacher_weights","teacher"),("reference_weights","reference")):
            require(train_receipt["inputs"].get(input_key)==launch["inputs"][launch_key]["path"], f"Training input {input_key} differs from launch")
        evaluation = load_json(paths["evaluation_val.json"])
        if evaluation is None:
            result["status"]="pending_evaluation"
            return result
        check_identity(evaluation,seed,arm,"evaluation")
        require(evaluation.get("split")=="val" and evaluation.get("endpoint")==ENDPOINT, "Evaluation split/endpoint mismatch")
        require(same_path(evaluation.get("checkpoint"),checkpoint), "Evaluation checkpoint mismatch")
        require(evaluation.get("metric_units")=="fraction_0_to_1", "Evaluation metric units mismatch")
        require(evaluation.get("official_test_accessed") is False and evaluation.get("single_seed_exploratory") is True, "Evaluation exposure/exploratory mismatch")
        require(all(isinstance(evaluation.get(k),(int,float)) and not isinstance(evaluation[k],bool) and math.isfinite(evaluation[k]) and 0<=evaluation[k]<=1 for k in METRICS), "Evaluation metrics missing/nonfinite/out-of-range")
        eval_receipt = load_json(paths["eval_evidence/run_receipt.json"])
        if eval_receipt is None:
            result["status"]="pending_evaluation_evidence"
            return result
        eval_command = check_receipt(paths["eval_evidence/run_receipt.json"],eval_receipt,seed,arm,"eval",evaluation)
        require(same_path(flag(eval_command,"--run"),run), "Evaluation receipt run command mismatch")
        require(same_path(eval_receipt["inputs"].get("checkpoint"),checkpoint) and eval_receipt["inputs"].get("endpoint")==ENDPOINT, "Evaluation receipt endpoint mismatch")
        result.update(status="completed",metrics={k:float(evaluation[k]) for k in METRICS},
                      single_seed_exploratory=True, checkpoint=str(checkpoint),
                      teacher_weights=launch["inputs"]["teacher"]["path"],reference_weights=launch["inputs"]["reference"]["path"],
                      epochs=200,split="val",endpoint=ENDPOINT,metric_units="fraction_0_to_1")
    except (InvalidEvidence, ValueError, TypeError, KeyError, OSError) as error:
        result.update(status="invalid_evidence",errors=[f"{type(error).__name__}: {error}"])
    return result


def aggregate(cells):
    require(len(cells)==6 and {(r["seed"],r["arm"]) for r in cells}=={(s,a) for s in SEEDS for a in ARMS}, "Expected exactly six distinct frozen cells")
    pairs=[]
    for seed in SEEDS:
        by_arm={r["arm"]:r for r in cells if r["seed"]==seed}
        complete=all(by_arm[a]["status"]=="completed" for a in ARMS)
        row={"seed":seed,"status":"completed" if complete else "pending_pair", "difference_pp":None}
        if complete:
            row["difference_pp"]={k:100*(by_arm["paired"]["metrics"][k]-by_arm["weight0"]["metrics"][k]) for k in METRICS}
            row["primary_direction"]="positive" if row["difference_pp"]["mAP50_95"]>0 else "negative" if row["difference_pp"]["mAP50_95"]<0 else "zero"
        pairs.append(row)
    out={"complete_endpoints":sum(r["status"]=="completed" for r in cells),"complete_seed_pairs":sum(p["status"]=="completed" for p in pairs),
         "seed_pairs":pairs,"three_seed_summary":None,"status":"pending"}
    if out["complete_endpoints"]==6:
        require(len({r["teacher_weights"] for r in cells})==1 and len({r["reference_weights"] for r in cells})==1,"Frozen teacher/reference differ across cells")
        summary={}
        for metric in METRICS:
            summary[metric]={}
            for arm in ARMS:
                values=[100*next(r for r in cells if r["seed"]==s and r["arm"]==arm)["metrics"][metric] for s in SEEDS]
                summary[metric][arm]={"mean_pct":statistics.mean(values),"sample_sd_pct":statistics.stdev(values),"values_pct":values}
            differences=[p["difference_pp"][metric] for p in pairs]
            summary[metric]["paired_minus_weight0"]={"mean_pp":statistics.mean(differences),"sample_sd_pp":statistics.stdev(differences),"values_pp":differences}
        out.update(status="completed_descriptive",three_seed_summary=summary)
    return out


def write_snapshot(base, output):
    cells=[collect_cell(base,s,a) for s in SEEDS for a in ARMS]
    report={"schema":"rgbir-oev1-three-seed-endpoints-v1","captured_at_utc":datetime.now(timezone.utc).isoformat(),
            "base":str(base),"method_id":METHOD,"student_seeds":list(SEEDS),"teacher_seed":42,"reference_seed":42,
            "primary_estimand":"100 * (paired.mAP50_95 - weight0.mAP50_95), per matched student seed",
            "aggregation":"All three complete pairs only; sample SD with ddof=1; no p-value", "cells":cells,
            "limitations":LIMITATIONS}
    try:
        report.update(aggregate(cells))
    except InvalidEvidence as error:
        report.update(status="invalid_cross_run_evidence",three_seed_summary=None,errors=[str(error)])
    output.mkdir(parents=True,exist_ok=False)
    (output/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines=["# OEv1 三seed固定端点汇总", "",f"> 状态：{report['status']}；完整端点 {sum(r['status']=='completed' for r in cells)}/6。本报告不读取训练CSV中的占位指标。", "",
           f"UTC采集时间：{report['captured_at_utc']}。主量为同seed paired−weight0 mAP50–95百分点，固定E200 last/EMA与完整开发val。", "",
           "| student seed | arm | 状态 | mAP50–95 (%) |", "|---|---|---|---:|"]
    for row in cells:
        value=f"{100*row['metrics']['mAP50_95']:.6f}" if row['metrics'] else "—"
        lines.append(f"| {row['seed']} | {row['arm']} | {row['status']} | {value} |")
    lines += ["", "## 配对主量", "", "| seed | paired−weight0 (pp) | 方向 |", "|---|---:|---|"]
    for pair in report.get("seed_pairs",[]):
        value=f"{pair['difference_pp']['mAP50_95']:+.6f}" if pair["difference_pp"] else "—"
        lines.append(f"| {pair['seed']} | {value} | {pair.get('primary_direction','待完成')} |")
    if report.get("three_seed_summary"):
        primary=report["three_seed_summary"]["mAP50_95"]["paired_minus_weight0"]
        lines += ["",f"三seed主量 mean±sample SD：{primary['mean_pp']:+.6f} ± {primary['sample_sd_pp']:.6f} pp（ddof=1）。每臂及次指标完整统计见summary.json。"]
    else:
        lines += ["", "三seed汇总未产生；缺失或无效端点不能用CSV、历史baseline或单seed指标补位。"]
    lines += ["", "## 解释边界", "", "教师与参考模型固定seed42，本轮只覆盖学生训练随机性。单run的exploratory标记保留；三seed汇总也不自动变成confirmatory证据。paired−weight0仅支持整套干预净效果，尚缺shuffled/same-modal四臂归因，不能宣称跨模态独特收益或避免负迁移。三点不输出p值。", "", "## 原始证据路径", ""]
    for row in cells:
        lines.append(f"- seed{row['seed']} {row['arm']}：`{row['run']}`" + (f"；问题：{' | '.join(row['errors'])}" if row["errors"] else ""))
    (output/"endpoint_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,default=Path("/mnt/dataset/yudongfang/projects/RGBT_campaign"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    report=write_snapshot(args.base,args.output)
    print(json.dumps({"status":report["status"],"complete_endpoints":sum(r["status"]=="completed" for r in report["cells"]),"output":str(args.output)}))


if __name__=="__main__":
    main()
