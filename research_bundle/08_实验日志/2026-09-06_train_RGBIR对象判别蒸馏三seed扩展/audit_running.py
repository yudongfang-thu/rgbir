"""CPU-only, read-only snapshot of a running object-evidence experiment.

Original run files are never modified. This script does not import torch, load
weights, or acquire a GPU. Report windows are fixed first/last five full epochs.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def stats(values):
    values = [float(x) for x in values if x is not None]
    finite = [x for x in values if math.isfinite(x)]
    return {"count": len(values), "nonfinite": len(values)-len(finite),
            "mean": statistics.mean(finite) if finite else None,
            "min": min(finite) if finite else None,
            "max": max(finite) if finite else None,
            "sd": statistics.stdev(finite) if len(finite)>1 else None}


def summarize_kd(rows):
    keys = ["native_total", "loss_unweighted", "weighted_kd_total", "total_loss",
            "base_count", "eligible_count", "selected_count", "nominal_dose",
            "student_evidence_selected_mean", "teacher_evidence_selected_mean",
            "reference_evidence_selected_mean", "quality_selected_mean",
            "target_clipped_count"]
    out = {key: stats([r.get(key) for r in rows]) for key in keys}
    out.update(logged_batches=len(rows), zero_selection_batches=sum(r["selected_count"] == 0 for r in rows),
               zero_kd_batches=sum(r["loss_unweighted"] == 0 for r in rows),
               selected_sum=sum(r["selected_count"] for r in rows),
               base_sum=sum(r["base_count"] for r in rows))
    out["selected_fraction_of_base"] = out["selected_sum"]/max(1, out["base_sum"])
    out["kd_over_native"] = stats([r["weighted_kd_total"]/r["native_total"] for r in rows if r["native_total"]>0])
    return out


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"command": args, "exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def process_snapshot(run):
    ps = command(["ps", "-eo", "pid=,ppid=,user=,rss=,args="])
    all_rows = []
    for line in ps["stdout"].splitlines():
        parts = line.split(None, 4)
        if len(parts)==5:
            all_rows.append(dict(pid=int(parts[0]), ppid=int(parts[1]), user=parts[2], rss_kib=int(parts[3]), args=parts[4]))
    matched = [r for r in all_rows if str(run) in r["args"] and "train_object_evidence.py" in r["args"]]
    for row in matched:
        allowed = {"CUDA_VISIBLE_DEVICES", "JSTARS_RESOURCE_LEASE_ID", "JSTARS_RESOURCE_LEASE_FILE", "JSTARS_RESOURCE_GPU_IDS"}
        try:
            env = (Path("/proc")/str(row["pid"])/"environ").read_bytes().split(b"\0")
            row["resource_environment"] = {k.decode():v.decode(errors="replace") for e in env if b"=" in e for k,v in [e.split(b"=",1)] if k.decode(errors="replace") in allowed}
        except (FileNotFoundError, PermissionError):
            row["resource_environment"] = None
    return {"matching_processes": matched, "matching_process_rss_mib_sum": sum(r["rss_kib"] for r in matched)/1024,
            "note": "RSS sum includes shared resident pages repeatedly; process descendants include train and prebuilt val loader workers."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Refuse accidental reuse so every running-state snapshot remains reviewable.
    args.output.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc)
    source_files = ["progress.json", "results.csv", "kd_batches.jsonl", "launch_manifest.json", "runtime_ready.json"]
    texts = {name: (args.run/name).read_text(encoding="utf-8") for name in source_files}
    progress = json.loads(texts["progress.json"])
    csv_rows = [{k.strip(): float(v) for k,v in row.items()} for row in csv.DictReader(io.StringIO(texts["results.csv"]))]
    complete_epochs = int(max((r["epoch"] for r in csv_rows), default=0))
    kd_rows, ignored_partial_lines = [], 0
    for line in texts["kd_batches.jsonl"].splitlines():
        try:
            kd_rows.append(json.loads(line))
        except json.JSONDecodeError:
            ignored_partial_lines += 1
    first_csv, last_csv = csv_rows[:5], csv_rows[-5:]
    first_kd = [r for r in kd_rows if 0 <= r["epoch"] < min(5, complete_epochs)]
    last_kd = [r for r in kd_rows if max(0,complete_epochs-5) <= r["epoch"] < complete_epochs]
    train_keys = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    csv_all_finite = all(math.isfinite(v) for r in csv_rows for v in r.values())
    kd_numeric_nonfinite = sum(not math.isfinite(v) for r in kd_rows for v in r.values() if isinstance(v,(int,float)))
    guard_path = Path("/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/runs/.project_resource_leases.json")
    guard = read_json(guard_path)
    leases = [v for v in (guard or {}).get("leases",{}).values() if "rgbir_oev1_full_paired_s42_attempt1" in json.dumps(v)]
    summary = {
        "schema": "rgbir-object-evidence-running-health-v1", "captured_at_utc": now.isoformat(),
        "run": str(args.run), "scope": "Read-only descriptive training-health audit; no AP or gain inference",
        "progress": progress, "completed_epochs_from_results_csv": complete_epochs,
        "csv_first_epoch": csv_rows[0] if csv_rows else None, "csv_latest_complete_epoch": csv_rows[-1] if csv_rows else None,
        "native_first_five_complete_epochs": {k: stats([r[k] for r in first_csv]) for k in train_keys},
        "native_last_five_complete_epochs": {k: stats([r[k] for r in last_csv]) for k in train_keys},
        "csv_all_numeric_finite": csv_all_finite, "logged_kd_numeric_nonfinite_count": kd_numeric_nonfinite,
        "kd_log_count": len(kd_rows), "ignored_partial_jsonl_lines": ignored_partial_lines,
        "kd_all_logged_batches": summarize_kd(kd_rows),
        "kd_first_five_complete_epochs": summarize_kd(first_kd),
        "kd_last_five_complete_epochs": summarize_kd(last_kd),
        "amp_skip_fraction": progress["amp_skips"]/max(1,progress["update_attempts"]),
        "updates_reconciled": progress["optimizer_updates"]+progress["amp_skips"]==progress["update_attempts"],
        "runtime": json.loads(texts["runtime_ready.json"]), "launch": json.loads(texts["launch_manifest.json"]),
        "queue": read_json(args.artifact_root/"full_queue_status.json"),
        "failure_receipt_exists": (args.run/"failure_receipt.json").exists(),
        "completion_receipt_exists": (args.run/"completion_receipt.json").exists(),
        "formal_gradient_log_exists": (args.run/"gradient_checks.jsonl").exists(),
        "guard_source": str(guard_path), "matching_guard_leases": leases,
        "process_snapshot": process_snapshot(args.run),
        "gpu_snapshot": command(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]),
        "gpu_process_snapshot": command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"]),
        "source_files": [{"path":str(args.run/name),"bytes":len(text.encode('utf-8'))} for name,text in texts.items()],
        "limitations": [
            "CSV metrics/val fields are zero placeholders because validate/final_eval are overridden; final standardized last/EMA val has not run.",
            "KD is logged on batches 1/2/3 then every 100; window means are descriptive logged-batch means, not all-batch exact means.",
            "Nonzero loss and selection do not prove all parameter gradients remain useful or aligned. Formal run has no gradient-norm trace; canary only verified initial nonzero gradient.",
            "Training-loss decrease cannot establish development AP improvement, causal transfer, or avoidance of negative transfer.",
            "Seed42 is incomplete and weight0 is queued. Additional seeds do not replace shuffled/same-modal attribution.",
            "Student/reference/teacher evidence means reflect different models and changing augmented batches; these are not calibrated probabilities or AP.",
            "All source files are sampled separately while training continues; timestamps and batch counts may differ slightly."
        ]
    }
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    snapshots = args.output/"sources"
    snapshots.mkdir()
    for name,text in texts.items():
        if name != "kd_batches.jsonl":
            (snapshots/name).write_text(text,encoding="utf-8")
    compact_keys = [k for k in kd_rows[0] if k not in {"selected_object_ids","student_files","config"}] if kd_rows else []
    with (snapshots/"kd_logged_scalar_snapshot.jsonl").open("w",encoding="utf-8") as f:
        for row in kd_rows:
            f.write(json.dumps({k:row.get(k) for k in compact_keys},ensure_ascii=False,allow_nan=False)+"\n")
    lines = ["# RGBIR OEv1 seed42 运行健康只读审计", "",
        "> 结论：当前训练与蒸馏信号没有出现日志可见的数值崩溃或选样归零；训练尚未完成，尚无检测性能或跨模态净收益结论。", "",
        f"采集时间（UTC）：{summary['captured_at_utc']}。原 run 只读，无 GPU 计算、无权重读取、无训练状态修改。", "",
        "## 训练进度与数值", "",
        f"CSV 已完整写出 {complete_epochs} / {progress['epochs']} 轮；progress 当前第 {progress['epoch']} 轮。真实更新 {progress['optimizer_updates']}，尝试 {progress['update_attempts']}，AMP skip {progress['amp_skips']}（{100*summary['amp_skip_fraction']:.4f}%）。更新计数可核对：{summary['updates_reconciled']}。",
        f"已记录训练 batch {progress['batches']}，累计入选对象实例 {progress['selected_objects']}（同一对象可在不同 epoch / 增强中重复，不能当成独立对象数）。", "",
        "| 原生训练损失 | 首5个完整epoch均值 | 末5个完整epoch均值 |", "|---|---:|---:|"]
    for k in train_keys:
        lines.append(f"| {k} | {summary['native_first_five_complete_epochs'][k]['mean']:.6f} | {summary['native_last_five_complete_epochs'][k]['mean']:.6f} |")
    lines += ["", f"CSV 数值全部 finite：{csv_all_finite}；KD 日志数值 nonfinite：{kd_numeric_nonfinite}；失败回执存在：{summary['failure_receipt_exists']}。", "",
        "## KD 运行信号", "", "固定首5轮与末5个完整轮次比较；每100 batch采样一次，前3 batch额外记录，不是全量均值。", "",
        "| KD 日志指标 | 首5个完整epoch | 末5个完整epoch |", "|---|---:|---:|"]
    for k in ["loss_unweighted","weighted_kd_total","selected_count","nominal_dose","student_evidence_selected_mean","teacher_evidence_selected_mean","reference_evidence_selected_mean","quality_selected_mean"]:
        lines.append(f"| {k} 均值 | {summary['kd_first_five_complete_epochs'][k]['mean']:.6f} | {summary['kd_last_five_complete_epochs'][k]['mean']:.6f} |")
    all_kd = summary["kd_all_logged_batches"]
    lines += ["", f"全部 {len(kd_rows)} 个已记录 batch 中，选样数为0：{all_kd['zero_selection_batches']}；KD为0：{all_kd['zero_kd_batches']}。末5轮加权KD/native损失比均值 {100*summary['kd_last_five_complete_epochs']['kd_over_native']['mean']:.3f}%。", "",
        "入选数与质量仍非零，KD仍参与总损失；这些日志支持‘辅助分支在工作’，不能推出‘有效避免负迁移’。长训未逐步记录梯度范数，不能仅凭loss非零断言后期梯度强度未退化；canary只证明初始实现梯度非零及零权重等价。", "",
        "## 资源与身份", "",
        f"runtime 可见设备={summary['runtime']['cuda_visible_devices']}，batch={summary['runtime']['batch']}，train workers={summary['runtime']['workers']}；frozen模型不进入optimizer={summary['runtime']['frozen_models_outside_optimizer']}；学生独立导出={summary['runtime']['student_only_state']}。", "",
        f"相关进程 RSS 相加 {summary['process_snapshot']['matching_process_rss_mib_sum']/1024:.2f} GiB；RSS包含共享页面重复计数，实际子进程也包含预建验证loader。GPU快照与guard租约原值见 summary.json。", "",
        "## 结果口径与下一步", "",
        "results.csv 的 precision/recall/AP/val-loss 均为关闭中间验证后的占位0，不能报道为AP=0。按冻结协议完成200轮，再由标准 last/EMA full-val 评估读取性能；weight0仍在同卡队列，未得到净收益比较。保持方法与阈值冻结，扩展到seed0/123只检验重复性；三seed paired−weight0仍不替代shuffled/same-modal归因。", "",
        "## 证据路径", "", f"原始run：`{args.run}`。本报告脚本 `audit_running.py`；完整机读摘要 `summary.json`；本次原始小文件与KD标量快照见 `sources/`。不同文件在训练继续期间分别读取，少量batch时差属于采样时差。"]
    (args.output/"health_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"completed_epochs":complete_epochs,"current_epoch":progress["epoch"],"amp_skip_fraction":summary["amp_skip_fraction"],"kd_log_count":len(kd_rows)},ensure_ascii=False))


if __name__ == "__main__":
    main()
