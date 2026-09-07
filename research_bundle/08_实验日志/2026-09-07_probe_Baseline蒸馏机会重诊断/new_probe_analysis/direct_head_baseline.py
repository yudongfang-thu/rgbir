"""Descriptive existing-head sanity at frozen confidence .25; no fit/parameter selection."""
import json
from pathlib import Path

import numpy as np

import analyze_baseline_probe as a


def main():
    root = Path(__file__).resolve().parent
    out = root / "direct_head_baseline_v1"
    out.mkdir(exist_ok=False)
    results = {}
    for dataset in ("dronevehicle", "llvip"):
        raw = root.parent / "remote_exports" / (dataset + "_full_attempt1")
        rows = a.read_rows(raw / "objects.jsonl")
        train = np.array([r["split"] == "train" for r in rows])
        dev = ~train
        bg = np.array([r.get("is_background", False) for r in rows])
        y = np.array([r["class"] for r in rows], dtype=int)
        groups, _ = a.group_columns(rows, train)
        with np.load(root / (dataset + "_analysis_v1") / "probe_predictions.npz", allow_pickle=False) as cached:
            np.testing.assert_array_equal(y, cached["y"])
            roi = cached["roi_common_valid"].copy()
        with np.load(raw / "logits.npz", allow_pickle=False) as logits:
            nc = logits["N42_cls"].shape[1]
            result = {"dataset": dataset, "input_directory": str(raw), "threshold": a.CONFIDENCE,
                      "rule": "Same-anchor native sigmoid(max class logit) >= .25: argmax foreground class; otherwise annotation-background class nc.",
                      "scope": "Descriptive original detector-head readout on GT-associated anchors, not AP/object-matched detection or a refitted classifier. Teacher unpaired positions do not certify same-object information.",
                      "models": {}}
            for model in ("N42", "T42", "N0"):
                if model + "_cls" not in logits:
                    continue
                z = logits[model + "_cls"].astype(np.float64)
                pred = z.argmax(axis=1)
                maximum = z.max(axis=1)
                confidence = np.exp(-np.logaddexp(0, -maximum))
                pred[confidence < a.CONFIDENCE] = nc
                result["models"][model] = {
                    "main_fixed_anchor": a.evaluation_views(y, pred, dev, nc + 1, bg, groups),
                    "auxiliary_gt_roi_common_valid": a.evaluation_views(y, pred, dev & roi, nc + 1, bg, groups)}
        result = a.plain(result)
        a.write_json(out / (dataset + "_summary.json"), result)
        results[dataset] = result
    lines = ["# 原生head直接读出 sanity（固定 conf=.25）", "",
             "**结论：该对照只诊断固定ridge是否低估现有head可读信息；所有原始probe、α及结果保持原样。**", "",
             "|数据集|模型|cohort|dev N|accuracy %|balanced %|前景macro recall %|背景recall %|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for dataset, result in results.items():
        for model, cohorts in result["models"].items():
            for cohort, v in cohorts.items():
                nums = [v["all"]["accuracy"], v["all"]["balanced_accuracy"], v["foreground"]["balanced_accuracy"], v["background"]["accuracy"]]
                lines.append(f"|{dataset}|{model}|{cohort}|{v['all']['n']}|" + "|".join("NA" if n is None else f"{100*n:.3f}" for n in nums) + "|")
    lines += ["", "输入是此前冻结的共同anchor raw类别logits，直接按原有conf=.25规则拒识背景，其余argmax类别，不拟合参数、不调阈值。各原生head分别使用本模型logits，GT-ROI有效cohort复用冻结probe保存的相同行掩码。", "",
              "这仍是GT关联位置的读出，不能称AP或常规检测召回率。共同有效ROI表只是相同行集合，直接head并未使用ROI特征。对未配对GT位置的IR读出不能声称同对象。", "",
              "若原生head的少数类召回显著高于N_logits ridge，说明该固定线性读出器/正则与类别不平衡限制了probe；不能由高维feature相对弱ridge的差值宣称logits缺信息、feature是更优KD载体或预期训练增益。", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
