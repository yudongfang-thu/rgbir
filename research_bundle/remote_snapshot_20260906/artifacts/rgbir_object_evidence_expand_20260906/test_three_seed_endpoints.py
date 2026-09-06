"""Constructed CPU fixtures for endpoint math, identity, and pending behavior."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import yaml
import analyze_three_seed_endpoints as analyzer


def write_json(path, record):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record),encoding="utf-8")


def fixture(base,seed,arm,value=0.5):
    run=analyzer.run_path(base,seed,arm)
    run.mkdir(parents=True)
    checkpoint=run/"weights/last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"CPU fixture only, never loaded as a model")
    identity={"method_id":analyzer.METHOD,"arm":arm,"seed":seed}
    command=["python","train_object_evidence.py","--arm",arm,"--output",str(run),"--seed",str(seed)]
    inputs={"model":{"path":str(base/"fixed_init.pt")},**{k:{"path":str(v)} for k,v in analyzer.fixed_frozen_inputs(base).items()}}
    launch={**identity,"command":command,"inputs":inputs,"canary_max_updates":None,"test_accessed":False}
    actual={"seed":seed,"epochs":200,"batch":32,"imgsz":640}
    (run/"args.yaml").write_text(yaml.safe_dump(actual))
    # Frozen template default must not override the actual CLI/args seed.
    (run/"protocol_config.yaml").write_text(yaml.safe_dump({"seed":42,"epochs":200}))
    completion={**identity,"status":"training_completed","epochs_configured":200,"last_epoch":200,
                "checkpoint":str(checkpoint),"official_test_accessed":False,"single_seed_exploratory":True}
    evaluation={**identity,**{k:value for k in analyzer.METRICS},"checkpoint":str(checkpoint),
                "endpoint":analyzer.ENDPOINT,"split":"val","metric_units":"fraction_0_to_1",
                "official_test_accessed":False,"single_seed_exploratory":True}
    for name,record in (("launch_manifest.json",launch),("completion_receipt.json",completion),("evaluation_val.json",evaluation)):
        write_json(run/name,record)
    for kind,metric in (("train",completion),("eval",evaluation)):
        root=run/("run_evidence" if kind=="train" else "eval_evidence")
        root.mkdir()
        groups={}
        for group in ("trainer","loss","config","split_roster"):
            file=root/f"source_{group}.txt"
            file.write_text(f"fixture {group}")
            groups[group]=[file.name]
        snapshot=root/"metric_snapshot.json"
        write_json(snapshot,metric)
        receipt={"schema":"jstars-run-receipt-v1","terminal_status":"COMPLETED","seed":seed,"run_kind":kind,
                 "dataset":"dronevehicle","method_identity":"PROTOCOL-ADAPTED","data_role":"development_train" if kind=="train" else "development_val",
                 "job_id":f"fixture_{kind}_{arm}_{seed}","environment":{},"resources":{"gpu_ids":[4],"cuda_pid_counts":{"4":1},"per_gpu_peak_vram_mib":{"4":1},"peak_rss_mib":1},
                 "test_exposure":{"dataset":"dronevehicle","test_status":"UNVERIFIED_SEALED","confirmatory":False},
                 "source_snapshots":groups,"metric_snapshots":[snapshot.name]}
        if kind=="train":
            receipt.update(command={"argv":command},inputs={"method_id":analyzer.METHOD,"arm":arm,"canary":False,
                "initial_weights":inputs["model"]["path"],"teacher_weights":inputs["teacher"]["path"],"reference_weights":inputs["reference"]["path"]})
        else:
            receipt.update(command={"argv":["python","evaluate_object_evidence.py","--run",str(run)]},
                inputs={"method_id":analyzer.METHOD,"arm":arm,"checkpoint":str(checkpoint),"endpoint":analyzer.ENDPOINT})
        write_json(root/"run_receipt.json",receipt)
    return run


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="oev1_endpoint_fixture_")
        self.base=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def test_three_seed_math_sample_sd_and_exploratory(self):
        # Three independent seed contrasts are exactly 1, 2, and 3 pp.
        for seed,difference in zip(analyzer.SEEDS,(.01,.02,.03)):
            fixture(self.base,seed,"weight0",.50)
            fixture(self.base,seed,"paired",.50+difference)
        cells=[analyzer.collect_cell(self.base,s,a) for s in analyzer.SEEDS for a in analyzer.ARMS]
        self.assertTrue(all(c["status"]=="completed" for c in cells),cells)
        summary=analyzer.aggregate(cells)
        primary=summary["three_seed_summary"]["mAP50_95"]["paired_minus_weight0"]
        self.assertAlmostEqual(primary["mean_pp"],2)
        self.assertAlmostEqual(primary["sample_sd_pp"],1)
        self.assertEqual(summary["complete_seed_pairs"],3)
        self.assertTrue(all(c["single_seed_exploratory"] for c in cells))
    def test_missing_seed_does_not_produce_three_seed_mean(self):
        for arm in analyzer.ARMS:
            fixture(self.base,42,arm,.5)
        cells=[analyzer.collect_cell(self.base,s,a) for s in analyzer.SEEDS for a in analyzer.ARMS]
        report=analyzer.aggregate(cells)
        self.assertEqual(report["complete_endpoints"],2)
        self.assertEqual(report["complete_seed_pairs"],1)
        self.assertIsNone(report["three_seed_summary"])
        self.assertEqual(report["seed_pairs"][1]["difference_pp"]["mAP50_95"],0)
    def test_csv_zero_never_becomes_metric(self):
        run=analyzer.run_path(self.base,42,"paired")
        run.mkdir(parents=True)
        (run/"results.csv").write_text("epoch,metrics/mAP50-95(B)\n85,0\n")
        result=analyzer.collect_cell(self.base,42,"paired")
        self.assertEqual(result["status"],"pending_training")
        self.assertIsNone(result["metrics"])
    def test_default_config_42_does_not_override_actual_seed0(self):
        fixture(self.base,0,"paired")
        self.assertEqual(analyzer.collect_cell(self.base,0,"paired")["status"],"completed")
    def test_wrong_actual_seed_fails(self):
        run=fixture(self.base,0,"paired")
        (run/"args.yaml").write_text("seed: 42\nepochs: 200\nbatch: 32\nimgsz: 640\n")
        self.assertEqual(analyzer.collect_cell(self.base,0,"paired")["status"],"invalid_evidence")
    def test_wrong_arm_receipt_fails(self):
        run=fixture(self.base,123,"paired")
        path=run/"eval_evidence/run_receipt.json"
        r=analyzer.load_json(path);r["inputs"]["arm"]="weight0";write_json(path,r)
        self.assertEqual(analyzer.collect_cell(self.base,123,"paired")["status"],"invalid_evidence")
    def test_wrong_frozen_teacher_is_rejected_before_aggregation(self):
        run=fixture(self.base,42,"paired")
        path=run/"launch_manifest.json"
        r=analyzer.load_json(path);r["inputs"]["teacher"]["path"]=str(self.base/"wrong_teacher.pt");write_json(path,r)
        self.assertEqual(analyzer.collect_cell(self.base,42,"paired")["status"],"invalid_evidence")
    def test_missing_terminal_eval_receipt_stays_pending(self):
        run=fixture(self.base,42,"paired")
        (run/"eval_evidence/run_receipt.json").unlink()
        result=analyzer.collect_cell(self.base,42,"paired")
        self.assertEqual(result["status"],"pending_evaluation_evidence")
        self.assertIsNone(result["metrics"])
    def test_wrong_endpoint_epoch_and_metric_snapshot_fail(self):
        for name,change in (("epoch",{"last_epoch":199}),("endpoint",{"endpoint":"best_ema"}),("metric_snapshot",{"mAP50_95":.9})):
            with self.subTest(name=name):
                subbase=self.base/name
                run=fixture(subbase,42,"paired")
                target=run/("completion_receipt.json" if name=="epoch" else "evaluation_val.json")
                record=analyzer.load_json(target);record.update(change);write_json(target,record)
                self.assertEqual(analyzer.collect_cell(subbase,42,"paired")["status"],"invalid_evidence")
    def test_snapshot_wont_overwrite_previous_output(self):
        output=self.base/"snapshot1"
        analyzer.write_snapshot(self.base,output)
        before=(output/"summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            analyzer.write_snapshot(self.base,output)
        self.assertEqual((output/"summary.json").read_bytes(),before)


if __name__=="__main__":
    unittest.main(verbosity=2)
