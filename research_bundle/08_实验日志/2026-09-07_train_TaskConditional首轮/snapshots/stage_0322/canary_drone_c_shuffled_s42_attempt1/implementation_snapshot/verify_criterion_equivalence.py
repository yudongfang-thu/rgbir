"""Exact old OEv1 N/C vs task-conditional N/C comparison on one shared real graph.

This entry point MUST be launched through project_resource_guard. It performs
no optimizer updates, never loads test images, and defaults to the first real
full B32 training batch. Syntax checking this file is not a GPU validation.

Acceptance requires bitwise-equal losses, native items and raw score/DFL
gradients for old weight0/new n and old paired/new c, an active C contribution
on at least one inspected batch, unchanged model state, frozen auxiliaries,
and equal RNG effects. Numerical tolerances are diagnostic only.

The reusable verify_same_prediction() function accepts an already prepared
batch/prediction/native criterion so the root can also attach this verification
to a lease-bound canary without repeating the student forward.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from criterion_items import clone_items, compare_items

from legacy_bridge import legacy, LEGACY, HERE
from task_criterion import TaskCriterion

# The trainer monkey-patches legacy.EvidenceCriterion when constructed. Taking
# TaskCriterion's actual historical base also works when this reusable helper is
# imported AFTER trainer setup; it cannot accidentally compare new-vs-new.
HistoricalEvidenceCriterion = TaskCriterion.__bases__[0]
if HistoricalEvidenceCriterion.__name__ != "EvidenceCriterion":
    raise RuntimeError("TaskCriterion no longer directly inherits the pinned historical criterion.")


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state().clone(),
                cuda=[state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [])


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _same_rng(a, b):
    na, nb = a["numpy"], b["numpy"]
    return (a["python"] == b["python"] and na[0] == nb[0] and
            np.array_equal(na[1], nb[1]) and na[2:] == nb[2:] and
            torch.equal(a["cpu"], b["cpu"]) and len(a["cuda"]) == len(b["cuda"]) and
            all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"])))


def _gradient(total, targets):
    gradients = torch.autograd.grad(total.sum(), targets, retain_graph=True, allow_unused=True)
    return [None if g is None else g.detach().clone() for g in gradients]


def _tensor_comparison(a, b):
    if a is None or b is None:
        return dict(exact=a is None and b is None, max_abs_error=None,
                    left_is_none=a is None, right_is_none=b is None)
    equal_shape = tuple(a.shape) == tuple(b.shape)
    return dict(exact=equal_shape and a.dtype == b.dtype and bool(torch.equal(a, b)),
                max_abs_error=float((a.float()-b.float()).abs().max()) if equal_shape and a.numel() else 0.0,
                shape=list(a.shape), dtype=str(a.dtype))


def _raw_comparison(left, right):
    return {name: _tensor_comparison(a, b)
            for name, a, b in zip(("scores", "dfl_logits"), left, right)}


def _snapshot_state(models):
    return {name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            for name, model in models.items()}


def _unchanged_state(models, snapshot):
    changed = []
    for name, model in models.items():
        now = model.state_dict()
        if set(now) != set(snapshot[name]):
            changed.append(name + ":state_keys")
            continue
        for key, before in snapshot[name].items():
            if not torch.equal(now[key].detach().cpu(), before):
                changed.append(name + ":" + key)
    return changed


def verify_same_prediction(prediction, batch, *, native, teacher, reference, cfg,
                           trainer, output, historical_class=HistoricalEvidenceCriterion):
    """Check both mappings using the SAME prediction, batch and native instance.

    No backward-to-parameters and no optimizer step are performed. Model state
    is measured after the caller's student forward, so expected train-mode BN
    updates during that forward are not misclassified as criterion mutations.
    Temporary assigner hooks installed by verification TaskCriterion instances
    are removed afterward; pre-existing trainer hooks are left intact.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    actual_batch = int(batch["img"].shape[0])
    if actual_batch != 32 or int(cfg["batch"]) != 32:
        raise ValueError("Historical endpoint reuse verification requires unchanged full B32.")
    if "teacher_batch" not in batch or "strong_img" not in batch:
        raise ValueError("A real preprocessed paired batch with independent teacher labels is required.")
    raw = legacy.raw_prediction(prediction)
    targets = (raw["scores"], raw["boxes"])
    if not all(t.requires_grad for t in targets):
        raise ValueError("Student raw score and DFL tensors must have their real autograd graph.")
    models = dict(student=trainer.model, teacher=teacher, reference=reference)
    if any(p.requires_grad for model in (teacher, reference) for p in model.parameters()):
        raise ValueError("Teacher and reference must already be frozen.")
    if any(p.grad is not None for model in (teacher, reference) for p in model.parameters()):
        raise ValueError("Teacher/reference have pre-existing gradients.")
    state_before = _snapshot_state(models)
    rng_before = _rng_state()
    hooks = getattr(native.assigner, "_forward_hooks", {})
    existing_hook_keys = set(hooks)
    cfg = copy.deepcopy(cfg)
    # This coefficient is unused by N/C; set only if the formal L calibration
    # has not happened. C weight, evidence policy and teacher identity stay exact.
    if cfg.get("localization_coefficient") is None:
        cfg["localization_coefficient"] = 0.0
    result = dict(protocol="tc_legacy_criterion_exact_equivalence_v1", actual_batch=actual_batch,
                  same_prediction_object=True, same_native_criterion_instance=True,
                  teacher_labels_present=True, optimizer_updates=0, pairs=[])
    try:
        # Establish native repeatability separately. A native CUDA nondeterminism
        # failure must not be reported as a newly introduced C mismatch.
        native_runs = []
        for _ in range(2):
            _restore_rng(rng_before)
            total, items = native(prediction, batch)
            native_runs.append((total.detach().clone(), clone_items(items),
                                _gradient(total, targets), _rng_state()))
        native_raw = _raw_comparison(native_runs[0][2], native_runs[1][2])
        result["native_repeat"] = dict(loss=_tensor_comparison(native_runs[0][0], native_runs[1][0]),
            items=compare_items(native_runs[0][1], native_runs[1][1]), gradients=native_raw,
            rng_effect_equal=_same_rng(native_runs[0][3], native_runs[1][3]))
        for old_arm, new_arm in (("weight0", "n"), ("paired", "c")):
            calls = []
            for label, cls, arm in (("old_"+old_arm, historical_class, old_arm),
                                    ("new_"+new_arm, TaskCriterion, new_arm)):
                folder = output / label
                folder.mkdir()
                view = SimpleNamespace(model=trainer.model, epoch=getattr(trainer, "epoch", 0),
                    real_updates=0, wall_started=time.time(), save_dir=folder)
                criterion = cls(native, teacher, reference, cfg, arm, view, sanity=False)
                _restore_rng(rng_before)
                total, items = criterion(prediction, batch)
                gradient = _gradient(total, targets)
                calls.append(dict(loss=total.detach().clone(), items=clone_items(items),
                    gradient=gradient, rng=_rng_state(), criterion=criterion))
            left, right = calls
            gradients = _raw_comparison(left["gradient"], right["gradient"])
            pair = dict(old_arm=old_arm, new_arm=new_arm,
                loss=_tensor_comparison(left["loss"], right["loss"]),
                native_items=compare_items(left["items"], right["items"]),
                raw_gradients=gradients, rng_effect_equal=_same_rng(left["rng"], right["rng"]),
                old_calls=left["criterion"].calls, new_calls=right["criterion"].calls,
                old_selected_C=left["criterion"].selected_total,
                new_selected_C=right["criterion"].selected_total,
                old_loss=float(left["loss"].sum()), new_loss=float(right["loss"].sum()))
            pair["exact"] = (pair["loss"]["exact"] and pair["native_items"]["exact"] and
                all(row["exact"] for row in gradients.values()) and pair["rng_effect_equal"] and
                pair["old_selected_C"] == pair["new_selected_C"])
            if new_arm == "n":
                pair["native_scalar_loss_exact"] = bool(torch.equal(left["loss"], native_runs[0][0].sum()))
                pair["native_raw_gradients"] = _raw_comparison(left["gradient"], native_runs[0][2])
                pair["exact"] = (pair["exact"] and pair["native_scalar_loss_exact"] and
                    all(row["exact"] for row in pair["native_raw_gradients"].values()))
            else:
                differences = []
                for value, base in zip(left["gradient"], native_runs[0][2]):
                    if value is None and base is None:
                        differences.append(0.0)
                    else:
                        value = torch.zeros_like(base) if value is None else value
                        base = torch.zeros_like(value) if base is None else base
                        differences.append(float((value.float()-base.float()).norm()))
                pair["C_contribution_raw_gradient_l2"] = dict(zip(("scores", "dfl_logits"), differences))
                pair["C_active"] = pair["old_selected_C"] > 0 and any(value > 0 for value in differences)
            result["pairs"].append(pair)
        changed = _unchanged_state(models, state_before)
        result["changed_state_keys"] = changed
        result["frozen_auxiliaries_no_grad"] = all(
            not p.requires_grad and p.grad is None
            for model in (teacher, reference) for p in model.parameters())
        repeat = result["native_repeat"]
        native_exact = (repeat["loss"]["exact"] and repeat["items"]["exact"] and
                        all(row["exact"] for row in repeat["gradients"].values()) and repeat["rng_effect_equal"])
        result["arithmetic_and_gradient_exact"] = (native_exact and not changed and
            result["frozen_auxiliaries_no_grad"] and all(pair["exact"] for pair in result["pairs"]))
        result["active_C_exercised"] = any(pair.get("C_active", False) for pair in result["pairs"])
        result["status"] = ("PASSED" if result["arithmetic_and_gradient_exact"] and result["active_C_exercised"]
                            else "INCONCLUSIVE_ZERO_C" if result["arithmetic_and_gradient_exact"] else "FAILED")
        result["criterion_scope_only"] = True
        result["not_proven_here"] = ["historical-vs-new loader sample-stream equality",
            "24 successful optimizer updates", "formal geometry coverage", "scientific benefit"]
        legacy.write_json(output/"equivalence.json", result)
        return result
    finally:
        # Restore the caller's RNG; diagnostic forwards must not change its run.
        _restore_rng(rng_before)
        for key in set(hooks)-existing_hook_keys:
            hooks.pop(key, None)


def run(args):
    from train_task_conditional import build_trainer, validate_config
    cfg = yaml.safe_load(args.config.read_text())
    cfg["seed"] = args.seed
    validate_config(cfg, "c", formal=False)
    lease = legacy.require_bound_lease_from_environment()
    if len(lease["gpus"]) != 1:
        raise ValueError("Exactly one bound GPU is required.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    shutil.copy2(args.config, args.output/"protocol_config.yaml")
    start = time.time()
    trainer = None
    try:
        trainer = build_trainer(cfg, args.config, args.output, "c", max_steps=1)
        trainer._setup_train()
        original = trainer.criterion_ref
        iterator = iter(trainer.train_loader)
        records = []
        for index in range(args.batches):
            batch = trainer.preprocess_batch(next(iterator))
            if int(batch["img"].shape[0]) != 32:
                raise ValueError("Verification encountered a non-full batch.")
            trainer.model.train()
            with torch.autocast(device_type="cuda", enabled=bool(trainer.amp)):
                prediction = trainer.model(batch["img"])
                records.append(verify_same_prediction(prediction, batch, native=original.native,
                    teacher=original.teacher, reference=original.reference, cfg=cfg, trainer=trainer,
                    output=args.output/("batch_%03d" % (index+1))))
            legacy.write_json(args.output/("batch_%03d_sources.json" % (index+1)), dict(
                im_file=list(batch["im_file"]), pair_info=batch["pair_info"],
                actual_batch=32, input_shape=list(batch["img"].shape),
                native_target_count=int(batch["bboxes"].shape[0]),
                teacher_target_count=int(batch["teacher_batch"]["bboxes"].shape[0])))
            del prediction, batch
        torch.cuda.synchronize()
        exact = all(record["arithmetic_and_gradient_exact"] for record in records)
        active = any(record["active_C_exercised"] for record in records)
        receipt = dict(status="ACCEPTED" if exact and active else "INCONCLUSIVE_ZERO_C" if exact else "FAILED",
            protocol="tc_legacy_criterion_exact_equivalence_v1", legacy_C_N_equivalence=bool(exact and active),
            batch_count=len(records), batch_size=32, seed=cfg["seed"], amp=bool(trainer.amp),
            optimizer_updates=0, test_accessed=False, dataset=cfg["dataset"],
            model=cfg["model"], teacher=cfg["teacher"], reference=cfg["reference"],
            config=str(args.config), legacy_source=str(LEGACY/"train_object_evidence.py"),
            new_source=str(HERE/"task_criterion.py"), records=records,
            criterion_scope_only=True, seconds=time.time()-start,
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            resources=legacy.bound_lease_resource_record_from_environment(),
            environment=dict(torch=str(torch.__version__), ultralytics=str(legacy.ultralytics.__version__)))
        legacy.write_json(args.output/"completion_receipt.json", receipt)
        legacy.emit_bound_run_receipt(run_dir=args.output/"run_evidence", method_identity=cfg["method_identity"],
            dataset=cfg["dataset"], data_role="development_train", seed=cfg["seed"], run_kind="eval",
            trainers=[Path(__file__), HERE/"criterion_items.py", HERE/"train_task_conditional.py", HERE/"tracked_pair_data.py"],
            losses=[HERE/"task_criterion.py", HERE/"localization_loss.py", LEGACY/"train_object_evidence.py",
                    LEGACY/"object_evidence_loss.py", *legacy.implementation_files(original.native)],
            configs=[args.config, Path(cfg["paths"]["student_data_yaml"]), Path(cfg["paths"]["privileged_data_yaml"])],
            split_rosters=[Path(cfg["paths"]["paired_train_mapping"])],
            metric_files=[args.output/"completion_receipt.json"], environment=receipt["environment"],
            inputs=dict(method_id=cfg["method_id"], purpose="criterion_exact_equivalence", optimizer_updates=0,
                        initial_weights=cfg["model"], teacher_weights=cfg["teacher"], reference_weights=cfg["reference"]))
        print(json.dumps({"status": receipt["status"], "output": str(args.output)}), flush=True)
        if not receipt["legacy_C_N_equivalence"]:
            raise RuntimeError("Exact criterion acceptance not established; see completion_receipt.json.")
    except BaseException as error:
        legacy.write_json(args.output/"failure_receipt.json", dict(status="failed", error=repr(error),
            traceback=traceback.format_exc(), seconds=time.time()-start, optimizer_updates=0,
            test_accessed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 42, 123), default=42)
    parser.add_argument("--batches", type=int, choices=(1, 2, 3, 4), default=1,
                        help="The fixed first 1-4 full training batches; no metric-based batch selection.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
