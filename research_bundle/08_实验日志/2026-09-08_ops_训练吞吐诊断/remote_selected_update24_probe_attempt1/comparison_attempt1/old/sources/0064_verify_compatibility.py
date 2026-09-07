"""Lease-bound, exact historical/new N/C0 compatibility, with real training.

Fixed scope per seed/arm: thirty complete paired CPU loader batches, including
per-worker augmentation RNG continuations, then 24 successful optimizer updates
on each of the historical and new trainers. Historical state/gradient snapshots
are retained on the data disk; the new path compares in memory and does not save
duplicate state snapshots. No image corpus and no hashes are written.

The module imports no project runtime at import time, so --self-test can verify
the comparison/observation helpers on CPU without Ultralytics or CUDA.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import functools
import gc
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

LOADER_BATCHES = 30
SUCCESSFUL_UPDATES = 24
SEEDS = (0, 42, 123)
RNG_FIELD = "_compatibility_worker_rng"


def cpu_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone())


def restore_cpu_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def all_rng_state():
    state = cpu_rng_state()
    state["cuda"] = [row.clone().cpu() for row in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []
    return state


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    return copy.deepcopy(value)


def tensor_bits_equal(left, right):
    if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.layout != torch.strided or right.layout != torch.strided:
        raise ValueError("Sparse/non-strided tensors require an explicit comparison policy")
    # Byte equality also compares AMP NaN/Inf patterns without claiming that a
    # nonfinite gradient was usable. Actual optimizer calls/skips are checked too.
    def raw_bytes(value):
        value = value.detach().cpu().contiguous()
        if value.dtype == torch.bfloat16:
            value = value.view(torch.int16)  # equal element size works on old CPU torch too
        return value.numpy().tobytes()
    return raw_bytes(left) == raw_bytes(right)


def differences(expected, actual, path="root", allow_pair_extensions=False, limit=20):
    """Exact recursive comparison; only new tracked pair_info keys may differ."""
    errors = []
    if isinstance(expected, torch.Tensor):
        if not tensor_bits_equal(expected, actual):
            errors.append(path + ": tensor differs (dtype/shape/bytes)")
    elif isinstance(expected, np.ndarray):
        if (not isinstance(actual, np.ndarray) or expected.dtype != actual.dtype
                or expected.shape != actual.shape or expected.tobytes() != actual.tobytes()):
            errors.append(path + ": ndarray differs")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [path + ": expected dict"]
        missing, extra = set(expected)-set(actual), set(actual)-set(expected)
        if missing:
            errors.append(path + ": missing keys " + repr(sorted(missing, key=str)))
        if extra and not (allow_pair_extensions and ".pair_info" in path):
            errors.append(path + ": extra keys " + repr(sorted(extra, key=str)))
        for key in expected:
            if key in actual:
                errors += differences(expected[key], actual[key], path + "." + str(key), allow_pair_extensions, limit)
            if len(errors) >= limit:
                break
    elif isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            return [path + ": sequence type/length differs"]
        for i, (left, right) in enumerate(zip(expected, actual)):
            errors += differences(left, right, path + "[%d]" % i, allow_pair_extensions, limit)
            if len(errors) >= limit:
                break
    elif type(expected) is not type(actual) or expected != actual:
        errors.append(path + ": scalar/type differs")
    return errors[:limit]


def require_equal(expected, actual, path, allow_pair_extensions=False):
    errors = differences(expected, actual, path, allow_pair_extensions)
    if errors:
        raise AssertionError("Exact compatibility failed: " + "; ".join(errors))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def load_trace(path):
    # All traces are produced by this verifier from the user's trusted models.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class RNGObservedDataset(torch.utils.data.Dataset):
    """Read-only worker RNG observation around the real paired __getitem__."""
    def __init__(self, dataset):
        self.inner = dataset
        if hasattr(dataset, "__getitems__"):
            raise RuntimeError("A new batched dataset retrieval path needs separate compatibility review")

    def __len__(self):
        return len(self.inner)

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __getitem__(self, index):
        before = cpu_rng_state()
        item = self.inner[index]
        after = cpu_rng_state()
        item = dict(item)
        if RNG_FIELD in item:
            raise RuntimeError("RNG observation field already exists")
        worker = torch.utils.data.get_worker_info()
        item[RNG_FIELD] = dict(index=int(index), worker_id=-1 if worker is None else worker.id,
                               worker_seed=None if worker is None else worker.seed,
                               before=before, after=after)
        return item

    def collate_fn(self, rows):
        copies = [dict(row) for row in rows]
        records = [row.pop(RNG_FIELD) for row in copies]
        result = self.inner.collate_fn(copies)
        result[RNG_FIELD] = records
        return result


def shutdown_loader(loader):
    if loader is None:
        return
    for name in ("iterator", "_iterator"):
        iterator = getattr(loader, name, None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


def make_loader_pair(cfg):
    """Same native data constructor/recipe and loader implementation as trainer."""
    import yaml
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from runtime import ORIGINAL_DATASET, TrackedDualLabelRGBIRDataset
    keys = ("imgsz", "batch", "nbs", "workers", "seed", "epochs", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
            "cos_lr", "close_mosaic", "patience", "amp", "deterministic")
    overrides = {key: cfg[key] for key in keys}
    overrides.update(cfg["augmentation"])
    overrides.update(task="detect", mode="train", cache=False, rect=False, multi_scale=False,
                     compile=False, channels_last=False)
    args = get_cfg(overrides=overrides)
    base = []
    for key in ("student_data_yaml", "privileged_data_yaml"):
        data_path = cfg["paths"][key]
        if "test" in yaml.safe_load(Path(data_path).read_text()):
            raise ValueError("Only train/dev YAML allowed")
        data = check_det_dataset(data_path, autodownload=False)
        base.append(build_yolo_dataset(args, data["train"], cfg["batch"], data, mode="train", rect=False, stride=32))
    copies = copy.deepcopy(base)
    mapping = json.loads(Path(cfg["paths"]["paired_train_mapping"]).read_text())
    old = RNGObservedDataset(ORIGINAL_DATASET(base[0], base[1], mapping, max_teacher_cache=cfg["teacher_cache_images"]))
    new = RNGObservedDataset(TrackedDualLabelRGBIRDataset(copies[0], copies[1], mapping, max_teacher_cache=cfg["teacher_cache_images"]))
    # device controls native worker count/pinning only; tensors remain on CPU.
    # The bound single CUDA visibility therefore matches the actual trainer.
    before = cpu_rng_state()
    old_loader = build_dataloader(old, batch=cfg["batch"], workers=cfg["workers"], shuffle=True,
                                 rank=-1, drop_last=False, device="cuda")
    after_old = cpu_rng_state()
    restore_cpu_rng(before)
    new_loader = build_dataloader(new, batch=cfg["batch"], workers=cfg["workers"], shuffle=True,
                                 rank=-1, drop_last=False, device="cuda")
    require_equal(after_old, cpu_rng_state(), "loader_construction_rng")
    return old_loader, new_loader


def verify_loaders(cfg, output):
    seed_process(cfg["seed"])
    old, new = make_loader_pair(cfg)
    rows = []
    try:
        if old.num_workers != cfg["workers"] or new.num_workers != cfg["workers"]:
            raise RuntimeError("Native loader did not retain the requested worker count")
        old_iter, new_iter = iter(old), iter(new)
        for index in range(LOADER_BATCHES):
            before = cpu_rng_state()
            expected = next(old_iter)
            after = cpu_rng_state()
            restore_cpu_rng(before)
            actual = next(new_iter)
            require_equal(after, cpu_rng_state(), "loader_batch_%d_parent_rng" % index)
            if expected["img"].shape[0] != cfg["batch"] or actual["img"].shape[0] != cfg["batch"]:
                raise RuntimeError("Thirty complete batches required; encountered a partial batch")
            require_equal(expected, actual, "loader_batch_%d" % index, allow_pair_extensions=True)
            row = dict(batch=index+1, actual_B=int(expected["img"].shape[0]),
                       files=list(expected["im_file"]), all_original_fields_exact=True,
                       all_rgb_ir_pixels_exact=True, all_rgb_ir_labels_exact=True,
                       worker_rng_before_after_exact=True, parent_rng_continuation_exact=True,
                       dataset_indices=[value["index"] for value in expected[RNG_FIELD]],
                       worker_ids=[value["worker_id"] for value in expected[RNG_FIELD]],
                       worker_seeds=[value["worker_seed"] for value in expected[RNG_FIELD]])
            rows.append(row)
            append_json(output / "loader_batches.jsonl", row)
            # Small RNG/source evidence only. No training images are serialized.
            torch.save(dict(batch=index+1, files=expected["im_file"], rng=expected[RNG_FIELD]),
                       output / ("loader_rng_%03d.pt" % (index+1)))
            del expected, actual
        return dict(status="PASSED", loader_batches=len(rows), images_compared=len(rows)*cfg["batch"],
                    workers=old.num_workers, exact_pixels_labels_original_fields=True,
                    worker_rng_exact=True, parent_rng_exact=True, batches=rows,
                    dataset_instrumentation="read-only RNG observer around original __getitem__",
                    image_corpus_saved=False)
    finally:
        shutdown_loader(old)
        shutdown_loader(new)


def model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def gradients(model):
    return {key: None if value.grad is None else value.grad.detach().cpu().clone()
            for key, value in model.named_parameters()}


def finite_gradient_count(values):
    finite = nonfinite = absent = 0
    for value in values.values():
        if value is None:
            absent += 1
        elif bool(torch.isfinite(value).all()):
            finite += 1
        else:
            nonfinite += 1
    return dict(finite_tensors=finite, nonfinite_tensors=nonfinite, absent_gradients=absent)


def assert_auxiliaries(trainer):
    criterion = trainer.criterion_ref
    param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    for name in ("teacher", "reference"):
        model = getattr(criterion, name)
        if model.training or any(p.requires_grad or p.grad is not None or id(p) in param_ids for p in model.parameters()):
            raise AssertionError("Frozen auxiliary lifecycle violated: " + name)
    if set(trainer.model.state_dict()) != set(trainer.ema.ema.state_dict()):
        raise AssertionError("Student and EMA state identities differ")


class TrajectoryObserver:
    def __init__(self, trainer, output, historical_output, historical, arm):
        self.trainer = trainer
        self.output = output
        self.historical_output = historical_output
        self.historical = historical
        self.arm = arm
        self.attempts = 0
        self.batches = 0
        self.optimizer_calls = 0
        self.applied_gradients = None
        self.last_batch = None
        self.aux_initial = None

    def check_or_save(self, relative, value, allow_pair_extensions=False):
        path = self.historical_output / relative
        if self.historical:
            if path.exists():
                raise FileExistsError(path)
            torch.save(value, path)
        else:
            if not path.is_file():
                raise FileNotFoundError("Historical trace missing: " + str(path))
            expected = load_trace(path)
            if differences(expected, value, relative, allow_pair_extensions):
                # Preserve the actual failing RNG continuation without copying
                # another full model/optimizer trace or any training images.
                def rng_only(row):
                    if not isinstance(row, dict):
                        return {}
                    return {key: clone_cpu(item) if key == 'rng' else rng_only(item)
                            for key,item in row.items()
                            if key == 'rng' or key in ('before','after') and isinstance(item,dict)}
                failed = self.output / ('mismatch_rng_' + Path(relative).stem + '.pt')
                if failed.exists():
                    raise FileExistsError(failed)
                torch.save(dict(expected=rng_only(expected), actual=rng_only(value)), failed)
            require_equal(expected, value, relative, allow_pair_extensions)

    def install_before_train(self):
        trainer = self.trainer
        old_builder = trainer.build_dataset
        @functools.wraps(old_builder)
        def dataset(*args, **kwargs):
            result = old_builder(*args, **kwargs)
            mode = kwargs.get("mode", args[1] if len(args)>1 else "train")
            return RNGObservedDataset(result) if mode == "train" else result
        trainer.build_dataset = dataset
        old_preprocess = trainer.preprocess_batch
        @functools.wraps(old_preprocess)
        def preprocess(batch):
            self.batches += 1
            if int(batch["img"].shape[0]) != int(trainer.batch_size):
                raise AssertionError("Compatibility short training encountered incomplete batch")
            if RNG_FIELD not in batch:
                raise AssertionError("Actual training loader lacks worker RNG trace")
            metadata = clone_cpu({key:value for key,value in batch.items() if key not in ("img","strong_img")})
            self.check_or_save("batch_%04d.pt" % self.batches, metadata, allow_pair_extensions=True)
            self.last_batch = dict(batch=self.batches, files=list(batch["im_file"]),
                                   actual_B=int(batch["img"].shape[0]),
                                   worker_ids=[r["worker_id"] for r in batch[RNG_FIELD]])
            append_json(self.output/"training_batches.jsonl", dict(self.last_batch,
                fields_and_worker_rng_recorded=self.historical, fields_and_worker_rng_exact=not self.historical))
            return old_preprocess(batch)
        trainer.preprocess_batch = preprocess
        trainer.add_callback("on_train_start", self.on_train_start)

    def on_train_start(self, trainer):
        from runtime import ORIGINAL_CRITERION, ORIGINAL_DATASET, TrackedDualLabelRGBIRDataset
        from independent_criterion import IndependentCriterion
        expected_criterion = ORIGINAL_CRITERION if self.historical else IndependentCriterion
        expected_dataset = ORIGINAL_DATASET if self.historical else TrackedDualLabelRGBIRDataset
        if type(trainer.criterion_ref) is not expected_criterion:
            raise AssertionError("Criterion monkey-patch resolved to wrong original/new class")
        if not isinstance(trainer.train_loader.dataset, RNGObservedDataset) or type(trainer.train_loader.dataset.inner) is not expected_dataset:
            raise AssertionError("Dataset monkey-patch resolved to wrong original/new class")
        if trainer.train_loader.num_workers != trainer.args.workers:
            raise AssertionError("Actual training workers differ")
        assert_auxiliaries(trainer)
        self.aux_initial = {name:model_state(getattr(trainer.criterion_ref,name)) for name in ("teacher","reference")}
        initial = dict(student=model_state(trainer.model), ema=model_state(trainer.ema.ema),
                       optimizer=clone_cpu(trainer.optimizer.state_dict()), scaler=clone_cpu(trainer.scaler.state_dict()),
                       rng=all_rng_state(), auxiliaries=self.aux_initial,
                       trainable_names=[name for name,p in trainer.model.named_parameters() if p.requires_grad],
                       optimizer_updates=trainer.real_updates, attempts=trainer.update_attempts,
                       amp_skips=trainer.skipped_amp_updates, ema_updates=trainer.ema.updates,
                       amp=bool(trainer.amp), batch=int(trainer.batch_size), workers=trainer.train_loader.num_workers)
        self.check_or_save("initial_trace.pt",initial)
        write_json(self.output/"initial_comparison.json",dict(status="RECORDED" if self.historical else "EXACT",
            criterion_class=type(trainer.criterion_ref).__name__, dataset_class=type(trainer.train_loader.dataset.inner).__name__,
            criterion_file=inspect.getfile(type(trainer.criterion_ref)),
            dataset_file=inspect.getfile(type(trainer.train_loader.dataset.inner)),
            all_student_parameters_buffers=True, all_ema_parameters_buffers=True, rng=True))
        old_optimizer_call = trainer.optimizer.step
        @functools.wraps(old_optimizer_call)
        def optimizer_call(*args, **kwargs):
            self.optimizer_calls += 1
            self.applied_gradients = gradients(trainer.model)
            if finite_gradient_count(self.applied_gradients)["nonfinite_tensors"]:
                raise FloatingPointError("Nonfinite gradients actually reached optimizer.step")
            return old_optimizer_call(*args, **kwargs)
        trainer.optimizer.step = optimizer_call
        original_step = trainer.optimizer_step
        @functools.wraps(original_step)
        def step():
            self.attempts += 1
            if self.attempts > 96:
                raise RuntimeError("More than 96 attempts without the fixed 24 successes; technical failure")
            self.applied_gradients = None
            before_calls = self.optimizer_calls
            before_updates = trainer.real_updates
            before = dict(gradients_scaled=gradients(trainer.model),
                          grad_scale=float(trainer.scaler.get_scale()),
                          scaler=clone_cpu(trainer.scaler.state_dict()), rng=all_rng_state(),
                          native_and_kd_total=clone_cpu(trainer.loss), native_items=clone_cpu(trainer.loss_items),
                          accumulate=int(trainer.accumulate), epoch=int(trainer.epoch),
                          batches=self.batches, optimizer_updates=trainer.real_updates,
                          attempts=trainer.update_attempts, skips=trainer.skipped_amp_updates)
            original_step()
            actual_call_delta=self.optimizer_calls-before_calls
            actual_update_delta=trainer.real_updates-before_updates
            if actual_call_delta not in (0,1) or actual_update_delta != actual_call_delta:
                raise AssertionError("AMP successful-update counter differs from real optimizer.step calls")
            if trainer.update_attempts != self.attempts:
                raise AssertionError("Attempt counter differs from observed optimizer_step calls")
            assert_auxiliaries(trainer)
            after = dict(student=model_state(trainer.model), optimizer=clone_cpu(trainer.optimizer.state_dict()),
                         ema=model_state(trainer.ema.ema), ema_updates=trainer.ema.updates,
                         scaler=clone_cpu(trainer.scaler.state_dict()), rng=all_rng_state(),
                         optimizer_updates=trainer.real_updates, attempts=trainer.update_attempts,
                         amp_skips=trainer.skipped_amp_updates, actual_optimizer_calls=self.optimizer_calls,
                         applied_gradients=self.applied_gradients)
            self.check_or_save("attempt_%03d.pt" % self.attempts,dict(before=before,after=after))
            row=dict(attempt=self.attempts,successful_updates=trainer.real_updates,
                     real_optimizer_called=bool(actual_call_delta),amp_skips=trainer.skipped_amp_updates,
                     ema_updates=trainer.ema.updates,grad_scale_before=before["grad_scale"],
                     grad_scale_after=float(trainer.scaler.get_scale()),
                     scaled_gradient_finiteness=finite_gradient_count(before["gradients_scaled"]),
                     all_parameter_gradients_recorded=self.historical,
                     all_parameter_gradients_exact=not self.historical,
                     student_optimizer_ema_scaler_rng_exact=not self.historical,
                     batch=self.batches)
            append_json(self.output/"trajectory_attempts.jsonl",row)
            if trainer.real_updates > SUCCESSFUL_UPDATES:
                raise AssertionError("Short trajectory exceeded frozen successful-update budget")
        trainer.optimizer_step = step

    def finish(self):
        trainer=self.trainer
        if trainer.real_updates != SUCCESSFUL_UPDATES or self.optimizer_calls != SUCCESSFUL_UPDATES:
            raise AssertionError("Exactly 24 real successful optimizer updates required")
        current_aux={name:model_state(getattr(trainer.criterion_ref,name)) for name in ("teacher","reference")}
        require_equal(self.aux_initial,current_aux,"frozen_auxiliaries_final")
        if self.arm == "C0":
            checks=trainer.criterion_ref.gradient_checks
            if trainer.criterion_ref.selected_total <= 0 or not any(row.get("kd_score_gradient_l2",0)>0 for row in checks):
                raise AssertionError("C0 compatibility never exercised a nonzero KD gradient")
        result=dict(status="RECORDED" if self.historical else "EXACT", successful_updates=trainer.real_updates,
                    update_attempts=trainer.update_attempts,amp_skips=trainer.skipped_amp_updates,
                    ema_updates=trainer.ema.updates,batches=self.batches,real_optimizer_calls=self.optimizer_calls,
                    selected_objects=trainer.criterion_ref.selected_total,
                    frozen_auxiliary_parameters_buffers_unchanged=True,
                    gradient_checks=trainer.criterion_ref.gradient_checks)
        # These gradient checks must agree as well (including zero-KD N behavior).
        self.check_or_save("final_summary.pt",dict(result,status="COMMON"))
        write_json(self.output/"trajectory_summary.json",result)
        return result


def run_trajectory(cfg, config_path, output, historical_output, historical):
    from train_independent import build_trainer
    from runtime import legacy, ORIGINAL_CRITERION, ORIGINAL_DATASET
    seed_process(cfg["seed"])
    output.mkdir(parents=True,exist_ok=False)
    initialization = {'before_trainer':dict(rng=cpu_rng_state(),
        events_module_loaded='ultralytics.utils.events' in sys.modules)}
    trainer=observer=None
    try:
        trainer=build_trainer(cfg,config_path,output,arm=cfg["arm"],max_steps=SUCCESSFUL_UPDATES,historical=historical)
        initialization['after_trainer_constructor'] = dict(rng=cpu_rng_state(),
            events_module_loaded='ultralytics.utils.events' in sys.modules)
        torch.save(initialization, output/'initialization_rng_stages.pt')
        observer=TrajectoryObserver(trainer,output,historical_output,historical,cfg["arm"])
        observer.install_before_train()
        trainer.train()
        return observer.finish()
    finally:
        if trainer is not None:
            shutdown_loader(getattr(trainer,"train_loader",None))
            shutdown_loader(getattr(trainer,"test_loader",None))
        legacy.EvidenceCriterion=ORIGINAL_CRITERION
        legacy.DualLabelRGBIRDataset=ORIGINAL_DATASET
        trainer=observer=None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def require_data_output(path):
    resolved=Path(path).resolve()
    if os.name != 'posix' or not str(resolved).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('Real compatibility traces must be on /mnt/dataset/yudongfang, never the system disk')
    here=Path(__file__).resolve().parent
    if resolved == here or here in resolved.parents:
        raise ValueError('Output cannot be inside source tree')
    return resolved


def run_isolated_phase(cfg, config_path, output, phase, historical_output=None):
    """A fresh interpreter per phase preserves native first-import behavior.

    Even the loader phase exits before training: native pinned-memory loading
    may initialize CUDA, so the coordinating parent must remain CPU-only.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError('Compatibility coordinator must never own a CUDA context')
    command=[sys.executable,str(Path(__file__).resolve()),'--config',str(config_path),
             '--output',str(output),'--seed',str(cfg['seed']),'--arm',cfg['arm'],
             '--worker-stage',phase]
    if historical_output is not None:
        command += ['--historical-output',str(historical_output)]
    # Inherit the one existing lease. No new session, lease, GPU choice or
    # concurrent workload is introduced, and wait for process exit before next.
    subprocess.run(command,check=True)
    receipt=json.loads((output/'worker_receipt.json').read_text(encoding='utf-8'))
    if (receipt.get('status') != 'COMPLETED' or receipt.get('worker_stage') != phase
            or receipt.get('configuration') != cfg or not receipt.get('fresh_interpreter')):
        raise ValueError('Isolated worker receipt does not identify this phase/config')
    from evidence_bindings import validate_execution_binding
    validate_execution_binding(receipt['execution_binding'],cfg,'compatibility',Path(__file__).resolve().parent)
    if torch.cuda.is_initialized():
        raise RuntimeError('Child execution unexpectedly initialized coordinator CUDA')
    return receipt


def run_worker(args):
    """Internal lease-bound child entry; never bypasses the formal scope checks."""
    import yaml
    from runtime import legacy
    from train_independent import validate_execution
    cfg=yaml.safe_load(args.config.read_text())
    if cfg['seed'] != int(args.seed) or cfg['arm'] != args.arm or cfg.get('source') != 'paired':
        raise ValueError('Worker arguments differ from its effective configuration')
    for key,expected in (('epochs',200),('imgsz',640),('batch',32),('nbs',64),('workers',4)):
        if cfg.get(key) != expected:
            raise ValueError('Worker changed frozen recipe: '+key)
    validate_execution(cfg,formal=False)
    if len(legacy.require_bound_lease_from_environment()['gpus']) != 1:
        raise RuntimeError('Worker requires the inherited single bound lease')
    output=require_data_output(args.output)
    torch.set_num_threads(4)
    started=time.time()
    try:
        if args.worker_stage == 'loader':
            output.mkdir(parents=True,exist_ok=False)
            result=verify_loaders(cfg,output)
            write_json(output/'loader_receipt.json',result)
        else:
            historical=args.worker_stage == 'historical'
            historical_output=output if historical else require_data_output(args.historical_output)
            result=run_trajectory(cfg,args.config,output,historical_output,historical)
        from evidence_bindings import capture_execution_binding
        receipt=dict(status='COMPLETED',worker_stage=args.worker_stage,configuration=cfg,
            fresh_interpreter=True,pid=os.getpid(),parent_pid=os.getppid(),result=result,
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0.,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0.,
            resources=legacy.bound_lease_resource_record_from_environment(),seconds=time.time()-started,
            execution_binding=str(capture_execution_binding(output,cfg,'compatibility',Path(__file__).resolve().parent)))
        write_json(output/'worker_receipt.json',receipt)
    except BaseException as error:
        if output.is_dir():
            write_json(output/'worker_failure_receipt.json',dict(status='FAILED',worker_stage=args.worker_stage,
                error=repr(error),traceback=traceback.format_exc(),seconds=time.time()-started))
        raise


def verify_one(cfg, config_path, output):
    output.mkdir(parents=True,exist_ok=False)
    loader_dir=output/"loader_evidence"
    effective=output/'worker_config.yaml'
    import yaml
    effective.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    started=time.time()
    loader_worker=run_isolated_phase(cfg,effective,loader_dir,'loader')
    loaders=loader_worker['result']
    write_json(output/"loader_receipt.json",loaders)
    historical_output=output/"historical"
    old_worker=run_isolated_phase(cfg,effective,historical_output,'historical')
    new_worker=run_isolated_phase(cfg,effective,output/'new','new',historical_output)
    old,new=old_worker['result'],new_worker['result']
    result=dict(status="ACCEPTED",seed=cfg["seed"],arm=cfg["arm"],dataset=cfg["dataset"],
                model=cfg["model"],teacher=cfg["teacher"],reference=cfg["reference"],
                successful_updates=new["successful_updates"],loader_batches=loaders["loader_batches"],
                trajectory_exact=True,worker_rng_exact=True,all_original_loader_fields_exact=True,
                gradient_scope="all named student parameters before optimizer and actual unscaled/clipped gradients reaching optimizer.step",
                old=old,new=new,seconds=time.time()-started,official_test_accessed=False,
                execution_model='sequential_fresh_interpreters_in_one_existing_lease',
                coordinator_cuda_initialized=False,
                isolated_workers=[dict(stage=w['worker_stage'],pid=w['pid'],
                    execution_binding=w['execution_binding'],gpu_allocated_peak_mib=w['gpu_allocated_peak_mib'],
                    gpu_reserved_peak_mib=w['gpu_reserved_peak_mib']) for w in (loader_worker,old_worker,new_worker)],
                image_corpus_saved=False,
                limits=["fixed first 30 full batches and first 24 successful updates; not whole-E200 identity",
                        "instrumented worker RNG observers are deterministic read-only wrappers",
                        "historical first_batch.pt is preserved by its unchanged native canary path"])
    return result


def snapshot_sources(output):
    here=Path(__file__).resolve().parent
    snapshot=output/"implementation_snapshot"
    shutil.copytree(here,snapshot,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    return [dict(relative=str(path.relative_to(here)).replace("\\","/"),accepted_copy=str(snapshot/path.relative_to(here)))
            for path in sorted(here.rglob("*.py")) if "__pycache__" not in path.parts]


def run(args):
    import yaml
    from runtime import legacy
    from train_independent import validate_execution
    cfg=yaml.safe_load(args.config.read_text())
    cfg.update(arm=args.arm,source="paired",classification_coefficient=0. if args.arm=="N" else .1,
               localization_coefficient=0.)
    seeds=SEEDS if args.seed=="all" else (int(args.seed),)
    cfg["seed"]=seeds[0]
    for key, expected in (("epochs",200),("imgsz",640),("batch",32),("nbs",64),("workers",4)):
        if cfg.get(key) != expected:
            raise ValueError("Compatibility must retain the frozen recipe: " + key)
    validate_execution(cfg,formal=False)
    lease=legacy.require_bound_lease_from_environment()
    if len(lease["gpus"]) != 1:
        raise RuntimeError("Exactly one bound shared resource lease required")
    resolved=require_data_output(args.output)
    resolved.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise RuntimeError('Coordinator must remain CPU-only; use a fresh CLI process')
    sources=snapshot_sources(resolved)
    (resolved/"effective_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    records=[]
    started=time.time()
    try:
        for seed in seeds:
            current=dict(cfg,seed=seed)
            record=verify_one(current,args.config,resolved/("seed%d_%s"%(seed,args.arm)))
            record["source_files"]=sources
            write_json(resolved/("seed%d_%s"%(seed,args.arm))/"compatibility_receipt.json",record)
            records.append(record)
        receipt=dict(status="ACCEPTED",seed=seeds[0] if len(seeds)==1 else "all",seeds=list(seeds),
            arm=args.arm,dataset=cfg["dataset"],model=cfg["model"],teacher=cfg["teacher"],reference=cfg["reference"],
            successful_updates=min(row["successful_updates"] for row in records),
            loader_batches=min(row["loader_batches"] for row in records),trajectory_exact=True,
            worker_rng_exact=True,source_files=sources,records=records,seconds=time.time()-started,
            gpu_allocated_peak_mib=max(w['gpu_allocated_peak_mib'] for row in records for w in row['isolated_workers']),
            gpu_reserved_peak_mib=max(w['gpu_reserved_peak_mib'] for row in records for w in row['isolated_workers']),
            peak_memory_source='maximum_of_sequential_worker_receipts',coordinator_cuda_initialized=False,
            resources=legacy.bound_lease_resource_record_from_environment(),official_test_accessed=False,
            protocol="independent_kd_30batch_24update_exact_v1")
        from evidence_bindings import capture_execution_binding
        receipt['execution_binding']=str(capture_execution_binding(resolved,cfg,'compatibility',Path(__file__).resolve().parent))
        write_json(resolved/"compatibility_receipt.json",receipt)
        print(json.dumps(dict(status="ACCEPTED",seeds=list(seeds),arm=args.arm,output=str(resolved))),flush=True)
    except BaseException as error:
        write_json(resolved/"failure_receipt.json",dict(status="FAILED",error=repr(error),
            traceback=traceback.format_exc(),arm=args.arm,seeds=list(seeds),completed_records=records,
            source_files=sources,seconds=time.time()-started,official_test_accessed=False))
        raise


def self_test():
    import unittest
    class HelperTests(unittest.TestCase):
        def test_exact_tensor_dtype_shape_and_nan_bits(self):
            a=torch.tensor([1.,float("inf"),float("nan")])
            self.assertTrue(tensor_bits_equal(a,a.clone()))
            self.assertFalse(tensor_bits_equal(a,a.double()))
            self.assertTrue(differences(dict(a=a),dict(a=torch.ones(3))))
        def test_pair_extensions_are_only_allowed_in_pair_info(self):
            a=dict(pair_info=[dict(weak_source="x")],cls=torch.tensor([1]))
            b=dict(pair_info=[dict(weak_source="x",rgb_matrix=[1])],cls=torch.tensor([1]))
            self.assertTrue(differences(a,b))
            self.assertFalse(differences(a,b,allow_pair_extensions=True))
            b["extra"]=1
            self.assertTrue(differences(a,b,allow_pair_extensions=True))
        def test_observer_does_not_change_rng_or_fields(self):
            class Tiny(torch.utils.data.Dataset):
                def __len__(self): return 2
                def __getitem__(self,index):
                    return dict(img=torch.rand(2),index=index,x=random.random(),y=np.random.random())
                def collate_fn(self,rows): return dict(rows=rows)
            dataset=Tiny()
            observed=RNGObservedDataset(Tiny())
            random.seed(4);np.random.seed(4);torch.manual_seed(4)
            before=cpu_rng_state();expected=dataset[0];after=cpu_rng_state()
            restore_cpu_rng(before);actual=observed[0]
            require_equal(after,cpu_rng_state(),"rng")
            metadata=actual.pop(RNG_FIELD)
            require_equal(expected,actual,"sample")
            require_equal(before,metadata["before"],"before")
            require_equal(after,metadata["after"],"after")
        def test_clone_and_tree_scope(self):
            a=dict(x=[torch.ones(2),np.arange(3)],name="x")
            b=clone_cpu(a)
            self.assertFalse(differences(a,b))
            b["x"][0][0]=2
            self.assertTrue(differences(a,b))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HelperTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--seed",choices=("0","42","123","all"))
    p.add_argument("--arm",choices=("N","C0"))
    p.add_argument("--self-test",action="store_true")
    p.add_argument('--worker-stage',choices=('loader','historical','new'),help=argparse.SUPPRESS)
    p.add_argument('--historical-output',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.self_test:
        self_test();return
    if any(getattr(args,key) is None for key in ("config","output","seed","arm")):
        p.error("--config, --output, --seed and --arm are required for real compatibility")
    if args.worker_stage:
        if args.seed == 'all' or args.worker_stage == 'new' and args.historical_output is None:
            p.error('Internal worker needs one seed and new worker needs historical output')
        run_worker(args)
    else:
        run(args)


if __name__=="__main__":
    main()
