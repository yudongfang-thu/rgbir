"""Fresh-process old/selected-only C1 24-update diagnostic; no production admission.

Only worker mode imports project runtime. CPU comparison uses fixed tolerances,
reports true byte equality separately, and retains mismatches without repairs.
No call to train/verify run(), evidence binding, or hash functions.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback
import types

import numpy as np
import torch

ATOL, RTOL = 1e-6, 1e-5
UPDATES, MAX_ATTEMPTS, WARMUP_BATCHES = 24, 96, 6
COEFFICIENT = 0.09227393550836771


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_bytes(t):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def compare_tree(a, b):
    """Finite values use |new-old|<=atol+rtol*|old|, calculated in float64.

    Nonfinite values never count as finite close: separately require identical
    positions and bytes for numeric_agreement. AMP skip validity is a separate
    lifecycle check; matching NaNs do not establish usable gradients.
    """
    result = dict(bitwise_exact=True, numeric_agreement=True, all_finite=True,
                  leaves=0, elements=0, nonfinite_elements=0,
                  finite_outside_tolerance=0, max_abs=0., mismatches=[])

    def bad(path, reason, numeric=True):
        result['bitwise_exact'] = False
        if numeric:
            result['numeric_agreement'] = False
        if len(result['mismatches']) < 20:
            result['mismatches'].append(dict(path=path, reason=reason))

    def floating(x, y, exact, path):
        original_x, original_y = x.reshape(-1), y.reshape(-1)
        x, y = x.double().reshape(-1), y.double().reshape(-1)
        finite_x, finite_y = torch.isfinite(x), torch.isfinite(y)
        finite = finite_x & finite_y
        nonfinite = ~finite
        result['elements'] += x.numel()
        result['nonfinite_elements'] += int(nonfinite.sum())
        result['all_finite'] &= not bool(nonfinite.any())
        if bool(nonfinite.any()):
            # Original dtypes/NaN payloads are checked by the caller's bytes
            # when all values are nonfinite or via the original masked subset.
            same_nonfinite = bool(torch.equal(finite_x, finite_y)) and tensor_bytes(original_x[nonfinite]) == tensor_bytes(original_y[nonfinite])
            if not same_nonfinite:
                bad(path, 'nonfinite pattern/value differs')
        if bool(finite.any()):
            delta = (y[finite]-x[finite]).abs()
            outside = delta > ATOL + RTOL*x[finite].abs()
            count = int(outside.sum())
            result['finite_outside_tolerance'] += count
            result['max_abs'] = max(result['max_abs'], float(delta.max()))
            if count:
                bad(path, 'finite values outside fixed tolerance')
        if not exact:
            bad(path, 'bytes differ', numeric=False)

    def visit(x, y, path):
        if isinstance(x, torch.Tensor):
            result['leaves'] += 1
            if not isinstance(y, torch.Tensor) or x.dtype != y.dtype or x.shape != y.shape or x.layout != y.layout:
                bad(path, 'tensor type/dtype/shape/layout differs'); return
            if x.layout != torch.strided:
                raise ValueError('Explicit sparse tensor policy required')
            exact = tensor_bytes(x) == tensor_bytes(y)
            if x.is_floating_point():
                floating(x, y, exact, path)
            else:
                result['elements'] += x.numel()
                if not exact: bad(path, 'integer/bool/complex tensor bytes differ')
        elif isinstance(x, np.ndarray):
            if not isinstance(y, np.ndarray) or x.dtype != y.dtype or x.shape != y.shape:
                bad(path, 'numpy type/dtype/shape differs'); return
            visit(torch.from_numpy(x), torch.from_numpy(y), path)
        elif isinstance(x, dict):
            if not isinstance(y, dict): bad(path, 'mapping type differs'); return
            if set(x) != set(y): bad(path, 'mapping keys differ')
            for key in x:
                if key in y: visit(x[key], y[key], path+'.'+str(key))
        elif isinstance(x, (list, tuple)):
            if type(x) is not type(y) or len(x) != len(y):
                bad(path, 'sequence type/length differs'); return
            for i, (xx, yy) in enumerate(zip(x, y)): visit(xx, yy, path+'[%d]' % i)
        elif type(x) is not type(y):
            bad(path, 'scalar type differs')
        elif isinstance(x, float):
            result['leaves'] += 1
            floating(torch.tensor([x], dtype=torch.float64), torch.tensor([y], dtype=torch.float64),
                     struct.pack('d', x) == struct.pack('d', y), path)
        else:
            result['leaves'] += 1
            if x != y: bad(path, 'scalar value differs')

    visit(a, b, 'root')
    return result


def require_exact(a, b, name):
    result = compare_tree(a, b)
    if not result['bitwise_exact']:
        raise AssertionError(name+': '+str(result['mismatches']))


def file_stat(path):
    path = Path(path)
    info = path.stat()
    return dict(path=str(path.resolve()), bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def explicit_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clone_function(fn, **bindings):
    namespace = dict(fn.__globals__, **bindings)
    copied = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    copied.__kwdefaults__ = fn.__kwdefaults__
    return copied


def selection_snapshot(s, helper):
    identity_keys = ('valid_levels', 'selected', 'labels', 'base_to_matched', 'matched_to_base',
                     'selected_matched_indices', 'matched_object_ids', 'base_object_ids', 'eligible')
    identity = {k: getattr(s, k) for k in identity_keys}
    identity['regions'] = [vars(r) for r in s.regions]
    floating = {k: getattr(s, k) for k in ('quality', 'c0_loss', 'c0_stats')}
    floating.update(selected_student_delta=s.student_delta[s.selected],
                    selected_teacher_delta=s.teacher_delta[s.selected])
    return dict(identity=helper.clone_cpu(identity), floating=helper.clone_cpu(floating),
        delta_collection_scope='selected_S_T_only',
        uncollected_learning_delta_fields=['unselected_student_delta','unselected_teacher_delta','reference_delta'])


class Observer:
    """Uses release helper primitives, with C1-specific local binding and scope."""
    def __init__(self, helper, trainer, output, criterion_type):
        self.h, self.t, self.out, self.criterion_type = helper, trainer, output, criterion_type
        self.batches = self.attempts = self.optimizer_calls = 0
        self.audit_seconds = 0.
        self.applied = None
        self.timings = []
        self.selected_calls = 0

    def audit(self, fn):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        self.audit_seconds += time.perf_counter()-started
        return result

    def save(self, name, factory):
        def persist():
            path = self.out/name
            if path.exists(): raise FileExistsError(path)
            torch.save(factory(), path)
        self.audit(persist)

    def install(self):
        trainer, helper = self.t, self.h
        builder, preprocess = trainer.build_dataset, trainer.preprocess_batch
        @functools.wraps(builder)
        def observed_dataset(*args, **kwargs):
            ds = builder(*args, **kwargs)
            mode = kwargs.get('mode', args[1] if len(args)>1 else 'train')
            return helper.RNGObservedDataset(ds) if mode == 'train' else ds
        trainer.build_dataset = observed_dataset
        @functools.wraps(preprocess)
        def observed_preprocess(batch):
            self.batches += 1
            if self.batches > MAX_ATTEMPTS*4:
                raise RuntimeError('Bounded batch safety cap exceeded')
            if int(batch['img'].shape[0]) != 32 or helper.RNG_FIELD not in batch:
                raise AssertionError('Full B32/observed worker flow required')
            self.save('batch_%04d.pt' % self.batches,
                lambda: dict(metadata=helper.clone_cpu({k:v for k,v in batch.items() if k not in ('img','strong_img')}),
                             parent_rng=helper.all_rng_state(), image_shapes={k:tuple(batch[k].shape) for k in ('img','strong_img')}))
            return preprocess(batch)
        trainer.preprocess_batch = observed_preprocess
        trainer.add_callback('on_train_start', self.on_start)
        trainer.add_callback('on_train_batch_start', self.batch_start)
        trainer.add_callback('on_train_batch_end', self.batch_end)

    def batch_start(self, trainer):
        torch.cuda.synchronize()
        self.batch_started, self.batch_audit_started = time.perf_counter(), self.audit_seconds
        self.batch_diagnostics_started = getattr(trainer.criterion_ref,'selected_only_diagnostics_seconds_total',0.)

    def batch_end(self, trainer):
        torch.cuda.synchronize()
        span = time.perf_counter()-self.batch_started
        audit = self.audit_seconds-self.batch_audit_started
        diagnostics = getattr(trainer.criterion_ref,'selected_only_diagnostics_seconds_total',0.)-self.batch_diagnostics_started
        row = dict(batch=self.batches, wall_seconds=span, audit_seconds=audit,
                   wall_minus_audit_seconds=span-audit, warmup=self.batches <= WARMUP_BATCHES,
                   extra_full_diagnostics_seconds=diagnostics,
                   wall_minus_audit_and_extra_diagnostics_seconds=span-audit-diagnostics,
                   successful_updates=trainer.real_updates, amp_skips=trainer.skipped_amp_updates)
        self.timings.append(row)
        self.save('loss_%04d.pt' % self.batches,lambda:dict(
            actual_B=trainer.criterion_ref.last_stats['actual_B'],
            coefficient=trainer.criterion_ref.last_stats['coefficient'],
            values={k:trainer.criterion_ref.last_stats[k] for k in ('native_total','loss_unweighted',
                'weighted_kd_total','total_loss','target_loss_unweighted','off_target_loss_unweighted')}))

    def on_start(self, trainer):
        import runtime
        h = self.h
        if type(trainer.criterion_ref) is not self.criterion_type:
            raise AssertionError('Private diagnostic criterion binding failed')
        if not isinstance(trainer.train_loader.dataset, h.RNGObservedDataset) or type(trainer.train_loader.dataset.inner) is not runtime.TrackedDualLabelRGBIRDataset:
            raise AssertionError('Actual tracked dual-label training loader required')
        if trainer.train_loader.num_workers != 4 or not bool(trainer.amp):
            raise AssertionError('B32/workers4/AMP formal recipe changed')
        h.assert_auxiliaries(trainer)
        self.aux_initial = self.audit(lambda: {k:h.model_state(getattr(trainer.criterion_ref,k)) for k in ('teacher','reference')})
        self.save('initial.pt', lambda: dict(student=h.model_state(trainer.model), ema=h.model_state(trainer.ema.ema),
            optimizer=h.clone_cpu(trainer.optimizer.state_dict()), scaler=h.clone_cpu(trainer.scaler.state_dict()),
            rng=h.all_rng_state(), auxiliaries=self.aux_initial,
            trainable_names=[n for n,p in trainer.model.named_parameters() if p.requires_grad],
            updates=trainer.real_updates, attempts=trainer.update_attempts, amp_skips=trainer.skipped_amp_updates,
            ema_updates=trainer.ema.updates, amp=bool(trainer.amp), batch=int(trainer.batch_size), workers=trainer.train_loader.num_workers))
        optimizer_call, original_step = trainer.optimizer.step, trainer.optimizer_step
        @functools.wraps(optimizer_call)
        def observed_optimizer(*args, **kwargs):
            self.optimizer_calls += 1
            def capture_applied():
                values = h.gradients(trainer.model)
                if h.finite_gradient_count(values)['nonfinite_tensors']:
                    raise FloatingPointError('Nonfinite gradients reached optimizer.step')
                return values
            self.applied = self.audit(capture_applied)
            return optimizer_call(*args, **kwargs)
        trainer.optimizer.step = observed_optimizer
        @functools.wraps(original_step)
        def observed_step():
            self.attempts += 1
            if self.attempts > MAX_ATTEMPTS: raise RuntimeError('24 successes not reached in 96 attempts')
            self.applied = None
            calls, updates = self.optimizer_calls, trainer.real_updates
            before = self.audit(lambda: dict(gradients_scaled=h.gradients(trainer.model),
                losses=dict(total=h.clone_cpu(trainer.loss), native_items=h.clone_cpu(trainer.loss_items)),
                control=dict(scaler=h.clone_cpu(trainer.scaler.state_dict()), rng=h.all_rng_state(),
                    accumulate=int(trainer.accumulate), epoch=int(trainer.epoch), batches=self.batches,
                    updates=trainer.real_updates, attempts=trainer.update_attempts, skips=trainer.skipped_amp_updates)))
            original_step()
            delta = self.optimizer_calls-calls
            if delta not in (0,1) or trainer.real_updates-updates != delta or trainer.update_attempts != self.attempts:
                raise AssertionError('Observed optimizer/scaler counters disagree')
            self.audit(lambda: h.assert_auxiliaries(trainer))
            self.save('attempt_%03d.pt' % self.attempts, lambda: dict(before=before,
                after=dict(student=h.model_state(trainer.model), optimizer=h.clone_cpu(trainer.optimizer.state_dict()),
                    ema=h.model_state(trainer.ema.ema), applied_gradients=self.applied,
                    control=dict(ema_updates=trainer.ema.updates, scaler=h.clone_cpu(trainer.scaler.state_dict()),
                        rng=h.all_rng_state(), updates=trainer.real_updates, attempts=trainer.update_attempts,
                        amp_skips=trainer.skipped_amp_updates, actual_optimizer_calls=self.optimizer_calls))))
            if trainer.real_updates > UPDATES: raise AssertionError('24-update budget exceeded')
        trainer.optimizer_step = observed_step

    def record_selection(self, selection):
        self.selected_calls += 1
        if self.selected_calls != self.batches: raise AssertionError('Not one complete-batch C1 selection per batch')
        self.save('selection_%04d.pt' % self.batches, lambda: selection_snapshot(selection,self.h))

    def finish(self):
        t, h = self.t, self.h
        if t.real_updates != UPDATES or self.optimizer_calls != UPDATES or self.attempts > MAX_ATTEMPTS:
            raise AssertionError('Exactly 24 real optimizer updates required')
        current_aux = self.audit(lambda: {k:h.model_state(getattr(t.criterion_ref,k)) for k in ('teacher','reference')})
        self.audit(lambda: require_exact(self.aux_initial, current_aux, 'frozen auxiliary states'))
        checks = t.criterion_ref.gradient_checks
        if t.criterion_ref.selected_total <= 0 or not any(math.isfinite(r.get('kd_score_gradient_l2',0)) and r.get('kd_score_gradient_l2',0)>0 for r in checks):
            raise AssertionError('C1 did not exercise finite nonzero KD score gradients')
        result = dict(updates=t.real_updates, attempts=t.update_attempts, amp_skips=t.skipped_amp_updates,
            ema_updates=t.ema.updates, batches=self.batches, optimizer_calls=self.optimizer_calls,
            selected_calls=self.selected_calls, selected_objects=t.criterion_ref.selected_total,
            frozen_auxiliaries_unchanged=True)
        self.save('final.pt',lambda:dict(control=result, gradient_checks=h.clone_cpu(checks)))
        return result


def validate_cfg(cfg):
    for key, value in dict(arm='C1', source='paired', seed=42, epochs=200, imgsz=640, batch=32, nbs=64,
                           workers=4, amp=True, classification_coefficient=COEFFICIENT, localization_coefficient=0.).items():
        if cfg.get(key) != value: raise ValueError('Frozen C1 config mismatch: '+key)
        if type(value) in (bool,int) and type(cfg.get(key)) is not type(value):
            raise ValueError('Frozen config scalar type mismatch: '+key)
    if Path(cfg['model']).name != 'yolo11n.pt': raise ValueError('Original yolo11n initialization required')


def require_output(path):
    path = path.resolve()
    if os.name != 'posix' or not str(path).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('GPU diagnostic output must be on the project data disk')
    return path


def snapshot(args, out, cfg):
    sources = [Path(__file__).resolve(), args.candidate_source.resolve()]
    sources += sorted(p for p in args.reference_dir.rglob('*.py') if '__pycache__' not in p.parts)
    target = out/'sources'; target.mkdir()
    rows = []
    for index, source in enumerate(sources):
        dest = target/('%04d_' % index+source.name)
        shutil.copyfile(source,dest)
        if source.read_bytes() != dest.read_bytes(): raise AssertionError('Source snapshot differs')
        rows.append(dict(file_stat(source), copy=str(dest), byte_identity=True))
    shutil.copyfile(args.config, target/'input_config.yaml')
    write_json(out/'source_manifest.json',dict(new_hash_computed=False, files=rows,
        config=file_stat(args.config), models={k:file_stat(cfg[k]) for k in ('model','teacher','reference')}))


def run_worker(args):
    import yaml
    ref = args.reference_dir.resolve()
    sys.path.insert(0,str(ref))
    import runtime
    import train_independent as training
    import independent_criterion as criterion_module
    import selection_adapter as selection_module
    import classification_logit as classification_module
    for module, name in [(runtime,'runtime.py'),(training,'train_independent.py'),
                         (criterion_module,'independent_criterion.py'),(selection_module,'selection_adapter.py'),
                         (classification_module,'classification_logit.py')]:
        if Path(module.__file__).resolve() != ref/name: raise RuntimeError('Wrong release module: '+name)
    helper = explicit_module('_update24_release_helpers',ref/'verify_compatibility.py')
    candidate = explicit_module('_update24_selected_only_candidate',args.candidate_source.resolve())
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    validate_cfg(cfg); training.validate_execution(cfg,formal=False)
    lease = runtime.legacy.require_bound_lease_from_environment()
    if len(lease['gpus']) != 1: raise RuntimeError('One inherited global lease required')
    out = require_output(args.output); out.mkdir(parents=True,exist_ok=False)
    snapshot(args,out,cfg)
    before_inputs = {k:file_stat(cfg[k]) for k in ('model','teacher','reference')}
    torch.set_num_threads(4)
    helper.seed_process(cfg['seed'])
    holder = {}
    if args.worker == 'old':
        def observed_selector(*a, **kw):
            selected = selection_module.build_classification_selection(*a,**kw)
            holder['observer'].record_selection(selected)
            return selected
        private_call = clone_function(criterion_module.IndependentCriterion.__call__, build_classification_selection=observed_selector)
        private_criterion = type('DiagnosticC1Criterion',(criterion_module.IndependentCriterion,),{'__call__':private_call})
    else:
        api = candidate.make_api(selection_module,classification_module)
        private_criterion = candidate.make_criterion_type(criterion_module,api,
            selection_observer=lambda criterion,payload:holder['observer'].record_selection(payload))
    build = clone_function(training.build_trainer, IndependentCriterion=private_criterion)
    trainer = observer = None
    started = time.perf_counter()
    try:
        trainer = build(cfg,args.config,out,arm='C1',max_steps=UPDATES,historical=False)
        observer = Observer(helper,trainer,out,private_criterion)
        holder['observer'] = observer
        observer.install()
        torch.cuda.reset_peak_memory_stats()
        trainer.train()
        training_span = time.perf_counter()-started
        training_audit = observer.audit_seconds
        result = observer.finish()
        criterion = trainer.criterion_ref
        coverage = dict(thin_learning_batches=getattr(criterion,'thin_learning_batches',0),
            full_diagnostics_batches=getattr(criterion,'full_diagnostics_batches',observer.batches if args.worker=='old' else 0),
            fallback_batches=getattr(criterion,'fallback_batches',0),
            extra_full_diagnostics_seconds=getattr(criterion,'selected_only_diagnostics_seconds_total',0.))
        if args.worker=='selected_only' and not (
                coverage['thin_learning_batches']==observer.batches and coverage['fallback_batches']==0
                and coverage['full_diagnostics_batches']==observer.batches):
            raise AssertionError('Every actual candidate batch must exercise thin learning with original sanity full diagnostics; fallback is not this diagnostic')
        require_exact(before_inputs,{k:file_stat(cfg[k]) for k in before_inputs},'input model stat')
        warm = [r['wall_minus_audit_seconds'] for r in observer.timings if not r['warmup']]
        warm_thin = [r['wall_minus_audit_and_extra_diagnostics_seconds'] for r in observer.timings if not r['warmup']]
        receipt = dict(status='COMPLETED_24_UPDATE_DIAGNOSTIC',worker=args.worker,pid=os.getpid(),parent_pid=os.getppid(),
            configuration=cfg,result=result,source_manifest=str(out/'source_manifest.json'),new_hash_computed=False,
            candidate='selected_only_v1_original_pool',candidate_coverage=coverage,
            long_training_switch_admitted=False,official_test_accessed=False,fresh_interpreter=True,
            timing=dict(training_span_seconds=training_span,training_audit_copy_save_check_seconds=training_audit,
                training_span_minus_audit_seconds=training_span-training_audit,
                all_audit_including_final_check_seconds=observer.audit_seconds,
                timing_scope='Synchronized observed batch spans exclude loader fetch. Audit overhead removed; instrumentation still changes overlap. Not production throughput.',
                warmup_batches=WARMUP_BATCHES,post_warmup_batches=len(warm),
                post_warmup_batch_seconds=warm,batches=observer.timings),
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            resources=runtime.legacy.bound_lease_resource_record_from_environment())
        receipt['timing'].update(extra_full_diagnostics_seconds=coverage['extra_full_diagnostics_seconds'],
            post_warmup_minus_audit_and_extra_diagnostics_seconds=warm_thin,
            production_throughput_claim=False,
            diagnostic_timing_caveat='Both full observed wall and audit-subtracted wall retained. Detached full diagnostics are additionally timed on the candidate; subtraction changes no learning graph and is not uninstrumented epoch throughput.')
        write_json(out/'worker_receipt.json',receipt)
    except BaseException as error:
        write_json(out/'failure_receipt.json',dict(status='FAILED_DIAGNOSTIC',worker=args.worker,error=repr(error),
            traceback=traceback.format_exc(),new_hash_computed=False,long_training_switch_admitted=False))
        raise
    finally:
        if trainer is not None:
            helper.shutdown_loader(getattr(trainer,'train_loader',None));helper.shutdown_loader(getattr(trainer,'test_loader',None))
        runtime.legacy.EvidenceCriterion = runtime.ORIGINAL_CRITERION
        runtime.legacy.DualLabelRGBIRDataset = runtime.ORIGINAL_DATASET


def compare_runs(old, new, output):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    receipts = [json.loads((p/'worker_receipt.json').read_text(encoding='utf-8')) for p in (old,new)]
    for worker, receipt in zip(('old','selected_only'),receipts):
        if receipt['status'] != 'COMPLETED_24_UPDATE_DIAGNOSTIC' or receipt['worker'] != worker or receipt['result']['updates'] != UPDATES or receipt['new_hash_computed'] is not False:
            raise AssertionError('Complete, no-hash, 24-update old/new receipts required')
        r = receipt['result']
        if not (UPDATES <= r['attempts'] <= MAX_ATTEMPTS and r['optimizer_calls'] == UPDATES
                and r['amp_skips'] == r['attempts']-UPDATES and 0 < r['batches'] <= MAX_ATTEMPTS*4
                and r['selected_calls'] == r['batches'] and r['frozen_auxiliaries_unchanged'] is True):
            raise AssertionError('Incomplete or contradictory trajectory counters')
    coverage = receipts[1]['candidate_coverage']
    if not (coverage['thin_learning_batches']==receipts[1]['result']['batches']
            and coverage['full_diagnostics_batches']==receipts[1]['result']['batches']
            and coverage['fallback_batches']==0):
        raise AssertionError('Candidate receipt did not exercise every batch through thin learning')
    require_exact(receipts[0]['configuration'],receipts[1]['configuration'],'worker configurations')
    rows = []
    def compare(name, category, a, b):
        rows.append(dict(file=name,category=category,**compare_tree(a,b)))
    files_old = {p.name for p in old.glob('*.pt') if p.name.startswith(('initial.','batch_','selection_','loss_','attempt_','final.','first_batch.'))}
    files_new = {p.name for p in new.glob('*.pt') if p.name.startswith(('initial.','batch_','selection_','loss_','attempt_','final.','first_batch.'))}
    for files, receipt in zip((files_old,files_new),receipts):
        r = receipt['result']
        expected = {'initial.pt','final.pt','first_batch.pt'}
        expected.update('batch_%04d.pt' % i for i in range(1,r['batches']+1))
        expected.update('selection_%04d.pt' % i for i in range(1,r['batches']+1))
        expected.update('loss_%04d.pt' % i for i in range(1,r['batches']+1))
        expected.update('attempt_%03d.pt' % i for i in range(1,r['attempts']+1))
        if files != expected:raise AssertionError('State inventory does not close against receipt counters')
    manifests = [json.loads((p/'source_manifest.json').read_text(encoding='utf-8')) for p in (old,new)]
    source_exact = manifests[0]['new_hash_computed'] is False and manifests[1]['new_hash_computed'] is False
    source_exact &= len(manifests[0]['files']) == len(manifests[1]['files']) > 0
    for a,b in zip(manifests[0]['files'],manifests[1]['files']):
        source_exact &= a['path'] == b['path'] and a['byte_identity'] is True and b['byte_identity'] is True
        # Resolve snapshot filenames against the supplied directories so that
        # a later copied CPU-only audit does not depend on remote absolute paths.
        pa,pb = old/'sources'/Path(a['copy']).name,new/'sources'/Path(b['copy']).name
        source_exact &= pa.read_bytes() == pb.read_bytes()
    source_exact &= (old/'sources'/'input_config.yaml').read_bytes() == (new/'sources'/'input_config.yaml').read_bytes()
    inputs_exact = compare_tree(manifests[0]['models'],manifests[1]['models'])['bitwise_exact']
    inventory_equal = files_old == files_new
    for name in sorted(files_old & files_new):
        a,b = load_pt(old/name),load_pt(new/name)
        if name.startswith('attempt_'):
            for part in ('before','after'):
                for key in a[part]: compare(name, key,a[part][key],b[part][key])
        elif name.startswith('selection_'):
            for item in (a,b):
                if item.get('delta_collection_scope')!='selected_S_T_only':
                    raise AssertionError('Selection snapshot includes unsupported delta comparison scope')
            for key in ('identity','floating'): compare(name,'selection_'+key,a[key],b[key])
        elif name.startswith('loss_'):
            compare(name,'per_batch_losses',a,b)
        elif name == 'final.pt':
            for key in a: compare(name,'final_'+key,a[key],b[key])
        else:
            category = 'initial' if name=='initial.pt' else 'first_batch_pixels' if name=='first_batch.pt' else 'flow_worker_rng_gt'
            compare(name,category,a,b)
    categories = {}
    for row in rows:
        cat = categories.setdefault(row['category'],dict(records=0,bitwise_exact=True,numeric_agreement=True,
            all_finite=True,max_abs=0.,finite_outside_tolerance=0,nonfinite_elements=0))
        cat['records'] += 1
        for k in ('bitwise_exact','numeric_agreement','all_finite'):cat[k] &= row[k]
        cat['max_abs'] = max(cat['max_abs'],row['max_abs'])
        for k in ('finite_outside_tolerance','nonfinite_elements'):cat[k] += row[k]
    strict = ('initial','first_batch_pixels','flow_worker_rng_gt','selection_identity','control','final_control')
    trajectory = ('per_batch_losses','losses','gradients_scaled','student','optimizer','ema','applied_gradients')
    contracts_exact = source_exact and inputs_exact and inventory_equal and all(categories.get(k,{}).get('bitwise_exact',False) for k in strict)
    trajectory_exact = contracts_exact and all(categories.get(k,{}).get('bitwise_exact',False) for k in trajectory)
    trajectory_numeric = contracts_exact and all(categories.get(k,{}).get('numeric_agreement',False) for k in trajectory)
    result = dict(status='COMPLETED_COMPARISON_NOT_PRODUCTION_ADMISSION',atol=ATOL,rtol=RTOL,
        tolerance_rule='abs(new-old)<=atol+rtol*abs(old), float64 comparison; matching nonfinite bits reported separately from finiteness',
        file_inventory_exact=inventory_equal,only_old=sorted(files_old-files_new),only_new=sorted(files_new-files_old),
        source_snapshots_byte_exact=source_exact,input_model_stats_exact=inputs_exact,
        candidate_coverage=coverage,
        initial_flow_selection_and_control_exact=contracts_exact,trajectory_bitwise_exact=trajectory_exact,
        trajectory_numeric_agreement=trajectory_numeric,categories=categories,records=rows,
        mathematical_definition_claim='Same original C0 selection/base and selected per-object masked-LME expression is a source-level claim, not proved by a trajectory test.',
        compared_delta_scope='selected_S_T_only',
        uncollected_learning_delta_fields=['unselected_student_delta','unselected_teacher_delta','reference_delta'],
        intermediate_pool_precision_separate=True,long_training_switch_admitted=False,new_hash_computed=False,
        comparison_cpu_seconds=time.perf_counter()-started,
        limits=['24 updates only, not E200 identity','Metadata/source/GT/worker RNG compared every batch; only native first_batch.pt contains pixels',
                'Frozen T/R initial parameters and within-run final states exact; complete raw T/R forward tensors are not serialized',
                'Per-batch synchronized timing excludes loader fetch and does not estimate uninstrumented whole-epoch throughput'])
    write_json(output/'comparison.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--candidate-source',type=Path,default=Path(__file__).resolve().parents[1]/'selected_only_v1.py')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',choices=('old','selected_only'),help=argparse.SUPPRESS)
    parser.add_argument('--compare-only',nargs=2,type=Path,metavar=('OLD','NEW'))
    args = parser.parse_args()
    if args.compare_only:
        result=compare_runs(*args.compare_only,args.output)
        print(json.dumps({k:result[k] for k in ('status','trajectory_bitwise_exact','trajectory_numeric_agreement')}));return
    if args.reference_dir is None or args.config is None:parser.error('--reference-dir and --config required')
    if args.worker:run_worker(args);return
    if torch.cuda.is_initialized():raise RuntimeError('Coordinator must remain CUDA-free')
    out=require_output(args.output);out.mkdir(parents=True,exist_ok=False)
    for worker in ('old','selected_only'):
        command=[sys.executable,str(Path(__file__).resolve()),'--reference-dir',str(args.reference_dir.resolve()),
            '--config',str(args.config.resolve()),'--candidate-source',str(args.candidate_source.resolve()),
            '--output',str(out/worker),'--worker',worker]
        subprocess.run(command,check=True)
    if torch.cuda.is_initialized():raise RuntimeError('Coordinator initialized CUDA')
    result=compare_runs(out/'old',out/'selected_only',out/'comparison')
    write_json(out/'receipt.json',dict(status=result['status'],coordinator_cuda_initialized=False,
        fresh_sequential_workers=True,new_hash_computed=False,long_training_switch_admitted=False,
        trajectory_bitwise_exact=result['trajectory_bitwise_exact'],trajectory_numeric_agreement=result['trajectory_numeric_agreement']))


if __name__=='__main__':main()
