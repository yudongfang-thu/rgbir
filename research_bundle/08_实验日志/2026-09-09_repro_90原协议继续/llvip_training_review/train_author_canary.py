"""Original LLVIP train(...) canary, paper B8/1280/E200; no new training loop.

CPU preflight compares the available generic COCO initialization architecture to
an author-released LLVIP checkpoint. This does not establish historical identity.
Canary needs an existing single-GPU lease and stops after actual SGD updates.
"""
import argparse
from copy import deepcopy
import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback


class CanaryComplete(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'canary'])
    p.add_argument('--root', type=Path, default=Path('/mnt/dataX/ydf/projects/RGBT_campaign_90'))
    p.add_argument('--weights', type=Path, required=True, help='Available generic COCO80 initialization')
    p.add_argument('--reference-weights', type=Path, required=True, help='Author released LLVIP baseline, architecture check only')
    p.add_argument('--modality', choices=['visible', 'infrared'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--successful-updates', type=int, default=24)
    p.add_argument('--max-batches', type=int, default=256)
    p.add_argument('--memory-fraction', type=float, default=.68)
    a = p.parse_args()
    assert 1 <= a.successful_updates < a.max_batches < 1504
    assert 0 < a.memory_fraction <= .68
    for key in ('root', 'weights', 'reference_weights', 'output'):
        setattr(a, key, getattr(a, key).resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, a.output/'executed_wrapper.py')
    shutil.copy2(Path(__file__).with_name('loss_index_compat.py'), a.output/'loss_index_compat.py')
    start = time.perf_counter()
    state = dict(status='STARTED', identity='PAPER-RECONSTRUCTED', mode=a.mode,
                 successful_optimizer_updates=0, scaler_step_attempts=0, amp_skips=0,
                 batches=[], update_events=[], ap_computed=False, training_completed=False,
                 new_hash_computed=False, original_source_modified=False)
    def save(name, obj):
        (a.output/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str)+'\n', encoding='utf-8')
    def event(kind, **kw):
        row = dict(event=kind, elapsed_seconds=time.perf_counter()-start, **kw)
        with (a.output/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, default=str)+'\n')
        print(json.dumps(row, default=str), flush=True)
    torch = None
    resource_record = None
    handles = []
    try:
        source = a.root/'external_reproductions/llvip_author_baseline/author_source/yolov5'
        assert a.weights.is_file() and a.reference_weights.is_file()
        os.environ.update(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', PYTHONDONTWRITEBYTECODE='1',
                          OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                          WANDB_MODE='disabled', WANDB_DISABLED='true',
                          MPLCONFIGDIR=str(a.root/'cache/llvip_author_matplotlib'),
                          TORCH_HOME=str(a.root/'cache/torch'))
        if a.mode == 'preflight': os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.dont_write_bytecode = True
        sys.modules['wandb'] = None
        sys.path.insert(0, str(a.root))
        lease = None
        if a.mode == 'canary':
            from tools.project_resource_guard import require_bound_lease_from_environment, bound_lease_resource_record_from_environment
            lease = require_bound_lease_from_environment()
            resource_record = bound_lease_resource_record_from_environment
            assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
        sys.path.insert(0, str(source))
        os.chdir(source)
        import numpy as np
        np.int = int
        np.float = float
        import torch
        import yaml
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        import train as author
        from utils.callbacks import Callbacks
        import utils.datasets as datasets
        from loss_index_compat import install
        assert not torch.cuda.is_initialized()
        save('loss_cpu_preflight.json', install(author.ComputeLoss, torch))

        ckpt = torch.load(a.weights, map_location='cpu', weights_only=False)
        ref = torch.load(a.reference_weights, map_location='cpu', weights_only=False)
        initial = ckpt['model']
        reference = ref.get('ema')
        if reference is None: reference = ref['model']
        assert len(initial.names) == 80 and len(reference.names) == 1
        assert ckpt.get('epoch') == -1 and ckpt.get('optimizer') is None
        keys = ('backbone', 'head', 'depth_multiple', 'width_multiple')
        architecture = {k: initial.yaml.get(k) == reference.yaml.get(k) for k in keys}
        model = author.Model(deepcopy(reference.yaml), ch=3, nc=1)
        target = model.state_dict()
        intersection = author.intersect_dicts(initial.float().state_dict(), target, exclude=[])
        missing = [k for k in target if k not in intersection]
        identity = dict(status='CPU_ARCHITECTURE_AND_TRANSFER_CHECKED', historical_initialization_identity=False,
                        generic_path=str(a.weights), generic_bytes=a.weights.stat().st_size,
                        reference_path=str(a.reference_weights), reference_bytes=a.reference_weights.stat().st_size,
                        generic_yaml=initial.yaml, reference_yaml=reference.yaml, architecture_fields_match=architecture,
                        target_parameters=sum(x.numel() for x in model.parameters()),
                        transferred=len(intersection), target_items=len(target), unmatched_target_keys=missing,
                        generic_classes=80, cuda_initialized=torch.cuda.is_initialized())
        save('initialization_cpu.json', identity)
        assert all(architecture.values()), f'Initialization/reference architecture mismatch: {architecture}; see initialization_cpu.json'
        assert len(intersection) / len(target) >= .95, 'Unexpectedly incomplete original model transfer'
        assert not torch.cuda.is_initialized()
        del ckpt, ref, initial, reference, model, target, intersection
        gc.collect()

        argv = sys.argv
        try:
            sys.argv = [str(source/'train.py')]
            opt = author.parse_opt()
        finally:
            sys.argv = argv
        data_path = a.root/'data_author_protocol/llvip_baseline_attempt1'/a.modality/'data.yaml'
        data = yaml.safe_load(data_path.read_text())
        assert data['nc'] == 1 and data['names'] == ['person'] and 'download' not in data
        for split, count in (('train', 12025), ('val', 3463)):
            roster = [Path(x) for x in Path(data[split]).read_text().splitlines() if x]
            assert len(roster) == len({x.stem for x in roster}) == count
            assert all(x.is_file() for x in roster)
        opt.weights, opt.data = str(a.weights), str(data_path)
        opt.imgsz, opt.batch_size, opt.epochs, opt.patience = 1280, 8, 200, 0
        opt.device, opt.project, opt.name = '0', str(a.output), 'author_train'
        opt.save_dir = str(a.output/'author_train')
        hyp = yaml.safe_load(Path(opt.hyp).read_text())
        hyp.update(lr0=.0032, lrf=.12, momentum=.843, weight_decay=.00036)
        save('plan.json', dict(opt=vars(opt), hyp=hyp, lease=lease, modality=a.modality,
                              identity='PAPER-RECONSTRUCTED', paper_overrides=['lr0', 'lrf', 'momentum', 'weight_decay'],
                              patience_note='Explicitly disable current-code early stop to budget full paper E200',
                              allocator_cap=a.memory_fraction, original_train='train.train(hyp,opt,device,callbacks)',
                              target_successful_updates=a.successful_updates, max_batches=a.max_batches,
                              author_source_commit_preexisting='c1a655cce437ebfd990a97b04fc48fbb99f4c47b',
                              compatibility=['CPU-validated two integer loss bounds', 'legacy pickle', 'NumPy aliases',
                                             'no-digest dataset cache key', 'optional wandb disabled',
                                             'batch image plotting/TensorBoard JIT trace skipped; no extra diagnostic model forward'],
                              training_population='official train12025; original val loader scans official test3463',
                              official_test_authorized=True, primary_annotations='previous', ap_disabled=True,
                              steady_optimizer_regime_observed=False))
        if a.mode == 'preflight':
            state['status'] = 'CPU_PREFLIGHT_PASS'
            return 0

        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction, 0)
        # This is the same explicit cache-metadata policy as the baseline evaluator.
        def no_digest(paths):
            return ('explicit_path_size_no_digest_v1', tuple((str(x), Path(x).stat().st_size if Path(x).is_file() else None) for x in paths))
        datasets.get_hash = no_digest
        # Calling train directly avoids main() git/network/requirements actions.
        original_download = author.attempt_download
        def local_weights(path, *args, **kwargs):
            assert Path(path).resolve() == a.weights and Path(path).is_file()
            return str(a.weights)
        author.attempt_download = local_weights
        # The original logger traces a B1 FP32 model at first batch-end; that is
        # an additional diagnostic forward, not the measured training operation.
        original_batch_logging = author.Loggers.on_train_batch_end
        def batch_logging_without_graph(self, ni, model, imgs, targets, paths, plots, sync_bn):
            return original_batch_logging(self, ni, model, imgs, targets, paths, False, sync_bn)
        author.Loggers.on_train_batch_end = batch_logging_without_graph
        def no_ap(*args, **kwargs): raise RuntimeError('Canary reached forbidden epoch-end evaluation')
        author.val.run = no_ap
        original_loader = author.create_dataloader
        class TimedLoader:
            def __init__(self, loader): self.loader = loader
            def __len__(self): return len(self.loader)
            def __getattr__(self, key): return getattr(self.loader, key)
            def __iter__(self):
                it = iter(self.loader)
                while len(state['batches']) < a.max_batches:
                    tick = time.perf_counter()
                    try: batch = next(it)
                    except StopIteration: return
                    row = dict(batch=len(state['batches'])+1, input_shape=list(batch[0].shape),
                               loader_seconds=time.perf_counter()-tick,
                               source_stems=[Path(x).stem for x in batch[2]])
                    assert row['input_shape'] == [8, 3, 1280, 1280]
                    state['batches'].append(row)
                    state['batch_start'] = time.perf_counter()
                    yield batch
                raise RuntimeError('Successful update target not reached inside batch bound')
        def checked_loader(*args, **kwargs):
            loader, dataset = original_loader(*args, **kwargs)
            training = bool(kwargs.get('augment', False))
            assert len(dataset) == (12025 if training else 3463)
            event('original_loader', training=training, images=len(dataset), batches=len(loader))
            return (TimedLoader(loader) if training else loader), dataset
        author.create_dataloader = checked_loader

        original_model = author.Model
        def forward_check(module, inputs):
            if not state['batches'] or state['batches'][-1]['batch'] != 1: return
            event('actual_training_input', shape=list(inputs[0].shape), dtype=str(inputs[0].dtype),
                  autocast=torch.is_autocast_enabled('cuda'), grad=torch.is_grad_enabled())
            assert torch.is_autocast_enabled('cuda') and torch.is_grad_enabled()
        def conv_check(module, inputs, output):
            if not state['batches'] or state['batches'][-1]['batch'] != 1: return
            event('actual_first_conv', dtype=str(output.dtype), shape=list(output.shape))
            assert output.dtype == torch.float16
        def observed_model(*args, **kwargs):
            model = original_model(*args, **kwargs)
            handles.append(model.register_forward_pre_hook(forward_check))
            handles.append(next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d)).register_forward_hook(conv_check))
            return model
        author.Model = observed_model
        sgd_init = torch.optim.SGD.__init__
        def after_sgd(optimizer, args, kwargs):
            state['successful_optimizer_updates'] += 1
        def observed_sgd(self, *args, **kwargs):
            sgd_init(self, *args, **kwargs)
            handles.append(self.register_step_post_hook(after_sgd))
        torch.optim.SGD.__init__ = observed_sgd
        scaler_step = author.amp.GradScaler.step
        def observed_scaler_step(self, optimizer, *args, **kwargs):
            before = state['successful_optimizer_updates']
            state['scaler_step_attempts'] += 1
            result = scaler_step(self, optimizer, *args, **kwargs)
            skipped = before == state['successful_optimizer_updates']
            state['amp_skips'] += int(skipped)
            row = dict(batch=len(state['batches']), attempt=state['scaler_step_attempts'],
                       successful_updates=state['successful_optimizer_updates'], skipped=skipped,
                       scale_before_update=float(self.get_scale()))
            state['update_events'].append(row)
            return result
        author.amp.GradScaler.step = observed_scaler_step
        callbacks = Callbacks()
        def after_batch(*args, **kwargs):
            torch.cuda.synchronize(0)
            row = state['batches'][-1]
            row.update(training_seconds=time.perf_counter()-state['batch_start'],
                       successful_updates=state['successful_optimizer_updates'],
                       allocated_mib=torch.cuda.memory_allocated()/2**20,
                       reserved_mib=torch.cuda.memory_reserved()/2**20)
            event('completed_train_batch', **row)
            if state['successful_optimizer_updates'] >= a.successful_updates:
                raise CanaryComplete()
        callbacks.register_action('on_train_batch_end', callback=after_batch)
        event('calling_original_train')
        author.train(hyp, opt, torch.device('cuda:0'), callbacks)
        raise RuntimeError('Original trainer returned before canary target')
    except CanaryComplete:
        state['status'] = 'CANARY_PASS'
        state['stop_point'] = 'Original on_train_batch_end after final scaler.update/zero_grad/EMA'
    except Exception as exc:
        state['status'] = 'RESOURCE_FAILURE_OOM' if torch is not None and isinstance(exc, torch.OutOfMemoryError) else 'TECHNICAL_FAILURE'
        state['error'], state['traceback'] = repr(exc), traceback.format_exc()
    finally:
        for h in handles: h.remove()
        state.pop('batch_start', None)
        rows = [r for r in state['batches'][4:] if 'training_seconds' in r]
        if rows:
            total = [r['training_seconds']+r['loader_seconds'] for r in rows]
            state['post_startup_timing'] = dict(excluded_first_batches=4, measured_batches=len(rows),
                mean_training_seconds=statistics.mean(r['training_seconds'] for r in rows),
                mean_loader_seconds=statistics.mean(r['loader_seconds'] for r in rows),
                mean_batch_seconds=statistics.mean(total), median_batch_seconds=statistics.median(total),
                e200_train_only_extrapolated_hours=statistics.mean(total)*1504*200/3600,
                excludes=['epoch-end validation', 'checkpoint saving', 'startup'],
                regime='Original early warmup; not steady LR/gradient accumulation, no 10h admission by itself')
        if torch is not None:
            state['torch_version'] = torch.__version__
            state['cuda_initialized'] = torch.cuda.is_initialized()
            if torch.cuda.is_initialized():
                state['torch_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
                state['torch_peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        if resource_record:
            try: state['lease_resources'] = resource_record(wait_seconds=0)
            except Exception as exc: state['lease_resource_error'] = repr(exc)
        state['elapsed_seconds'] = time.perf_counter()-start
        state['finished'] = datetime.datetime.now().astimezone().isoformat()
        save('receipt.json' if state['status'] in ('CPU_PREFLIGHT_PASS', 'CANARY_PASS') else 'failure.json', state)
        print(json.dumps(state, default=str), flush=True)
    return 0 if state['status'] in ('CPU_PREFLIGHT_PASS', 'CANARY_PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
