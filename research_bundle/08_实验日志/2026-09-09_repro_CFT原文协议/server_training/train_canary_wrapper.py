"""Two successful original CFT optimizer steps; no AP or reusable checkpoint.

Run only through the existing project resource lease. No installs or downloads.
The original train_rgb_ir, model, loss, optimizer, AMP, augmentation and loader
remain in use. This is an early-warmup technical canary, not a speed benchmark.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace


class CanaryComplete(Exception):
    pass


def make_opt(root, source, weights, output):
    return SimpleNamespace(
        weights=str(weights), cfg=str(source/'models/transformer/yolov5l_fusion_transformerx3_llvip.yaml'),
        data=str(root/'data_author_protocol/llvip_attempt1/previous/LLVIP.yaml'),
        hyp=str(source/'data/hyp.scratch.yaml'), epochs=200, batch_size=32,
        total_batch_size=32, img_size=[1024, 1024], rect=False, resume=False,
        nosave=False, notest=False, noautoanchor=False, evolve=False, bucket='',
        cache_images=False, image_weights=False, device='0', multi_scale=False,
        single_cls=False, adam=False, sync_bn=False, local_rank=-1, workers=8,
        project=str(output), entity=None, name='author_train_canary', exist_ok=False,
        quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False,
        bbox_interval=-1, save_period=-1, artifact_alias='latest', world_size=1,
        global_rank=-1, save_dir=str(output/'author_train'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/dataX/ydf/projects/RGBT_campaign_90'))
    parser.add_argument('--weights', type=Path, required=True, help='Locally downloaded author generic COCO yolov5l.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-fraction', type=float, default=0.65, help='Explicit CUDA allocator cap, not batch adaptation')
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.weights = args.weights.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = dict(status='STARTED', successful_optimizer_steps=0, batches_entered=0,
                 steps=[], batches=[], source_unchanged=True, ap_computed=False,
                 training_completed=False, new_hash_computed=False)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    def event(kind, **details):
        with (args.output/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=kind, elapsed=time.perf_counter()-started, **details))+'\n')
        print(kind, json.dumps(details), flush=True)
    torch = None
    guard_record = None
    hook_handles = []
    try:
        root = args.root.resolve()
        source = root/'external_reproductions/cft/author_source'
        assert args.weights.is_file() and args.weights.suffix == '.pt'
        assert 0 < args.memory_fraction <= 1
        os.environ.update(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', PYTHONDONTWRITEBYTECODE='1',
                          OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                          WANDB_MODE='disabled', WANDB_DISABLED='true',
                          MPLCONFIGDIR=str(root/'cache/cft_matplotlib'), TORCH_HOME=str(root/'cache/torch'))
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        from tools.project_resource_guard import require_bound_lease_from_environment, bound_lease_resource_record_from_environment
        lease = require_bound_lease_from_environment()
        guard_record = bound_lease_resource_record_from_environment
        assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1, 'This canary tests original total B32 on one leased GPU'
        sys.path.insert(0, str(source))
        os.chdir(source)
        import numpy as np
        np.int = int
        np.float = float
        import torch
        import yaml
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
        # Use the original optional-logging branch; no W&B initialization/network.
        sys.modules['wandb'] = None
        import train as author_train
        import global_var
        global_var._init()
        global_var.set_value('flag_visual_training_dataset', False)
        author_train.set_logging(-1)
        opt = make_opt(root, source, args.weights.resolve(), args.output.resolve())
        for path in (opt.cfg, opt.data, opt.hyp):
            assert Path(path).is_file(), path
        hyp = yaml.safe_load(Path(opt.hyp).read_text())
        data = yaml.safe_load(Path(opt.data).read_text())
        assert data['nc'] == 1 and data['names'] == ['person']
        assert 'download' not in data, 'Local-only canary cannot execute a dataset download directive'
        for split, expected in (('train', 12025), ('val', 3463)):
            paths = [[Path(x) for x in Path(data[split+'_'+mod]).read_text().splitlines() if x] for mod in ('rgb', 'ir')]
            assert len(paths[0]) == len(paths[1]) == expected
            assert [x.stem for x in paths[0]] == [x.stem for x in paths[1]]
            assert all(x.is_file() for modal in paths for x in modal)
        plan = dict(opt=vars(opt).copy(), hyp=hyp.copy(), lease=lease, original_train_function='train.train_rgb_ir',
                    stop_after_successful_optimizer_steps=2, max_training_batches=16,
                    input_population='complete official train12025; original val loader scans official test3463',
                    official_test_authorized_for_author_reproduction=True, primary_annotations='previous',
                    initialization='author generic COCO yolov5l.pt',
                    initialization_author_url='https://drive.google.com/file/d/12OFGLF73CqTgOCMJAycZ8lB4eW19D0nb/view',
                    allocator_memory_fraction=args.memory_fraction, torch_version=torch.__version__,
                    logging_only_changes=['wandb optional branch disabled', 'tb_writer=None', 'label-distribution plotting skipped'],
                    formula_changes=[], runtime_aliases=['np.int=int', 'np.float=float', 'legacy torch.load allowed'],
                    outcome_use='resource and execution feasibility only; no AP selection; no 10-hour guarantee',
                    started=datetime.datetime.now().astimezone().isoformat())
        save('plan.json', plan)

        original_load = torch.load
        def inspected_load(path, *pos, **kw):
            payload = original_load(path, *pos, **kw)
            if isinstance(path, (str, Path)) and Path(path).resolve() == args.weights.resolve():
                model = payload.get('model')
                assert model is not None and len(model.names) == 80, 'Expected generic COCO80 initialization, not LLVIP-trained weights'
                assert payload.get('optimizer') is None and payload.get('epoch') == -1, 'Unexpected resume/trained checkpoint; do not change author resume behavior'
                if 'initialization' not in state:
                    state['initialization'] = dict(path=str(args.weights.resolve()), bytes=args.weights.stat().st_size,
                                                   classes=len(model.names), epoch=payload.get('epoch'), optimizer_is_none=True)
                    save('initialization.json', state['initialization'])
            return payload
        torch.load = inspected_load
        original_intersect = author_train.intersect_dicts
        def recorded_intersect(a, b, exclude=()):
            result = original_intersect(a, b, exclude=exclude)
            save('initialization_transfer.json', dict(transferred=len(result), target_items=len(b),
                                                     excluded=list(exclude), transferred_keys=list(result)))
            return result
        author_train.intersect_dicts = recorded_intersect
        author_train.plot_labels = lambda *a, **kw: event('diagnostic_label_plot_skipped')
        # Protect the canary from accidentally reaching epoch-end AP after a logic error.
        def no_evaluation(*a, **kw):
            raise RuntimeError('Canary unexpectedly reached evaluation before successful two-step stop')
        author_train.test.test = no_evaluation

        original_loader = author_train.create_dataloader_rgb_ir
        class TimedTrainLoader:
            def __init__(self, loader): self.loader = loader
            def __len__(self): return len(self.loader)
            def __getattr__(self, name): return getattr(self.loader, name)
            def __iter__(self):
                iterator = iter(self.loader)
                while True:
                    if state['batches_entered'] >= 16:
                        raise RuntimeError('No two successful AMP optimizer steps within 16 batches')
                    t = time.perf_counter()
                    try: batch = next(iterator)
                    except StopIteration: return
                    state['batches_entered'] += 1
                    row = dict(index=state['batches_entered'], input_shape=list(batch[0].shape),
                               loader_seconds=time.perf_counter()-t,
                               source_stems=[Path(p).stem for p in batch[2]])
                    state['batches'].append(row)
                    state['batch_started'] = time.perf_counter()
                    event('train_batch', **row)
                    yield batch
                    torch.cuda.synchronize()
                    row['training_seconds'] = time.perf_counter()-state['batch_started']
        def checked_loader(*pos, **kw):
            loader, dataset = original_loader(*pos, **kw)
            is_train = bool(kw.get('augment', False))
            expected = 12025 if is_train else 3463
            assert len(dataset) == expected
            assert [Path(p).stem for p in dataset.img_files_rgb] == [Path(p).stem for p in dataset.img_files_ir]
            event('original_loader_ready', split='train' if is_train else 'official_test', images=len(dataset), batches=len(loader))
            return (TimedTrainLoader(loader) if is_train else loader), dataset
        author_train.create_dataloader_rgb_ir = checked_loader

        original_sgd_init = torch.optim.SGD.__init__
        def after_step(optimizer, step_args, step_kwargs):
            torch.cuda.synchronize()
            state['successful_optimizer_steps'] += 1
            row = dict(step=state['successful_optimizer_steps'], batch=state['batches_entered'],
                       since_batch_started_seconds=time.perf_counter()-state['batch_started'],
                       lr=[float(group['lr']) for group in optimizer.param_groups],
                       allocated_mib=torch.cuda.memory_allocated()/2**20,
                       reserved_mib=torch.cuda.memory_reserved()/2**20)
            state['steps'].append(row)
            event('successful_optimizer_step', **row)
            if state['successful_optimizer_steps'] == 2:
                raise CanaryComplete('Original SGD.step returned successfully twice; AMP-skipped steps are not counted')
        def observed_sgd_init(self, *pos, **kw):
            original_sgd_init(self, *pos, **kw)
            hook_handles.append(self.register_step_post_hook(after_step))
        torch.optim.SGD.__init__ = observed_sgd_init
        event('calling_original_train_rgb_ir')
        author_train.train_rgb_ir(hyp, opt, torch.device('cuda:0'), tb_writer=None)
        raise RuntimeError('Original trainer returned without the required two-step canary stop')
    except CanaryComplete:
        state['status'] = 'CANARY_TWO_SUCCESSFUL_OPTIMIZER_STEPS'
        state['stop_point'] = 'post-SGD-step hook; second scaler.update/zero_grad/EMA not executed; no resume checkpoint produced'
        state['timing_scope'] = 'First warmup batches only; cannot establish steady-state E200 completion within 10 hours'
    except Exception as exc:
        state['status'] = 'RESOURCE_FAILURE_OOM' if torch is not None and isinstance(exc, torch.OutOfMemoryError) else 'TECHNICAL_FAILURE'
        state['error'] = repr(exc)
        state['traceback'] = traceback.format_exc()
    finally:
        for handle in hook_handles: handle.remove()
        state.pop('batch_started', None)
        state['elapsed_seconds'] = time.perf_counter()-started
        if torch is not None and torch.cuda.is_initialized():
            state['torch_max_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
            state['torch_max_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        if guard_record:
            try: state['lease_resources'] = guard_record(wait_seconds=0)
            except Exception as exc: state['lease_resource_record_error'] = repr(exc)
        state['finished'] = datetime.datetime.now().astimezone().isoformat()
        save('receipt.json' if state['status'] == 'CANARY_TWO_SUCCESSFUL_OPTIMIZER_STEPS' else 'failure.json', state)
        print(json.dumps(state), flush=True)
    return 0 if state['status'] == 'CANARY_TWO_SUCCESSFUL_OPTIMIZER_STEPS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
