"""In-memory parameter/buffer exchange on two accepted N checkpoints; eval only."""
import argparse
import copy
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import yaml

SCOPE = 'BN_PARAMETER_SWAP_DIAGNOSTIC'
EFFECTIVE = dict(imgsz=640, batch=32, workers=4, quantize=None, conf=.001, iou=.7,
                 max_det=300, agnostic_nms=False, single_cls=False, rect=True,
                 augment=False, half=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def stat(path):
    p = Path(path).resolve(); s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


def tensor_groups(model):
    """Include every registered buffer, also nonpersistent buffers."""
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    require(not set(parameters) & set(buffers), 'Parameter/buffer name collision')
    require(set(model.state_dict()).issubset(set(parameters) | set(buffers)),
            'Unclassified/aliased state_dict keys; refuse implicit state omission')
    return parameters, buffers


def compose_model(initial, finetuned, variant):
    """Strict raw dtype/shape/key check before the shared FP32 eval conversion."""
    import torch
    require(variant in ('parameter_only', 'buffer_only'), 'Unknown swap variant')
    require(type(initial) is type(finetuned), 'Model class differs')
    groups = [tensor_groups(model) for model in (initial, finetuned)]
    details = []
    bn_keys = set()
    for prefix, module in initial.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn_keys.update((prefix+'.' if prefix else '')+name for name in
                           ('running_mean', 'running_var', 'num_batches_tracked'))
    for kind, left, right in zip(('parameter', 'buffer'), groups[0], groups[1]):
        require(set(left) == set(right), kind + ' keys differ')
        for name in sorted(left):
            a, b = left[name], right[name]
            require(a.dtype == b.dtype and a.shape == b.shape, 'dtype/shape differs: '+name)
            require(a.layout == b.layout, 'Tensor layout differs: '+name)
            if a.is_floating_point():
                require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'Nonfinite source tensor: '+name)
            details.append(dict(name=name, kind=kind, dtype=str(a.dtype), shape=list(a.shape),
                                changed=not torch.equal(a.detach().cpu(), b.detach().cpu()),
                                bn_statistic=kind=='buffer' and name in bn_keys))
    hybrid = copy.deepcopy(initial)
    target = tensor_groups(hybrid)
    sources = (groups[1][0], groups[0][1]) if variant == 'parameter_only' else (groups[0][0], groups[1][1])
    with torch.no_grad():
        for output, source in zip(target, sources):
            for name, value in output.items():
                value.copy_(source[name].detach())
                require(torch.equal(value.detach().cpu(), source[name].detach().cpu()), 'Swap did not copy exactly: '+name)
    hybrid.float().eval()
    for value in hybrid.parameters():
        value.requires_grad_(False); value.grad = None
    for output, source in zip(tensor_groups(hybrid), sources):
        for name, value in output.items():
            expected = source[name].detach().float() if source[name].is_floating_point() else source[name].detach()
            require(torch.equal(value.detach().cpu(), expected.cpu()), 'FP32 projection differs: '+name)
    changed_buffers = [r for r in details if r['kind']=='buffer' and r['changed']]
    report = dict(scope=SCOPE, variant=variant, strict_key_dtype_shape=True,
        parameter_source='finetuned' if variant=='parameter_only' else 'initial',
        buffer_source='initial' if variant=='parameter_only' else 'finetuned',
        registered_parameters=len(groups[0][0]), registered_buffers=len(groups[0][1]),
        changed_parameter_keys=[r['name'] for r in details if r['kind']=='parameter' and r['changed']],
        changed_buffer_keys=[r['name'] for r in changed_buffers],
        non_bn_changed_buffer_keys=[r['name'] for r in changed_buffers if not r['bn_statistic']],
        all_changed_buffers_are_bn_statistics=all(r['bn_statistic'] for r in changed_buffers),
        tensors=details, fp32_projection_exact=True, gradients_enabled=False,
        parameters_require_grad=False, optimizer_created=False, checkpoint_written=False,
        metadata_and_unregistered_cache_source='initial model; native evaluator may rebuild inference caches',
        new_hash_computed=False)
    return hybrid, report


def checkpoint_module(path, torch):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    require(isinstance(payload, dict), 'Expected native checkpoint dictionary')
    model = payload.get('ema')
    selected = 'ema'
    if model is None:
        model = payload.get('model'); selected = 'model'
    require(isinstance(model, torch.nn.Module), 'Checkpoint lacks module model/EMA')
    return model.cpu(), selected


def copy_sources(output, sources):
    directory = output/'source_copies'; directory.mkdir()
    rows = []
    for i, source in enumerate(sorted({Path(p).resolve() for p in sources}, key=str)):
        target = directory/('%04d_' % i + source.name)
        shutil.copyfile(source, target)
        require(source.read_bytes()==target.read_bytes(), 'Source copy differs')
        rows.append(dict(stat(source), copy=str(target), byte_identity=True))
    write_new(output/'source_manifest.json', dict(files=rows, new_hash_computed=False))


def run(args):
    output = args.output.resolve()
    require(os.name=='posix' and Path('/mnt/dataset/yudongfang') in output.parents, 'Data-disk output required')
    require(not output.exists(), 'Use a new immutable output directory')
    initial_stat, finetuned_stat = stat(args.initial), stat(args.finetuned)
    require(initial_stat['path'] != finetuned_stat['path'], 'Two distinct accepted inputs required')
    cfg = yaml.safe_load(args.native_config.read_text(encoding='utf-8'))
    require(cfg['dataset']=='dronevehicle' and cfg['expected_nc']==5 and cfg['expected_val_images']==1469,
            'Only the fixed Drone native evaluator is supported')
    ref = args.reference_dir.resolve(); sys.path.insert(0, str(ref))
    import torch
    import runtime
    import evaluator_profile as profile
    import evaluate_independent as helpers
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class
    from ultralytics.utils.nms import non_max_suppression
    for module, name in ((runtime,'runtime.py'), (profile,'evaluator_profile.py'), (helpers,'evaluate_independent.py')):
        require(Path(module.__file__).resolve()==ref/name, 'Wrong pinned helper: '+name)
    require(len(runtime.legacy.require_bound_lease_from_environment()['gpus'])==1, 'One existing globallease required')
    versions = dict(torch=str(torch.__version__), ultralytics=str(runtime.legacy.ultralytics.__version__))
    require(versions=={k:cfg[k+'_version'] for k in versions}, 'Pinned runtime differs')
    binding = profile.validate_evaluation_profile_binding(args.native_profile_binding, cfg, ref)
    require(binding['actual_effective_kwargs']==EFFECTIVE, 'Native evaluation contract differs')
    data = cfg['paths']['student_data_yaml']; canonical = profile.dev_roster(data)
    require(len(canonical)==1469 and len(set(canonical))==1469, 'Complete unique dev required')
    output.mkdir(parents=True, exist_ok=False)
    (output/'development_roster.txt').write_text(''.join(p+'\n' for p in canonical), encoding='utf-8')
    sources = {Path(__file__), ref/'evaluator_profile.py', ref/'evaluate_independent.py',
               args.native_config, args.native_profile_binding, Path(data)}
    sources.update(Path(p) for p in runtime.legacy.implementation_files(
        YOLO.val, DetectionValidator, BaseValidator, AutoBackend, check_det_dataset,
        DetMetrics, Metric, ap_per_class, non_max_suppression))
    started = time.perf_counter(); captured = {}; evaluated = {}; hybrid = None
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    try:
        initial, initial_field = checkpoint_module(args.initial, torch)
        finetuned, ft_field = checkpoint_module(args.finetuned, torch)
        require(initial.names==finetuned.names, 'Checkpoint class names differ')
        require(torch.equal(initial.stride, finetuned.stride), 'Checkpoint strides differ')
        hybrid, composition = compose_model(initial, finetuned, args.variant)
        composition.update(initial_checkpoint=initial_stat, finetuned_checkpoint=finetuned_stat,
                           initial_payload_field=initial_field, finetuned_payload_field=ft_field)
        write_new(output/'composition.json', composition)
        del initial, finetuned
        # YOLO.val receives self.model directly. Replace it before val and assert
        # object identity at the validator call and actual module forward.
        model = YOLO(str(args.initial.resolve()), task='detect')
        model.model = hybrid; model.ckpt = None; model.predictor = None
        evidence = helpers.make_evidence_validator(DetectionValidator)
        class MemoryValidator(evidence):
            def __call__(self, *positional, **keyword):
                require(keyword.get('model') is hybrid, 'Validator did not receive the swapped in-memory module')
                evaluated['validator_received_memory_model'] = True
                return super().__call__(*positional, **keyword)
        def check_forward(module, inputs):
            require(module is hybrid and not torch.is_grad_enabled() and not module.training,
                    'Unexpected training/gradient path during diagnostic')
            require(not any(p.requires_grad for p in module.parameters()), 'Eval model has trainable parameters')
            require(all(not p.is_floating_point() or p.dtype==torch.float32 for p in module.parameters()), 'Eval is not FP32')
            evaluated['forward_memory_identity'] = True
            evaluated['forward_calls'] = evaluated.get('forward_calls', 0)+1
        hybrid.register_forward_pre_hook(check_forward)
        def on_start(v):
            require(not any(p.requires_grad for p in hybrid.parameters()), 'Eval model has trainable parameters')
            require(all(not p.is_floating_point() or p.dtype==torch.float32 for p in hybrid.parameters()), 'Eval is not FP32')
            captured.update(helpers.capture_contract(v, canonical, versions))
            require(captured['effective_kwargs']==EFFECTIVE, 'Actual native kwargs differ')
            labels = v.dataloader.dataset.labels
            require(len(labels)==1469 and sum(len(x['cls']) for x in labels)==22462, 'Full dev GT changed')
            captured.update(scope=SCOPE, endpoint='in_memory_'+args.variant, gt_objects_before_inference=22462,
                            initial_checkpoint=initial_stat, finetuned_checkpoint=finetuned_stat,
                            formal_training_endpoint=False)
            captured['runtime_sources'] = profile.capture_runtime_sources(v, model, sources)
        def on_end(v):
            helpers.verify_population(v, captured)
            with gzip.open(v.save_dir/'objects.jsonl.gz', 'rt', encoding='utf-8') as f:
                rows = [json.loads(line) for line in f if line.strip()]
            require(sum(len(r['gt_classes']) for r in rows)==22462, 'Captured GT population differs')
            captured['gt_objects_captured'] = 22462
        model.add_callback('on_val_start', on_start); model.add_callback('on_val_end', on_end)
        torch.cuda.reset_peak_memory_stats()
        metrics = model.val(validator=MemoryValidator, data=data, split='val', imgsz=640, batch=32,
            workers=4, device='0', conf=.001, iou=.7, max_det=300, agnostic_nms=False,
            single_cls=False, rect=True, augment=False, quantize=None, plots=False, save_json=False,
            verbose=False, project=str(output), name='native_capture', exist_ok=False)
        values = profile.metric_record(metrics, model.names)
        require({p['class_id'] for p in values['per_class']}==set(range(5)) and len(values['per_class'])==5,
                'Five unique classes required')
        for metric in ('AP50','AP75','mAP50_95','precision','recall'):
            require(math.isfinite(values[metric]) and 0<=values[metric]<=1, 'Invalid metric fraction')
        for metric in ('AP50','AP75','mAP50_95'):
            require(all(math.isfinite(p[metric]) and 0<=p[metric]<=1 for p in values['per_class']), 'Invalid per-class AP')
        require(evaluated.get('validator_received_memory_model') and evaluated.get('forward_memory_identity') and evaluated.get('forward_calls',0)>0, 'No verified memory-model inference')
        require(stat(args.initial)==initial_stat and stat(args.finetuned)==finetuned_stat, 'Input checkpoint stat changed')
        require(all(p.grad is None for p in hybrid.parameters()), 'Unexpected parameter gradient')
        copy_sources(output, sources)
        write_new(output/'evaluation_contract.json', captured)
        write_new(output/'swap_evaluation_receipt.json', dict(status='BN_PARAMETER_SWAP_EVALUATION_COMPLETED',
            scope=SCOPE, variant=args.variant, dataset='dronevehicle', seed=42, single_seed=True,
            endpoint='in_memory_'+args.variant, formal_training_endpoint=False,
            initial_checkpoint=initial_stat, finetuned_checkpoint=finetuned_stat,
            native_profile_binding=str(args.native_profile_binding), native_config=str(args.native_config),
            full_dev_images=1469, full_dev_gt_objects=22462, metric_units='fraction_0_to_1', **values,
            model_identity=evaluated, changed_buffers_all_bn=composition['all_changed_buffers_are_bn_statistics'],
            seconds=time.perf_counter()-started, resources=runtime.legacy.bound_lease_resource_record_from_environment(),
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            new_hash_computed=False, official_test_accessed=False, checkpoint_written=False,
            optimizer_created=False, formal_e200_complete=False, formal_paper_gain_claim=False))
    except BaseException as error:
        write_new(output/'failure.json', dict(status='BN_PARAMETER_SWAP_FAILED', scope=SCOPE,
            error=repr(error), traceback=traceback.format_exc(), new_hash_computed=False,
            seconds=time.perf_counter()-started, checkpoint_written=False))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('reference-dir','initial','finetuned','native-config','native-profile-binding','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--variant', choices=('parameter_only','buffer_only'), required=True)
    run(parser.parse_args())
