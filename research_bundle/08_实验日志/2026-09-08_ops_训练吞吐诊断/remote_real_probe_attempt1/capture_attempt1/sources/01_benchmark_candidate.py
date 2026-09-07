"""CPU equivalence, synthetic pool timings, or replay of a real raw-batch bundle.

CUDA execution must run under the existing bound global lease. No optimizer,
data resampling, hyperparameter search, source mutation, or production patch.
"""
import argparse
import copy
import importlib.util
import json
import statistics
import sys
import time
import types
from pathlib import Path

import torch
from pool_block16 import BLOCK_SIZE, pool_relative_logits_block16

ATOL, RTOL = 1e-6, 1e-5


def load_reference(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path))
    import selection_adapter as selection
    import classification_logit as classification
    for module, name in [(selection, 'selection_adapter.py'), (classification, 'classification_logit.py')]:
        if Path(module.__file__).resolve() != path/name:
            raise RuntimeError('Reference module identity mismatch: '+str(module.__file__))
    return selection, classification


def bound_cuda(reference_dir):
    if str(torch.__version__) != '2.10.0+cu128':
        raise RuntimeError('CUDA benchmark requires pinned torch 2.10.0+cu128')
    sys.path.insert(0, str(Path(reference_dir).resolve()))
    import runtime
    if Path(runtime.__file__).resolve() != Path(reference_dir).resolve()/'runtime.py':
        raise RuntimeError('runtime identity mismatch')
    lease = runtime.legacy.require_bound_lease_from_environment()
    if len(lease['gpus']) != 1:
        raise RuntimeError('Exactly one existing bound GPU lease required')
    if str(runtime.legacy.ultralytics.__version__) != '8.4.115':
        raise RuntimeError('Pinned ultralytics mismatch')
    return runtime


def cloned_selector(selection, pool):
    fn = selection.build_classification_selection
    # Per-function globals clone: the original module/function binding survives.
    namespace = dict(fn.__globals__, pool_relative_logits=pool)
    result = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


def tensor_difference(a, b):
    if a is None or b is None:
        return dict(pass_numeric=a is b, exact=a is b, none=True)
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(pass_numeric=False, exact=False, shape_a=list(a.shape), shape_b=list(b.shape))
    exact = bool(torch.equal(a, b))
    if not a.is_floating_point():
        return dict(pass_numeric=exact, exact=exact)
    delta = (a.double()-b.double()).abs()
    return dict(pass_numeric=bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)), exact=exact,
                max_abs=float(delta.max()) if delta.numel() else 0., elements=a.numel())


def rng_state(device):
    return (torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None)


def rng_equal(a, b):
    return torch.equal(a[0], b[0]) and (a[1] is b[1] if a[1] is None or b[1] is None else torch.equal(a[1], b[1]))


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def selector_eval(selector, classification, data):
    student = dict(data['student'])
    student['scores'] = student['scores'].detach().clone().requires_grad_(True)
    student['boxes'] = student['boxes'].detach().clone().requires_grad_(True)
    auxiliaries={}
    for name in ('teacher','reference'):
        auxiliaries[name]=dict(data[name])
        for key in ('scores','boxes'):
            auxiliaries[name][key]=data[name][key].detach().clone().requires_grad_(True)
    selected = selector(student, auxiliaries['teacher'], auxiliaries['reference'], data['batch'],
        strides=tuple(data['strides']), config=data['evidence'], selection_seed=20260907)
    loss, stats = classification.classification_loss_from_selection(selected, off_target_weight=.25)
    grad_inputs=(student['scores'],student['boxes'],auxiliaries['teacher']['scores'],auxiliaries['teacher']['boxes'],
                 auxiliaries['reference']['scores'],auxiliaries['reference']['boxes'])
    grads = torch.autograd.grad(loss,grad_inputs,allow_unused=True)
    if any(g is not None for g in grads[1:]):
        raise AssertionError('C1 must reach only student scores, not DFL/T/R raw tensors')
    fields = ('student_delta', 'teacher_delta', 'reference_delta', 'valid_levels', 'selected', 'labels',
              'base_to_matched', 'matched_to_base', 'selected_matched_indices', 'quality', 'eligible', 'c0_loss')
    result = {k: getattr(selected, k).detach() for k in fields}
    result.update(loss=loss.detach(), score_gradient=grads[0], box_gradient=grads[1],
                  teacher_score_gradient=grads[2],teacher_box_gradient=grads[3],
                  reference_score_gradient=grads[4],reference_box_gradient=grads[5],
                  matched_object_ids=selected.matched_object_ids if hasattr(selected, 'matched_object_ids') else selected.object_ids,
                  base_object_ids=selected.base_object_ids, c0_stats=selected.c0_stats,
                  selected_count=stats['selected_count'])
    return result


def compare_selector(selection, classification, data):
    before = rng_state(data['student']['scores'].device)
    old = selector_eval(selection.build_classification_selection, classification, data)
    new = selector_eval(cloned_selector(selection, pool_relative_logits_block16), classification, data)
    differences = {k: tensor_difference(old[k], new[k]) for k in old if isinstance(old[k], torch.Tensor) or old[k] is None}
    identities = {k: old[k] == new[k] for k in ('matched_object_ids', 'base_object_ids', 'c0_stats', 'selected_count')}
    unchanged = bool(rng_equal(before, rng_state(data['student']['scores'].device)))
    return dict(pass_numeric=all(x['pass_numeric'] for x in differences.values()) and all(identities.values()) and unchanged,
                differences=differences, exact_identities=identities, rng_unchanged=unchanged)


def pool_case(m, c, anchors, seed):
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(c, anchors, generator=generator)*4
    rank = torch.rand(m, anchors, generator=generator)
    foreground, background = rank < .07, (rank >= .12) & (rank < .35)
    if m:
        foreground[0] = False  # invalid foreground
    if m > 1:
        background[1] = False  # invalid background
    return scores, foreground, background


def compare_pool(old, args):
    outputs = []
    before = rng_state(args[0].device)
    for fn in (old, pool_relative_logits_block16):
        scores = args[0].detach().clone().requires_grad_(True)
        output, valid = fn(scores, *args[1:])
        weights = torch.linspace(.1, 1., max(output.numel(), 1), device=scores.device)[:output.numel()].reshape(output.shape)
        gradient, = torch.autograd.grad((output*weights).sum(), scores)
        outputs.append((output.detach(), valid, gradient))
    fields = dict(zip(('delta', 'valid', 'student_gradient'), [tensor_difference(a,b) for a,b in zip(*outputs)]))
    # Frozen T/R use the same kernel under no_grad; also compare their outputs.
    with torch.no_grad():
        for name, scale in [('teacher', .7), ('reference', 1.3)]:
            a, av = old(args[0]*scale, *args[1:]); b, bv = pool_relative_logits_block16(args[0]*scale, *args[1:])
            fields[name+'_delta'] = tensor_difference(a,b)
            fields[name+'_valid'] = tensor_difference(av,bv)
    unchanged = bool(rng_equal(before, rng_state(args[0].device)))
    return dict(pass_numeric=all(v['pass_numeric'] for v in fields.values()) and unchanged,
                differences=fields, rng_unchanged=unchanged)


def cpu_checks(reference_dir, selection, classification):
    checks = []
    for m, c, a in [(0,1,31),(1,1,31),(2,5,31),(15,5,97),(16,1,97),(17,5,97),(33,8,97),(64,5,257)]:
        checks.append(dict(kind='pool', shape=[m,c,a], **compare_pool(selection.pool_relative_logits, pool_case(m,c,a,1717+m))))
    spec = importlib.util.spec_from_file_location('frozen_fixture_tests', Path(reference_dir)/'test_classification_logit.py')
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    for name, boxes, labels, indices, classes, batch_size in [
        ('empty', [], [], [], 1, 1),
        ('single_class', [(16.,16.,40.,40.)], [0], [0], 1, 1),
        ('mixed_invalid', [(1.,1.,2.,2.),(16.,16.,40.,40.),(45.,45.,60.,60.)], [0,0,1], [0,0,0], 5, 1),
        ('multi_image_empty_middle', [(16.,16.,40.,40.),(16.,16.,40.,40.),(45.,45.,60.,60.)], [0,1,4], [0,2,2], 5, 3)]:
        batch = fixtures.labels(boxes, classes=labels, indices=indices)
        batch['teacher_batch'] = copy.deepcopy(batch)
        data = dict(student=fixtures.evidence_raw(0,batch,classes,batch_size),
                    teacher=fixtures.evidence_raw(4,batch['teacher_batch'],classes,batch_size,False),
                    reference=fixtures.evidence_raw(1,batch,classes,batch_size,False), batch=batch,
                    strides=(8,16,32), evidence={'input_size':64})
        checks.append(dict(kind='full_selection', case=name, **compare_selector(selection,classification,data)))
    return checks


def timed(fn, device, track_peak=True):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        if track_peak:torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter()-started
    resource = dict(seconds=seconds)
    if device.type == 'cuda' and track_peak:
        resource.update(allocated_peak_mib=torch.cuda.max_memory_allocated(device)/2**20,
                        reserved_peak_mib=torch.cuda.max_memory_reserved(device)/2**20)
    return result, resource


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu-check','synthetic','bundle'), required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    args=parser.parse_args()
    if args.mode=='cpu-check' and args.device!='cpu':
        parser.error('cpu-check does not initialize CUDA')
    if args.iterations<1 or args.warmup<0:
        parser.error('Positive iterations and nonnegative warmup required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    selection, classification=load_reference(args.reference_dir)
    runtime=bound_cuda(args.reference_dir) if args.device=='cuda' else None
    device=torch.device(args.device)
    report=dict(status='STARTED', block_size=BLOCK_SIZE, atol=ATOL, rtol=RTOL,
                mode=args.mode, device=str(device), torch=str(torch.__version__),
                reference_dir=str(args.reference_dir.resolve()), no_source_mutation=True,
                new_hash_computed=False, optimizer_updates=0,
                scope='Kernel/raw-score diagnostic; not 24-update or long-training admission')
    try:
        if args.mode=='cpu-check':
            report['checks']=cpu_checks(args.reference_dir,selection,classification)
            passed=all(r['pass_numeric'] for r in report['checks'])
        else:
            samples=[]
            if args.mode=='synthetic':
                cases=[pool_case(32,5,6400,1717),pool_case(128,5,1600,1818)]
                report['checks']=[compare_pool(selection.pool_relative_logits,tuple(v.to(device) for v in case)) for case in cases]
                for ci,case in enumerate(cases):
                    case=tuple(v.to(device) for v in case)
                    for iteration in range(args.warmup+args.iterations):
                        for name,fn in ((('old',selection.pool_relative_logits),('block16',pool_relative_logits_block16)) if iteration%2==0 else (('block16',pool_relative_logits_block16),('old',selection.pool_relative_logits))):
                            def run(fn=fn):
                                z=case[0].detach().clone().requires_grad_(True)
                                out,_=fn(z,*case[1:]); out.square().mean().backward()
                            _,record=timed(run,device)
                            if iteration>=args.warmup: samples.append(dict(case=ci,variant=name,iteration=iteration-args.warmup,**record))
                report['timing_scope']='Pool forward plus raw-score backward; S model/loader excluded'
            else:
                if args.bundle is None: raise ValueError('--bundle is required')
                try: data=torch.load(args.bundle,map_location='cpu',weights_only=False)
                except TypeError: data=torch.load(args.bundle,map_location='cpu')
                if data.get('bundle_kind')!='real_frozen_flow_raw_v1': raise ValueError('Unrecognized real-batch bundle')
                report['bundle_provenance']=data['provenance']
                data=move(data,device)
                report['checks']=[compare_selector(selection,classification,data)]
                alternate=cloned_selector(selection,pool_relative_logits_block16)
                for iteration in range(args.warmup+args.iterations):
                    variants=[('old',selection.build_classification_selection),('block16',alternate)]
                    if iteration%2: variants.reverse()
                    for name,fn in variants:
                        _,record=timed(lambda: selector_eval(fn,classification,data),device)
                        if iteration>=args.warmup:samples.append(dict(case=0,variant=name,iteration=iteration-args.warmup,**record))
                report['timing_scope']='Complete selector plus KD/statistics plus raw-score gradient; model/loader excluded'
            report['samples']=samples
            report['median_seconds']={str(ci):{name:statistics.median(r['seconds'] for r in samples if r['case']==ci and r['variant']==name) for name in ('old','block16')} for ci in sorted({r['case'] for r in samples})}
            passed=all(r['pass_numeric'] for r in report['checks'])
        report['status']='PASS_NUMERIC_EQUIVALENCE_ONLY' if passed else 'FAIL_EQUIVALENCE'
        if runtime is not None:report['resources']=runtime.legacy.bound_lease_resource_record_from_environment()
    except Exception as error:
        report.update(status='FAILED',error=repr(error))
        raise
    finally:
        (args.output/'receipt.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],output=str(args.output)),ensure_ascii=False))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
