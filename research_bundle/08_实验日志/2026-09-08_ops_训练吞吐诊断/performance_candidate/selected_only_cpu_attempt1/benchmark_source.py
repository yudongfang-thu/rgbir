"""CPU truth checks / lease-bound replay of selected-only training C1 candidate."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import statistics

import torch

import benchmark_candidate as shared
from selected_only_v1 import make_api, full_diagnostics_due


def evaluate(adapter, classification, api, data, variant, full_diagnostics=False):
    raw = {}
    for name in ('student','teacher','reference'):
        raw[name] = dict(data[name])
        for key in ('scores','boxes'):
            raw[name][key] = data[name][key].detach().clone().requires_grad_(True)
    args = [raw[name] for name in ('student','teacher','reference')] + [data['batch']]
    kwargs = dict(strides=data['strides'],config=data['evidence'],selection_seed=20260907)
    if variant == 'old':
        payload = adapter.build_classification_selection(*args, **kwargs)
        loss, stats = classification.classification_loss_from_selection(payload)
        observed, fallback, diagnostic_seconds = True, None, 0.
    else:
        payload = api.build(*args, full_diagnostics=full_diagnostics, **kwargs)
        loss, stats = api.loss(payload)
        observed, fallback, diagnostic_seconds = payload.thin_path_used,payload.fallback_reason,payload.diagnostics_seconds
    grads = torch.autograd.grad(loss, [raw[n][k] for n in ('student','teacher','reference')
                                     for k in ('scores','boxes')], allow_unused=True)
    if grads[0] is None or any(value is not None for value in grads[1:]):
        raise AssertionError('KD gradient must reach only student scores, including differentiable empty zero')
    scalar_fields = ('common_count','valid_region_count','reference_candidate_count','base_count',
        'teacher_correct_base_count','eligible_count','selected_count','normalizer','nominal_dose')
    result = {name:getattr(payload,name).detach() for name in ('valid_levels','selected','labels',
        'base_to_matched','matched_to_base','selected_matched_indices','quality','eligible','c0_loss')}
    result.update(loss=loss.detach(),score_gradient=grads[0],box_gradient=grads[1],
        teacher_score_gradient=grads[2],teacher_box_gradient=grads[3],
        reference_score_gradient=grads[4],reference_box_gradient=grads[5],
        selected_student_delta=payload.student_delta[payload.selected].detach(),
        selected_teacher_delta=payload.teacher_delta[payload.selected].detach(),
        matched_object_ids=payload.matched_object_ids,base_object_ids=payload.base_object_ids,
        c0_stats=payload.c0_stats,counts={k:stats[k] for k in scalar_fields},
        regions=[{k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in vars(r).items()} for r in payload.regions],
        stats=stats,thin_path_used=observed,fallback_reason=fallback,diagnostics_seconds=diagnostic_seconds)
    return result


def check(adapter,classification,api,data,expect_thin=True,full_diagnostics=False):
    device=data['student']['scores'].device
    state=shared.rng_state(device)
    old=evaluate(adapter,classification,api,data,'old')
    new=evaluate(adapter,classification,api,data,'selected_only',full_diagnostics)
    differences={k:shared.tensor_difference(old[k],new[k]) for k in old
                 if isinstance(old[k],torch.Tensor) or k.endswith('_gradient')}
    identities={k:old[k]==new[k] for k in ('matched_object_ids','base_object_ids','c0_stats','counts')}
    regions_exact=len(old['regions'])==len(new['regions'])
    for a,b in zip(old['regions'],new['regions']):
        for k in a:
            regions_exact &= bool(torch.equal(a[k],b[k])) if isinstance(a[k],torch.Tensor) else a[k]==b[k]
    identities['all_background_masks_and_region_identity_exact']=regions_exact
    identities['expected_thin_or_fallback']=new['thin_path_used'] is expect_thin
    identities['rng_unchanged']=bool(shared.rng_equal(state,shared.rng_state(device)))
    identities['reference_diagnostic_missing_not_zero']=(
        new['stats']['reference_delta_class_mean_base_valid'] is None if expect_thin and not full_diagnostics
        else old['stats']['reference_delta_class_mean_base_valid']==new['stats']['reference_delta_class_mean_base_valid'])
    return dict(pass_numeric=all(r['pass_numeric'] for r in differences.values()) and all(identities.values()),
        differences=differences,identities=identities,counts=new['counts'],fallback_reason=new['fallback_reason'],
        full_diagnostics=full_diagnostics)


def fixture_data(reference_dir, fp16=True):
    spec=importlib.util.spec_from_file_location('_selected_only_fixture',Path(reference_dir)/'test_classification_logit.py')
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    cases=[('empty',[],[],[],1,1),
        ('one_class',[(16.,16.,40.,40.)],[0],[0],1,1),
        ('multi_invalid',[(1.,1.,2.,2.),(16.,16.,40.,40.),(45.,45.,60.,60.)],[0,0,1],[0,0,0],5,1),
        ('multi_image_empty_middle',[(16.,16.,40.,40.),(16.,16.,40.,40.),(45.,45.,60.,60.)],[0,1,4],[0,2,2],5,3),
        ('many_objects',[(4.+i%4*14,4.+i//4*14,14.+i%4*14,14.+i//4*14) for i in range(12)],
         [i%5 for i in range(12)],[0]*12,5,1)]
    for name,boxes,labels,indices,classes,batch_size in cases:
        batch=fixtures.labels(boxes,classes=labels,indices=indices)
        batch['teacher_batch']=copy.deepcopy(batch)
        data=dict(student=fixtures.evidence_raw(0,batch,classes,batch_size),
            teacher=fixtures.evidence_raw(4,batch['teacher_batch'],classes,batch_size,False),
            reference=fixtures.evidence_raw(1,batch,classes,batch_size,False),batch=batch,
            strides=(8,16,32),evidence={'input_size':64})
        if fp16:
            for key in ('student','teacher','reference'):data[key]['scores']=data[key]['scores'].half()
        yield name,data
    # Shared GT background exclusion must still include a teacher-unmatched RGB
    # object. Keep raw predictions from the full scene, remove only its IR label.
    data=copy.deepcopy(data)
    for key in ('batch_idx','cls','bboxes'):
        data['batch']['teacher_batch'][key]=data['batch']['teacher_batch'][key][:-1]
    yield 'unmatched_rgb_gt_in_background_exclusion',data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--mode',choices=('cpu-check','bundle'),required=True)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--iterations',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=2)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.mode=='cpu-check' and args.device!='cpu':raise ValueError('CPU checks must remain CPU-only')
    if args.iterations<1 or args.warmup<0:raise ValueError('Invalid fixed measurement counts')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    adapter,classification=shared.load_reference(args.reference_dir)
    api=make_api(adapter,classification)
    runtime=shared.bound_cuda(args.reference_dir) if args.device=='cuda' else None
    device=torch.device(args.device)
    receipt=dict(status='STARTED',candidate='selected_only_v1_original_pool',atol=shared.ATOL,rtol=shared.RTOL,
        torch=str(torch.__version__),new_hash_computed=False,optimizer_updates=0,long_training_switch_admitted=False,
        reference_dir=str(args.reference_dir.resolve()),checks=[])
    try:
        if args.mode=='cpu-check':
            for name,data in fixture_data(args.reference_dir,True):
                for full in (False,True):
                    receipt['checks'].append(dict(case=name,**check(adapter,classification,api,data,True,full)))
            for name,data in fixture_data(args.reference_dir,False):
                receipt['checks'].append(dict(case=name+'_fp32_fallback',**check(adapter,classification,api,data,False)))
            cadence=[full_diagnostics_due(i,100,False,False) for i in (1,2,3,4,99,100,101)]
            if cadence != [True,True,True,False,False,True,False]:raise AssertionError('Original logging cadence changed')
            if not full_diagnostics_due(4,100,True,False) or not full_diagnostics_due(4,100,False,True):
                raise AssertionError('Sanity/observer diagnostics omitted')
            receipt['cadence_truth_pass']=True
            receipt['cuda_initialized']=torch.cuda.is_initialized()
        else:
            if args.bundle is None:raise ValueError('--bundle required')
            try:data=torch.load(args.bundle,map_location='cpu',weights_only=False)
            except TypeError:data=torch.load(args.bundle,map_location='cpu')
            if data.get('bundle_kind')!='real_frozen_flow_raw_v1':raise ValueError('Unrecognized real bundle')
            receipt['bundle_provenance']=data['provenance']
            data=shared.move(data,device)
            receipt['raw_score_dtypes']={k:str(data[k]['scores'].dtype) for k in ('student','teacher','reference')}
            expect_thin=all(data[k]['scores'].dtype==torch.float16 for k in ('student','teacher','reference'))
            receipt['checks']=[check(adapter,classification,api,data,expect_thin,full) for full in (False,True)]
            samples=[]
            for iteration in range(args.warmup+args.iterations):
                variants=('old','selected_only') if iteration%2==0 else ('selected_only','old')
                for variant in variants:
                    _,measurement=shared.timed(lambda:evaluate(adapter,classification,api,data,variant),device)
                    if iteration>=args.warmup:samples.append(dict(variant=variant,iteration=iteration-args.warmup,**measurement))
            receipt['samples']=samples
            receipt['median_seconds']={v:statistics.median(s['seconds'] for s in samples if s['variant']==v) for v in ('old','selected_only')}
            receipt['timing_scope']='Original complete selection+stats+raw-score backward versus thin selection+minimal stats+raw-score backward; no model/loader/optimizer, no full diagnostics on thin timed path'
        passed=all(row['pass_numeric'] for row in receipt['checks'])
        receipt['status']='PASS_NUMERIC_EQUIVALENCE_ONLY' if passed else 'FAIL_EQUIVALENCE'
        if runtime is not None:receipt['resources']=runtime.legacy.bound_lease_resource_record_from_environment()
    except BaseException as error:
        receipt.update(status='FAILED',error=repr(error));raise
    finally:
        (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],output=str(args.output))))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
