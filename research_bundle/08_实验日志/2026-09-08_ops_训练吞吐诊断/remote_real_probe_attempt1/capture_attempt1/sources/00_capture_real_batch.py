"""Bound-lease diagnostic capture: fixed natural flow, staged forward/backward.

Loads a separately named checkpoint; never writes model weights or steps an
optimizer. Raw-bundle replay isolates the pool candidate from model/data changes.
S is train-mode on its private in-memory copy. T/R are frozen eval models.
"""
import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from benchmark_candidate import (bound_cuda, load_reference, move, timed,
                                  cloned_selector, selector_eval)


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def signature(path):
    info=Path(path).stat()
    return dict(bytes=info.st_size,mtime_ns=info.st_mtime_ns)


def compact_raw(raw):
    # C1 reads feature shapes/version identities, never feature values. Preserve
    # every logical B,C,H,W/dtype with a zero-stride one-element CPU storage.
    return dict(scores=raw['scores'].detach().cpu(),boxes=raw['boxes'].detach().cpu(),
                feats=[torch.zeros(1,dtype=feature.dtype).expand(feature.shape) for feature in raw['feats']])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--student-checkpoint',type=Path,required=True)
    parser.add_argument('--coverage-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batches',type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not str(args.output.resolve()).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('Remote output must be a new directory on the project data disk')
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if cfg.get('arm')!='C1':raise ValueError('This capture is scoped to frozen C1')
    if (cfg['batch'],cfg['workers'],cfg['imgsz']) != (32,4,640):
        raise ValueError('Preserve frozen B32/workers4/640')
    coefficient=float(cfg['classification_coefficient'])
    if not math.isfinite(coefficient) or coefficient<=0:raise ValueError('Actual positive C1 coefficient required')
    selection,classification=load_reference(args.reference_dir)
    runtime=bound_cuda(args.reference_dir)
    import coverage_probe as coverage
    if Path(coverage.__file__).resolve()!=args.reference_dir.resolve()/'coverage_probe.py':
        raise RuntimeError('Coverage module identity mismatch')
    previous=json.loads((args.coverage_dir/'coverage_receipt.json').read_text(encoding='utf-8'))
    if previous['status']!='COMPLETED' or previous['actual_batches']!=64 or previous['dataset']!=cfg['dataset']:
        raise ValueError('Original completed natural coverage required')
    traces=[json.loads(x) for x in (args.coverage_dir/'natural_batches.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(traces)!=64 or [x['batch'] for x in traces]!=list(range(64)):
        raise ValueError('Original 64-batch trace incomplete')
    roster,_=coverage.load_roster(previous['roster_path'],cfg['dataset'])
    paths={'student':args.student_checkpoint,'teacher':Path(cfg['teacher']),'reference':Path(cfg['reference'])}
    before={k:signature(p) for k,p in paths.items()}
    args.output.mkdir(parents=True)
    snapshot=args.output/'sources'; snapshot.mkdir()
    source_paths=[Path(__file__),Path(__file__).with_name('benchmark_candidate.py'),Path(__file__).with_name('pool_block16.py')]
    source_paths += [args.reference_dir/name for name in ('selection_adapter.py','classification_logit.py','runtime.py','coverage_probe.py',
        'task_conditional_reference/legacy_oev1/object_evidence_loss.py','task_conditional_reference/legacy_oev1/train_object_evidence.py')]
    source_manifest=[]
    for index,path in enumerate(source_paths):
        target=snapshot/f'{index:02d}_{path.name}'; shutil.copyfile(path,target)
        if target.read_bytes()!=path.read_bytes():raise AssertionError('Source copy differs')
        source_manifest.append(dict(path=str(path.resolve()),snapshot=str(target),**signature(path)))
    shutil.copyfile(args.config,snapshot/'input_config.yaml')
    shutil.copyfile(args.student_checkpoint.parents[1]/'args.yaml',snapshot/'student_checkpoint_args.yaml')
    write(args.output/'source_manifest.json',dict(files=source_manifest,new_hash_computed=False))
    report=dict(status='STARTED',device='cuda',torch=str(torch.__version__),dataset=cfg['dataset'],
                batches_requested=args.batches,optimizer_updates=0,new_hash_computed=False,
                source_module=str(args.reference_dir.resolve()),model_files={k:str(p) for k,p in paths.items()},
                model_signatures_before=before,block_size_candidate_not_used_in_capture=16,
                timing_scope='Synchronized serial stage wall clock; instrumentation removes normal overlap. Not production throughput.',
                backward_scope='Unscaled native+B*lambda*C1 backward on a private train-mode student, no optimizer/scaler/EMA update',
                flow_scope='Original fixed calibration/probe natural flow seed20260907; not formal trainer sampler',
                long_training_switch_admitted=False,stages=[],batches=[])
    loader=None; iterator=None
    try:
        torch.set_num_threads(4)
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.use_deterministic_algorithms(True,warn_only=True)
        data=runtime.legacy.check_det_dataset(cfg['paths']['student_data_yaml'],autodownload=False)
        names=data['names']
        models={}
        for name,path in paths.items():
            expected=cfg['paths']['privileged_data_yaml'] if name=='teacher' else cfg['paths']['student_data_yaml']
            models[name]=runtime.legacy.load_frozen(path,expected,names).cuda()
        student_model=models['student'].train().requires_grad_(True)
        student_args=yaml.safe_load((args.student_checkpoint.parents[1]/'args.yaml').read_text(encoding='utf-8'))
        student_model.args=SimpleNamespace(**student_args)
        native_criterion=student_model.init_criterion()
        strides=tuple(int(v) for v in student_model.stride)
        loader=coverage.build_natural_loader(cfg,seed=20260907)
        report['loader_metadata']=loader.coverage_metadata
        iterator=iter(loader)
        device=torch.device('cuda')

        def stage(name,index,fn):
            result,record=timed(fn,device)
            report['stages'].append(dict(batch=index,stage=name,**record))
            return result

        for index in range(args.batches):
            start=time.perf_counter()
            raw_batch=stage('loader_wait',index,lambda:next(iterator))
            actual=coverage.summarize_batch(raw_batch,index,roster)
            if json.loads(json.dumps(actual))!=traces[index]['images']:
                raise AssertionError('Source/augmentation/labels differ from original natural flow')
            if len(raw_batch['im_file'])!=32:raise AssertionError('Batch size changed')
            write(args.output/f'trace_{index:02d}.json',dict(batch=index,images=actual))
            batch=stage('preprocess',index,lambda:runtime.to_device(raw_batch,device))
            student_model.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=bool(cfg['amp'])):
                prediction=stage('student_forward',index,lambda:student_model(batch['img']))
                s=runtime.legacy.raw_prediction(prediction)
                with torch.no_grad():
                    t=stage('teacher_forward',index,lambda:runtime.legacy.raw_prediction(models['teacher'](batch['strong_img'])))
                    r=stage('reference_forward',index,lambda:runtime.legacy.raw_prediction(models['reference'](batch['img'])))
                native,items=stage('native_loss',index,lambda:native_criterion(prediction,batch))
                selected=stage('selection_total_original',index,lambda:selection.build_classification_selection(s,t,r,batch,
                    strides=strides,config=cfg['evidence'],selection_seed=cfg['seed']+index+1))
                kd,stats=stage('classification_loss_and_statistics',index,lambda:classification.classification_loss_from_selection(selected,off_target_weight=.25))
                total=native.sum()+len(batch['img'])*coefficient*kd
            # Independent effective-KD check. Separate from all phase timers and
            # subtracted from the pipeline wall span; it never writes .grad.
            torch.cuda.synchronize(); kd_probe_started=time.perf_counter()
            kd_gradient,=torch.autograd.grad(kd,s['scores'],retain_graph=True,allow_unused=True)
            kd_gradient_finite=kd_gradient is not None and bool(torch.isfinite(kd_gradient).all())
            kd_gradient_norm=float(kd_gradient.float().norm()) if kd_gradient is not None else 0.
            if not kd_gradient_finite or not math.isfinite(kd_gradient_norm) or kd_gradient_norm<=0:
                raise FloatingPointError('Fixed real batch has no finite nonzero independent KD score gradient; do not replace batch')
            del kd_gradient
            torch.cuda.synchronize(); kd_probe_seconds=time.perf_counter()-kd_probe_started
            stage('student_native_plus_KD_backward',index,lambda:total.backward())
            if any(p.grad is not None for name in ('teacher','reference') for p in models[name].parameters()):
                raise AssertionError('Frozen auxiliary gradient leakage')
            gradients=[p.grad for p in student_model.parameters() if p.grad is not None]
            finite_gradients=bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
            if not finite_gradients:raise FloatingPointError('Student backward missing or nonfinite')
            gradient_tensor_count=len(gradients)
            del gradients
            pipeline_span_seconds=time.perf_counter()-start
            pipeline_seconds=pipeline_span_seconds-kd_probe_seconds
            # Images and unused feature values are absent; actual scores/DFL
            # remain intact. Shape-only features are explicit, never evidence
            # of feature KD. No model forward is repeated here.
            keep_batch={k:batch[k] for k in ('batch_idx','cls','bboxes','teacher_batch','pair_info','im_file')}
            bundle=dict(bundle_kind='real_frozen_flow_raw_v1',student=compact_raw(s),teacher=compact_raw(t),
                        reference=compact_raw(r),batch=move(keep_batch,'cpu'),strides=strides,evidence=cfg['evidence'],
                        provenance=dict(dataset=cfg['dataset'],batch_index=index,source_trace_exact=True,
                            model_files={k:str(p) for k,p in paths.items()},model_signatures=before,
                            input_uint8_normalized_once=True,student_mode='train',teacher_reference_mode='eval',
                            amp=bool(cfg['amp']),seed=20260907,optimizer_updates=0,
                            feature_values_omitted=True,feature_role='Only original logical shape/dtype used by C1 layout; zero-stride placeholder storage',
                            trace_file=f'trace_{index:02d}.json'))
            torch.save(bundle,args.output/f'raw_batch_{index:02d}.pt')
            del selected,total,native,kd,prediction,s,t,r
            student_model.zero_grad(set_to_none=True)
            # Separate instrumented replay: pool wall time includes explicit
            # sync on every call. It is NOT subtracted from the ordinary stage.
            pool_records=[]
            replay=move(bundle,device)
            def timed_pool(*pool_args,**pool_kwargs):
                # The surrounding replay owns its peak interval. Inner calls
                # synchronize for wall time but must not reset that interval.
                value,record=timed(lambda:selection.pool_relative_logits(*pool_args,**pool_kwargs),device,track_peak=False)
                pool_records.append(dict(objects=pool_args[1].shape[0],classes=pool_args[0].shape[0],anchors=pool_args[0].shape[1],**record))
                return value
            replay_selector=cloned_selector(selection,timed_pool)
            stage('separate_pool_profile_replay_selection_KD_score_grad',index,
                  lambda:selector_eval(replay_selector,classification,replay))
            write(args.output/f'pool_component_replay_{index:02d}.json',dict(scope='Separately synchronized pool replay; not additive to live pipeline',calls=pool_records))
            report['batches'].append(dict(batch=index,source_trace_exact=True,common_count=stats['common_count'],
                base_count=stats['base_count'],selected_count=stats['selected_count'],
                pipeline_serial_wall_seconds=pipeline_seconds,pool_replay_calls=len(pool_records),
                pipeline_serial_wall_excludes_independent_KD_probe=True,
                pipeline_span_including_KD_probe_seconds=pipeline_span_seconds,
                independent_KD_probe_seconds=kd_probe_seconds,independent_KD_raw_score_gradient_l2=kd_gradient_norm,
                independent_KD_raw_score_gradient_finite=kd_gradient_finite,
                pool_replay_sum_seconds=sum(x['seconds'] for x in pool_records),
                finite_student_gradients=finite_gradients,student_gradient_tensor_count=gradient_tensor_count))
            del replay,bundle,raw_batch,batch,stats
            write(args.output/'progress.json',report)
        report['model_signatures_after']={k:signature(p) for k,p in paths.items()}
        if report['model_signatures_after']!=before:raise AssertionError('Input checkpoint changed during capture')
        report['status']='COMPLETED_DIAGNOSTIC_CAPTURE_NOT_TRAINING_ADMISSION'
    except Exception as error:
        report.update(status='FAILED',error=repr(error))
        raise
    finally:
        if iterator is not None and hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        report['resources']=runtime.legacy.bound_lease_resource_record_from_environment()
        write(args.output/'receipt.json',report)
    print(json.dumps(dict(status=report['status'],output=str(args.output))))


if __name__=='__main__':main()
