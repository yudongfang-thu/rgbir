"""Original N/C0 loss, generic pretrain, fixed2048 E8; no C1 fast-path migration."""
import argparse
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
import types
import yaml
from screen_common import (SCOPE,ENDPOINT,Deadline,ExecutionDeadline,FlowAudit,canary_decision,
    copy_sources,data_output,load_config,read,stat,write_new,append_json,validate_amp_prior,
    setup_with_validated_amp)

def cloned(fn,**bindings):
    f=types.FunctionType(fn.__code__,dict(fn.__globals__,**bindings),fn.__name__,fn.__defaults__,fn.__closure__)
    f.__kwdefaults__=fn.__kwdefaults__;return f

def private_builder(runtime,training,criterion,cfg,canary):
    # The accepted train_independent.build_trainer still owns arm mapping and
    # historical=False. Only frozen data validation and observational sanity are
    # private bindings; all N/C0 loss arithmetic is the original superclass.
    legacy=runtime.legacy
    class ScreenCriterion(criterion.IndependentCriterion):
        def __call__(self,prediction,batch):
            active=any(r.get('kd_score_gradient_l2',0)>0 for r in self.gradient_checks)
            self.sanity=bool(canary and (self.calls==0 or (cfg['arm']=='C0' and not active)))
            return super().__call__(prediction,batch)
    def frozen(path,expected_data,names):
        key=('privileged_data_yaml' if str(path)==cfg['teacher'] else
            'student_data_yaml' if str(path)==cfg['reference'] else None)
        if key is None or str(expected_data)!=cfg['paths'][key]:raise ValueError('Unexpected frozen data binding')
        return legacy.load_frozen(path,cfg['auxiliary_data_identity'][key],names)
    old_build=cloned(legacy.build_trainer,EvidenceCriterion=ScreenCriterion,
        DualLabelRGBIRDataset=runtime.TrackedDualLabelRGBIRDataset,load_frozen=frozen)
    private_legacy=types.SimpleNamespace(**dict(vars(legacy),build_trainer=old_build))
    return cloned(training.build_trainer,legacy=private_legacy,IndependentCriterion=ScreenCriterion)

def check_canary(path,cfg,config,reference):
    if path is None:raise ValueError('Actual same-arm new canary required')
    c=read(path)
    if (c.get('status')!='SUBSET_SCREEN_CANARY_COMPLETED' or c.get('scope')!=SCOPE
        or c.get('endpoint')!=ENDPOINT or c.get('arm')!=cfg['arm']
        or canary_decision(c['successful_updates'],c['batches'],c['attempts'])!='completed'):
        raise ValueError('Corresponding subset canary incomplete')
    if Path(c['configuration']).read_bytes()!=Path(config).read_bytes():raise ValueError('Canary config bytes differ')
    flow=read(c['flow_prefix_receipt']);init=read(c['initialization_receipt'])
    if (flow.get('status')!='PASS' or flow.get('prefix_batches')!=30 or flow.get('exact_pixels_and_labels') is not True
        or Path(flow['reference_dir']).resolve()!=Path(reference).resolve() or init.get('status')!='PASS'):
        raise ValueError('Canary flow/initialization acceptance missing')
    return c

def run(args):
    deadline=Deadline(args.wall_seconds);cfg=load_config(args.config);output=data_output(args.output)
    reference=data_output(args.flow_reference_dir)
    if output.exists():raise FileExistsError(output)
    if args.write_flow_reference and not (args.canary and cfg['arm']=='N'):raise ValueError('Only N canary may write reference')
    if not args.canary:check_canary(args.canary_receipt,cfg,args.config,reference)
    ref=args.reference_dir.resolve();sys.path.insert(0,str(ref))
    import torch
    import runtime
    import train_independent as training
    import independent_criterion as criterion
    import ultralytics.engine.trainer as native_trainer
    for m,n in [(runtime,'runtime.py'),(training,'train_independent.py'),(criterion,'independent_criterion.py')]:
        if Path(m.__file__).resolve()!=ref/n:raise RuntimeError('Wrong pinned module: '+n)
    training.validate_execution(cfg,formal=False);legacy=runtime.legacy
    if len(legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('Sole global GPU lease required')
    amp_prior=validate_amp_prior(cfg,args.amp_prior)
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(args.config,output/'short_screen_config.yaml')
    source_paths=list(Path(__file__).parent.glob('*.py'))+list(ref.rglob('*.py'))+[args.config,args.amp_prior]
    source_paths += [Path(v) for v in cfg['paths'].values()]+[Path(v) for v in cfg['auxiliary_data_identity'].values()]
    for key in ('student_data_yaml','privileged_data_yaml'):
        d=yaml.safe_load(Path(cfg['paths'][key]).read_text(encoding='utf-8'))
        roster=Path(d['train']);roster=roster if roster.is_absolute() else Path(d['path'])/roster
        if len([r for r in roster.read_text(encoding='utf-8').splitlines() if r.strip()])!=2048:raise ValueError('Train roster not2048')
        source_paths.append(roster)
    source_paths += [Path(p) for p in legacy.implementation_files(legacy.DetectionTrainer)]
    copy_sources(output,[p for p in source_paths if '__pycache__' not in p.parts])
    before={k:stat(cfg[k]) for k in ('model','teacher','reference')}
    torch.set_num_threads(4);trainer=None;flow=None;times=[];timing_state={};setup_record={};bn_initial={};cleaned=False
    def sync():torch.cuda.synchronize()
    started=deadline.started
    def cleanup():
        nonlocal cleaned
        if trainer is not None and not cleaned:
            for name in ('train_loader','test_loader'):
                iterator=getattr(getattr(trainer,name,None),'iterator',None)
                if callable(getattr(iterator,'_shutdown_workers',None)):iterator._shutdown_workers()
            cleaned=True
    try:
        with deadline.armed():
            flow=FlowAudit(reference,args.write_flow_reference,cfg,output,torch)
            build=private_builder(runtime,training,criterion,cfg,args.canary)
            # Exact accepted entry call; max_steps=None avoids an early24-only
            # stop and leaves this wrapper's >=30batch/<=48attempt guard in charge.
            trainer=build(cfg,args.config,output,arm=cfg['arm'],max_steps=None,historical=False)
            original_setup=trainer._setup_train
            def setup():
                deadline.check();t=time.perf_counter()
                bypass=setup_with_validated_amp(trainer,original_setup,native_trainer)
                if trainer.optimizer.state or trainer.ema.updates!=0:raise AssertionError('Optimizer/EMA not fresh')
                current=trainer.model.state_dict();ema=trainer.ema.ema.state_dict()
                if current.keys()!=ema.keys() or any(not torch.equal(current[k],ema[k]) for k in current):raise AssertionError('Fresh EMA differs')
                ready=read(output/'runtime_ready.json')
                if tuple(ready[k] for k in ('train_images','val_images','train_batches','batch','workers'))!=(2048,1469,64,32,4):
                    raise ValueError('Actual population/batch/worker mismatch')
                bn=[(n,m) for n,m in trainer.model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
                if not bn or any(not m.track_running_stats or not m.training for _,m in bn):raise ValueError('BN not normal training')
                for name,m in bn:
                    for key in ('running_mean','running_var','num_batches_tracked'):
                        bn_initial[name+'.'+key]=getattr(m,key).detach().cpu().clone()
                init=flow.initial(trainer.model,cfg)
                init.update(initialization=before,fresh_optimizer=True,fresh_ema=True,ema_updates=0,
                    bn_running_statistics='normal_training',batchnorm_modules=len(bn),actual_amp=True)
                write_new(output/'initialization_check.json',init)
                setup_record.update(seconds=time.perf_counter()-started,native_setup_seconds=time.perf_counter()-t,
                    bypass=bypass,amp_prior=amp_prior,
                    initial_state_audit_seconds=flow.initial_seconds,batchnorm_modules=len(bn))
                write_new(output/'setup_receipt.json',setup_record)
                sync();timing_state['previous_end']=time.perf_counter();deadline.check()
            trainer._setup_train=setup
            original_preprocess=trainer.preprocess_batch
            def preprocess(batch):
                deadline.check();index=trainer.batch_visits+1
                if args.canary and trainer.update_attempts>=48:raise RuntimeError('Canary attempt cap reached before next batch')
                # Includes full RGB/IR augmented uint8 tensors and all dual GT.
                flow.batch(batch,index)
                return original_preprocess(batch)
            trainer.preprocess_batch=preprocess
            def batch_end(t):
                sync();now=time.perf_counter();index=t.batch_visits
                elapsed=now-timing_state['previous_end'];audit=flow.batch_seconds.get(index,0.)
                if elapsed<=audit:raise RuntimeError('Invalid full-batch timing interval')
                row=dict(batch=index,seconds=elapsed,flow_audit_seconds=audit,
                    excluding_flow_seconds=elapsed-audit,optimizer_attempts=t.update_attempts,
                    successful_updates=t.real_updates,amp_skips=t.skipped_amp_updates)
                times.append(row);append_json(output/'batch_timing.jsonl',row)
                if args.canary:
                    if canary_decision(t.real_updates,index,t.update_attempts)=='completed':t.stop=True
                deadline.check();timing_state['previous_end']=time.perf_counter()
            trainer.add_callback('on_train_batch_end',batch_end)
            torch.cuda.reset_peak_memory_stats();trainer.train();deadline.check()
            if args.canary:
                if canary_decision(trainer.real_updates,trainer.criterion_ref.calls,trainer.update_attempts)!='completed':
                    raise RuntimeError('Canary did not satisfy full success/batch contract')
                checks=trainer.criterion_ref.gradient_checks
                if not checks or not all(r.get('weight0_exact_loss_gradient') is True for r in checks):raise RuntimeError('Native weight0 identity check absent')
                if cfg['arm']=='C0' and not any(r.get('kd_score_gradient_l2',0)>0 for r in checks):raise RuntimeError('No actual nonzero C0 gradient')
                if any(not math.isfinite(float(v)) for r in checks for k,v in r.items() if 'gradient_l2' in k):raise RuntimeError('Nonfinite actual gradient evidence')
            elif trainer.epoch+1!=8 or trainer.criterion_ref.calls!=512:
                raise RuntimeError('E8 incomplete: expected exactly8 epochs/512 batches')
            if len(times)!=trainer.criterion_ref.calls:raise RuntimeError('Every real training batch must have full-stage timing')
            flow_result=flow.finish()
            if {k:stat(cfg[k]) for k in before}!=before:raise ValueError('Initial input checkpoint stat changed')
            for m in (trainer.model,trainer.ema.ema):
                if any(not bool(torch.isfinite(v).all()) for v in m.state_dict().values() if v.is_floating_point()):raise FloatingPointError('Nonfinite student/EMA')
            state=trainer.model.state_dict()
            bn_changed=[k for k,v in bn_initial.items() if not torch.equal(v,state[k].detach().cpu())]
            result=dict(status='SUBSET_SCREEN_CANARY_COMPLETED' if args.canary else 'SUBSET_SCREEN_TRAINING_COMPLETED',
                scope=SCOPE,endpoint=ENDPOINT,single_seed=True,arm=cfg['arm'],seed=42,dataset='dronevehicle',
                epochs_configured=8,last_epoch=trainer.epoch+1,independent_lr_horizon=8,expected_train_images=2048,
                configuration=str(output/'short_screen_config.yaml'),config_copy=str(output/'short_screen_config.yaml'),
                initialization=before,model=cfg['model'],teacher=cfg['teacher'],reference=cfg['reference'],
                classification_coefficient=cfg['classification_coefficient'],localization_coefficient=0.,
                training_subset_identity=cfg['paths'],bn_running_statistics='normal_training',
                bn_running_buffers_changed=len(bn_changed),fresh_optimizer=True,fresh_ema=True,actual_amp=True,
                successful_updates=trainer.real_updates,optimizer_updates=trainer.real_updates,
                attempts=trainer.update_attempts,amp_skips=trainer.skipped_amp_updates,ema_updates=trainer.ema.updates,
                batches=trainer.criterion_ref.calls,selected_objects=trainer.criterion_ref.selected_total,
                gradient_checks=trainer.criterion_ref.gradient_checks,checkpoint=stat(output/'weights/last.pt'),
                flow_prefix_receipt=str(output/'flow_check.json'),initialization_receipt=str(output/'initialization_check.json'),
                first30_pixels_labels_exact=True,flow_reference=str(reference),setup_seconds=setup_record['seconds'],
                flow_audit_seconds=flow.seconds,initial_state_audit_seconds=flow.initial_seconds,
                batch_duration_seconds=[r['seconds'] for r in times],
                normal_cadence_batch_seconds=[r['excluding_flow_seconds'] for r in times],
                warmup_excluded_batch_seconds=[r['excluding_flow_seconds'] for r in times if r['batch']>6],
                cadence_scope='synchronized_complete_batch_intervals_including_loader_S_T_R_loss_backward_optimizer; flowIO_subtracted; diagnostic_not_production_epoch',
                seconds=time.perf_counter()-started,wall_limit_seconds=args.wall_seconds,
                resources=legacy.bound_lease_resource_record_from_environment(),
                gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
                canary_receipt=stat(args.canary_receipt) if args.canary_receipt else None,
                nonzero_kd_gradient=any(r.get('kd_score_gradient_l2',0)>0 for r in trainer.criterion_ref.gradient_checks),
                formal_e200_complete=False,formal_paper_gain_claim=False,new_hash_computed=False,official_test_accessed=False)
            cleanup();deadline.check();result['loader_workers_cleaned']=True
            result['seconds']=time.perf_counter()-started
            deadline.check();write_new(output/('canary.json' if args.canary else 'short_training_receipt.json'),result)
            print(result['status'],cfg['arm'],result['seconds'],flush=True)
    except BaseException as error:
        write_new(output/'short_training_failure.json',dict(status='SUBSET_SCREEN_INCOMPLETE' if isinstance(error,ExecutionDeadline) else 'SUBSET_SCREEN_FAILED',
            scope=SCOPE,arm=cfg['arm'],error=repr(error),traceback=traceback.format_exc(),
            seconds=time.perf_counter()-started,batches=getattr(trainer,'batch_visits',0),
            optimizer_updates=getattr(trainer,'real_updates',0),new_hash_computed=False,official_test_accessed=False))
        raise
    finally:
        cleanup()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir','config','output','flow-reference-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--amp-prior',type=Path,default=Path(__file__).with_name('validated_amp_prior.json'))
    p.add_argument('--wall-seconds',type=float,required=True);p.add_argument('--canary',action='store_true')
    p.add_argument('--write-flow-reference',action='store_true');p.add_argument('--canary-receipt',type=Path)
    run(p.parse_args())
if __name__=='__main__':main()
