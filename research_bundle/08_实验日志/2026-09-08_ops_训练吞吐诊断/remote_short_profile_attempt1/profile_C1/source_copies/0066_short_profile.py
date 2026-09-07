"""Authorized E20-recipe throughput probe: 24 updates, actual statistics cadence.

This is a separate performance diagnostic, not E20 training or equivalence
admission. Root supplies one existing globallease; this entry never queues jobs.
No per-step tensor/state snapshots, dev/AP, hash or formal completion receipts.
"""
import argparse
import functools
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

from screen_common import copy_sources,data_output,load_config,stat,write_new
from train_short_screen import candidate_builder,cloned

SUCCESSFUL_UPDATES=24
MAX_ATTEMPTS=96
WARMUP_BATCHES=6


class ContinuousWindow:
    """CPU-only bookkeeping. Boundary timestamps must follow CUDA synchronization.

    Intermediate timestamps are CPU submission boundaries, not isolated GPU
    batch latency. Their differences include the next loader fetch and callback
    work; only the complete start/end synchronized span is authoritative.
    """
    def __init__(self):
        self.rows=[];self.warm_end=None;self.warm_updates=None;self.final_end=None

    def add(self,batch,now,updates,attempts,skips,details):
        if batch!=len(self.rows)+1 or updates+skips!=attempts:raise AssertionError('Counter/sequence mismatch')
        if not math.isfinite(now) or self.rows and now<self.rows[-1]['end_perf_counter']:
            raise ValueError('Invalid monotonic timing')
        if attempts>MAX_ATTEMPTS or updates>SUCCESSFUL_UPDATES:raise RuntimeError('Bounded update budget exceeded')
        row=dict(details,batch=batch,end_perf_counter=now,successful_updates=updates,attempts=attempts,amp_skips=skips,
                 timing_warmup=batch<=WARMUP_BATCHES)
        self.rows.append(row)
        if batch==WARMUP_BATCHES:self.warm_end=now;self.warm_updates=updates
        if updates==SUCCESSFUL_UPDATES:self.final_end=now

    def summary(self):
        if self.warm_end is None or self.final_end is None or len(self.rows)<=WARMUP_BATCHES:
            raise ValueError('Complete synchronized post-warmup window required')
        seconds=self.final_end-self.warm_end
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('Invalid continuous elapsed time')
        batches=len(self.rows)-WARMUP_BATCHES
        updates=SUCCESSFUL_UPDATES-self.warm_updates
        return dict(warmup_batches=WARMUP_BATCHES,post_warmup_batches=batches,
            post_warmup_successful_updates=updates,continuous_post_warmup_seconds=seconds,
            seconds_per_batch=seconds/batches,images_per_second=batches*32/seconds,
            interval_includes_between_batch_loader_wait=True,
            per_batch_timestamp_deltas_are_not_isolated_gpu_latencies=True,
            periodic_100_batch_event_observed=any(r['batch']%100==0 for r in self.rows))


def first_composition_only_type(base):
    """Keep the first original sanity/composition check, then native cadence.

    max_steps still controls the existing real-update stop. No learning loss,
    selection rule, shared-epoch observer or log_every_batches rule changes.
    """
    call=base.__call__
    def profiled_call(self,prediction,batch):
        self.sanity=self.calls==0
        try:return call(self,prediction,batch)
        finally:self.sanity=False
    return type('FirstCompositionThenNativeCadenceCriterion',(base,),{'__call__':profiled_call})


def candidate_counters(arm,batches,criterion):
    if arm!='C1':return dict(applicable=False)
    counts={key:getattr(criterion,key,None) for key in ('thin_learning_batches','fallback_batches','full_diagnostics_batches')}
    if counts['thin_learning_batches']!=batches or counts['fallback_batches']!=0:
        raise AssertionError('C1 did not execute the thin path on every profiled batch: '+repr(counts))
    if type(counts['full_diagnostics_batches']) is not int or not 1<=counts['full_diagnostics_batches']<=batches:
        raise AssertionError('C1 full-diagnostics cadence counter missing')
    return dict(applicable=True,**counts)


def run(args):
    cfg=load_config(args.config)
    if cfg['arm']=='C1':
        if args.candidate_source is None:raise ValueError('C1 requires explicit accepted candidate source for this probe')
        candidate=dict(kind='criterion_factory',source=str(args.candidate_source.resolve()))
    else:
        if args.candidate_source is not None:raise ValueError('N/C0 must use the unmodified criterion path')
        candidate=dict(kind='original')
    ref=args.reference_dir.resolve();sys.path.insert(0,str(ref))
    import torch
    import runtime
    import train_independent as training
    import independent_criterion as criterion
    import selection_adapter as selection
    import classification_logit as classification
    for module,name in [(runtime,'runtime.py'),(training,'train_independent.py'),(criterion,'independent_criterion.py'),
                         (selection,'selection_adapter.py'),(classification,'classification_logit.py')]:
        if Path(module.__file__).resolve()!=ref/name:raise RuntimeError('Wrong pinned module: '+name)
    training.validate_execution(cfg,formal=False)
    if len(runtime.legacy.require_bound_lease_from_environment()['gpus'])!=1:raise RuntimeError('One inherited globallease required')
    output=data_output(args.output)
    if output.exists():raise FileExistsError(output)
    build,extra=candidate_builder(training,criterion,selection,candidate,classification)
    base=build.__globals__['IndependentCriterion']
    profile_type=first_composition_only_type(base)
    build=cloned(build,IndependentCriterion=profile_type)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.config,output/'profile_config.yaml')
    sources=list(Path(__file__).parent.glob('*.py'))+list(ref.rglob('*.py'))+extra+[args.config]
    sources += [Path(p) for p in runtime.legacy.implementation_files(runtime.legacy.DetectionTrainer)]
    copy_sources(output,[p for p in sources if '__pycache__' not in p.parts])
    before_inputs={key:stat(cfg[key]) for key in ('model','teacher','reference')}
    torch.set_num_threads(4)
    trainer=None;window=ContinuousWindow();state=dict(batch=0,optimizer_calls=0,last_batch=None)
    started=time.perf_counter()
    try:
        trainer=build(cfg,args.config,output,arm=cfg['arm'],max_steps=SUCCESSFUL_UPDATES,historical=False)
        original_preprocess=trainer.preprocess_batch
        @functools.wraps(original_preprocess)
        def preprocess(batch):
            state['batch']+=1
            if state['batch']>MAX_ATTEMPTS*4 or len(batch['im_file'])!=32:raise RuntimeError('Bounded complete B32 batches required')
            state['last_batch']=dict(files=list(batch['im_file']),actual_B=32,
                rgb_gt_count=int(batch['cls'].shape[0]),ir_gt_count=int(batch['strong_cls'].shape[0]))
            return original_preprocess(batch)
        trainer.preprocess_batch=preprocess
        def on_start(t):
            if type(t.criterion_ref) is not profile_type or not bool(t.amp) or t.train_loader.num_workers!=4:
                raise AssertionError('Actual criterion/AMP/workers differ')
            if len(t.train_loader.dataset)!=17990 or len(t.train_loader)!=563:raise AssertionError('Full Drone training roster changed')
            original=t.optimizer.step
            @functools.wraps(original)
            def counted_step(*a,**kw):
                state['optimizer_calls']+=1
                return original(*a,**kw)
            t.optimizer.step=counted_step
            # Start/end only: no per-batch GPU synchronization is introduced.
            torch.cuda.synchronize();state['train_started']=time.perf_counter()
        def on_batch_end(t):
            if state['batch']==WARMUP_BATCHES or t.real_updates==SUCCESSFUL_UPDATES:
                torch.cuda.synchronize()
            now=time.perf_counter()
            if t.real_updates!=state['optimizer_calls']:raise AssertionError('Scalers/counters differ from actual optimizer calls')
            details=dict(state['last_batch'],epoch=int(t.epoch),accumulate=int(t.accumulate),
                ema_updates=int(t.ema.updates),lr=[float(g['lr']) for g in t.optimizer.param_groups],
                momentum=[float(g.get('momentum',0.)) for g in t.optimizer.param_groups],
                sanity_after_call=bool(t.criterion_ref.sanity),
                full_diagnostics_collected=t.criterion_ref.last_stats.get('full_diagnostics_collected'),
                thin_learning_batches=getattr(t.criterion_ref,'thin_learning_batches',None),
                fallback_batches=getattr(t.criterion_ref,'fallback_batches',None),
                full_diagnostics_batches=getattr(t.criterion_ref,'full_diagnostics_batches',None))
            window.add(state['batch'],now,t.real_updates,t.update_attempts,t.skipped_amp_updates,details)
        trainer.add_callback('on_train_start',on_start)
        trainer.add_callback('on_train_batch_end',on_batch_end)
        torch.cuda.reset_peak_memory_stats()
        trainer.train()
        if trainer.real_updates!=SUCCESSFUL_UPDATES or state['optimizer_calls']!=SUCCESSFUL_UPDATES:
            raise AssertionError('Exactly24 successful real optimizer calls required')
        checks=trainer.criterion_ref.gradient_checks
        if not checks or checks[0]['batch']!=1:raise AssertionError('First original composition check missing')
        if cfg['arm']!='N' and not any(math.isfinite(r.get('kd_score_gradient_l2',0)) and r.get('kd_score_gradient_l2',0)>0 for r in checks):
            raise AssertionError('No finite nonzero KD score gradient in original first check')
        if {key:stat(cfg[key]) for key in before_inputs}!=before_inputs:raise AssertionError('Input model stat changed')
        candidate_execution=candidate_counters(cfg['arm'],state['batch'],trainer.criterion_ref)
        summary=window.summary()
        receipt=dict(status='SHORT_SCREEN_THROUGHPUT_PROFILED',scope='PERFORMANCE_ONLY',arm=cfg['arm'],seed=42,
            epochs_configured=20,full_epochs_completed=0,successful_updates=trainer.real_updates,
            actual_optimizer_calls=state['optimizer_calls'],attempts=trainer.update_attempts,
            amp_skips=trainer.skipped_amp_updates,ema_updates=trainer.ema.updates,batches=state['batch'],
            classification_coefficient=cfg['classification_coefficient'],candidate=candidate,
            candidate_execution=candidate_execution,
            first_composition_check=checks,statistics_cadence='first original composition check; thereafter sanity=False, original first3/every100/shared-epoch observer retained',
            timing=summary,batch_sequence=window.rows,whole_probe_seconds=time.perf_counter()-started,
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            resources=runtime.legacy.bound_lease_resource_record_from_environment(),
            new_hash_computed=False,official_test_accessed=False,dev_or_ap_computed=False,
            equivalence_test_repeated=False,production_or_E20_training_admitted=False,
            no_per_step_tensor_state_snapshots=True,
            limitations=['Short continuous post-warmup sample, not a completed production epoch',
                'The original native max_steps setup retains its initial_student.pt and first_batch.pt; both precede the timed warmup boundary',
                'Per100 statistics may not occur within24 updates; future frequency/cost needs budget margin',
                'Single first-epoch E20 LR is shared with E200 here; this is not an E20 learning result'])
        write_new(output/'short_profile_receipt.json',receipt)
    except BaseException as error:
        write_new(output/'short_profile_failure.json',dict(status='SHORT_PROFILE_FAILED',error=repr(error),
            traceback=traceback.format_exc(),batches=state['batch'],actual_optimizer_calls=state['optimizer_calls'],
            batch_sequence=window.rows,new_hash_computed=False,dev_or_ap_computed=False))
        raise
    finally:
        if trainer is not None:
            for name in ('train_loader','test_loader'):
                loader=getattr(trainer,name,None);iterator=getattr(loader,'iterator',None)
                if callable(getattr(iterator,'_shutdown_workers',None)):iterator._shutdown_workers()
        runtime.legacy.EvidenceCriterion=runtime.ORIGINAL_CRITERION
        runtime.legacy.DualLabelRGBIRDataset=runtime.ORIGINAL_DATASET


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-source',type=Path);run(p.parse_args())


if __name__=='__main__':main()
