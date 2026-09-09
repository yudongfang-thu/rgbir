"""Independent C1 implementation benchmark: 106 real batches, no evaluation.

Run only inside the existing project global lease. Each implementation starts
from the same original E200 configuration; no resumption of an active run.
No per-step tensor/state files. Six warmup batches precede a synchronized
100-batch wall window including loader, optimizer, AMP and ordinary statistics.
"""
import argparse, functools, importlib.util, json, math, os
from pathlib import Path
import shutil, sys, time, traceback, types

TOTAL_BATCHES=106
WARMUP_BATCHES=6
MIN_SUCCESSFUL_UPDATES=24

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def explicit(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def cloned(fn,**bindings):
    out=types.FunctionType(fn.__code__,dict(fn.__globals__,**bindings),fn.__name__,fn.__defaults__,fn.__closure__)
    out.__kwdefaults__=fn.__kwdefaults__;return out

def first_sanity_only(base):
    original=base.__call__
    def call(self,prediction,batch):
        self.sanity=self.calls==0
        try:return original(self,prediction,batch)
        finally:self.sanity=False
    return type('CadenceFirstSanityThenOriginalStatistics',(base,),{'__call__':call})

def stat(path):
    s=Path(path).stat();return dict(path=str(Path(path).resolve()),size_bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def summarize(rows,warm_boundary,final_boundary):
    if len(rows)!=TOTAL_BATCHES or [r['batch'] for r in rows]!=list(range(1,TOTAL_BATCHES+1)):
        raise ValueError('Require exactly106 consecutive full B32 batches')
    if rows[-1]['successful_updates']<MIN_SUCCESSFUL_UPDATES:raise ValueError('Fewer than24 actual successful updates')
    if any(r['successful_updates']+r['amp_skips']!=r['attempts'] for r in rows):raise ValueError('AMP counter mismatch')
    if warm_boundary is None or final_boundary is None:raise ValueError('Synchronized boundaries absent')
    seconds=final_boundary-warm_boundary
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('Invalid timing')
    return dict(warmup_batches=6,timed_batches=100,timed_images=3200,continuous_seconds=seconds,
                seconds_per_batch=seconds/100,images_per_second=3200/seconds,
                successful_updates_in_window=rows[-1]['successful_updates']-rows[5]['successful_updates'],
                includes_loader_optimizer_amp_native_statistics=True,includes_batch100_statistics=True,
                excludes_setup_first6_warmup_and_final_checkpoint=True,full_epoch_completed=False)

def run(args):
    import yaml
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.arm=='N':
        cfg=dict(cfg,arm='N',classification_coefficient=0.)
    expected=dict(arm=args.arm,source='paired',seed=42,epochs=200,batch=32,nbs=64,workers=4,imgsz=640,
                  amp=True,classification_coefficient=0. if args.arm=='N' else 0.09227393550836771,localization_coefficient=0.)
    for k,v in expected.items():
        if cfg.get(k)!=v:raise ValueError('Unexpected frozen baseline configuration: '+k)
    if cfg.get('log_every_batches')!=100:raise ValueError('Original100-batch log cadence required')
    if args.mode=='original' and args.candidate_source is not None:raise ValueError('Original mode cannot load candidate')
    if args.mode=='candidate' and args.candidate_source is None:raise ValueError('Explicit candidate factory needed')
    ref=args.reference_dir.resolve();sys.path.insert(0,str(ref))
    import torch, runtime, train_independent as training, independent_criterion as criterion
    import selection_adapter as selection, classification_logit as classification
    for m in (runtime,training,criterion,selection,classification):
        if Path(m.__file__).resolve().parent!=ref:raise ValueError('Wrong pinned module '+m.__name__)
    training.validate_execution(cfg,formal=False)
    lease=runtime.legacy.require_bound_lease_from_environment()
    if len(lease['gpus'])!=1:raise ValueError('One real inherited project globallease required')
    output=args.output.resolve()
    project_root=Path(os.environ.get('RGBIR90_PROJECT_ROOT', '/mnt/dataX/ydf/projects/RGBT_campaign_90')).resolve()
    if os.name!='posix' or project_root not in output.parents:
        raise ValueError('Benchmark output must be a new data-disk attempt')
    if output.exists():raise FileExistsError(output)
    base=criterion.IndependentCriterion;extra=[]
    if args.mode=='candidate':
        path=args.candidate_source.resolve();sys.path.insert(0,str(path.parent))
        candidate=explicit(path,'_cadence106_candidate')
        api=candidate.make_api(selection,classification)
        base=candidate.make_criterion_type(criterion,api)
        if not isinstance(base,type) or not issubclass(base,criterion.IndependentCriterion):raise TypeError('Bad factory class')
        extra=list(path.parent.glob('*.py'))
    private=first_sanity_only(base)
    build=cloned(training.build_trainer,IndependentCriterion=private)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.config,output/'benchmark_config.yaml')
    (output/'benchmark_effective_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    source_paths=[Path(__file__).resolve(),args.config.resolve(),*[Path(m.__file__).resolve() for m in (runtime,training,criterion,selection,classification)],*extra]
    sources=output/'sources';sources.mkdir();source_rows=[]
    for i,path in enumerate(sorted(set(source_paths),key=str)):
        raw=path.read_bytes();dest=sources/f'{i:02d}_{path.name}';dest.write_bytes(raw)
        source_rows.append(dict(source=str(path),copy=str(dest),size_bytes=len(raw),content_binding='preserved_source_copy_no_new_hash'))
    write_new(output/'source_manifest.json',dict(files=source_rows,weights_hashed=False,new_hashes_computed=False,server='90'))
    before={k:stat(cfg[k]) for k in ('model','teacher','reference')}
    state=dict(batch=0,optimizer_calls=0,warm_boundary=None,final_boundary=None,last_batch=None,rows=[])
    trainer=None;started=time.perf_counter();torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    try:
        # max_steps=None suppresses initial_student.pt/first_batch.pt and per-step
        # canary sanity. The explicit callback bounds this performance-only run.
        trainer=build(cfg,args.config,output,arm=args.arm,max_steps=None,historical=False)
        preprocess=trainer.preprocess_batch
        @functools.wraps(preprocess)
        def pre(batch):
            state['batch']+=1
            if state['batch']>TOTAL_BATCHES or len(batch['im_file'])!=32:raise RuntimeError('Full-batch bound violated')
            state['last_batch']=dict(files=list(batch['im_file']),actual_B=32,
                rgb_gt_count=int(batch['cls'].shape[0]),ir_gt_count=int(batch['strong_cls'].shape[0]))
            return preprocess(batch)
        trainer.preprocess_batch=pre
        def on_start(t):
            if type(t.criterion_ref) is not private or not bool(t.amp) or t.train_loader.num_workers!=4:
                raise AssertionError('Criterion/AMP/loader contract changed')
            if len(t.train_loader.dataset)!=17990 or len(t.train_loader)!=563:raise AssertionError('Drone roster changed')
            original=t.optimizer.step
            @functools.wraps(original)
            def counted(*a,**kw):
                state['optimizer_calls']+=1;return original(*a,**kw)
            t.optimizer.step=counted
            torch.cuda.synchronize();state['train_start']=time.perf_counter()
        def on_end(t):
            batch=state['batch']
            if batch in (WARMUP_BATCHES,TOTAL_BATCHES):torch.cuda.synchronize()
            now=time.perf_counter()
            if batch==WARMUP_BATCHES:state['warm_boundary']=now
            if batch==TOTAL_BATCHES:state['final_boundary']=now;t.stop=True
            if t.real_updates!=state['optimizer_calls']:raise AssertionError('Actual optimizer call count differs')
            c=t.criterion_ref;s=c.last_stats
            state['rows'].append(dict(state['last_batch'],batch=batch,epoch=int(t.epoch),
                successful_updates=t.real_updates,attempts=t.update_attempts,amp_skips=t.skipped_amp_updates,
                ema_updates=t.ema.updates,accumulate=int(t.accumulate),
                native_total=s.get('native_total'),loss_unweighted=s.get('loss_unweighted'),total_loss=s.get('total_loss'),
                selected_count=s.get('selected_count'),base_count=s.get('base_count'),
                full_diagnostics_collected=s.get('full_diagnostics_collected'),
                full_diagnostics_batches=getattr(c,'full_diagnostics_batches',None),
                thin_learning_batches=getattr(c,'thin_learning_batches',None),
                fallback_batches=getattr(c,'fallback_batches',None),end_cpu_submission_timestamp=now))
        trainer.add_callback('on_train_start',on_start);trainer.add_callback('on_train_batch_end',on_end)
        trainer.train();torch.cuda.synchronize();returned=time.perf_counter()
        summary=summarize(state['rows'],state['warm_boundary'],state['final_boundary'])
        c=trainer.criterion_ref
        if c.calls!=TOTAL_BATCHES or state['optimizer_calls']<24:raise AssertionError('Actual workload incomplete')
        logs=[json.loads(x) for x in (output/'kd_batches.jsonl').read_text().splitlines()]
        if not {1,2,3,100}.issubset({x['batch'] for x in logs}):raise AssertionError('Native first3/every100 cadence missing')
        if args.arm!='N' and (not c.gradient_checks or not any(math.isfinite(x.get('kd_score_gradient_l2',0)) and x.get('kd_score_gradient_l2',0)>0 for x in c.gradient_checks)):
            raise AssertionError('Original first finite nonzero KD gradient check absent')
        if not all(bool(torch.isfinite(v).all()) for model in (trainer.model,trainer.ema.ema) for v in model.state_dict().values() if v.is_floating_point()):
            raise FloatingPointError('Final student/EMA contains nonfinite state')
        if {k:stat(cfg[k]) for k in before}!=before:raise AssertionError('Frozen model inputs stat changed')
        candidate_counters={k:getattr(c,k,None) for k in ('thin_learning_batches','fallback_batches','full_diagnostics_batches')}
        if args.mode=='candidate' and args.arm=='C1':
            if candidate_counters['thin_learning_batches']!=TOTAL_BATCHES or candidate_counters['fallback_batches']!=0:
                raise AssertionError('Candidate silently fell back or did not execute106 times')
            if state['rows'][99]['full_diagnostics_collected'] is not True:raise AssertionError('Candidate batch100 full statistics absent')
        receipt=dict(status='CADENCE106_COMPLETED',scope='NEW_IMPLEMENTATION_PERFORMANCE_ONLY',mode=args.mode,arm=args.arm,
            candidate_source=str(args.candidate_source) if args.candidate_source else None,seed=42,
            batches=106,successful_updates=trainer.real_updates,actual_optimizer_calls=state['optimizer_calls'],
            attempts=trainer.update_attempts,amp_skips=trainer.skipped_amp_updates,ema_updates=trainer.ema.updates,
            timing=summary,whole_probe_seconds=returned-started,
            training_start_to_end_seconds=state['final_boundary']-state['train_start'],
            after_timed_window_seconds=returned-state['final_boundary'],
            ordinary_kd_log_batches=[x['batch'] for x in logs],first_gradient_checks=c.gradient_checks,
            final_student_ema_finite=True,candidate_execution=candidate_counters,batch_sequence=state['rows'],
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            resources=runtime.legacy.bound_lease_resource_record_from_environment(),
            checkpoint_is_partial_epoch_performance_artifact=True,
            no_full_epoch_claim=True,no_e200_continuation_claim=True,no_dev_or_test_evaluation=True,
            no_per_step_tensor_or_state_snapshot=True,weights_hashed=False,new_hashes_computed=False,server='90',
            limitations=['First-epoch warmup schedule and short window, not mature E200 steady state',
                'A partial first-epoch checkpoint may be emitted by native trainer after the timed window; never count it as completed epoch',
                'Numeric implementation acceptance is separate; floating-point drift need not be bitwise zero'])
        write_new(output/'cadence106_receipt.json',receipt)
    except BaseException as error:
        write_new(output/'cadence106_failure.json',dict(status='CADENCE106_FAILED',error=repr(error),traceback=traceback.format_exc(),
            batches=state['batch'],actual_optimizer_calls=state['optimizer_calls'],rows=state['rows']))
        raise
    finally:
        if trainer is not None:
            for name in ('train_loader','test_loader'):
                iterator=getattr(getattr(trainer,name,None),'iterator',None)
                if callable(getattr(iterator,'_shutdown_workers',None)):iterator._shutdown_workers()
        runtime.legacy.EvidenceCriterion=runtime.ORIGINAL_CRITERION
        runtime.legacy.DualLabelRGBIRDataset=runtime.ORIGINAL_DATASET

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['original','candidate'],required=True)
    p.add_argument('--arm',choices=['C1','N'],default='C1')
    p.add_argument('--candidate-source',type=Path);run(p.parse_args())

if __name__=='__main__':main()
