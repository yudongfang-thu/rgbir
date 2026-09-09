"""Fresh, full E200 C1 run through the accepted native trainer and fast factory.

No resume or batch-limit option. Requires the original global GPU lease. The
scientific configuration is an exact frozen per-seed copy of original C1.
"""
import argparse, hashlib, importlib.util, json, os, shutil, sys, types
from pathlib import Path

def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def clone(fn, **bindings):
    value=types.FunctionType(fn.__code__,dict(fn.__globals__,**bindings),fn.__name__,fn.__defaults__,fn.__closure__)
    value.__kwdefaults__=fn.__kwdefaults__
    return value

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_inputs(plan_path, seed, output):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if seed not in (0,42,123):raise ValueError('Unregistered seed')
    cell=plan['cells'][str(seed)]
    if Path(output).resolve()!=Path(cell['output']).resolve():raise ValueError('Unregistered output attempt')
    if Path(output).exists():raise FileExistsError('Preserve original attempt: '+str(output))
    for row in plan['frozen_source_files']:
        if sha(row['path'])!=row['sha256']:raise ValueError('Changed source: '+row['path'])
    if sha(cell['config'])!=cell['config_sha256']:raise ValueError('Frozen per-seed config changed')
    for row in plan['admission_inputs']:
        if sha(row['path'])!=row['sha256']:raise ValueError('Changed acceptance input: '+row['path'])
    candidate=json.loads(Path(plan['cadence_receipt']).read_text())
    if candidate.get('status')!='CADENCE106_COMPLETED' or candidate.get('mode')!='candidate' or candidate.get('actual_optimizer_calls',0)<24:
        raise ValueError('New-implementation actual-update/resource evidence missing')
    if not candidate.get('final_student_ema_finite') or candidate['candidate_execution']['fallback_batches']!=0:
        raise ValueError('Accepted candidate execution evidence changed')
    return plan,cell

def run(args):
    import yaml
    plan,cell=verify_inputs(args.plan,args.seed,args.output)
    cfg=yaml.safe_load(Path(cell['config']).read_text())
    if cfg.get('seed')!=args.seed or cfg.get('arm')!='C1' or cfg.get('source')!='paired':raise ValueError('Cell identity differs')
    if cfg.get('epochs')!=200 or cfg.get('batch')!=32 or cfg.get('workers')!=4 or cfg.get('log_every_batches')!=100:
        raise ValueError('Full frozen E200 recipe required')
    if cfg.get('classification_coefficient')!=0.09227393550836771 or cfg.get('localization_coefficient')!=0.:
        raise ValueError('Frozen loss dose differs')
    ref=Path(plan['reference_dir']).resolve()
    sys.path.insert(0,str(ref))
    import runtime,train_independent as training,independent_criterion as criterion
    import selection_adapter as selection,classification_logit as classification
    for module in (runtime,training,criterion,selection,classification):
        if Path(module.__file__).resolve().parent!=ref:raise ValueError('Wrong pinned module: '+module.__name__)
    # Original readiness is retained as scientific configuration/calibration
    # evidence. New implementation eligibility is separately bound above.
    training.validate_execution(cfg,formal=True)
    lease=runtime.legacy.require_bound_lease_from_environment()
    if len(lease['gpus'])!=1:raise ValueError('Exactly one original global lease GPU required')
    if os.name!='posix' or not str(args.output.resolve()).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('Formal outputs must stay on the project data disk')
    spec=importlib.util.spec_from_file_location('_c1_fast_production_factory',plan['candidate_source'])
    candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
    api=candidate.make_api(selection,classification)
    fast_type=candidate.make_criterion_type(criterion,api)
    if not issubclass(fast_type,criterion.IndependentCriterion):raise TypeError('Unexpected criterion factory')
    fast_build=clone(training.build_trainer,IndependentCriterion=fast_type)
    holder={}
    def build(bound_cfg, config_path, output, arm=None, max_steps=None, historical=False):
        if max_steps is not None or historical:raise ValueError('Formal fresh E200 only')
        snap=Path(output)/'fastpath_implementation_snapshot';snap.mkdir(exist_ok=False)
        paths=[Path(__file__).resolve(),Path(args.plan).resolve(),Path(cell['config']),Path(plan['candidate_source'])]
        records=[]
        for i,p in enumerate(paths):
            target=snap/f'{i:02d}_{p.name}';shutil.copyfile(p,target)
            records.append(dict(source=str(p),copy=str(target),sha256=sha(target)))
        write_new(Path(output)/'fastpath_launch_manifest.json',dict(
            implementation_id=plan['implementation_id'],scientific_method_id=cfg['method_id'],
            fresh_common_initialization=True,resume=False,old_run_continuation=False,
            initialization_path=cfg['model'],seed=args.seed,epochs=200,
            candidate_version=candidate.VERSION,candidate_source=plan['candidate_source'],
            candidate_sha256=sha(plan['candidate_source']),original_reference=str(ref),
            old_readiness_scope='original science/calibration; not new implementation acceptance',
            new_implementation_admission=plan['admission_inputs'],sources=records,
            cadence_benchmark_not_an_epoch=True,observer_schedule_unchanged=True,
            full_diagnostics_schedule='first3, every100, original sanity and fixed shared-gradient observer',
            bound_lease_id=lease.get('lease_id'),physical_gpu_ids=lease['gpus']))
        trainer=fast_build(bound_cfg,config_path,output,arm=arm,max_steps=None,historical=False)
        holder['trainer']=trainer
        def on_start(t):
            if type(t.criterion_ref) is not fast_type:raise AssertionError('Fast factory not executed')
            if getattr(t.args,'resume',False):raise AssertionError('Resume is prohibited')
            if len(t.train_loader.dataset)!=17990 or len(t.train_loader)!=563 or t.train_loader.num_workers!=4:
                raise AssertionError('Original full training population changed')
            write_new(Path(output)/'fresh_initialization_receipt.json',dict(
                status='FRESH_NATIVE_BUILDER_STARTED',seed=args.seed,model=cfg['model'],
                epochs=t.epochs,resume=False,train_images=len(t.train_loader.dataset),
                batches_per_epoch=len(t.train_loader),implementation_id=plan['implementation_id'],
                criterion_class=type(t.criterion_ref).__name__,optimizer_updates=t.real_updates,
                no_old_student_checkpoint_loaded=True,claim='same initializer path and original seeded builder; not old-trajectory continuation'))
        trainer.add_callback('on_train_start',on_start)
        return trainer
    native_args=argparse.Namespace(config=Path(cell['config']),output=args.output,arm='C1',source='paired',seed=args.seed,max_steps=None)
    try:
        clone(training.run,build_trainer=build)(native_args)
        t=holder['trainer'];c=t.criterion_ref
        if t.epoch+1!=200:raise RuntimeError('E200 terminal endpoint missing')
        receipt=dict(status='FAST_C1_E200_COMPLETED',implementation_id=plan['implementation_id'],
            scientific_method_id=cfg['method_id'],seed=args.seed,fresh_initialization=True,resume=False,
            last_epoch=200,batches=c.calls,optimizer_updates=t.real_updates,
            thin_learning_batches=getattr(c,'thin_learning_batches',None),
            full_diagnostics_batches=getattr(c,'full_diagnostics_batches',None),
            fallback_batches=getattr(c,'fallback_batches',None),
            original_training_receipt=str(args.output/'completion_receipt.json'),
            candidate_source=plan['candidate_source'],candidate_sha256=sha(plan['candidate_source']),
            checkpoint=str(args.output/'weights/last.pt'),official_test_accessed=False,
            independent_full_dev_evaluation_pending=True,old_runs_modified=False)
        write_new(args.output/'fastpath_completion_receipt.json',receipt)
        runtime.legacy.emit_bound_run_receipt(run_dir=args.output/'fastpath_run_evidence',
            method_identity=cfg['method_identity'],dataset=cfg['dataset'],data_role='development_train',
            seed=args.seed,run_kind='train',trainers=[Path(__file__),ref/'train_independent.py'],
            losses=[Path(plan['candidate_source']),ref/'independent_criterion.py',ref/'selection_adapter.py',ref/'classification_logit.py'],
            configs=[Path(cell['config']),args.plan],split_rosters=[Path(cfg['paths']['paired_train_mapping'])],
            metric_files=[args.output/'completion_receipt.json',args.output/'fastpath_completion_receipt.json'],
            environment=dict(torch=cfg['torch_version'],ultralytics=cfg['ultralytics_version']),
            inputs=dict(implementation_id=plan['implementation_id'],fresh_common_initialization=True,
                initial_weights=cfg['model'],candidate_sha256=sha(plan['candidate_source']),old_run_continuation=False))
    except BaseException as error:
        if args.output.is_dir() and not (args.output/'fastpath_failure_receipt.json').exists():
            write_new(args.output/'fastpath_failure_receipt.json',dict(status='FAILED',error=repr(error),seed=args.seed,old_runs_modified=False))
        raise
    finally:
        runtime.legacy.EvidenceCriterion=runtime.ORIGINAL_CRITERION
        runtime.legacy.DualLabelRGBIRDataset=runtime.ORIGINAL_DATASET

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(0,42,123),required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
