"""Independent feature-gradient-matched FT3; preserved native control path."""
import argparse
from pathlib import Path
import shutil
import sys
import time
import traceback
import types
import yaml
from feature_gm_common import ENDPOINT,copy_sources,data_output,explicit,load_config,read,stat,write_new

def cloned(fn,**bindings):
    f=types.FunctionType(fn.__code__,dict(fn.__globals__,**bindings),fn.__name__,fn.__defaults__,fn.__closure__)
    f.__kwdefaults__=fn.__kwdefaults__;return f

def build_private(runtime,criterion,selection,classification,cfg,args):
    legacy=runtime.legacy
    p=Path(cfg['direction_screen']['candidate_source'])
    if p.read_bytes()!=Path(cfg['direction_screen']['approved_candidate_source']).read_bytes():
        raise ValueError('Approved selected-only source differs')
    m=explicit('_direction_selected_only',p)
    from feature_gm_criterion import make_type
    ScreenCriterion=make_type(m.make_api(selection,classification))
    def frozen(path,expected_data,names):
        key='privileged_data_yaml' if str(path)==cfg['teacher'] else 'student_data_yaml'
        if str(path) not in (cfg['teacher'],cfg['reference']) or str(expected_data)!=cfg['paths'][key]:
            raise ValueError('Unexpected frozen binding')
        return legacy.load_frozen(path,cfg['auxiliary_data_identity'][key],names)
    builder=cloned(legacy.build_trainer,EvidenceCriterion=ScreenCriterion,
        DualLabelRGBIRDataset=runtime.TrackedDualLabelRGBIRDataset,load_frozen=frozen)
    return builder,[p]

def install_bn_freeze(trainer):
    import torch
    original=trainer._model_train
    def model_train():
        original()
        for m in trainer.model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    trainer._model_train=model_train
    return model_train

def bn_state(model):
    import torch
    return {n+'.'+k:v.detach().cpu().clone() for n,m in model.named_modules()
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)
            for k,v in m.named_buffers(recurse=False) if v is not None}


def run(args):
    cfg=load_config(args.config);cfg['canary_execution']=bool(args.canary);output=data_output(args.output)
    if output.exists():raise FileExistsError(output)
    if not args.canary:
        if args.canary_receipt is None:raise ValueError('Executed corresponding canary required')
        c=read(args.canary_receipt)
        if c.get('status')!='DIRECTION_CANARY_COMPLETED' or c.get('arm')!=cfg['arm'] or c.get('successful_updates',0)<24 or c.get('scope')!='FEATURE_RELATION_GM_FT3' or c.get('dataset')!='drone' or c.get('bn_running_buffers_unchanged') is not True:
            raise ValueError('Invalid new-path canary')
        if Path(c['config_copy']).read_bytes()!=args.config.read_bytes():raise ValueError('Canary config differs')
    ref=args.reference_dir.resolve();sys.path.insert(0,str(ref))
    import torch
    import runtime
    import independent_criterion as criterion
    import selection_adapter as selection
    import classification_logit as classification
    for m,n in [(runtime,'runtime.py'),(criterion,'independent_criterion.py'),(selection,'selection_adapter.py'),(classification,'classification_logit.py')]:
        if Path(m.__file__).resolve()!=ref/n:raise ValueError('Wrong pinned module: '+n)
    legacy=runtime.legacy
    if str(torch.__version__)!=cfg['torch_version'] or str(legacy.ultralytics.__version__)!=cfg['ultralytics_version']:
        raise ValueError('Pinned runtime differs')
    if len(legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('One global GPU lease required')
    for k in ('student_data_yaml','privileged_data_yaml'):
        if 'test' in yaml.safe_load(Path(cfg['paths'][k]).read_text()):raise ValueError('Train/dev-only YAML required')
    build,extra=build_private(runtime,criterion,selection,classification,cfg,args)
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(args.config,output/'feature_gm_config.yaml')
    data_sources=[Path(cfg['paths'][k]) for k in
                  ('student_data_yaml','privileged_data_yaml','paired_train_mapping')]
    data_sources += [Path(p) for p in cfg['auxiliary_data_identity'].values()]
    for key in ('student_data_yaml','privileged_data_yaml'):
        data=yaml.safe_load(Path(cfg['paths'][key]).read_text(encoding='utf-8'))
        roster=Path(data['train'])
        if not roster.is_absolute():roster=Path(data['path'])/roster
        data_sources.append(roster)
    copy_sources(output,list(Path(__file__).parent.glob('*.py'))+list(ref.rglob('*.py'))+extra+
                 [args.config]+data_sources+[Path(p) for p in legacy.implementation_files(legacy.DetectionTrainer)])
    before={k:stat(cfg[k]) for k in ('model','teacher','reference')}
    torch.set_num_threads(4);trainer=None;started=time.perf_counter();batch_times=[]
    try:
        # Deliberately not resume: .pt initializes all student tensors, while
        # the new three-epoch optimizer/scheduler/EMA start from their own zero.
        trainer=build(cfg,args.config,output,'weight0' if cfg['arm']=='N' else 'paired',max_steps=None)
        original_setup=trainer._setup_train
        def setup():
            original_setup()
            payload=torch.load(cfg['model'],map_location='cpu',weights_only=False)
            initial=(payload.get('ema') or payload['model']).float()
            current=trainer.model.state_dict();expected=initial.state_dict()
            if set(current)!=set(expected):raise AssertionError('Warm-start state keys differ')
            bad=[k for k in current if not torch.equal(current[k].detach().cpu(),expected[k].detach().cpu())]
            if bad:raise AssertionError('Warm-start tensors changed: '+str(bad[:12]))
            if trainer.optimizer.state or trainer.ema.updates!=0:raise AssertionError('Optimizer/EMA did not start fresh')
            if any(not torch.equal(current[k],trainer.ema.ema.state_dict()[k]) for k in current):
                raise AssertionError('Fresh EMA initial state differs')
            ready=read(output/'runtime_ready.json')
            if (ready['train_images'],ready['val_images'],ready['train_batches'],ready['batch'],ready['workers'])!=(2048,cfg['expected_val_images'],64,32,4):
                raise ValueError('Dataset/batch/worker contract differs')
            write_new(output/'initialization_check.json',dict(status='PASS_FULL_STATE_WARM_START',
                initial_checkpoint=before['model'],state_tensors=len(current),head_included=True,
                fresh_optimizer=True,fresh_ema=True,teacher_reference_isolated=True,new_hash_computed=False))
            frozen_bn.update(bn_state(trainer.model))
            del payload,initial,expected,current
        trainer._setup_train=setup
        install_bn_freeze(trainer)
        frozen_bn={}
        original_preprocess=trainer.preprocess_batch
        def preprocess(batch):
            batch=original_preprocess(batch)
            if trainer.batch_visits<=30:
                row=dict(batch=trainer.batch_visits,im_file=list(batch['im_file']))
                for k in ('cls','bboxes','batch_idx'):row[k]=batch[k].detach().cpu().tolist()
                for k in ('cls','bboxes','batch_idx'):row['teacher_'+k]=batch['teacher_batch'][k].detach().cpu().tolist()
                legacy.append_json(output/'sample_stream.jsonl',row)
            batch_times.append(time.perf_counter())
            return batch
        trainer.preprocess_batch=preprocess
        if args.canary:
            original_step=trainer.optimizer_step
            def step():
                original_step()
                if trainer.real_updates>=24:trainer.stop=True
            trainer.optimizer_step=step
        torch.cuda.reset_peak_memory_stats();trainer.train()
        bn_after=bn_state(trainer.model)
        if set(bn_after)!=set(frozen_bn) or any(not torch.equal(bn_after[k],frozen_bn[k]) for k in frozen_bn):
            raise AssertionError('Frozen BN buffers changed')
        if args.canary:
            if trainer.real_updates<24:raise RuntimeError('Fewer than 24 successful canary updates')
            active=any(max(x.get('kd_gradient_l2',0),x.get('kd_score_gradient_l2',0))>0 for x in trainer.criterion_ref.gradient_checks)
            if cfg['arm']!='N' and (not active or trainer.criterion_ref.selected_total<=0):raise RuntimeError('No nonzero selected KD signal')
        elif trainer.epoch+1!=3 or trainer.criterion_ref.calls!=192:
            raise RuntimeError('Independent FT did not complete exactly 3 epochs / 192 batches')
        if {k:stat(cfg[k]) for k in before}!=before:raise ValueError('Input checkpoint stat changed')
        result=dict(status='DIRECTION_CANARY_COMPLETED' if args.canary else 'DIRECTION_TRAINING_COMPLETED',
            scope='FEATURE_RELATION_GM_FT3',single_seed=True,arm=cfg['arm'],seed=42,dataset=cfg['dataset'],
            epochs_configured=3,last_epoch=trainer.epoch+1,endpoint=ENDPOINT,formal_e200_complete=False,
            config_copy=str(output/'feature_gm_config.yaml'),model=cfg['model'],teacher=cfg['teacher'],reference=cfg['reference'],
            classification_coefficient=cfg['classification_coefficient'],localization_coefficient=cfg['localization_coefficient'],kd_coefficient=cfg['kd_coefficient'],
            bn_running_buffers_unchanged=True,bn_buffer_count=len(frozen_bn),bn_affine_trainable=True,
            checkpoint=stat(output/'weights/last.pt'),successful_updates=trainer.real_updates,
            optimizer_updates=trainer.real_updates,attempts=trainer.update_attempts,amp_skips=trainer.skipped_amp_updates,
            ema_updates=trainer.ema.updates,batches=trainer.criterion_ref.calls,selected_objects=trainer.criterion_ref.selected_total,
            gradient_checks=trainer.criterion_ref.gradient_checks,seconds=time.perf_counter()-started,
            steady_batch_intervals_seconds=[b-a for a,b in zip(batch_times[6:-1],batch_times[7:])],
            new_hash_computed=False,official_test_accessed=False,
            resources=legacy.bound_lease_resource_record_from_environment(),
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            canary_receipt=str(args.canary_receipt.resolve()) if args.canary_receipt else None,
            formal_paper_gain_claim=False)
        # These are existing candidate counters, read directly at termination;
        # never substitute the last sparse log or invent zero for missing fields.
        if False:
            names=('thin_learning_batches','full_diagnostics_batches','fallback_batches')
            result['candidate_counter_fields_missing']=[n for n in names if not hasattr(trainer.criterion_ref,n)]
            for name in names:
                if hasattr(trainer.criterion_ref,name):result[name]=getattr(trainer.criterion_ref,name)
        write_new(output/('canary.json' if args.canary else 'completion_receipt.json'),result)
        print(result['status'],cfg['arm'],result['seconds'],flush=True)
    except BaseException as error:
        write_new(output/'feature_gm_failure.json',dict(status='FEATURE_GM_TRAINING_FAILED',scope='FEATURE_RELATION_GM_FT3',error=repr(error),
            traceback=traceback.format_exc(),seconds=time.perf_counter()-started,new_hash_computed=False))
        raise
    finally:
        if trainer is not None:
            for name in ('train_loader','test_loader'):
                it=getattr(getattr(trainer,name,None),'iterator',None)
                if callable(getattr(it,'_shutdown_workers',None)):it._shutdown_workers()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir','config','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--canary',action='store_true');p.add_argument('--canary-receipt',type=Path)
    run(p.parse_args())

if __name__=='__main__':main()
