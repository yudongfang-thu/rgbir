"""Independent E8 entry; actual root admission required to execute."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import types

from screen_common import (ENDPOINT,copy_sources,data_output,explicit,load_config,
                           read,require_admission,stat,write_new)


def cloned(fn,**bindings):
    new=types.FunctionType(fn.__code__,dict(fn.__globals__,**bindings),fn.__name__,fn.__defaults__,fn.__closure__)
    new.__kwdefaults__=fn.__kwdefaults__;return new


def candidate_builder(training,criterion,selection,candidate,classification=None):
    kind=candidate['kind']
    if kind=='original':return training.build_trainer,[]
    if kind not in ('pool','selector','criterion_factory'):raise ValueError('C1 candidate remains unresolved')
    path=Path(candidate['source']).resolve()
    sys.path.insert(0,str(path.parent))
    module=explicit('_short_screen_candidate',path)
    if kind=='criterion_factory':
        if classification is None:raise ValueError('Criterion factory requires the pinned classification module')
        # selected_only_v1 explicitly binds both selector AND its matching thin
        # loss/statistics. Its factory expects the criterion module, not a class.
        api=module.make_api(selection,classification)
        private=module.make_criterion_type(criterion,api)
        if not isinstance(private,type) or not issubclass(private,criterion.IndependentCriterion):
            raise TypeError('Candidate factory must return a private IndependentCriterion subclass')
        return cloned(training.build_trainer,IndependentCriterion=private),list(path.parent.glob('*.py'))
    if hasattr(module,'make_api') and hasattr(module,'make_criterion_type'):
        raise ValueError('This thin candidate requires criterion_factory; pool/selector-only would use the wrong loss/statistics')
    function=getattr(module,candidate['export'])
    if not callable(function):raise TypeError('Candidate export must be callable')
    select=cloned(selection.build_classification_selection,pool_relative_logits=function) if kind=='pool' else function
    call=cloned(criterion.IndependentCriterion.__call__,build_classification_selection=select)
    private=type('ShortScreenCandidateCriterion',(criterion.IndependentCriterion,),{'__call__':call})
    return cloned(training.build_trainer,IndependentCriterion=private),list(path.parent.glob('*.py'))


def run(args):
    cfg=load_config(args.config)
    admission=require_admission(args.admission,cfg,args.config,Path(__file__),'training')
    candidate=admission['candidate']
    if cfg['arm']!='C1' and candidate['kind']!='original':raise ValueError('Only C1 may load an admitted candidate')
    if candidate['kind']!='original':
        if Path(candidate['source']).read_bytes()!=Path(candidate['approved_source_copy']).read_bytes():
            raise ValueError('Candidate source differs from approved copy')
    output=data_output(args.output)
    if output.exists():raise FileExistsError(output)
    ref=args.reference_dir.resolve();sys.path.insert(0,str(ref))
    import torch
    import runtime
    import train_independent as training
    import independent_criterion as criterion
    import selection_adapter as selection
    import classification_logit as classification
    for module,name in [(runtime,'runtime.py'),(training,'train_independent.py'),(criterion,'independent_criterion.py'),(selection,'selection_adapter.py'),(classification,'classification_logit.py')]:
        if Path(module.__file__).resolve()!=ref/name:raise RuntimeError('Wrong release module: '+name)
    # formal=False does not enforce E200/readiness; this entry owns all E8
    # completion/admission checks and never changes the formal validator.
    training.validate_execution(cfg,formal=False)
    if len(runtime.legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('One bound globallease required')
    build,extra=candidate_builder(training,criterion,selection,candidate,classification)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.config,output/'short_screen_config.yaml')
    shutil.copyfile(args.admission,output/'short_screen_admission.json')
    paths=list(Path(__file__).parent.glob('*.py'))+list(ref.rglob('*.py'))+extra+[args.config,args.admission]
    paths += [Path(p) for p in runtime.legacy.implementation_files(runtime.legacy.DetectionTrainer)]
    copy_sources(output,[p for p in paths if '__pycache__' not in p.parts])
    before={k:stat(cfg[k]) for k in ('model','teacher','reference')}
    torch.set_num_threads(4);trainer=None;started=time.perf_counter()
    try:
        trainer=build(cfg,args.config,output,arm=cfg['arm'],max_steps=None,historical=False)
        trainer.train()
        if trainer.epoch+1!=8:raise RuntimeError('SHORT_SCREEN did not complete exactly8 epochs')
        ready=read(output/'runtime_ready.json')
        if ready['train_images']!=17990 or ready['val_images']!=1469:raise ValueError('Training population changed')
        if {k:stat(cfg[k]) for k in before}!=before:raise ValueError('Input model stat changed')
        checkpoint=output/'weights'/'last.pt'
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        result=dict(status='SHORT_SCREEN_TRAINING_COMPLETED',scope='SHORT_SCREEN',single_seed=True,
            arm=cfg['arm'],seed=42,dataset='dronevehicle',epochs_configured=8,last_epoch=8,endpoint=ENDPOINT,
            independent_lr_horizon=8,not_e200_prefix=True,formal_e200_complete=False,
            configuration=str(output/'short_screen_config.yaml'),checkpoint=stat(checkpoint),
            model=cfg['model'],teacher=cfg['teacher'],reference=cfg['reference'],
            classification_coefficient=cfg['classification_coefficient'],localization_coefficient=0.,
            optimizer_updates=trainer.real_updates,attempts=trainer.update_attempts,amp_skips=trainer.skipped_amp_updates,
            ema_updates=trainer.ema.updates,batches=trainer.criterion_ref.calls,selected_objects=trainer.criterion_ref.selected_total,
            candidate=candidate,seconds=time.perf_counter()-started,new_hash_computed=False,official_test_accessed=False,
            resources=runtime.legacy.bound_lease_resource_record_from_environment(),
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            training_source_manifest=str(output/'source_manifest.json'),formal_paper_gain_claim=False)
        write_new(output/'short_training_receipt.json',result)
    except BaseException as error:
        write_new(output/'short_training_failure.json',dict(status='SHORT_SCREEN_TRAINING_FAILED',error=repr(error),
            traceback=traceback.format_exc(),seconds=time.perf_counter()-started,new_hash_computed=False))
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
    p.add_argument('--admission',type=Path);args=p.parse_args();run(args)


if __name__=='__main__':main()
