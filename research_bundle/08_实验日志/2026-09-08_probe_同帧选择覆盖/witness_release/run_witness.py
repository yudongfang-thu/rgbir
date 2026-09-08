"""One original LLVIP batch; traced GT identities, raw states, actual selector, no updates."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

SCOPE='SAME_FORWARD_DETECTOR_WITNESS'

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def stream_row(batch):
    row=dict(batch=1,im_file=list(batch['im_file']))
    for k in ('cls','bboxes','batch_idx'):row[k]=batch[k].detach().cpu().tolist()
    for k in ('cls','bboxes','batch_idx'):row['teacher_'+k]=batch['teacher_batch'][k].detach().cpu().tolist()
    return row

def first_record(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():return json.loads(line)
    raise ValueError('Missing original first-batch record')

def run(args):
    sys.path.insert(0,str(args.screen_release.resolve()))
    from direction_common import load_config,stat,copy_sources,data_output
    from train_direction import build_private,cloned,install_bn_freeze,bn_state
    cfg=load_config(args.config)
    if cfg['dataset']!='llvip' or cfg['arm']!='N' or cfg['model']!=cfg['reference']:
        raise ValueError('Frozen LLVIP initialization/reference identity differs')
    output=data_output(args.output);output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.reference_dir.resolve()))
    import torch,runtime,independent_criterion,selection_adapter,classification_logit
    from mapping_trace import make_traced_dataset,batch_identity_contract
    from export_selection import export_selection
    from detector_witness import analyze_witness
    legacy=runtime.legacy
    if str(torch.__version__)!=cfg['torch_version'] or str(legacy.ultralytics.__version__)!=cfg['ultralytics_version']:
        raise ValueError('Pinned framework versions changed')
    if len(legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('One existing lease required')
    if Path(runtime.__file__).resolve()!=args.reference_dir.resolve()/'runtime.py':raise ValueError('Wrong runtime')
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    sources=list(Path(__file__).parent.glob('*.py'))+list(args.reference_dir.rglob('*.py'))
    sources += [args.config,args.screen_release/'train_direction.py',args.screen_release/'direction_common.py',
        args.screen_release/'direction_criterion.py',Path(cfg['direction_screen']['candidate_source'])]
    copy_sources(output,sources);shutil.copyfile(args.config,output/'inherited_training_config.yaml')
    inputs={k:stat(cfg[k]) for k in ('model','teacher','reference')}
    started=time.perf_counter();trainer=None
    try:
        builder,_=build_private(runtime,independent_criterion,selection_adapter,classification_logit,cfg,args)
        builder=cloned(builder,DualLabelRGBIRDataset=make_traced_dataset(runtime.TrackedDualLabelRGBIRDataset))
        trainer=builder(cfg,args.config,output,'weight0',max_steps=None)
        trainer._setup_train();trainer.epoch=0;install_bn_freeze(trainer)()
        initial={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
        payload=torch.load(cfg['model'],map_location='cpu',weights_only=False)
        expected=(payload.get('ema') or payload['model']).float().state_dict()
        if set(initial)!=set(expected) or any(not torch.equal(initial[k],v) for k,v in expected.items()):
            raise AssertionError('Full student initialization differs')
        del payload,expected
        criterion=trainer.criterion_ref
        if criterion.teacher.training or criterion.reference.training:
            raise AssertionError('Frozen models must be eval')
        auxiliary_bn={k:bn_state(m) for k,m in [('teacher',criterion.teacher),('reference',criterion.reference)]}
        batch=trainer.preprocess_batch(next(iter(trainer.train_loader)))
        current=stream_row(batch);expected_stream=first_record(args.expected_stream)
        expected_stream.pop('arm',None)
        if current!=expected_stream:raise AssertionError('Traced first batch differs from completed original stream')
        write_new(output/'first_batch_stream.json',current)
        identity=batch_identity_contract(batch,frame_id='llvip_first32_seed42')
        write_new(output/'identity_contract.json',identity)
        api=sys.modules['_direction_selected_only'].make_api(selection_adapter,classification_logit)
        with torch.no_grad(),torch.amp.autocast('cuda',enabled=bool(trainer.amp)):
            student=legacy.raw_prediction(trainer.model(batch['img']))
            teacher=legacy.raw_prediction(criterion.teacher(batch['strong_img']))
            reference=legacy.raw_prediction(criterion.reference(batch['img']))
            exported=export_selection(batch,student,teacher,reference,api=api,
                evidence_config=criterion.evidence_cfg,strides=tuple(int(s) for s in trainer.model.stride),
                selection_seed=cfg['seed']+1,identity_contract=identity)
            previous_objects=[json.loads(line) for line in args.previous_objects.read_text(encoding='utf-8').splitlines() if line.strip()]
            if exported['records']!=previous_objects:raise AssertionError('Old same-batch object states or original selection changed')
            witnesses=analyze_witness(batch,{'S':student,'T':teacher,'R':reference},
                {'S':trainer.model,'T':criterion.teacher,'R':criterion.reference},
                evidence_config=criterion.evidence_cfg,strides=tuple(int(s) for s in trainer.model.stride),existing_export=exported)
        if any(v.grad is not None for m in (trainer.model,criterion.teacher,criterion.reference) for v in m.parameters()):
            raise AssertionError('Unexpected parameter gradient')
        if trainer.optimizer.state or trainer.ema.updates!=0 or trainer.real_updates!=0 or trainer.update_attempts!=0:
            raise AssertionError('Unexpected optimizer/EMA update')
        if set(initial)!=set(trainer.model.state_dict()) or any(not torch.equal(v,trainer.model.state_dict()[k].detach().cpu()) for k,v in initial.items()):
            raise AssertionError('Student parameters or buffers changed')
        for name,model in [('teacher',criterion.teacher),('reference',criterion.reference)]:
            after=bn_state(model)
            if set(after)!=set(auxiliary_bn[name]) or any(not torch.equal(after[k],v) for k,v in auxiliary_bn[name].items()):
                raise AssertionError('Auxiliary BN changed')
        if inputs!={k:stat(cfg[k]) for k in inputs}:raise AssertionError('Input file stat changed')
        for name,rows in [('objects',exported['records']),('ir_objects',exported['ir_records']),('frames',exported['frames'])]:
            with (output/(name+'.jsonl')).open('x',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        for name,rows in [('witness_objects',witnesses['records']),('witness_ir_objects',witnesses['ir_records']),('witness_frames',witnesses['frames'])]:
            with (output/(name+'.jsonl')).open('x',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        write_new(output/'witness_contract.json',{k:v for k,v in witnesses.items() if k not in ('records','ir_records','frames')})
        write_new(output/'export_contract.json',{k:v for k,v in exported.items() if k not in ('records','ir_records','frames')})
        receipt=dict(status=SCOPE+'_COMPLETED',scope=SCOPE,dataset='llvip',seed=42,batches=1,frames=len(exported['frames']),
            objects_n=len(exported['records']),training=0,backward=0,optimizer_updates=0,ema_updates=0,
            optimizer_constructed_for_original_loader_setup=True,optimizer_state_empty=True,
            actual_batch_size=int(batch['img'].shape[0]),actual_amp=bool(trainer.amp),student_mode='train_with_BN_frozen',
            teacher_reference_mode='eval',student_full_state_unchanged=True,auxiliary_gradients_absent=True,
            previous_objects_exact=True,previous_objects=stat(args.previous_objects),head_decode_nonstate_caches_may_change=True,
            raw_forward_counts=dict(student=1,teacher=1,reference=1),selector_counts=exported['selector_counts'],
            identity_contract=identity,first_batch_stream_exact=True,expected_stream=stat(args.expected_stream),
            matched_stream_fields=list(current),historical_pixel_tensor_comparison=False,
            initialization=inputs,resources=legacy.bound_lease_resource_record_from_environment(),
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            seconds=time.perf_counter()-started,new_hash_computed=False,official_test_accessed=False,
            full_dev_evaluated=False,formal_paper_gain_claim=False,global_population_coverage_claim=False)
        if receipt['frames']!=32 or receipt['actual_batch_size']!=32:raise AssertionError('Fixed B32 changed')
        write_new(output/'completion_receipt.json',receipt)
        print(receipt['status'],receipt['objects_n'],receipt['seconds'],flush=True)
    except BaseException as e:
        write_new(output/'failure.json',dict(status=SCOPE+'_FAILED',error=repr(e),traceback=traceback.format_exc(),
            seconds=time.perf_counter()-started,new_hash_computed=False));raise
    finally:
        if trainer is not None:
            for name in ('train_loader','test_loader'):
                it=getattr(getattr(trainer,name,None),'iterator',None)
                if callable(getattr(it,'_shutdown_workers',None)):it._shutdown_workers()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('screen-release','reference-dir','config','expected-stream','previous-objects','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
