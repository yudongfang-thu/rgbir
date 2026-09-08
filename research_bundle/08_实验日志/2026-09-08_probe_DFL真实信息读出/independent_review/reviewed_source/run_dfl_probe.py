"""One new LLVIP first-batch raw DFL readout under the existing global lease."""
import argparse,inspect,json,shutil,sys,time,traceback
from pathlib import Path
STATUS='RAW_DFL_SINGLE_BATCH_COMPLETED'
SCOPE='RAW_DFL_SINGLE_BATCH'

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def stream(batch):
    r=dict(batch=1,im_file=list(batch['im_file']))
    for k in ('cls','bboxes','batch_idx'):r[k]=batch[k].detach().cpu().tolist();r['teacher_'+k]=batch['teacher_batch'][k].detach().cpu().tolist()
    return r

def setup_with_validated_amp(trainer,cfg_amp,prior_amp,trainer_module):
    if type(cfg_amp) is not bool or cfg_amp is not prior_amp:raise ValueError('Historical validated AMP differs')
    original=trainer_module.check_amp;calls=[]
    def validated(model):
        if model is not trainer.model:raise ValueError('Unexpected setup AMP model')
        calls.append(1);return cfg_amp
    trainer_module.check_amp=validated
    try:trainer._setup_train()
    finally:trainer_module.check_amp=original
    if len(calls)!=int(cfg_amp) or bool(trainer.amp)!=cfg_amp:raise ValueError('Probe AMP bypass/actual mode differs')
    return dict(probe_specific_setup_check_amp_bypass=True,validated_value=cfg_amp,calls=len(calls),
        original_binding_restored=True,setup_equivalence_claim=False,unrelated_model_forward=False)

def run(args):
    sys.path.insert(0,str(args.screen_release.resolve()))
    from direction_common import load_config,stat,data_output
    from train_direction import build_private,cloned,install_bn_freeze
    cfg=load_config(args.config)
    if cfg['dataset']!='llvip' or cfg['arm']!='N' or cfg['model']!=cfg['reference']:raise ValueError('Frozen LLVIP N/R identity differs')
    output=data_output(args.output);output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.reference_dir.resolve()));sys.path.insert(0,str(args.witness_release.resolve()))
    import torch,runtime,independent_criterion,selection_adapter,classification_logit
    from mapping_trace import make_traced_dataset,batch_identity_contract
    import dfl_export
    from dfl_export import export_dfl
    legacy=runtime.legacy
    if Path(runtime.__file__).resolve()!=args.reference_dir.resolve()/'runtime.py':raise ValueError('Wrong runtime')
    if Path(sys.modules['mapping_trace'].__file__).resolve()!=args.witness_release.resolve()/'mapping_trace.py':raise ValueError('Wrong traced loader')
    if Path(dfl_export.__file__).resolve()!=Path(__file__).resolve().with_name('dfl_export.py'):raise ValueError('Wrong DFL exporter')
    if str(torch.__version__)!=cfg['torch_version'] or str(legacy.ultralytics.__version__)!=cfg['ultralytics_version']:raise ValueError('Pinned versions differ')
    if len(legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('One original global lease required')
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();started=time.perf_counter();trainer=None
    inputs={k:stat(cfg[k]) for k in ('model','teacher','reference')}
    prior=read(args.previous_probe/'completion_receipt.json')
    if prior['status']!='SAME_FORWARD_DETECTOR_WITNESS_COMPLETED' or prior['initialization']!=inputs:raise ValueError('Historical model path/stat differs')
    anchor=lines(args.anchor_objects);old_objects=lines(args.previous_probe/'witness_objects.jsonl')
    if len(anchor)!=80 or len(old_objects)!=80:raise ValueError('Fixed 80 GT required')
    old={r['stable_rgb_gt_id']:r for r in old_objects}
    if len(old)!=80 or {r['stable_rgb_gt_id'] for r in anchor}!=set(old):raise ValueError('Historical anchor GT roster differs')
    for r in anchor:
        w=old[r['stable_rgb_gt_id']]
        for key in ('frame_id','image_index','rgb_global_row','ir_global_row','stable_ir_gt_id','rgb_gt_xyxy','ir_gt_xyxy'):
            if r[key]!=w[key]:raise ValueError('Anchor/GT identity mismatch: '+key)
        for name,m in r['native_iou50_matches'].items():
            previous=w['detector'][name]['native_witness']
            if (m is None)!=(previous is None):raise ValueError('Historical native match missing')
            if m is not None and any(m[k]!=previous[k] for k in ('anchor_index','box','confidence')):raise ValueError('Historical native anchor changed')
    try:
        builder,_=build_private(runtime,independent_criterion,selection_adapter,classification_logit,cfg,args)
        builder=cloned(builder,DualLabelRGBIRDataset=make_traced_dataset(runtime.TrackedDualLabelRGBIRDataset))
        trainer=builder(cfg,args.config,output,'weight0',max_steps=None)
        import ultralytics.engine.trainer as native_trainer_module
        amp_setup=setup_with_validated_amp(trainer,cfg['amp'],prior['actual_amp'],native_trainer_module)
        trainer.epoch=0;install_bn_freeze(trainer)()
        criterion=trainer.criterion_ref;models={'S':trainer.model,'R':criterion.reference,'T':criterion.teacher}
        if criterion.teacher.training or criterion.reference.training:raise ValueError('Auxiliary models must remain eval')
        states={name:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for name,m in models.items()}
        payload=torch.load(cfg['model'],map_location='cpu',weights_only=False);expected=(payload.get('ema') or payload['model']).float().state_dict()
        if set(states['S'])!=set(expected) or any(not torch.equal(states['S'][k],v) for k,v in expected.items()):raise ValueError('Full warm start differs')
        del payload,expected
        batch=trainer.preprocess_batch(next(iter(trainer.train_loader)));current=stream(batch)
        if current!=read(args.previous_probe/'first_batch_stream.json'):raise ValueError('First batch stream differs')
        identity=batch_identity_contract(batch,frame_id='llvip_first32_seed42')
        if identity!=read(args.previous_probe/'identity_contract.json'):raise ValueError('Full GT identity contract differs')
        if len(identity['frames'])!=32:raise ValueError('Full identity must preserve all32 frames including empty GT')
        if int(batch['img'].shape[0])!=32:raise ValueError('One B32 required')
        write(output/'first_batch_stream.json',current);write(output/'identity_contract.json',identity)
        counts={name:0 for name in models}
        def counter(name):
            def observe(module,inp,out):counts[name]+=1
            return observe
        handles=[m.register_forward_hook(counter(n)) for n,m in models.items()]
        try:
            with torch.no_grad(),torch.amp.autocast('cuda',enabled=bool(trainer.amp)):
                raw={'S':legacy.raw_prediction(models['S'](batch['img'])),'T':legacy.raw_prediction(models['T'](batch['strong_img'])),'R':legacy.raw_prediction(models['R'](batch['img']))}
                exported=export_dfl(batch,raw,models,anchor,selection_adapter.original,selection_adapter.evidence_config(criterion.evidence_cfg),tuple(int(x) for x in trainer.model.stride))
        finally:
            for h in handles:h.remove()
        if counts!={'S':1,'R':1,'T':1}:raise ValueError('Actual batch model forward counts differ')
        for name,m in models.items():
            after=m.state_dict()
            if set(after)!=set(states[name]) or any(not torch.equal(after[k].detach().cpu(),v) for k,v in states[name].items()):raise ValueError('Model full state changed: '+name)
            if any(p.grad is not None for p in m.parameters()):raise ValueError('Unexpected gradient: '+name)
        if trainer.optimizer.state or trainer.ema.updates!=0 or trainer.real_updates!=0 or trainer.update_attempts!=0:raise ValueError('Unexpected optimizer or EMA update')
        if inputs!={k:stat(cfg[k]) for k in inputs}:raise ValueError('Checkpoint stat changed')
        forward_id='raw_dfl_new_forward::'+output.name+'::'+str(time.time_ns())
        for filename,key in [('objects.jsonl','objects'),('anchor_distributions.jsonl','distributions')]:
            with (output/filename).open('x',encoding='utf-8') as f:
                for r in exported[key]:
                    r['current_forward_id']=forward_id;f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        contract={k:v for k,v in exported.items() if k not in ('objects','distributions')}
        contract.update(current_forward_id=forward_id,initialization=inputs,previous_probe=str(args.previous_probe),historical_same_forward_claim=False,objects_n=80,frames=32,unique_distributions=len(exported['distributions']))
        write(output/'dfl_contract.json',contract)
        dependencies=[]
        roots=[str(p.resolve()) for p in (args.reference_dir,args.screen_release,args.witness_release)]
        for module in list(sys.modules.values()):
            path=getattr(module,'__file__',None)
            if path and path.endswith('.py') and any(str(Path(path).resolve()).startswith(root+'/') for root in roots):dependencies.append(Path(path))
        dependencies += [Path(inspect.getsourcefile(type(models['S'].model[-1]))),Path(inspect.getsourcefile(type(models['S'].model[-1].dfl))),args.config,args.anchor_objects,args.previous_probe/'completion_receipt.json']
        write(output/'dependency_identity.json',dict(files=[stat(p) for p in dict.fromkeys(dependencies)],new_hash_computed=False))
        source=output/'sources';source.mkdir()
        for name in ('run_dfl_probe.py','dfl_export.py','test_dfl_cpu.py'):shutil.copyfile(Path(__file__).with_name(name),source/name)
        shutil.copyfile(args.config,output/'inherited_training_config.yaml')
        receipt=dict(status=STATUS,scope=SCOPE,current_forward_id=forward_id,dataset='llvip',seed=42,batches=1,frames=32,objects_n=80,unique_distributions=len(exported['distributions']),
            raw_forward_counts=dict(student=counts['S'],reference=counts['R'],teacher=counts['T']),raw_forward_counter_scope='Actual requested batch only; setup excluded',head_decode_readout_calls_per_model=1,
            optimizer_updates=0,ema_updates=0,backward=0,training=0,optimizer_constructed_for_loader_setup=True,optimizer_state_empty=True,
            first_batch_stream_exact=True,identity_exact=True,student_full_state_unchanged=True,auxiliary_full_state_unchanged=True,all_gradients_absent=True,
            student_mode='train_with_BN_frozen',teacher_reference_mode='eval',actual_amp=bool(trainer.amp),initialization=inputs,setup_amp=amp_setup,
            historical_same_forward_claim=False,historical_pixel_tensor_comparison=False,head_decode_nonstate_caches_may_change=True,new_NMS=False,new_matching=False,new_loss=False,
            resources=legacy.bound_lease_resource_record_from_environment(),gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            seconds=time.perf_counter()-started,new_hash_computed=False,official_test_accessed=False)
        write(output/'completion_receipt.json',receipt);print(STATUS,receipt['seconds'],flush=True)
    except BaseException as e:
        write(output/'failure.json',dict(status='RAW_DFL_SINGLE_BATCH_FAILED',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-started,new_hash_computed=False));raise
    finally:
        if trainer is not None:
            for name in ('train_loader','test_loader'):
                it=getattr(getattr(trainer,name,None),'iterator',None)
                if callable(getattr(it,'_shutdown_workers',None)):it._shutdown_workers()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('screen-release','reference-dir','config','witness-release','previous-probe','anchor-objects','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
