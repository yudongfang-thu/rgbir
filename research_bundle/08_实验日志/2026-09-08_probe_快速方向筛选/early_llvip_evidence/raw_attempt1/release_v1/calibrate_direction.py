"""Eight fixed subset batches; restore initial state, no optimizer update, frozen BN."""
import argparse,math,statistics,sys,time,traceback,shutil
from pathlib import Path
from direction_common import load_config,data_output,read,write_new,copy_sources,stat
from train_direction import build_private,install_bn_freeze,bn_state

def run(args):
    cfg=load_config(args.config);output=data_output(args.output)
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.reference_dir.resolve()))
    import torch,runtime,independent_criterion,selection_adapter,classification_logit
    from gradient_observation import shared_parameter_set
    legacy=runtime.legacy
    if len(legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('Existing one-GPU lease required')
    copy_sources(output,list(Path(__file__).parent.glob('*.py'))+[args.config])
    shutil.copyfile(args.config,output/'direction_config.yaml')
    torch.set_num_threads(4);trainer=None;started=time.perf_counter()
    rows=[];arms=['C1','C2','F-rel'] if cfg['dataset']=='drone' else ['L2-box','L2-GT']
    ratios={a:[] for a in arms};coef_c1=.09227393550836771
    try:
        build,_=build_private(runtime,independent_criterion,selection_adapter,classification_logit,cfg,args)
        trainer=build(cfg,args.config,output,'weight0',max_steps=None)
        trainer._setup_train();trainer.epoch=0;install_bn_freeze(trainer)()
        initial={k:v.detach().clone() for k,v in trainer.model.state_dict().items()}
        checkpoint=torch.load(cfg['model'],map_location='cpu',weights_only=False)
        expected=(checkpoint.get('ema') or checkpoint['model']).float().state_dict()
        if set(expected)!=set(initial) or any(not torch.equal(initial[k].cpu(),v) for k,v in expected.items()):
            raise AssertionError('Calibration state differs from complete input checkpoint')
        del checkpoint,expected
        frozen=bn_state(trainer.model);_,named=shared_parameter_set(trainer.model)
        params=[p for n,p in named];criterion=trainer.criterion_ref
        def gradient(loss):
            values=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            finite=all(bool(torch.isfinite(v).all()) for v in values if v is not None)
            n=float(sum((v.detach().double().square().sum() for v in values if v is not None),torch.zeros((),device=loss.device)).sqrt()) if finite else None
            return values,n
        def cosine(a,b):
            pairs=[(u.double(),v.double()) for u,v in zip(a,b) if u is not None and v is not None]
            na=sum(float(u.double().square().sum()) for u in a if u is not None)
            nb=sum(float(v.double().square().sum()) for v in b if v is not None)
            return sum(float((u*v).sum()) for u,v in pairs)/(na*nb)**.5 if na>0 and nb>0 else None
        torch.cuda.reset_peak_memory_stats()
        for bi,raw in enumerate(trainer.train_loader):
            if bi==8:break
            trainer.model.load_state_dict(initial,strict=True);trainer._model_train()
            batch=trainer.preprocess_batch(raw);criterion.calls=bi+1
            with torch.amp.autocast('cuda',enabled=bool(trainer.amp)):
                prediction=trainer.model(batch['img']);student=legacy.raw_prediction(prediction)
                native,_=criterion.native(prediction,batch)
                with torch.no_grad():
                    teacher=legacy.raw_prediction(criterion.teacher(batch['strong_img']))
                    reference=legacy.raw_prediction(criterion.reference(batch['img']))
                losses=criterion.losses(student,teacher,reference,batch,arms)
                b=int(batch['img'].shape[0]);ng,nn=gradient(native.sum())
                grads={};norms={}
                for arm,(loss,stats) in losses.items():grads[arm],norms[arm]=gradient(b*loss)
                target=coef_c1*norms['C1'] if cfg['dataset']=='drone' and norms['C1'] is not None else .1*nn if cfg['dataset']=='llvip' and nn is not None else None
                for arm in arms:
                    n=norms[arm]
                    if target is not None and n is not None and math.isfinite(target) and target>0 and n>0:ratios[arm].append(target/n)
                row=dict(batch=bi+1,files=list(batch['im_file']),native_norm=nn,unit_B_kd_norms=norms,
                    stats={a:s for a,(loss,s) in losses.items()},native_cosines={a:cosine(ng,g) if nn and norms[a] else None for a,g in grads.items()})
            rows.append(row);legacy.append_json(output/'calibration_batches.jsonl',row)
            del prediction,student,native,teacher,reference,losses,grads,ng,batch,raw
        if len(rows)!=8:raise RuntimeError('Need exactly eight fixed batches')
        if trainer.optimizer.state or trainer.ema.updates!=0:raise AssertionError('Calibration updated optimizer/EMA')
        if any(not torch.equal(v,bn_state(trainer.model)[k]) for k,v in frozen.items()):raise AssertionError('Frozen BN changed')
        coefficients={'N':0.};blocked={};details={}
        for arm in arms:
            r=ratios[arm];median=statistics.median(r) if r else None
            details[arm]=dict(finite_nonzero_batches=len(r),raw_median=median,ratios=r)
            if arm=='C1':coefficients[arm]=coef_c1;continue
            if len(r)<4:coefficients[arm]=None;blocked[arm]='Fewer than 4/8 finite nonzero gradient ratios';continue
            if arm.startswith('L2'):coefficients[arm]=min(1.,median)
            elif 0<median<=1:coefficients[arm]=median
            else:coefficients[arm]=None;blocked[arm]='Unclipped coefficient outside (0,1]'
        if cfg['dataset']=='llvip':
            coefficients['L2-GT']=coefficients['L2-box']
            if 'L2-box' in blocked:blocked['L2-GT']='Shared teacher-mask coefficient blocked: '+blocked['L2-box']
            else:blocked.pop('L2-GT',None)
        write_new(output/'calibration_receipt.json',dict(status='DIRECTION_CALIBRATION_COMPLETED',scope='EXPLORATORY_FIXED8_BNFROZEN',
            dataset=cfg['dataset'],coefficients=coefficients,blocked=blocked,details=details,batches=8,minimum_nonzero=4,
            seed=42,student_initial_checkpoint=stat(cfg['model']),parameter_names=[n for n,p in named],
            reset_all_parameters_buffers_each_batch=True,optimizer_updates=0,ema_updates=0,bn_running_buffers_unchanged=True,
            full_initial_checkpoint_state_verified=True,
            formal64_admission=False,seconds=time.perf_counter()-started,new_hash_computed=False,official_test_accessed=False,
            resources=legacy.bound_lease_resource_record_from_environment(),
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20))
    except BaseException as e:
        write_new(output/'failure.json',dict(status='DIRECTION_CALIBRATION_FAILED',error=repr(e),traceback=traceback.format_exc()))
        raise
    finally:
        if trainer is not None:
            for name in ('train_loader','test_loader'):
                it=getattr(getattr(trainer,name,None),'iterator',None)
                if callable(getattr(it,'_shutdown_workers',None)):it._shutdown_workers()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference-dir','config','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
