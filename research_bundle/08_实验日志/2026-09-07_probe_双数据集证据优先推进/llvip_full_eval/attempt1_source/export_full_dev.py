"""Read-only full development evaluation of historical LLVIP baselines."""
from pathlib import Path
import argparse, gc, gzip, inspect, json, random, shutil, sys, time, traceback

REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
RELEASE=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(RELEASE))

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def state(path):
    s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns)

def run(a):
    import numpy as np
    import torch, yaml
    from PIL import Image
    from runtime import legacy
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.metrics import ap_per_class
    from evaluator_profile import dev_roster,metric_record,assert_equal_metrics
    from evaluate_independent import make_evidence_validator,capture_contract,verify_population
    if len(legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('One shared lease required')
    versions=dict(torch=str(torch.__version__),ultralytics=str(legacy.ultralytics.__version__))
    if versions!={'torch':'2.10.0+cu128','ultralytics':'8.4.115'}:raise ValueError('Pinned versions differ')
    spec=json.loads(a.spec.read_text())['models'][a.model]
    checkpoint=Path(spec['checkpoint']);data_source=Path(spec['data'])
    roster=dev_roster(data_source)
    if len(roster)!=2406:raise ValueError('Expected complete LLVIP dev2406')
    if a.output.exists():raise FileExistsError('New attempt required')
    if not str(a.output.resolve()).startswith('/mnt/dataset/yudongfang/'):raise ValueError('Data disk only')
    a.output.mkdir(parents=True)
    started=time.time();initial=state(checkpoint)
    try:
        shutil.copyfile(a.spec,a.output/'input_spec.json')
        shutil.copyfile(data_source,a.output/'original_data.yaml')
        old_args=checkpoint.parents[1]/'args.yaml'
        shutil.copyfile(old_args,a.output/'original_args.yaml')
        source_dir=a.output/'sources';source_dir.mkdir()
        sources=[Path(__file__),RELEASE/'evaluate_independent.py',RELEASE/'evaluator_profile.py',
                 *map(Path,legacy.implementation_files(DetectionValidator,BaseValidator,YOLO.val,non_max_suppression,ap_per_class))]
        for i,p in enumerate(dict.fromkeys(sources)):shutil.copyfile(p,source_dir/(str(i)+'_'+p.name))
        shapes=set()
        for p in roster:
            with Image.open(p) as im:shapes.add(im.size)
        if len(shapes)!=1:raise ValueError('Canary requires full-roster uniform image dimensions')
        actual=roster[:64] if a.canary else roster
        data=yaml.safe_load(data_source.read_text())
        if 'test' in data:raise ValueError('Train/dev YAML only')
        lst=a.output/'evaluation_roster.txt';lst.write_text(''.join(p+'\n' for p in actual))
        subset=dict(path=str(a.output),train=str(lst),val=str(lst),names=data['names'])
        evaluation_data=a.output/'evaluation_only_data.yaml'
        evaluation_data.write_text(yaml.safe_dump(subset,allow_unicode=True))
        write(a.output/'population.json',dict(full_roster=roster,evaluated_roster=actual,
            full_shape_set=sorted(shapes),source_data=str(data_source),data_role='development_val',
            train_yaml_key_is_evaluation_alias=True,official_test_accessed=False))
        results={};contracts={};identities={};peak_reserved=0;peak_allocated=0
        for label,validator_class in ([('native',DetectionValidator),('capture',make_evidence_validator(DetectionValidator))]
                                     if a.canary else [('capture',make_evidence_validator(DetectionValidator))]):
            random.seed(20260907);np.random.seed(20260907);torch.manual_seed(20260907);torch.cuda.manual_seed_all(20260907)
            torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
            model=YOLO(str(checkpoint),task='detect')
            ckpt=model.ckpt
            if ckpt.get('epoch') not in (-1,199):raise ValueError('Unexpected historical E200 checkpoint epoch')
            if len(model.names)!=1:raise ValueError('Expected one LLVIP class')
            args=yaml.safe_load(old_args.read_text())
            if args.get('seed')!=42 or args.get('epochs')!=200:raise ValueError('Unexpected historical training identity')
            if Path(args['data']).resolve()!=data_source.resolve():raise ValueError('Training modality/data differs from requested evaluation')
            def names_dict(value):
                return {int(k):v for k,v in (enumerate(value) if isinstance(value,list) else value.items())}
            if names_dict(model.names)!=names_dict(data['names']):raise ValueError('Model/data class names or ordering differ')
            identities[label]=dict(path=str(checkpoint),stat=initial,checkpoint_epoch=ckpt.get('epoch'),
                ema_present=ckpt.get('ema') is not None,loaded_source='ema' if ckpt.get('ema') is not None else 'model',
                names=model.names,training_args=args,scope='historical_baseline_not_new_protocol_N')
            contract={}
            def on_start(v):contract.update(capture_contract(v,actual,versions))
            def on_end(v):
                if v.seen!=len(actual):raise ValueError('Incomplete actual population')
                if label=='capture':verify_population(v,contract)
                else:contract['observed_images']=int(v.seen)
            model.add_callback('on_val_start',on_start);model.add_callback('on_val_end',on_end)
            m=model.val(validator=validator_class,data=str(evaluation_data),split='val',imgsz=640,batch=32,
                workers=4,device='0',plots=False,save_json=False,verbose=False,project=str(a.output),
                name=label,exist_ok=False,rect=True,conf=.001,iou=.7,max_det=300,augment=False,
                agnostic_nms=False,single_cls=False,quantize=None)
            result=metric_record(m,model.names)
            if contract['effective_kwargs']['half'] or contract['effective_kwargs']['quantize'] is not None:
                raise ValueError('Actual evaluation precision is not FP32')
            results[label]=result;contracts[label]=contract
            peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved()/2**20)
            peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated()/2**20)
            write(a.output/(label+'_metrics.json'),result);write(a.output/(label+'_contract.json'),contract)
            del model,m,ckpt;gc.collect();torch.cuda.empty_cache()
        if a.canary:
            assert_equal_metrics(results['native'],results['capture'])
            if contracts['native']['effective_kwargs']!=contracts['capture']['effective_kwargs']:raise ValueError('Kwargs differ')
            if contracts['native']['actual_loader_roster']!=contracts['capture']['actual_loader_roster']:raise ValueError('Loader order differs')
        if state(checkpoint)!=initial:raise ValueError('Input checkpoint changed')
        write(a.output/'model_identity.json',identities)
        write(a.output/'summary.json',dict(status='completed',model=a.model,dataset='llvip',
            images=len(actual),full_population=2406,canary=a.canary,native_capture_exact=a.canary,
            metric_units='fraction_0_to_1',metrics=results['capture'],checkpoint=str(checkpoint),
            input_data=str(data_source),historical_training_args=str(old_args),
            objects=str(a.output/'capture/objects.jsonl.gz'),evaluation_contract=str(a.output/'capture_contract.json'),
            resources=legacy.bound_lease_resource_record_from_environment(),gpu_reserved_peak_mib=peak_reserved,
            gpu_allocated_peak_mib=peak_allocated,seconds=time.time()-started,official_test_accessed=False,
            training_modified=False,new_protocol_N=False,physical_alignment_verified=False))
    except BaseException as e:
        write(a.output/'failure_receipt.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.time()-started))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--model',choices=['N42','T42'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--canary',action='store_true');run(p.parse_args())
