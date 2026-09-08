"""Read-only full development evaluation of historical Drone IR42 teacher."""
from pathlib import Path
import argparse, gc, gzip, inspect, json, math, random, shutil, sys, time, traceback

REPO=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
RELEASE=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(RELEASE))

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def state(path):
    s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns)

def evaluation_aliases(data,data_source):
    """Preserve processed image paths so YOLO finds processed labels."""
    root=Path(data.get('path',Path(data_source).parent))
    if not root.is_absolute():root=Path(data_source).parent/root
    values=data['val'] if isinstance(data['val'],list) else [data['val']]
    paths=[]
    for value in values:
        p=Path(value);p=p if p.is_absolute() else root/p
        if not p.is_dir():raise ValueError('This frozen Drone IR val source must be a directory')
        paths.extend(x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'})
    aliases=sorted(map(str,paths))
    if len(aliases)!=len(set(aliases)):raise ValueError('Duplicate aliases')
    return aliases

SCOPE='DRONE_IR42_FULL_DEV_CAPTURE'
CLASS_NAMES=['car','freight car','truck','bus','van']

def label_array(lines):
    import numpy as np
    if any(len(row)!=5 for row in lines):raise ValueError('YOLO label requires five fields')
    array=np.asarray(lines,dtype=np.float32).reshape(-1,5)
    if (not np.isfinite(array).all() or (array[:,0]!=np.floor(array[:,0])).any()
        or (array[:,0]<0).any() or (array[:,0]>=5).any()
        or (array[:,1:]<0).any() or (array[:,1:]>1).any() or (array[:,3:]<=0).any()):
        raise ValueError('Invalid Drone IR normalized labels')
    return array

def validate_metrics(result,expected_class_ids):
    if [r['class_id'] for r in result['per_class']]!=expected_class_ids:
        raise ValueError('Missing, reordered or unexpected observed-class metrics')
    for key in ('AP50','AP75','mAP50_95','precision','recall'):
        if not math.isfinite(result[key]) or not 0<=result[key]<=1:raise ValueError('Invalid metric: '+key)
    for row in result['per_class']:
        if row['name']!=CLASS_NAMES[row['class_id']]:raise ValueError('Metric class name differs')
        for key in ('AP50','AP75','mAP50_95'):
            if not math.isfinite(row[key]) or not 0<=row[key]<=1:raise ValueError('Invalid per-class metric')

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
    from ultralytics.data.utils import img2label_paths
    from evaluator_profile import dev_roster,metric_record,assert_equal_metrics
    from evaluate_independent import make_evidence_validator,capture_contract,verify_population
    if len(legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('One shared lease required')
    versions=dict(torch=str(torch.__version__),ultralytics=str(legacy.ultralytics.__version__))
    if versions!={'torch':'2.10.0+cu128','ultralytics':'8.4.115'}:raise ValueError('Pinned versions differ')
    frozen=json.loads(a.spec.read_text());spec=frozen['models'][a.model]
    if frozen['scope']!=SCOPE:raise ValueError('Wrong capture scope')
    checkpoint=Path(spec['checkpoint']);data_source=Path(spec['data'])
    roster=dev_roster(data_source)
    if len(roster)!=1469:raise ValueError('Expected complete Drone IR dev1469')
    data=yaml.safe_load(data_source.read_text())
    if state(checkpoint)!=spec['checkpoint_stat']:raise ValueError('Frozen teacher checkpoint stat differs')
    if {int(k):v for k,v in data['names'].items()}!=dict(enumerate(CLASS_NAMES)):raise ValueError('Wrong IR class order')
    aliases=evaluation_aliases(data,data_source)
    if sorted(str(Path(p).resolve()) for p in aliases)!=roster:raise ValueError('Processed aliases do not bind the canonical roster')
    if aliases!=spec['aliases'] or roster!=spec['canonical']:raise ValueError('Frozen full dev roster differs')
    all_labels=img2label_paths(aliases);label_counts=[];expected_labels={}
    for alias,path in zip(aliases,all_labels):
        p=Path(path)
        if not p.is_file():raise ValueError('Missing required original processed GT: '+str(p))
        lines=[line.split() for line in p.read_text().splitlines() if line.strip()]
        array=label_array(lines)
        expected_labels[alias]=array
        label_counts.append(len(lines))
    if label_counts!=spec['label_counts'] or sum(label_counts)!=24490:raise ValueError('Frozen per-image IR GT counts differ')
    full_class_counts=np.bincount(np.concatenate(list(expected_labels.values()))[:,0].astype(int),minlength=5).tolist()
    if full_class_counts!=spec['class_counts']:raise ValueError('Frozen full IR class counts differ')
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
        if shapes!={(640,512)}:raise ValueError('Frozen original image dimensions differ')
        actual_aliases=aliases[:32] if a.canary else aliases
        actual=[str(Path(p).resolve()) for p in actual_aliases]
        expected_gt=sum(label_counts[:32] if a.canary else label_counts)
        class_counts=np.bincount(np.concatenate([expected_labels[p] for p in actual_aliases])[:,0].astype(int),minlength=5).tolist()
        if a.canary and (expected_gt!=514 or class_counts!=[447,24,30,13,0]):raise ValueError('Frozen first32 GT/classes differ')
        expected_class_ids=[i for i,n in enumerate(class_counts) if n>0]
        if 'test' in data:raise ValueError('Train/dev YAML only')
        lst=a.output/'evaluation_roster.txt';lst.write_text(''.join(p+'\n' for p in actual_aliases))
        subset=dict(path=str(a.output),train=str(lst),val=str(lst),names=data['names'])
        evaluation_data=a.output/'evaluation_only_data.yaml'
        evaluation_data.write_text(yaml.safe_dump(subset,allow_unicode=True))
        write(a.output/'population.json',dict(full_roster=roster,evaluated_roster=actual,
            evaluated_aliases=actual_aliases,original_label_files=all_labels,original_label_counts=label_counts,
            expected_evaluated_gt=expected_gt,full_gt=sum(label_counts),class_counts=class_counts,
            full_class_counts=full_class_counts,expected_metric_class_ids=expected_class_ids,
            empty_gt_images=sum(n==0 for n in label_counts),independent_ir_labels=True,
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
            if len(model.names)!=5:raise ValueError('Expected five Drone classes')
            args=yaml.safe_load(old_args.read_text())
            if args.get('seed')!=42 or args.get('epochs')!=200:raise ValueError('Unexpected historical training identity')
            if Path(args['data']).resolve()!=data_source.resolve():raise ValueError('Training modality/data differs from requested evaluation')
            def names_dict(value):
                return {int(k):v for k,v in (enumerate(value) if isinstance(value,list) else value.items())}
            if names_dict(model.names)!=names_dict(data['names']):raise ValueError('Model/data class names or ordering differ')
            identities[label]=dict(path=str(checkpoint),stat=initial,checkpoint_epoch=ckpt.get('epoch'),
                ema_present=ckpt.get('ema') is not None,loaded_source='ema' if ckpt.get('ema') is not None else 'model',
                names=model.names,training_args=args,scope='historical_native_infrared_teacher_seed42_E200')
            contract={}
            def on_start(v):
                contract.update(capture_contract(v,actual,versions))
                contract['alias_to_canonical']={p:str(Path(p).resolve()) for p in v.dataloader.dataset.im_files}
                observed=sum(len(row['cls']) for row in v.dataloader.dataset.labels)
                if observed!=expected_gt:raise ValueError('Actual loader GT differs from processed labels: '+str((observed,expected_gt)))
                for row in v.dataloader.dataset.labels:
                    expected=expected_labels[row['im_file']]
                    if not np.array_equal(np.asarray(row['cls']).reshape(-1),expected[:,0]) or not np.array_equal(row['bboxes'],expected[:,1:]):
                        raise ValueError('Loader per-image normalized GT differs from original processed file')
                contract['expected_gt']=expected_gt;contract['loader_gt']=observed
                contract['loader_per_image_labels_exact']=True
                contract['scope']=SCOPE;contract['independent_ir_labels']=True
                contract['expected_metric_class_ids']=expected_class_ids
            def on_end(v):
                if v.seen!=len(actual):raise ValueError('Incomplete actual population')
                if label=='capture':
                    verify_population(v,contract)
                    with gzip.open(v.save_dir/'objects.jsonl.gz','rt',encoding='utf-8') as f:
                        captured_gt=sum(len(json.loads(line)['gt_boxes']) for line in f if line.strip())
                    if captured_gt!=expected_gt:raise ValueError('Captured GT population differs')
                    contract['captured_gt']=captured_gt
                else:contract['observed_images']=int(v.seen)
            model.add_callback('on_val_start',on_start);model.add_callback('on_val_end',on_end)
            m=model.val(validator=validator_class,data=str(evaluation_data),split='val',imgsz=640,batch=32,
                workers=4,device='0',plots=False,save_json=False,verbose=False,project=str(a.output),
                name=label,exist_ok=False,rect=True,conf=.001,iou=.7,max_det=300,augment=False,
                agnostic_nms=False,single_cls=False,quantize=None,half=False)
            result=metric_record(m,model.names)
            validate_metrics(result,expected_class_ids)
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
        write(a.output/'summary.json',dict(status='completed',scope=SCOPE,model=a.model,dataset='dronevehicle',
            images=len(actual),full_population=1469,canary=a.canary,native_capture_exact=a.canary,
            gt_count=expected_gt,full_gt=24490,class_counts=class_counts,class_names=CLASS_NAMES,
            metric_units='fraction_0_to_1',metrics=results['capture'],checkpoint=str(checkpoint),
            input_data=str(data_source),historical_training_args=str(old_args),
            objects=str(a.output/'capture/objects.jsonl.gz'),evaluation_contract=str(a.output/'capture_contract.json'),
            resources=legacy.bound_lease_resource_record_from_environment(),gpu_reserved_peak_mib=peak_reserved,
            gpu_allocated_peak_mib=peak_allocated,seconds=time.time()-started,official_test_accessed=False,
            training_modified=False,new_protocol_N=False,physical_alignment_verified=False,
            new_hash_computed=False,optimizer_updates=0,independent_ir_labels=True))
    except BaseException as e:
        write(a.output/'failure_receipt.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.time()-started))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--model',choices=['T42'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--canary',action='store_true');run(p.parse_args())
