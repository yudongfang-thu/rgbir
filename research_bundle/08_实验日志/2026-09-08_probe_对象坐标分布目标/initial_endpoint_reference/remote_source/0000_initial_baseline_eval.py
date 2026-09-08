"""LLVIP initial checkpoint recheck under current full-dev native FP32 settings."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import yaml

SCOPE='INITIAL_BASELINE_NATIVE_RECHECK'
ENDPOINT='INITIAL_FIXED_LAST_EMA_RECHECK'
EFFECTIVE=dict(imgsz=640,batch=32,workers=4,quantize=None,conf=.001,iou=.7,max_det=300,
               agnostic_nms=False,single_cls=False,rect=True,augment=False,half=False)


def require(value,message):
    if not value:raise ValueError(message)


def stat(path):
    p=Path(path).resolve();s=p.stat()
    return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def copy_sources(output,sources):
    dest=output/'source_copies';dest.mkdir();rows=[]
    for i,p in enumerate(sorted({Path(p).resolve() for p in sources},key=str)):
        q=dest/('%04d_'%i+p.name);shutil.copyfile(p,q)
        require(p.read_bytes()==q.read_bytes(),'Source copy differs')
        rows.append(dict(stat(p),copy=str(q),byte_identity=True))
    write_new(output/'source_manifest.json',dict(files=rows,new_hash_computed=False))


def check_config(cfg,checkpoint):
    for k,v in dict(dataset='llvip',expected_nc=1,expected_val_images=2406,
                    imgsz=640,batch=32,workers=4,seed=42).items():
        require(cfg.get(k)==v,'Fixed LLVIP evaluation configuration differs: '+k)
    require(checkpoint.name=='last.pt' and checkpoint.parent.name=='weights','Fixed initial last checkpoint required')
    require(Path(cfg['model']).resolve()==checkpoint.resolve(),'Checkpoint differs from explicitly declared initialization')
    full=Path(cfg['paths']['student_data_yaml'])
    require(full.resolve()==Path(cfg['auxiliary_data_identity']['student_data_yaml']).resolve(),
            'Evaluation must use the original full visible data identity')
    return str(full)


def run(args):
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8-sig'))
    checkpoint=args.checkpoint.resolve();data=check_config(cfg,checkpoint);before=stat(checkpoint)
    output=args.output.resolve()
    require(os.name=='posix' and Path('/mnt/dataset/yudongfang') in output.parents,'Project data disk required')
    require(not output.exists() and checkpoint.parent.parent not in output.parents,'New output outside source run required')
    ref=args.reference_dir.resolve();sys.path.insert(0,str(ref))
    import torch
    import runtime
    import evaluator_profile as profile
    import evaluate_independent as helpers
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import DetMetrics,Metric,ap_per_class
    from ultralytics.utils.nms import non_max_suppression
    for module,name in ((runtime,'runtime.py'),(profile,'evaluator_profile.py'),(helpers,'evaluate_independent.py')):
        require(Path(module.__file__).resolve()==ref/name,'Wrong pinned helper: '+name)
    require(len(runtime.legacy.require_bound_lease_from_environment()['gpus'])==1,'One bound globallease required')
    versions=dict(torch=str(torch.__version__),ultralytics=str(runtime.legacy.ultralytics.__version__))
    require(versions=={k:cfg[k+'_version'] for k in versions},'Pinned environment differs')
    canonical=profile.dev_roster(data)
    require(len(canonical)==2406 and len(set(canonical))==2406,'Full unique LLVIP dev required')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.config,output/'requested_native_config.yaml')
    (output/'development_roster.txt').write_text(''.join(p+'\n' for p in canonical),encoding='utf-8')
    sources={Path(__file__),args.config,Path(data),ref/'runtime.py',ref/'evaluator_profile.py',ref/'evaluate_independent.py'}
    sources.update(Path(p) for p in runtime.legacy.implementation_files(YOLO.val,DetectionValidator,BaseValidator,
        AutoBackend,check_det_dataset,DetMetrics,Metric,ap_per_class,non_max_suppression))
    started=time.perf_counter();captured={};model=None

    def on_start(v):
        captured.update(helpers.capture_contract(v,canonical,versions))
        require(captured['effective_kwargs']==EFFECTIVE,'Actual native FP32 kwargs differ')
        labels=v.dataloader.dataset.labels
        count=sum(len(row['cls']) for row in labels)
        require(len(labels)==2406 and count==7879,'Actual LLVIP dev GT population differs')
        require(all((row['cls']==0).all() for row in labels),'Unexpected LLVIP GT class')
        names=v.dataloader.dataset.data['names']
        require(len(names)==1 and len(model.names)==1,'Single-class native mapping required')
        dataset_names=dict(enumerate(names)) if isinstance(names,list) else names
        model_names=dict(enumerate(model.names)) if isinstance(model.names,list) else model.names
        require(dataset_names==model_names,'Actual dataset/model class names differ')
        captured.update(scope=SCOPE,endpoint=ENDPOINT,dataset='llvip',gt_objects_before_inference=count,
            actual_class_names=names,checkpoint=before,formal_training_endpoint=False)
        captured['runtime_sources']=profile.capture_runtime_sources(v,model,sources)

    def on_end(v):
        helpers.verify_population(v,captured)
        with gzip.open(v.save_dir/'objects.jsonl.gz','rt',encoding='utf-8') as f:
            records=[json.loads(line) for line in f if line.strip()]
        require(sum(len(r['gt_classes']) for r in records)==7879 and
                all(len(r['gt_classes'])==len(r['gt_boxes']) for r in records),'Captured GT population differs')
        captured['gt_objects_captured']=7879

    try:
        torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
        model=YOLO(str(checkpoint),task='detect')
        model.add_callback('on_val_start',on_start);model.add_callback('on_val_end',on_end)
        metrics=model.val(validator=helpers.make_evidence_validator(DetectionValidator),data=data,split='val',
            imgsz=640,batch=32,workers=4,device='0',quantize=None,conf=.001,iou=.7,max_det=300,
            agnostic_nms=False,single_cls=False,rect=True,augment=False,plots=False,save_json=False,
            verbose=False,project=str(output),name='native_capture',exist_ok=False)
        values=profile.metric_record(metrics,model.names)
        require(len(values['per_class'])==1 and values['per_class'][0]['class_id']==0,'One actual class AP required')
        for k in ('AP50','AP75','mAP50_95','precision','recall'):
            require(type(values[k]) in (int,float) and math.isfinite(values[k]) and 0<=values[k]<=1,'Invalid native fraction')
        for k in ('AP50','AP75','mAP50_95'):
            require(math.isclose(values['per_class'][0][k],values[k],rel_tol=0,abs_tol=1e-12),'Single-class AP differs')
        require(stat(checkpoint)==before,'Initial checkpoint changed during inference')
        copy_sources(output,sources)
        write_new(output/'initial_evaluation_contract.json',captured)
        result=dict(status='INITIAL_BASELINE_EVALUATION_COMPLETED',scope=SCOPE,endpoint=ENDPOINT,
            dataset='llvip',seed=42,single_seed=True,checkpoint=before,full_dev_images=2406,full_dev_gt_objects=7879,
            observed_images=captured['observed_images'],gt_objects_captured=captured['gt_objects_captured'],
            metric_units='fraction_0_to_1',**values,actual_effective_kwargs=captured['effective_kwargs'],
            actual_evaluation_data_yaml=data,configuration=str(args.config.resolve()),
            configuration_role='Only native full-dev evaluation fields consumed; inherited FT training fields ignored',
            training_receipt_required=False,training_performed=False,formal_training_endpoint=False,
            evaluation_epoch_budget=None,prior_LLVIP_profile_binding_claim=False,accepted_endpoint_claim=False,
            matrix_changed=False,parameters_or_thresholds_changed=False,checkpoint_written=False,
            optimizer_created=False,formal_paper_gain_claim=False,formal_e200_complete=False,
            seconds=time.perf_counter()-started,resources=runtime.legacy.bound_lease_resource_record_from_environment(),
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            new_hash_computed=False,official_test_accessed=False)
        write_new(output/'initial_evaluation_receipt.json',result)
        return result
    except BaseException as error:
        write_new(output/'initial_evaluation_failure.json',dict(status='INITIAL_BASELINE_EVALUATION_FAILED',scope=SCOPE,
            error=repr(error),traceback=traceback.format_exc(),seconds=time.perf_counter()-started,
            checkpoint_written=False,new_hash_computed=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('reference-dir','config','checkpoint','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())
