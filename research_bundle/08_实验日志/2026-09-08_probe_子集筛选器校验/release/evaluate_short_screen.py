"""Independent SHORT_SCREEN full-dev native evaluation; rejects E200 receipts."""
import argparse
import gzip
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
import yaml

from screen_common import (SCOPE,ENDPOINT,Deadline,ExecutionDeadline,copy_sources,data_output,load_config,read,stat,write_new)

EFFECTIVE=dict(imgsz=640,batch=32,workers=4,quantize=None,conf=.001,iou=.7,max_det=300,
               agnostic_nms=False,single_cls=False,rect=True,augment=False,half=False)


def short_completion(run,cfg):
    if (run/'short_training_failure.json').exists():raise ValueError('Training failure conflicts with completion')
    result=read(run/'short_training_receipt.json')
    if (result.get('status')!='SUBSET_SCREEN_TRAINING_COMPLETED' or result.get('scope')!=SCOPE
        or result.get('single_seed') is not True or result.get('last_epoch')!=8 or result.get('epochs_configured')!=8
        or result.get('endpoint')!=ENDPOINT or result.get('formal_e200_complete') is not False
        or result.get('new_hash_computed') is not False or result.get('official_test_accessed') is not False
        or result.get('batches')!=512 or result.get('expected_train_images')!=2048
        or result.get('bn_running_statistics')!='normal_training' or result.get('loader_workers_cleaned') is not True):
        raise ValueError('Completed independent E8 SHORT_SCREEN receipt required')
    for key in ('arm','seed','dataset','model','teacher','reference','classification_coefficient','localization_coefficient'):
        if result.get(key)!=cfg[key]:raise ValueError('Short completion identity differs: '+key)
    checkpoint=run/'weights'/'last.pt'
    if result['checkpoint']!=stat(checkpoint):raise ValueError('Fixed last/EMA checkpoint stat differs')
    if Path(result['configuration']).resolve()!=(run/'short_screen_config.yaml').resolve():raise ValueError('Executed config path differs')
    for p,key in [(result['flow_prefix_receipt'],'exact_pixels_and_labels'),(result['initialization_receipt'],'cross_arm_initial_state_exact')]:
        r=read(p)
        if r.get('status')!='PASS' or r.get(key) is not True:raise ValueError('Actual training flow/initial state not exact')
    from train_short_screen import check_canary
    check_canary(Path(result['canary_receipt']['path']),cfg,run/'short_screen_config.yaml',result['flow_reference'])
    return result,checkpoint


def evaluation_projection(cfg,native_cfg,native_config_path,profile,binding,ref):
    # Same executed hourly-screen projection: subset train, unchanged FULL dev.
    if Path(cfg['native_contract_config']).resolve()!=Path(native_config_path).resolve():raise ValueError('Native config differs')
    for key in ('dataset','expected_nc','expected_val_images','imgsz','batch','workers','torch_version','ultralytics_version'):
        if cfg[key]!=native_cfg[key]:raise ValueError('Evaluation invariant differs: '+key)
    identity=profile.validate_evaluation_profile_binding(binding,native_cfg,ref)
    full_data=native_cfg['paths']['student_data_yaml']
    if Path(full_data).resolve()!=Path(cfg['auxiliary_data_identity']['student_data_yaml']).resolve():raise ValueError('Full data identity differs')
    canonical=profile.dev_roster(full_data)
    if len(canonical)!=1469 or canonical!=profile.dev_roster(cfg['paths']['student_data_yaml']):raise ValueError('Full subset dev roster differs')
    return identity,dict(kind='explicit_subset_training_to_original_full_dev_evaluation',
        training_model=cfg['model'],training_student_data_yaml=cfg['paths']['student_data_yaml'],
        native_contract_configuration=stat(native_config_path),actual_evaluation_data_yaml=full_data,
        native_profile_binding=stat(binding),subset_dev_roster_exact_to_full=True)


def run(args):
    deadline=Deadline(args.wall_seconds)
    cfg=load_config(args.run/'short_screen_config.yaml')
    completed,checkpoint=short_completion(args.run,cfg)
    output=data_output(args.output)
    if output.exists():raise FileExistsError(output)
    ref=args.reference_dir.resolve();sys.path.insert(0,str(ref))
    import torch
    import runtime
    import evaluator_profile as profile
    import evaluate_independent as formal_helpers
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import DetMetrics,Metric,ap_per_class
    from ultralytics.utils.nms import non_max_suppression
    for module,name in [(runtime,'runtime.py'),(profile,'evaluator_profile.py'),(formal_helpers,'evaluate_independent.py')]:
        if Path(module.__file__).resolve()!=ref/name:raise RuntimeError('Wrong native helper binding: '+name)
    if len(runtime.legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('One bound globallease required')
    versions=dict(torch=str(torch.__version__),ultralytics=str(runtime.legacy.ultralytics.__version__))
    if versions!={k:cfg[k+'_version'] for k in versions}:raise ValueError('Pinned environment changed')
    # Only pure roster/byte binding/metric helpers are reused. Never invoke the
    # E200 load_run_configuration/run/publish/emit paths or create their receipt.
    native_config=Path(cfg['native_contract_config'])
    native_cfg=yaml.safe_load(native_config.read_text(encoding='utf-8'))
    identity,projection=evaluation_projection(cfg,native_cfg,native_config,profile,args.native_profile_binding,ref)
    if identity['actual_effective_kwargs']!=EFFECTIVE:raise ValueError('Accepted native evaluator kwargs differ')
    canonical=profile.dev_roster(cfg['paths']['student_data_yaml'])
    if len(canonical)!=1469:raise ValueError('Full dev1469 required')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.run/'short_screen_config.yaml',output/'short_screen_config.yaml')
    shutil.copyfile(args.run/'short_training_receipt.json',output/'short_training_receipt_copy.json')
    (output/'development_roster.txt').write_text(''.join(p+'\n' for p in canonical),encoding='utf-8')
    captured={};sources=set(Path(__file__).parent.glob('*.py'))|{ref/'evaluator_profile.py',ref/'evaluate_independent.py'}
    sources.update(Path(p) for p in runtime.legacy.implementation_files(DetectionValidator,BaseValidator,AutoBackend,
        YOLO.val,check_det_dataset,DetMetrics,Metric,ap_per_class,non_max_suppression))
    started=deadline.started
    model=None
    def on_start(v):
        deadline.check()
        captured.update(formal_helpers.capture_contract(v,canonical,versions))
        if captured['effective_kwargs']!=EFFECTIVE:raise ValueError('Actual native settings changed')
        labels=v.dataloader.dataset.labels
        if len(labels)!=1469 or sum(len(x['cls']) for x in labels)!=22462:raise ValueError('Full dev GT labels missing or changed')
        captured['gt_objects_before_inference']=22462
        captured['runtime_sources']=profile.capture_runtime_sources(v,model,sources)
        captured['endpoint']=ENDPOINT;captured['scope']=SCOPE;captured['formal_e200_complete']=False
    def on_end(v):
        deadline.check()
        formal_helpers.verify_population(v,captured)
        with gzip.open(v.save_dir/'objects.jsonl.gz','rt',encoding='utf-8') as f:
            import json
            records=[json.loads(line) for line in f if line.strip()]
        total=sum(len(r['gt_classes']) for r in records)
        if total!=22462 or any(len(r['gt_classes'])!=len(r['gt_boxes']) for r in records):raise ValueError('Captured GT population changed')
        captured['gt_objects_captured']=total
    try:
        model=YOLO(str(checkpoint),task='detect')
        model.add_callback('on_val_start',on_start);model.add_callback('on_val_end',on_end)
        deadline.check()
        metrics=model.val(validator=formal_helpers.make_evidence_validator(DetectionValidator),
            data=cfg['auxiliary_data_identity']['student_data_yaml'],split='val',imgsz=640,batch=32,workers=4,device='0',
            conf=.001,iou=.7,max_det=300,agnostic_nms=False,single_cls=False,rect=True,augment=False,
            quantize=None,plots=False,save_json=False,verbose=False,project=str(output),name='native_capture',exist_ok=False)
        deadline.check()
        values=profile.metric_record(metrics,model.names)
        if {r['class_id'] for r in values['per_class']}!=set(range(5)):raise ValueError('All five GT classes must be present in native AP')
        if not all(math.isfinite(values[k]) and 0<=values[k]<=1 for k in ('AP50','AP75','mAP50_95','precision','recall')):raise ValueError('Invalid native metric fractions')
        for row in values['per_class']:
            if not all(math.isfinite(row[k]) and 0.<=row[k]<=1. for k in ('AP50','AP75','mAP50_95')):
                raise ValueError('Invalid per-class native AP fraction')
        if completed['checkpoint']!=stat(checkpoint):raise ValueError('Checkpoint changed during evaluation')
        sources.update([args.run/'short_screen_config.yaml',args.native_profile_binding,native_config])
        copy_sources(output,sources)
        write_new(output/'short_evaluation_contract.json',captured)
        deadline.check()
        write_new(output/'short_evaluation_receipt.json',dict(status='SUBSET_SCREEN_EVALUATION_COMPLETED',
            scope=SCOPE,single_seed=True,seed=42,arm=cfg['arm'],dataset='dronevehicle',endpoint=ENDPOINT,
            epochs=8,independent_lr_horizon=8,formal_e200_complete=False,formal_paper_gain_claim=False,
            metric_units='fraction_0_to_1',native_metric_definition='pinned Ultralytics native AP; not TIDE oracle',
            **values,checkpoint=completed['checkpoint'],full_dev_images=1469,full_dev_gt_objects=22462,
            native_profile_binding=str(args.native_profile_binding),seconds=time.perf_counter()-started,
            resources=runtime.legacy.bound_lease_resource_record_from_environment(),
            new_hash_computed=False,official_test_accessed=False,
            classification_coefficient=cfg['classification_coefficient'],expected_train_images=2048,
            training_subset_identity=cfg['paths'],initialization=completed['initialization'],
            bn_running_statistics='normal_training',training_configuration=stat(args.run/'short_screen_config.yaml'),
            training_completion=stat(args.run/'short_training_receipt.json'),evaluation_identity_projection=projection,
            wall_limit_seconds=args.wall_seconds))
    except BaseException as error:
        write_new(output/'short_evaluation_failure.json',dict(status='SUBSET_SCREEN_INCOMPLETE' if isinstance(error,ExecutionDeadline) else 'SUBSET_SCREEN_EVALUATION_FAILED',scope=SCOPE,error=repr(error),
            traceback=traceback.format_exc(),seconds=time.perf_counter()-started,new_hash_computed=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-profile-binding',type=Path,required=True);p.add_argument('--wall-seconds',type=float,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
