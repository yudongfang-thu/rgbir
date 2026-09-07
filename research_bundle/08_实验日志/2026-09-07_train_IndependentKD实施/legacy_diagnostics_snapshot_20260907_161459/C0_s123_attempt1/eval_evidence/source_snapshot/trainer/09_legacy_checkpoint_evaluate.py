"""Post-hoc evaluation of six actual legacy N/C0 checkpoints; new artifacts only.

CPU import/inspection is safe. GPU execution requires an independent code review,
a live shared resource lease and the successful frozen evaluation profile. This
module reuses the frozen validator/metrics; it never creates a training receipt.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import yaml

HERE=Path(__file__).resolve().parent
REVIEW_SCHEMA='rgbir-legacy-evaluation-wrapper-review-v1'
REVIEW_FILES=('legacy_checkpoint_evaluate.py','prepare_queue.py','test_legacy_endpoint_eval.py','README.md')
METRICS=('AP50','AP75','mAP50_95','precision','recall')
ARMS={'N':'weight0','C0':'paired'}
DERIVED_KEYS=('seed','arm','source','expected_val_images','evaluation_kind')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def copy_new(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:stream.write(Path(source).read_bytes())


def frozen_module(release,name):
    path=(Path(release)/(name+'.py')).resolve()
    if name in sys.modules:
        value=sys.modules[name]
        if Path(value.__file__).resolve()!=path:raise ValueError('Different release already imported: '+name)
        return value
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def require_review(path):
    path=Path(path);value=read(path)
    if value.get('schema')!=REVIEW_SCHEMA or value.get('status')!='ACCEPTED' or not value.get('reviewer'):
        raise ValueError('Legacy evaluation wrapper needs independent source review')
    rows=value.get('source_files',[])
    if len(rows)!=len(REVIEW_FILES) or {row.get('relative') for row in rows}!=set(REVIEW_FILES):
        raise ValueError('Review must bind all wrapper/queue/tests/rules source files')
    for row in rows:
        accepted=Path(row['accepted_copy'])
        if not accepted.is_absolute():accepted=path.parent/accepted
        actual=HERE/row['relative']
        if accepted.resolve()==actual.resolve() or accepted.read_bytes()!=actual.read_bytes():
            raise ValueError('Reviewed wrapper source bytes differ: '+row['relative'])
    return value


def derived_config(train_cfg,arm,seed):
    if arm not in ARMS or seed not in (0,42,123):raise ValueError('Only legacy N/C0 three seeds are in scope')
    value=copy.deepcopy(train_cfg)
    # Original YAML sometimes used seed42 as its default; actual train/eval
    # receipts, not that default, identify which seed was trained.
    value.update(seed=seed,arm=ARMS[arm],source='paired',expected_val_images=1469,
                 evaluation_kind='posthoc_legacy_checkpoint_diagnostics')
    return value


def load_legacy_metadata(run,arm,seed,cfg,analyzer):
    run=Path(run)
    if arm not in ARMS or seed not in (0,42,123):raise ValueError('Unsupported old arm/seed')
    spec=dict(arm=arm,source_arm=ARMS[arm],seed=seed,path=str(run),protocol_id='oev1_frozen_drone_e200')
    record=analyzer._legacy_loader().load_endpoint(spec,run.parent)
    if record['status']!='complete':raise ValueError('Invalid actual legacy endpoint: '+repr(record['issues']))
    complete,metric=read(run/'completion_receipt.json'),read(run/'evaluation_val.json')
    if complete.get('last_epoch')!=200 or complete.get('epochs_configured')!=200:
        raise ValueError('Original full E200 last/EMA endpoint is required')
    train_receipt,eval_receipt=read(run/'run_evidence/run_receipt.json'),read(run/'eval_evidence/run_receipt.json')
    train_cfg,_=analyzer._bound_configs(run,train_receipt,'run_evidence')
    eval_cfg,_=analyzer._bound_configs(run,eval_receipt,'eval_evidence')
    if cfg!=derived_config(train_cfg,arm,seed):raise ValueError('Requested evaluation config differs from original bound recipe/identity')
    if (train_cfg.get('dataset')!='dronevehicle' or train_cfg.get('expected_nc')!=5
        or len(record['roster'])!=1469 or train_cfg.get('epochs')!=200):
        raise ValueError('Only the actual Drone five-class full dev1469 protocol is supported')
    for key in ('model','imgsz','batch','workers','torch_version','ultralytics_version','paths'):
        if train_cfg.get(key)!=eval_cfg.get(key):raise ValueError('Original bound train/eval identity differs: '+key)
    for key,cfg_key in (('torch','torch_version'),('ultralytics','ultralytics_version')):
        if eval_receipt.get('environment',{}).get(key)!=cfg[cfg_key]:raise ValueError('Original evaluation environment differs')
    return dict(record=record,completion=complete,metric=metric,train_receipt=train_receipt,
                eval_receipt=eval_receipt,train_config=train_cfg,eval_config=eval_cfg)


def checkpoint_identity(run,complete):
    path=(Path(run)/'weights/last.pt').resolve()
    if Path(complete.get('checkpoint','')).resolve()!=path or not path.is_file():
        raise ValueError('Actual checkpoint is not the original run fixed last.pt')
    return path


def file_state(path):
    value=Path(path).stat()
    return dict(size=value.st_size,mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,
                device=value.st_dev,inode=value.st_ino)


def validate_output(output,run):
    output,run=Path(output).resolve(),Path(run).resolve()
    if output==run or run in output.parents or output in run.parents:
        raise ValueError('New diagnostics cannot be inside or replace the old run')
    if output.exists():raise FileExistsError('Preserve earlier diagnostics; select a new attempt')
    return output


def origin_files(run,metadata):
    run=Path(run)
    result=[run/name for name in ('completion_receipt.json','evaluation_val.json','evaluation_val_roster.txt')]
    for folder,key in (('run_evidence','train_receipt'),('eval_evidence','eval_receipt')):
        result.append(run/folder/'run_receipt.json')
        receipt=metadata[key]
        for group in receipt.get('source_snapshots',{}).values():result.extend(run/folder/name for name in group)
        result.extend(run/folder/name for name in receipt.get('metric_snapshots',[]))
    return list(dict.fromkeys(result))


def save_origins(run,metadata,output,checkpoint):
    records=[]
    for path in origin_files(run,metadata):
        relative=path.relative_to(run);destination=output/'origin_evidence'/relative
        copy_new(path,destination);records.append(dict(original=str(path.resolve()),copy=str(destination)))
    origin=dict(schema='rgbir-legacy-reevaluation-origin-v1',original_run=str(Path(run).resolve()),
        original_checkpoint=str(checkpoint),checkpoint_stat_before=file_state(checkpoint),
        original_arm=metadata['metric']['arm'],seed=metadata['metric']['seed'],input_files=records,
        original_training_config=metadata['train_config'],original_evaluation_config=metadata['eval_config'],
        new_training_receipt_created=False,original_results_modified=False,
        historical_native_library_bytes='not present in the original eval receipt; this new evaluation records current sources',
        original_evaluation_has_objects=False,official_test_accessed=False)
    write_new(output/'origin_manifest.json',origin)
    return origin


def verify_origins(origin):
    if file_state(origin['original_checkpoint'])!=origin['checkpoint_stat_before']:
        raise ValueError('Original checkpoint changed during read-only evaluation')
    for row in origin['input_files']:
        if Path(row['original']).read_bytes()!=Path(row['copy']).read_bytes():
            raise ValueError('Original legacy evidence changed during evaluation: '+row['original'])


def historical_comparison(old,new):
    for key in METRICS:
        if any(type(value.get(key)) not in (float,int) or not math.isfinite(value[key]) for value in (old,new)):
            raise ValueError('Five historical/current metrics must be finite real numbers')
    if old.get('metric_units')!='fraction_0_to_1':raise ValueError('Historical metric units differ')
    differences={key:new[key]-old[key] for key in METRICS}
    return dict(status='EXACT' if all(value==0.0 for value in differences.values()) else 'MISMATCH_REVIEW_REQUIRED',
                differences_fraction=differences,old_metrics={k:old[k] for k in METRICS},
                newly_evaluated_metrics={k:new[k] for k in METRICS},old_results_overwritten=False)


def require_all_classes(record,nc):
    rows=record.get('per_class',[])
    if len(rows)!=nc or {r.get('class_id') for r in rows}!=set(range(nc)):
        raise ValueError('Complete unique class inventory is required')
    for row in rows:
        for key in ('AP50','AP75','mAP50_95'):
            if type(row.get(key)) not in (float,int) or not math.isfinite(row[key]) or not 0<=row[key]<=1:
                raise ValueError('Invalid actual per-class metric')


def persist_checked_result(output,record,old,origin):
    """Keep raw mismatch evidence, but publish no completed metric on mismatch."""
    output=Path(output)
    write_new(output/'evaluation_observed.json',record)
    comparison=historical_comparison(old,record)
    write_new(output/'historical_metric_comparison.json',comparison)
    verify_origins(origin)
    if comparison['status']!='EXACT':
        raise ValueError('New/old five-metric mismatch; raw new evidence retained for review')
    result=dict(record,status='completed',historical_five_metrics_exact=True)
    target=output/'evaluation_val.json';write_new(target,result)
    return target,result


def run(args):
    require_review(args.review_receipt)
    release=Path(args.release).resolve()
    analyzer=frozen_module(release,'analyze_independent')
    evaluator=frozen_module(release,'evaluate_independent')
    profile_module=frozen_module(release,'evaluator_profile')
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    run_dir=Path(args.legacy_run).resolve()
    metadata=load_legacy_metadata(run_dir,args.arm,args.seed,cfg,analyzer)
    checkpoint=checkpoint_identity(run_dir,metadata['completion'])
    output=validate_output(args.output,run_dir)
    if os.name!='posix' or not str(output).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('Actual diagnostics must write to project data storage on94')
    profile=read(args.evaluation_profile)
    if (profile.get('status')!='COMPLETED' or profile.get('measurement_valid') is not True
        or profile.get('stage')!='evaluation_profile'):
        raise ValueError('Actual completed evaluation resource profile required')
    identity=profile_module.validate_evaluation_profile_binding(profile['evaluation_profile_binding'],cfg,release)
    if identity!=profile.get('profile_identity'):raise ValueError('Measured evaluation identity differs')
    binding=read(profile['evaluation_profile_binding'])
    canonical=profile_module.dev_roster(cfg['paths']['student_data_yaml'])
    if canonical!=metadata['record']['roster']:raise ValueError('Current actual dev differs from original complete roster')
    data_source=Path(cfg['paths']['student_data_yaml'])
    old_data=[run_dir/'eval_evidence'/p for p in metadata['eval_receipt']['source_snapshots']['config'] if 'rgb.data.yaml' in p]
    if len(old_data)!=1 or old_data[0].read_bytes()!=data_source.read_bytes():raise ValueError('Original bound RGB data YAML changed')
    import numpy as np
    import torch
    legacy=frozen_module(release,'runtime').legacy
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import DetMetrics,Metric,ap_per_class
    from ultralytics.utils.nms import non_max_suppression
    if len(legacy.require_bound_lease_from_environment()['gpus'])!=1:raise ValueError('One actual shared global GPU lease required')
    versions=dict(torch=str(torch.__version__),ultralytics=str(legacy.ultralytics.__version__))
    if versions!={'torch':cfg['torch_version'],'ultralytics':cfg['ultralytics_version']}:raise ValueError('Actual pinned environment differs')
    output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    try:
        copy_new(args.config,output/'evaluation_config.yaml')
        copy_new(args.review_receipt,output/'wrapper_review_receipt.json')
        copy_new(args.evaluation_profile,output/'evaluation_resource_profile_input.json')
        (output/'evaluation_val_roster.txt').write_text(''.join(p+'\n' for p in canonical),encoding='utf-8')
        origin=save_origins(run_dir,metadata,output,checkpoint)
        captured={}
        random.seed(20260907);np.random.seed(20260907);torch.manual_seed(20260907);torch.cuda.manual_seed_all(20260907)
        model=YOLO(str(checkpoint),task='detect')
        def on_start(validator):
            captured.update(evaluator.capture_contract(validator,canonical,versions))
            if captured['effective_kwargs']!=binding['actual_effective_kwargs']:
                raise ValueError('Actual new evaluation kwargs differ from accepted native profile')
        model.add_callback('on_val_start',on_start)
        model.add_callback('on_val_end',lambda validator:evaluator.verify_population(validator,captured))
        metrics=model.val(validator=evaluator.make_evidence_validator(DetectionValidator),
            data=cfg['paths']['student_data_yaml'],split='val',imgsz=cfg['imgsz'],batch=cfg['batch'],
            workers=cfg['workers'],device='0',plots=False,save_json=False,verbose=False,
            project=str(output),name='predictions',exist_ok=False)
        record=profile_module.metric_record(metrics,model.names)
        require_all_classes(record,cfg['expected_nc'])
        contract=output/'evaluation_contract.json';write_new(contract,captured)
        record.update(status='evaluated_pending_historical_check',metric_units='fraction_0_to_1',
            checkpoint=str(checkpoint),endpoint='fixed_budget_last_ema',split='val',arm=cfg['arm'],
            source=cfg['source'],seed=args.seed,dataset=cfg['dataset'],method_id=cfg['method_id'],
            normalized_method_arm=args.arm,evaluation_kind=cfg['evaluation_kind'],
            evaluation_contract=str(contract),objects=str(output/'predictions/objects.jsonl.gz'),
            training_origin=str(output/'origin_manifest.json'),official_test_accessed=False,seconds=time.time()-started)
        target,record=persist_checked_result(output,record,metadata['metric'],origin)
        implementations=legacy.implementation_files(DetectionValidator,BaseValidator,YOLO.val,
            check_det_dataset,DetMetrics,Metric,ap_per_class,non_max_suppression)
        # Record all real executed wrapper/helper sources. Do not relabel this as
        # the old seven-source canonical receipt or pretend it ran historically.
        trainers=[Path(evaluator.__file__),*implementations,Path(profile_module.__file__),Path(__file__)]
        legacy.emit_bound_run_receipt(run_dir=output/'eval_evidence',method_identity=cfg['method_identity'],
            dataset=cfg['dataset'],data_role='development_val',seed=args.seed,run_kind='eval',
            trainers=trainers,losses=[],configs=[output/'evaluation_config.yaml',data_source,contract,output/'origin_manifest.json'],
            split_rosters=[output/'evaluation_val_roster.txt'],
            metric_files=[target,output/'predictions/objects.jsonl.gz',output/'historical_metric_comparison.json'],
            environment=versions,inputs=dict(method_id=cfg['method_id'],arm=cfg['arm'],source=cfg['source'],
                checkpoint=str(checkpoint),endpoint='fixed_budget_last_ema',original_training_run=str(run_dir),
                evaluation_kind=cfg['evaluation_kind'],new_training_receipt_created=False))
        write_new(output/'reevaluation_receipt.json',dict(status='completed',schema='rgbir-legacy-reevaluation-v1',
            seed=args.seed,arm=cfg['arm'],normalized_method_arm=args.arm,checkpoint=str(checkpoint),
            old_run=str(run_dir),evaluation_val=str(target),evaluation_receipt=str(output/'eval_evidence/run_receipt.json'),
            full_dev_images=1469,all_class_metrics=cfg['expected_nc'],historical_five_metrics_exact=True,
            old_results_modified=False,new_training_receipt_created=False,official_test_accessed=False,
            resources=legacy.bound_lease_resource_record_from_environment(),
            seconds=time.time()-started))
        return record
    except BaseException as error:
        write_new(output/'failure_receipt.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),
            seconds=time.time()-started,old_results_modified=False,new_training_receipt_created=False,official_test_accessed=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('release','legacy-run','config','output','evaluation-profile','review-receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=tuple(ARMS),required=True)
    parser.add_argument('--seed',type=int,choices=(0,42,123),required=True)
    args=parser.parse_args();print(json.dumps(run(args),ensure_ascii=False))


if __name__=='__main__':main()
