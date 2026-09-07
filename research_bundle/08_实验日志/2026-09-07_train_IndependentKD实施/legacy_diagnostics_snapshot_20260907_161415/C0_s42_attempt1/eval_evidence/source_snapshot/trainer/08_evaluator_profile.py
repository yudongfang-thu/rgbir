"""Real old-N42 native/evidence evaluator parity and resource probe.

No training receipt is manufactured. Output is a new probe artifact directory;
the old N42 run is a read-only checkpoint/completion input.
"""
from __future__ import annotations
import argparse
import gc
import inspect
import json
import math
from pathlib import Path
import random
import time
import yaml

HERE = Path(__file__).resolve().parent
BINDING_SCHEMA = 'rgbir-evaluation-profile-binding-v1'
METRICS = ('AP50', 'AP75', 'mAP50_95', 'precision', 'recall')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def configuration_identity(cfg):
    keys = ('dataset', 'model', 'expected_nc', 'imgsz', 'batch', 'workers',
            'torch_version', 'ultralytics_version')
    if any(key not in cfg for key in keys):
        raise ValueError('Incomplete actual evaluation configuration')
    if cfg['dataset'] != 'dronevehicle' or cfg.get('expected_val_images') != 1469:
        raise ValueError('This fixed baseline probe is Drone full dev1469 only')
    return dict({key: cfg[key] for key in keys},
                student_data_yaml=cfg['paths']['student_data_yaml'],
                split='val', expected_val_images=1469)


def dev_roster(data_path):
    """Read only the governed dev split; train/test contents are never opened."""
    data_path = Path(data_path)
    data = yaml.safe_load(data_path.read_text(encoding='utf-8'))
    if 'test' in data:
        raise ValueError('A train/val-only YAML is required')
    root = Path(data.get('path', data_path.parent))
    if not root.is_absolute():
        root = data_path.parent/root
    sources = data['val'] if isinstance(data['val'], list) else [data['val']]
    result = []
    for source in sources:
        path = Path(source)
        if not path.is_absolute():
            path = root/path
        if any(part.lower() in ('test','testing','test2017','test-dev') for part in path.resolve().parts):
            raise ValueError('Sealed test path is not part of this probe')
        if path.is_dir():
            result.extend(p.resolve() for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() == '.txt':
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    item = Path(line.strip())
                    result.append((item if item.is_absolute() else path.parent/item).resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            result.append(path.resolve())
        else:
            raise ValueError('Unresolved development input: ' + str(path))
    if not result or len(set(result)) != len(result):
        raise ValueError('Empty or duplicate development roster')
    for path in result:
        if not path.is_file() or any(part.lower() in ('test','testing','test2017','test-dev') for part in path.parts):
            raise ValueError('Missing or forbidden development image')
    return sorted(map(str,result))


def assert_equal_metrics(native, evidence):
    for key in METRICS:
        if not all(math.isfinite(float(x[key])) for x in (native,evidence)):
            raise ValueError('Nonfinite metric: ' + key)
        if native[key] != evidence[key]:
            raise ValueError('Native/evidence metric mismatch: ' + key)
    if native.get('per_class') != evidence.get('per_class'):
        raise ValueError('Native/evidence per-class AP mismatch')
    return {key: evidence[key]-native[key] for key in METRICS}


def baseline_identity(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent.parent
    receipt_path = run/'completion_receipt.json'
    receipt = read_json(receipt_path)
    if (checkpoint.name != 'last.pt' or checkpoint.parent.name != 'weights' or
        receipt.get('status') != 'training_completed' or receipt.get('arm') not in ('weight0','N') or
        receipt.get('seed') != 42 or receipt.get('last_epoch') != 200 or
        receipt.get('epochs_configured') != 200 or receipt.get('official_test_accessed') is not False or
        Path(receipt.get('checkpoint','')).resolve() != checkpoint or not checkpoint.is_file()):
        raise ValueError('Probe requires the actual completed N42 fixed last checkpoint')
    return run,receipt_path,receipt


def make_binding(output, cfg, contract, sources):
    folder = Path(output)/'evaluation_profile_binding'
    folder.mkdir(exist_ok=False)
    data_path = Path(cfg['paths']['student_data_yaml']).resolve()
    data_copy = folder/'student_data.yaml'
    data_copy.write_bytes(data_path.read_bytes())
    records=[]
    unique=sorted({Path(p).resolve() for p in sources},key=str)
    for index,path in enumerate(unique):
        target=folder/'sources'/f'{index:03d}_{path.name}'
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
        records.append(dict(original=str(path),copy=str(target)))
    value=dict(schema=BINDING_SCHEMA, configuration=configuration_identity(cfg),
        actual_effective_kwargs=contract['effective_kwargs'], roster=contract['roster'],
        data_source=dict(original=str(data_path),copy=str(data_copy)), source_files=records,
        expected_val_images=1469, observed_images=contract['observed_images'],
        metric_equivalence='exact_native_vs_evidence_on_old_N42',
        profile_scope='evaluation_resources_and_evaluator_equivalence_only',
        official_test_accessed=False)
    path=folder/'binding.json'
    write_new(path,value)
    return path


def validate_evaluation_profile_binding(path, cfg, module_root=HERE):
    """Derive identity only after config/data/roster/actual-source byte checks."""
    binding=read_json(path)
    if (binding.get('schema')!=BINDING_SCHEMA or binding.get('configuration')!=configuration_identity(cfg)
        or binding.get('observed_images')!=1469 or binding.get('expected_val_images')!=1469
        or len(binding.get('roster',[]))!=1469 or len(set(binding.get('roster',[])))!=1469
        or binding.get('official_test_accessed') is not False):
        raise ValueError('Evaluation profile configuration or population differs')
    data=binding['data_source']
    if Path(data['original']).resolve()!=Path(cfg['paths']['student_data_yaml']).resolve():
        raise ValueError('Evaluation data source differs')
    if Path(data['original']).read_bytes()!=Path(data['copy']).read_bytes():
        raise ValueError('Evaluation dataset YAML bytes changed')
    if binding['roster']!=dev_roster(data['original']):
        raise ValueError('Evaluation development roster changed')
    records=binding.get('source_files',[])
    required={str((Path(module_root)/n).resolve()) for n in ('evaluator_profile.py','evaluate_independent.py')}
    if not required.issubset({r['original'] for r in records}):
        raise ValueError('Actual evaluator/profile source binding missing')
    for row in records:
        if Path(row['original']).read_bytes()!=Path(row['copy']).read_bytes():
            raise ValueError('Actual evaluator or loader source bytes changed: '+row['original'])
    return dict(configuration=binding['configuration'],
                actual_effective_kwargs=binding['actual_effective_kwargs'],
                expected_val_images=1469, evaluator_sources=[r['original'] for r in records],
                binding=str(Path(path).resolve()))


def metric_record(metrics, names):
    per_class=[]
    for row,ci in enumerate(metrics.box.ap_class_index):
        ap=metrics.box.all_ap[row]
        per_class.append(dict(class_id=int(ci),name=names[int(ci)],AP50=float(ap[0]),
                              AP75=float(ap[5]),mAP50_95=float(ap.mean())))
    return dict(AP50=float(metrics.box.map50),AP75=float(metrics.box.map75),
                mAP50_95=float(metrics.box.map),precision=float(metrics.box.mp),
                recall=float(metrics.box.mr),per_class=per_class)


def capture_runtime_sources(validator, loaded_model, sources):
    """Record live loader/network classes without inventing validator.model.

    In pinned 8.4.115 Model.val passes loaded_model.model to BaseValidator;
    BaseValidator wraps it in a *local* AutoBackend before on_val_start. The
    validator never exposes that backend as self.model. AutoBackend itself is
    bound separately alongside BaseValidator below.
    """
    records = {}
    objects = dict(loader=validator.dataloader, dataset=validator.dataloader.dataset,
                   network=loaded_model.model)
    for role, obj in objects.items():
        cls = type(obj)
        source = inspect.getsourcefile(cls)
        if not source or not Path(source).is_file():
            raise ValueError('Actual runtime source unavailable: ' + role)
        path = Path(source).resolve()
        sources.add(path)
        records[role] = dict(class_name=cls.__module__ + '.' + cls.__qualname__,
                             source_file=str(path))
    return records


def run(args):
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    configuration_identity(cfg)
    old_run,old_receipt_path,old_receipt=baseline_identity(args.checkpoint)
    output=args.output.resolve()
    if output==old_run or old_run in output.parents:
        raise ValueError('Probe artifacts must not be written inside the old run')
    if output.exists():
        raise FileExistsError('Preserve old probe attempts; use a new output')
    canonical=dev_roster(cfg['paths']['student_data_yaml'])
    if len(canonical)!=1469:
        raise ValueError('Full unique dev1469 is required')
    import numpy as np
    import torch
    from runtime import legacy
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class
    from ultralytics.utils.nms import non_max_suppression
    from evaluate_independent import make_evidence_validator,capture_contract,verify_population
    if len(legacy.require_bound_lease_from_environment()['gpus'])!=1:
        raise ValueError('Exactly one shared guard GPU lease is required')
    versions=dict(torch=str(torch.__version__),ultralytics=str(legacy.ultralytics.__version__))
    if versions['torch']!=cfg['torch_version'] or versions['ultralytics']!=cfg['ultralytics_version']:
        raise ValueError('Pinned evaluation environment differs')
    output.mkdir(parents=True,exist_ok=False)
    (output/'requested_config.yaml').write_bytes(args.config.read_bytes())
    (output/'old_N42_completion_copy.json').write_bytes(old_receipt_path.read_bytes())
    (output/'development_roster.txt').write_text(''.join(p+'\n' for p in canonical),encoding='utf-8')
    started=time.time()
    records,contracts,sources={},{},{Path(__file__),HERE/'evaluate_independent.py'}
    sources.update(Path(p) for p in legacy.implementation_files(DetectionValidator,BaseValidator,AutoBackend,YOLO.val,
        check_det_dataset,DetMetrics,Metric,ap_per_class,non_max_suppression))
    try:
        for label,validator_class in (('native',DetectionValidator),('evidence',make_evidence_validator(DetectionValidator))):
            # Fixed probe RNG only; no training is taking place in this process.
            random.seed(20260907);np.random.seed(20260907);torch.manual_seed(20260907)
            torch.cuda.manual_seed_all(20260907)
            torch.cuda.reset_peak_memory_stats()
            model=YOLO(str(args.checkpoint.resolve()),task='detect')
            captured={}
            def on_start(v):
                captured.update(capture_contract(v,canonical,versions))
                captured['evaluator_identity']['extension']='native_only' if label=='native' else 'read_only_post_metric_object_capture_v1'
                captured['runtime_sources']=capture_runtime_sources(v,model,sources)
            def on_end(v):
                if v.seen!=1469:
                    raise ValueError('Actual validator.seen differs from1469')
                captured['observed_images']=int(v.seen)
                if label=='evidence':
                    verify_population(v,captured)
            model.add_callback('on_val_start',on_start)
            model.add_callback('on_val_end',on_end)
            kwargs=dict(data=cfg['paths']['student_data_yaml'],split='val',imgsz=cfg['imgsz'],
                        batch=cfg['batch'],workers=cfg['workers'],device='0',plots=False,
                        save_json=False,verbose=False,project=str(output),name=label,exist_ok=False)
            metrics=model.val(validator=validator_class,**kwargs)
            record=metric_record(metrics,model.names)
            record.update(status='completed',metric_units='fraction_0_to_1',split='val',
                checkpoint=str(args.checkpoint.resolve()),baseline_arm='N42_weight0',
                resources=legacy.bound_lease_resource_record_from_environment(),
                gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
                requested_kwargs=kwargs,actual_loader_roster=captured['actual_loader_roster'],
                observed_images=captured['observed_images'],official_test_accessed=False)
            write_new(output/(label+'_metrics.json'),record)
            write_new(output/(label+'_contract.json'),captured)
            records[label],contracts[label]=record,captured
            del metrics,model
            gc.collect();torch.cuda.empty_cache()
        if contracts['native']['effective_kwargs']!=contracts['evidence']['effective_kwargs']:
            raise ValueError('Native/evidence actual kwargs differ')
        if contracts['native']['actual_loader_roster']!=contracts['evidence']['actual_loader_roster']:
            raise ValueError('Native/evidence actual loader sequence differs')
        differences=assert_equal_metrics(records['native'],records['evidence'])
        binding=make_binding(output,cfg,contracts['evidence'],sources)
        identity=validate_evaluation_profile_binding(binding,cfg,HERE)
        result=dict(status='evaluation_profile_completed',schema='rgbir-evaluator-parity-probe-v1',
            native_evidence_metrics_exact=True,metric_differences=differences,expected_val_images=1469,
            native_seen=1469,evidence_seen=1469,checkpoint=str(args.checkpoint.resolve()),
            baseline_completion_input=str(old_receipt_path),baseline_training_receipt_created=False,
            profile_identity=identity,evaluation_profile_binding=str(binding),
            resources=legacy.bound_lease_resource_record_from_environment(),
            official_test_accessed=False,seconds=time.time()-started,
            purpose='fixed_old_N42_evaluation_equivalence_and_resource_probe_only')
        write_new(output/'evaluator_profile_receipt.json',result)
        return result
    except BaseException as error:
        write_new(output/'failure_receipt.json',dict(status='failed',error=repr(error),
            official_test_accessed=False,baseline_training_receipt_created=False,seconds=time.time()-started))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
