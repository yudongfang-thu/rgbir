"""Independent full-dev native FP32 evaluation of fixed direction-screen last/EMA."""
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

SCOPE = 'DIRECTION_FT3_BNFROZEN'
ENDPOINT = 'DIRECTION_FT3_BNFROZEN_LAST_EMA'
ARMS = ('N', 'C1', 'C2', 'F-rel', 'L2-box', 'L2-GT')
DATASET_ARMS = {'drone': ('N', 'C1', 'C2', 'F-rel'), 'llvip': ('N', 'L2-box', 'L2-GT')}
POPULATIONS = {'drone': (1469, 22462, 5), 'llvip': (2406, 7879, 1)}
EFFECTIVE = dict(imgsz=640, batch=32, workers=4, quantize=None, conf=.001, iou=.7,
                 max_det=300, agnostic_nms=False, single_cls=False, rect=True,
                 augment=False, half=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stat(path):
    p = Path(path); s = p.stat()
    return dict(path=str(p.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def copy_sources(output, paths):
    dest = output/'source_copies'; dest.mkdir(exist_ok=False); rows = []
    for index, p in enumerate(sorted({Path(p).resolve() for p in paths}, key=str)):
        target = dest/('%04d_' % index + p.name)
        shutil.copyfile(p, target)
        if p.read_bytes() != target.read_bytes():
            raise ValueError('Source byte copy differs')
        rows.append(dict(stat(p), copy=str(target), byte_identity=True))
    write_new(output/'source_manifest.json', dict(files=rows, new_hash_computed=False))


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if cfg.get('dataset') not in POPULATIONS or cfg.get('arm') not in DATASET_ARMS[cfg['dataset']]:
        raise ValueError('Unknown direction dataset/arm')
    count, objects, nc = POPULATIONS[cfg['dataset']]
    fixed = dict(scope=SCOPE, seed=42, epochs=3, batch=32, workers=4, imgsz=640,
                 lr0=.0001, lrf=1., warmup_epochs=0., expected_val_images=count, expected_nc=nc,
                 expected_train_images=2048, freeze_bn_running_statistics=True)
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError('Direction fixed configuration differs')
    if cfg.get('direction_screen', {}).get('scope', SCOPE) != SCOPE:
        raise ValueError('Direction nested scope differs')
    return cfg


def training_completion(cfg, checkpoint, config_path=None):
    checkpoint = checkpoint.resolve()
    if checkpoint.name != 'last.pt' or checkpoint.parent.name != 'weights':
        raise ValueError('Fixed weights/last.pt required; best is not an endpoint')
    receipt_path = checkpoint.parent.parent/'completion_receipt.json'
    result = read(receipt_path)
    fixed = dict(status='DIRECTION_TRAINING_COMPLETED', scope=SCOPE, endpoint=ENDPOINT,
        single_seed=True, seed=42, last_epoch=3, epochs_configured=3, batches=192,
        formal_e200_complete=False, new_hash_computed=False, official_test_accessed=False,
        bn_running_buffers_unchanged=True)
    for key, value in fixed.items():
        if result.get(key) != value or (type(value) is bool and result.get(key) is not value):
            raise ValueError('Incomplete or mismatched direction endpoint: ' + key)
    for key in ('arm', 'seed', 'dataset', 'model', 'teacher', 'reference',
                'classification_coefficient', 'localization_coefficient', 'kd_coefficient'):
        if key not in cfg or result.get(key) != cfg[key]:
            raise ValueError('Training identity differs: ' + key)
    if result.get('checkpoint') != stat(checkpoint):
        raise ValueError('Training/evaluation checkpoint stat mismatch')
    saved_config = Path(result['config_copy'])
    if saved_config.resolve() != checkpoint.parent.parent/'direction_config.yaml':
        raise ValueError('Training config provenance differs from this run')
    if config_path is not None:
        if saved_config.read_bytes() != Path(config_path).read_bytes():
            raise ValueError('Evaluation CLI config differs from actual training config bytes')
    elif yaml.safe_load(saved_config.read_text(encoding='utf-8-sig')) != cfg:
        raise ValueError('Actual training config differs')
    return result, receipt_path


def evaluation_projection(cfg, native_cfg, native_path, binding, profile, ref):
    count, objects, nc = POPULATIONS[cfg['dataset']]
    allowed_native_dataset = 'dronevehicle' if cfg['dataset'] == 'drone' else 'llvip'
    if native_cfg.get('dataset') != allowed_native_dataset:
        raise ValueError('Native configuration belongs to another dataset')
    for key, expected in dict(expected_nc=nc, expected_val_images=count, imgsz=640, batch=32, workers=4).items():
        if native_cfg.get(key) != expected:
            raise ValueError('Native full-dev configuration differs: ' + key)
    for key in ('torch_version', 'ultralytics_version'):
        if cfg[key] != native_cfg[key]:
            raise ValueError('Native environment projection differs: ' + key)
    full_data = native_cfg['paths']['student_data_yaml']
    declared = cfg.get('auxiliary_data_identity', {}).get('student_data_yaml')
    if declared is None or Path(declared).resolve() != Path(full_data).resolve():
        raise ValueError('Full RGB identity must be explicitly declared')
    canonical = profile.dev_roster(full_data)
    subset_dev = profile.dev_roster(cfg['paths']['student_data_yaml'])
    if len(canonical) != count or len(set(canonical)) != count or subset_dev != canonical:
        raise ValueError('Subset dev roster must equal the complete native dev roster')
    prior = None
    if cfg['dataset'] == 'drone':
        if binding is None:
            raise ValueError('Drone requires its accepted native profile binding')
        prior = profile.validate_evaluation_profile_binding(binding, native_cfg, ref)
        if prior['actual_effective_kwargs'] != EFFECTIVE:
            raise ValueError('Accepted native effective kwargs differ')
    elif binding is not None:
        raise ValueError('Drone-only legacy binding must not be presented as LLVIP acceptance')
    return canonical, dict(
        kind='explicit_direction_training_to_full_dev_evaluation_projection', dataset=cfg['dataset'],
        training_model=cfg['model'], training_data_yaml=cfg['paths']['student_data_yaml'],
        actual_evaluation_data_yaml=full_data, native_configuration=str(native_path.resolve()),
        native_contract_model=native_cfg.get('model'),
        prior_binding_validated=prior is not None,
        native_profile_binding=str(binding.resolve()) if binding else None,
        profile_scope='existing_Drone_evaluator_binding' if prior else 'new_LLVIP_actual_native_profile_only',
        native_vs_capture_parity_rerun=False, accepted_endpoint_claim=False,
        subset_dev_roster_exact_to_full=True, training_identity_unchanged_claim=False,
        expected_dev_images=count, expected_dev_gt_objects=objects)


def validate_values(values, nc):
    metrics = ('AP50', 'AP75', 'mAP50_95', 'precision', 'recall')
    rows = values['per_class']
    if len(rows) != nc or sorted(r['class_id'] for r in rows) != list(range(nc)):
        raise ValueError('Incomplete or duplicate GT class metrics')
    if not all(type(r['class_id']) is int for r in rows):
        raise ValueError('Invalid class IDs')
    for item, keys in [(values, metrics)] + [(r, metrics[:3]) for r in rows]:
        if not all(type(item[k]) in (int, float) and math.isfinite(item[k]) and 0. <= item[k] <= 1. for k in keys):
            raise ValueError('Invalid native metric fraction')
    for key in metrics[:3]:
        if not math.isclose(sum(r[key] for r in rows)/nc, values[key], abs_tol=1e-12, rel_tol=0):
            raise ValueError('Per-class native AP mean differs')


def run(args):
    cfg = load_config(args.config)
    checkpoint = args.checkpoint.resolve()
    completed, completion_path = training_completion(cfg, checkpoint, args.config)
    native_cfg = yaml.safe_load(args.native_config.read_text(encoding='utf-8-sig'))
    output = args.output.resolve()
    if os.name != 'posix' or not str(output).startswith('/mnt/dataset/yudongfang/'):
        raise ValueError('Output must remain on project data disk')
    if output.exists() or checkpoint.parent.parent in output.parents:
        raise ValueError('Use a new evaluation directory outside the training run')
    ref = args.reference_dir.resolve(); sys.path.insert(0, str(ref))
    import torch
    import runtime
    import evaluator_profile as profile
    import evaluate_independent as helpers
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class
    from ultralytics.utils.nms import non_max_suppression
    for module, filename in ((runtime, 'runtime.py'), (profile, 'evaluator_profile.py'), (helpers, 'evaluate_independent.py')):
        if Path(module.__file__).resolve() != ref/filename:
            raise RuntimeError('Wrong pinned helper: ' + filename)
    if len(runtime.legacy.require_bound_lease_from_environment()['gpus']) != 1:
        raise ValueError('Exactly one bound global lease required')
    versions = dict(torch=str(torch.__version__), ultralytics=str(runtime.legacy.ultralytics.__version__))
    if versions != {key: cfg[key+'_version'] for key in versions}:
        raise ValueError('Pinned environment differs')
    canonical, projection = evaluation_projection(cfg, native_cfg, args.native_config,
        args.native_profile_binding, profile, ref)
    images, objects, nc = POPULATIONS[cfg['dataset']]
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, output/'direction_config.yaml')
    shutil.copyfile(completion_path, output/'training_completion_copy.json')
    shutil.copyfile(args.native_config, output/'native_config.yaml')
    (output/'development_roster.txt').write_text(''.join(p+'\n' for p in canonical), encoding='utf-8')
    write_new(output/'evaluation_identity_projection.json', projection)
    sources = {Path(__file__), ref/'evaluator_profile.py', ref/'evaluate_independent.py', ref/'runtime.py'}
    sources.update(Path(p) for p in runtime.legacy.implementation_files(DetectionValidator, BaseValidator,
        AutoBackend, YOLO.val, check_det_dataset, DetMetrics, Metric, ap_per_class, non_max_suppression))
    captured = {}; started = time.perf_counter(); model = None

    def on_start(v):
        captured.update(helpers.capture_contract(v, canonical, versions))
        if captured['effective_kwargs'] != EFFECTIVE:
            raise ValueError('Actual native FP32 settings differ')
        labels = v.dataloader.dataset.labels
        actual_objects = sum(len(row['cls']) for row in labels)
        if len(labels) != images or actual_objects != objects:
            raise ValueError('Actual complete dev GT population differs')
        actual_names = v.dataloader.dataset.data['names']
        if len(actual_names) != nc:
            raise ValueError('Actual dataset class mapping differs')
        captured.update(gt_objects_before_inference=actual_objects, endpoint=ENDPOINT, scope=SCOPE,
            dataset=cfg['dataset'], actual_class_names=actual_names,
            formal_e200_complete=False, evaluation_identity_projection=projection)
        captured['runtime_sources'] = profile.capture_runtime_sources(v, model, sources)

    def on_end(v):
        helpers.verify_population(v, captured)
        with gzip.open(v.save_dir/'objects.jsonl.gz', 'rt', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]
        total = sum(len(row['gt_classes']) for row in records)
        if total != objects or any(len(row['gt_classes']) != len(row['gt_boxes']) for row in records):
            raise ValueError('Captured complete dev GT population differs')
        captured['gt_objects_captured'] = total

    try:
        model = YOLO(str(checkpoint), task='detect')
        model.add_callback('on_val_start', on_start); model.add_callback('on_val_end', on_end)
        metrics = model.val(validator=helpers.make_evidence_validator(DetectionValidator),
            data=projection['actual_evaluation_data_yaml'], split='val', imgsz=640, batch=32, workers=4,
            device='0', conf=.001, iou=.7, max_det=300, agnostic_nms=False, single_cls=False,
            rect=True, augment=False, quantize=None, plots=False, save_json=False, verbose=False,
            project=str(output), name='native_capture', exist_ok=False)
        values = profile.metric_record(metrics, model.names); validate_values(values, nc)
        if completed['checkpoint'] != stat(checkpoint):
            raise ValueError('Checkpoint changed during independent evaluation')
        sources.update((args.config, completion_path, args.native_config,
            Path(cfg['paths']['student_data_yaml']), Path(projection['actual_evaluation_data_yaml'])))
        if args.native_profile_binding is not None:
            sources.add(args.native_profile_binding)
        copy_sources(output, sources)
        write_new(output/'direction_evaluation_contract.json', captured)
        write_new(output/'actual_native_evaluation_profile.json', dict(
            schema='direction-actual-native-profile-v1', dataset=cfg['dataset'], scope=SCOPE,
            effective_kwargs=captured['effective_kwargs'], runtime_sources=captured['runtime_sources'],
            actual_class_names=captured['actual_class_names'], observed_images=captured['observed_images'],
            gt_objects_before_inference=captured['gt_objects_before_inference'],
            gt_objects_captured=captured['gt_objects_captured'], profile_projection=projection,
            source_manifest=str(output/'source_manifest.json'), independent_endpoint_accepted=False,
            new_native_capture_equivalence_claim=False, new_hash_computed=False))
        receipt = dict(status='DIRECTION_EVALUATION_COMPLETED', scope=SCOPE, endpoint=ENDPOINT,
            seed=42, arm=cfg['arm'], dataset=cfg['dataset'], single_seed=True, epochs=3,
            independent_lr_horizon=3, formal_e200_complete=False, formal_paper_gain_claim=False,
            metric_units='fraction_0_to_1', native_metric_definition='pinned native AP; no TIDE oracle',
            **values, checkpoint=completed['checkpoint'], full_dev_images=images, full_dev_gt_objects=objects,
            observed_images=captured['observed_images'], gt_objects_captured=captured['gt_objects_captured'],
            training_configuration=str(args.config.resolve()), training_model=cfg['model'],
            training_completion=str(completion_path), evaluation_identity_projection=projection,
            kd_coefficient=completed['kd_coefficient'],
            bn_training_evidence={k:v for k,v in completed.items() if k.startswith('bn_')},
            seconds=time.perf_counter()-started, resources=runtime.legacy.bound_lease_resource_record_from_environment(),
            actual_native_profile=str(output/'actual_native_evaluation_profile.json'),
            accepted_endpoint_claim=False, new_hash_computed=False, official_test_accessed=False)
        write_new(output/'direction_evaluation_receipt.json', receipt)
        return receipt
    except BaseException as error:
        write_new(output/'direction_evaluation_failure.json', dict(status='DIRECTION_EVALUATION_FAILED',
            scope=SCOPE, error=repr(error), traceback=traceback.format_exc(),
            seconds=time.perf_counter()-started, new_hash_computed=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('reference-dir', 'config', 'checkpoint', 'output', 'native-config'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--native-profile-binding', type=Path)
    run(p.parse_args())


if __name__ == '__main__':
    main()
