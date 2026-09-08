"""Independent warm-start/subset hourly screen, evaluated on the pinned full dev."""
import argparse
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import yaml

from hourly_common import ENDPOINT, copy_sources, data_output, load_config, read, stat, write_new

EFFECTIVE = dict(imgsz=640, batch=32, workers=4, quantize=None, conf=.001, iou=.7,
                 max_det=300, agnostic_nms=False, single_cls=False, rect=True,
                 augment=False, half=False)


def hourly_completion(run, cfg, canary_receipt=None):
    """Bind the fixed new endpoint and its own executed canary, never E8/E200."""
    result = read(run / 'hourly_training_receipt.json')
    if (result.get('status') != 'HOURLY_SCREEN_TRAINING_COMPLETED'
            or result.get('scope') != 'HOURLY_SCREEN_FT'
            or result.get('single_seed') is not True
            or result.get('last_epoch') != 3 or result.get('epochs_configured') != 3
            or result.get('endpoint') != ENDPOINT
            or result.get('formal_e200_complete') is not False
            or result.get('new_hash_computed') is not False
            or result.get('official_test_accessed') is not False):
        raise ValueError('Completed independent hourly E3 training receipt required')
    for key in ('arm', 'seed', 'dataset', 'model', 'teacher', 'reference',
                'classification_coefficient', 'localization_coefficient'):
        if result.get(key) != cfg[key]:
            raise ValueError('Hourly completion identity differs: ' + key)
    checkpoint = run / 'weights' / 'last.pt'
    if result['checkpoint'] != stat(checkpoint):
        raise ValueError('Fixed hourly last/EMA checkpoint stat differs')
    recorded_canary = Path(result['canary_receipt']).resolve()
    canary_path = Path(canary_receipt).resolve() if canary_receipt is not None else recorded_canary
    if canary_path != recorded_canary:
        raise ValueError('Explicit canary differs from training receipt provenance')
    if not canary_path.is_file():
        raise FileNotFoundError(canary_path)
    canary = read(canary_path)
    if (canary.get('status') != 'HOURLY_SCREEN_CANARY_COMPLETED'
            or canary.get('arm') != cfg['arm']
            or type(canary.get('successful_updates')) is not int
            or canary['successful_updates'] < 24):
        raise ValueError('Corresponding successful hourly canary required')
    if Path(canary['config_copy']).read_bytes() != (run / 'hourly_config.yaml').read_bytes():
        raise ValueError('Canary and training configurations differ')
    return result, checkpoint, canary_path


def evaluation_projection(cfg, native_cfg, native_config_path, profile, binding, ref):
    """Validate the original evaluator identity separately from changed training."""
    if Path(cfg['native_contract_config']).resolve() != native_config_path.resolve():
        raise ValueError('Native contract configuration differs from declared projection')
    for key in ('dataset', 'expected_nc', 'expected_val_images', 'imgsz', 'batch',
                'workers', 'torch_version', 'ultralytics_version'):
        if cfg[key] != native_cfg[key]:
            raise ValueError('Training/evaluation invariant differs: ' + key)
    identity = profile.validate_evaluation_profile_binding(binding, native_cfg, ref)
    if identity['actual_effective_kwargs'] != EFFECTIVE:
        raise ValueError('Accepted native evaluator kwargs differ')
    full_data = native_cfg['paths']['student_data_yaml']
    if Path(cfg['auxiliary_data_identity']['student_data_yaml']).resolve() != Path(full_data).resolve():
        raise ValueError('Declared full RGB data identity differs from native evaluation')
    canonical = profile.dev_roster(full_data)
    subset_dev = profile.dev_roster(cfg['paths']['student_data_yaml'])
    if len(canonical) != 1469 or len(set(canonical)) != 1469 or subset_dev != canonical:
        raise ValueError('Subset configuration must retain the exact complete dev roster')
    projection = dict(
        kind='explicit_training_to_original_full_dev_evaluation_identity',
        training_model=cfg['model'], training_student_data_yaml=cfg['paths']['student_data_yaml'],
        training_expected_images=cfg['expected_train_images'],
        native_contract_config=str(native_config_path.resolve()),
        native_contract_model=native_cfg['model'], actual_evaluation_data_yaml=full_data,
        native_profile_binding=str(binding.resolve()), subset_dev_roster_exact_to_full=True,
        training_identity_unchanged_claim=False,
        scope='Original binding validates evaluator sources/data/kwargs; hourly checkpoint and training identity are separate.')
    return canonical, projection


def run(args):
    config_path = args.run / 'hourly_config.yaml'
    cfg = load_config(config_path)
    completed, checkpoint, canary_path = hourly_completion(args.run, cfg, args.canary_receipt)
    native_cfg = yaml.safe_load(args.native_contract_config.read_text(encoding='utf-8'))
    output = data_output(args.output)
    if output.exists():
        raise FileExistsError(output)
    ref = args.reference_dir.resolve()
    sys.path.insert(0, str(ref))
    import torch
    import runtime
    import evaluator_profile as profile
    import evaluate_independent as formal_helpers
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class
    from ultralytics.utils.nms import non_max_suppression
    for module, name in ((runtime, 'runtime.py'), (profile, 'evaluator_profile.py'),
                         (formal_helpers, 'evaluate_independent.py')):
        if Path(module.__file__).resolve() != ref / name:
            raise RuntimeError('Wrong native helper binding: ' + name)
    if len(runtime.legacy.require_bound_lease_from_environment()['gpus']) != 1:
        raise ValueError('One bound globallease required')
    versions = dict(torch=str(torch.__version__), ultralytics=str(runtime.legacy.ultralytics.__version__))
    if versions != {k: cfg[k + '_version'] for k in versions}:
        raise ValueError('Pinned environment changed')
    # Reuse only pure helpers. The original E200 run/load/publish path is never called.
    canonical, projection = evaluation_projection(
        cfg, native_cfg, args.native_contract_config, profile, args.native_profile_binding, ref)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / 'hourly_config.yaml')
    shutil.copyfile(args.run / 'hourly_training_receipt.json', output / 'hourly_training_receipt_copy.json')
    shutil.copyfile(canary_path, output / 'canary_receipt_copy.json')
    shutil.copyfile(args.native_contract_config, output / 'native_contract_config.yaml')
    (output / 'development_roster.txt').write_text(''.join(p + '\n' for p in canonical), encoding='utf-8')
    write_new(output / 'evaluation_identity_projection.json', projection)
    captured = {}
    sources = set(Path(__file__).parent.glob('*.py')) | {ref / 'evaluator_profile.py', ref / 'evaluate_independent.py'}
    sources.update(Path(p) for p in runtime.legacy.implementation_files(
        DetectionValidator, BaseValidator, AutoBackend, YOLO.val, check_det_dataset,
        DetMetrics, Metric, ap_per_class, non_max_suppression))
    started = time.perf_counter()
    model = None

    def on_start(v):
        captured.update(formal_helpers.capture_contract(v, canonical, versions))
        if captured['effective_kwargs'] != EFFECTIVE:
            raise ValueError('Actual native settings changed')
        labels = v.dataloader.dataset.labels
        if len(labels) != 1469 or sum(len(x['cls']) for x in labels) != 22462:
            raise ValueError('Full dev GT population changed')
        captured.update(gt_objects_before_inference=22462, endpoint=ENDPOINT,
                        scope='HOURLY_SCREEN_FT', formal_e200_complete=False,
                        evaluation_identity_projection=projection)
        captured['runtime_sources'] = profile.capture_runtime_sources(v, model, sources)

    def on_end(v):
        formal_helpers.verify_population(v, captured)
        with gzip.open(v.save_dir / 'objects.jsonl.gz', 'rt', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]
        total = sum(len(r['gt_classes']) for r in records)
        if total != 22462 or any(len(r['gt_classes']) != len(r['gt_boxes']) for r in records):
            raise ValueError('Captured GT population changed')
        captured['gt_objects_captured'] = total

    try:
        model = YOLO(str(checkpoint), task='detect')
        model.add_callback('on_val_start', on_start)
        model.add_callback('on_val_end', on_end)
        metrics = model.val(validator=formal_helpers.make_evidence_validator(DetectionValidator),
            data=projection['actual_evaluation_data_yaml'], split='val', imgsz=640, batch=32,
            workers=4, device='0', conf=.001, iou=.7, max_det=300, agnostic_nms=False,
            single_cls=False, rect=True, augment=False, quantize=None, plots=False,
            save_json=False, verbose=False, project=str(output), name='native_capture', exist_ok=False)
        values = profile.metric_record(metrics, model.names)
        if len(values['per_class']) != 5 or {r['class_id'] for r in values['per_class']} != set(range(5)):
            raise ValueError('All five unique GT classes required')
        if not all(math.isfinite(values[k]) and 0. <= values[k] <= 1.
                   for k in ('AP50', 'AP75', 'mAP50_95', 'precision', 'recall')):
            raise ValueError('Invalid native metric fraction')
        for row in values['per_class']:
            if not all(math.isfinite(row[k]) and 0. <= row[k] <= 1. for k in ('AP50', 'AP75', 'mAP50_95')):
                raise ValueError('Invalid per-class native AP fraction')
        if completed['checkpoint'] != stat(checkpoint):
            raise ValueError('Checkpoint changed during evaluation')
        sources.update((config_path, args.run / 'hourly_training_receipt.json', canary_path,
                        args.native_contract_config, args.native_profile_binding,
                        Path(cfg['paths']['student_data_yaml']), Path(projection['actual_evaluation_data_yaml'])))
        copy_sources(output, sources)
        write_new(output / 'hourly_evaluation_contract.json', captured)
        write_new(output / 'hourly_evaluation_receipt.json', dict(
            status='HOURLY_SCREEN_EVALUATION_COMPLETED', scope='HOURLY_SCREEN_FT', single_seed=True,
            seed=42, arm=cfg['arm'], dataset='dronevehicle', endpoint=ENDPOINT,
            epochs=3, independent_lr_horizon=3, formal_e200_complete=False, formal_paper_gain_claim=False,
            metric_units='fraction_0_to_1', native_metric_definition='pinned Ultralytics native AP; not TIDE oracle',
            **values, checkpoint=completed['checkpoint'], full_dev_images=1469, full_dev_gt_objects=22462,
            training_configuration=str(config_path), training_model=cfg['model'],
            native_profile_binding=str(args.native_profile_binding), canary_receipt=str(canary_path),
            evaluation_identity_projection=projection, seconds=time.perf_counter() - started,
            resources=runtime.legacy.bound_lease_resource_record_from_environment(),
            new_hash_computed=False, official_test_accessed=False))
    except BaseException as error:
        write_new(output / 'hourly_evaluation_failure.json', dict(
            status='HOURLY_SCREEN_EVALUATION_FAILED', error=repr(error), traceback=traceback.format_exc(),
            seconds=time.perf_counter() - started, new_hash_computed=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-profile-binding', type=Path, required=True)
    p.add_argument('--native-contract-config', type=Path, required=True)
    p.add_argument('--canary-receipt', type=Path)
    run(p.parse_args())


if __name__ == '__main__':
    main()
