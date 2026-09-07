"""Fixed E200 last/EMA dev evaluation with immutable, recoverable evidence.

Importing this module creates no GPU context. Object capture runs only after
native metric updates. A complete raw attempt precedes canonical publication.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import time
import traceback
import yaml

POPULATIONS = {'dronevehicle': 1469, 'llvip': 2406}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def copy_missing_or_equal(source, destination):
    """Finish a partial publication without replacing any existing bytes."""
    source, destination = Path(source), Path(destination)
    content = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError('Existing evidence differs; refusing overwrite: ' + str(destination))
        return
    # Publish a complete file atomically without replacing an existing name.
    # The temporary file is a derived copy, never an original run artifact.
    temporary = destination.with_name('.publication_' + str(time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise ValueError('Conflicting concurrently published evidence')
    finally:
        if temporary.exists():
            temporary.unlink()


def bound_configurations(run, receipt, folder):
    configurations = []
    for relative in receipt.get('source_snapshots', {}).get('config', []):
        path = Path(run) / folder / relative
        if path.suffix.lower() in ('.yaml', '.yml', '.json'):
            value = yaml.safe_load(path.read_text(encoding='utf-8'))
            if isinstance(value, dict) and 'model' in value and 'paths' in value:
                configurations.append(value)
    return configurations


def load_run_configuration(run, requested):
    run = Path(run)
    cfg = yaml.safe_load((run / 'protocol_config.yaml').read_text(encoding='utf-8'))
    if yaml.safe_load(Path(requested).read_text(encoding='utf-8')) != cfg:
        raise ValueError('Evaluate using the completed run protocol_config.yaml')
    if cfg.get('dataset') not in POPULATIONS or cfg.get('epochs') != 200:
        raise ValueError('Only authorized Drone/LLVIP E200 endpoints are supported')
    complete = read_json(run / 'completion_receipt.json')
    if (complete.get('status') != 'training_completed' or complete.get('last_epoch') != 200 or
            complete.get('epochs_configured') != 200 or complete.get('official_test_accessed') is not False):
        raise ValueError('Full E200 training must complete before endpoint evaluation')
    for key in ('method_id', 'arm', 'source', 'seed', 'dataset', 'model', 'teacher', 'reference',
                'classification_coefficient', 'localization_coefficient'):
        if complete.get(key) != cfg.get(key):
            raise ValueError('Completion differs from actual run configuration: ' + key)
    checkpoint = run / 'weights/last.pt'
    if Path(complete.get('checkpoint', '')).resolve() != checkpoint.resolve() or not checkpoint.is_file():
        raise ValueError('Completion checkpoint is not this run last/EMA')
    receipt = read_json(run / 'run_evidence/run_receipt.json')
    if (receipt.get('terminal_status') != 'COMPLETED' or receipt.get('run_kind') != 'train' or
            receipt.get('data_role') != 'development_train' or receipt.get('dataset') != cfg['dataset'] or
            receipt.get('seed') != cfg['seed']):
        raise ValueError('Completed bound training receipt is inconsistent')
    bound = bound_configurations(run, receipt, 'run_evidence')
    if not bound or any(value != cfg for value in bound):
        raise ValueError('Actual run configuration differs from training receipt snapshots')
    copies = [run / 'run_evidence' / relative for relative in receipt.get('metric_snapshots', [])]
    if not any(path.suffix == '.json' and read_json(path) == complete for path in copies):
        raise ValueError('Completion differs from bound training snapshot')
    return cfg, complete, checkpoint


def make_evidence_validator(native_validator):
    class EvidenceValidator(native_validator):
        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            # Do not mutate native tensors, metrics, matching state or RNG.
            with gzip.open(self.save_dir / 'objects.jsonl.gz', 'at', encoding='utf-8') as stream:
                for si, pred in enumerate(preds):
                    target = self._prepare_batch(si, batch)
                    row = dict(image=target['im_file'], canvas_shape=list(target['imgsz']),
                        original_shape=list(target['ori_shape']),
                        gt_boxes=target['bboxes'].detach().cpu().tolist(),
                        gt_classes=target['cls'].detach().cpu().tolist(),
                        pred_boxes=pred['bboxes'].detach().cpu().tolist(),
                        pred_classes=pred['cls'].detach().cpu().tolist(),
                        pred_confidence=pred['conf'].detach().cpu().tolist())
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
    return EvidenceValidator


def capture_contract(validator, canonical_roster, versions):
    expected = list(canonical_roster)
    actual = [str(Path(path).resolve()) for path in validator.dataloader.dataset.im_files]
    if (not expected or len(set(expected)) != len(expected) or len(actual) != len(expected) or
            len(set(actual)) != len(actual) or set(actual) != set(expected)):
        raise ValueError('Actual loader differs from full unique development roster')
    keys = ('imgsz', 'batch', 'workers', 'quantize', 'conf', 'iou', 'max_det',
            'agnostic_nms', 'single_cls', 'rect', 'augment')
    if not hasattr(validator.args, 'quantize'):
        raise ValueError('Pinned validator did not expose actual quantize precision')
    effective = {key: getattr(validator.args, key, None) for key in keys}
    effective['half'] = validator.args.quantize == 16
    return dict(schema='rgbir-evaluation-contract-v1', expected_val_images=len(expected),
        roster=expected, actual_loader_roster=actual, endpoint='fixed_budget_last_ema',
        official_test_accessed=False,
        evaluator_identity=dict(native='ultralytics.DetectionValidator', **versions,
            extension='read_only_post_metric_object_capture_v1'), effective_kwargs=effective)


def verify_population(validator, contract):
    with gzip.open(validator.save_dir / 'objects.jsonl.gz', 'rt', encoding='utf-8') as stream:
        observed = [str(Path(json.loads(line)['image']).resolve()) for line in stream]
    expected = contract['roster']
    if (validator.seen != len(expected) or len(observed) != len(expected) or
            len(set(observed)) != len(observed) or set(observed) != set(expected)):
        raise ValueError('Actual evaluated/object population differs from full roster')
    contract['observed_images'] = validator.seen


def validate_attempt(run, attempt):
    """Require raw metric/objects and a complete receipt before publication."""
    run, attempt = Path(run), Path(attempt)
    target = attempt / 'evaluation_val.json'
    result = read_json(target)
    receipt = read_json(attempt / 'eval_evidence/run_receipt.json')
    if (receipt.get('terminal_status') != 'COMPLETED' or receipt.get('run_kind') != 'eval' or
            receipt.get('data_role') != 'development_val' or result.get('status') != 'completed'):
        raise ValueError('Attempt lacks a completed evaluation receipt')
    cfg = yaml.safe_load((run / 'protocol_config.yaml').read_text(encoding='utf-8'))
    if (any(result.get(key) != cfg[key] for key in ('dataset', 'seed', 'arm', 'source', 'method_id')) or
            result.get('endpoint') != 'fixed_budget_last_ema' or result.get('split') != 'val' or
            result.get('official_test_accessed') is not False or
            Path(result.get('checkpoint', '')).resolve() != (run / 'weights/last.pt').resolve()):
        raise ValueError('Attempt metric identity differs from this completed run')
    if (receipt.get('dataset') != result['dataset'] or receipt.get('seed') != result['seed'] or
            any(receipt.get('inputs', {}).get(key) != result[key]
                for key in ('method_id', 'arm', 'source', 'checkpoint', 'endpoint'))):
        raise ValueError('Attempt receipt identity differs from metric')
    bound = bound_configurations(attempt, receipt, 'eval_evidence')
    if not bound or any(value != cfg for value in bound):
        raise ValueError('Attempt evaluator configuration differs from actual run')
    objects = attempt / 'objects.jsonl.gz'
    if Path(result.get('objects', '')).resolve() != objects.resolve():
        raise ValueError('Attempt objects path is not its own raw evidence')
    copies = [attempt / 'eval_evidence' / relative for relative in receipt.get('metric_snapshots', [])]
    for original in (target, objects):
        content = original.read_bytes()
        if not any(path.is_file() and path.read_bytes() == content for path in copies):
            raise ValueError('Raw metric/object bytes differ from receipt snapshots')
    for relatives in receipt.get('source_snapshots', {}).values():
        if any(not (attempt / 'eval_evidence' / relative).is_file() for relative in relatives):
            raise ValueError('Attempt source/config snapshot missing')
    return result, receipt


def publish_evaluation_attempt(run, attempt):
    """Idempotent publication; failed copying can be retried without GPU work."""
    run, attempt = Path(run), Path(attempt)
    result, _ = validate_attempt(run, attempt)
    source, destination = attempt / 'eval_evidence', run / 'eval_evidence'
    expected = {path.relative_to(source) for path in source.rglob('*') if path.is_file()}
    if destination.exists():
        actual = {path.relative_to(destination) for path in destination.rglob('*') if path.is_file()}
        if actual - expected:
            raise ValueError('Canonical receipt contains another attempt')
    if (run / 'evaluation_val.json').exists():
        if (run / 'evaluation_val.json').read_bytes() != (attempt / 'evaluation_val.json').read_bytes():
            raise ValueError('An original endpoint from another attempt already exists')
    # The metric is last; it never advertises a half-published receipt. Existing
    # identical copies remain untouched, and conflicting bytes are rejected.
    for relative in sorted(expected):
        copy_missing_or_equal(source / relative, destination / relative)
    copy_missing_or_equal(attempt / 'evaluation_val.json', run / 'evaluation_val.json')
    return result


def run(args):
    if args.attempt < 1:
        raise ValueError('Evaluation attempt must be positive')
    cfg, completed, checkpoint = load_run_configuration(args.run, args.config)
    name = 'eval_val' if args.attempt == 1 else 'eval_val_attempt' + str(args.attempt)
    attempt = args.run / name
    if (attempt / 'eval_evidence/run_receipt.json').is_file():
        result = publish_evaluation_attempt(args.run, attempt)
        print(json.dumps(dict(status='published_existing_attempt', result=result)), flush=True)
        return result
    if (args.run / 'evaluation_val.json').exists():
        raise FileExistsError('An original endpoint already exists')
    if attempt.exists():
        raise FileExistsError('Preserve the incomplete attempt; use a new --attempt')

    import torch
    from runtime import legacy
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import DetMetrics, Metric, ap_per_class
    from ultralytics.utils.nms import non_max_suppression
    from diagnose_opportunities import dataset_config, split_images
    if len(legacy.require_bound_lease_from_environment()['gpus']) != 1:
        raise ValueError('One bound GPU required')
    versions = dict(torch=str(torch.__version__), ultralytics=str(legacy.ultralytics.__version__))
    if versions['torch'] != cfg['torch_version'] or versions['ultralytics'] != cfg['ultralytics_version']:
        raise ValueError('Pinned evaluation environment differs')
    images = split_images(dataset_config(cfg['paths']['student_data_yaml']), 'val')
    canonical = [str(path.resolve()) for path in images]
    if len(canonical) != POPULATIONS[cfg['dataset']] or len(set(canonical)) != len(canonical):
        raise ValueError('Full development population differs')
    roster = args.run / 'evaluation_val_roster.txt'
    if roster.exists():
        if roster.read_text(encoding='utf-8').splitlines() != canonical:
            raise ValueError('Earlier evaluation roster differs')
    else:
        with roster.open('x', encoding='utf-8') as stream:
            stream.write(''.join(path + '\n' for path in canonical))
    started, captured = time.time(), {}
    model = YOLO(str(checkpoint), task='detect')
    model.add_callback('on_val_start', lambda validator: captured.update(capture_contract(validator, canonical, versions)))
    model.add_callback('on_val_end', lambda validator: verify_population(validator, captured))
    try:
        metrics = model.val(validator=make_evidence_validator(DetectionValidator),
            data=cfg['paths']['student_data_yaml'], split='val',
            imgsz=cfg['imgsz'], batch=cfg['batch'], workers=cfg['workers'], device='0',
            plots=False, save_json=False, verbose=False, project=str(args.run), name=name, exist_ok=False)
        contract = attempt / 'evaluation_contract.json'
        write_json_new(contract, captured)
        per_class = []
        for row, ci in enumerate(metrics.box.ap_class_index):
            ap = metrics.box.all_ap[row]
            per_class.append(dict(class_id=int(ci), name=model.names[int(ci)], AP50=float(ap[0]),
                                  AP75=float(ap[5]), mAP50_95=float(ap.mean())))
        result = dict(status='completed', AP50=float(metrics.box.map50), AP75=float(metrics.box.map75),
            mAP50_95=float(metrics.box.map), precision=float(metrics.box.mp), recall=float(metrics.box.mr),
            per_class=per_class, checkpoint=str(checkpoint), endpoint='fixed_budget_last_ema', split='val',
            arm=completed['arm'], source=completed['source'], seed=completed['seed'], dataset=cfg['dataset'],
            official_test_accessed=False, metric_units='fraction_0_to_1', method_id=cfg['method_id'],
            evaluation_contract=str(contract), objects=str(attempt / 'objects.jsonl.gz'), seconds=time.time()-started)
        target = attempt / 'evaluation_val.json'
        write_json_new(target, result)
        implementations = legacy.implementation_files(DetectionValidator, BaseValidator, YOLO.val,
            check_det_dataset, DetMetrics, Metric, ap_per_class, non_max_suppression)
        legacy.emit_bound_run_receipt(run_dir=attempt / 'eval_evidence', method_identity=cfg['method_identity'],
            dataset=cfg['dataset'], data_role='development_val', seed=completed['seed'], run_kind='eval',
            trainers=[Path(__file__), *implementations], losses=[],
            configs=[args.run / 'protocol_config.yaml', Path(cfg['paths']['student_data_yaml']), contract],
            split_rosters=[roster], metric_files=[target, attempt / 'objects.jsonl.gz'], environment=versions,
            inputs=dict(method_id=cfg['method_id'], arm=completed['arm'], source=completed['source'],
                checkpoint=str(checkpoint), endpoint='fixed_budget_last_ema', teacher_labels_used_by_kd=True))
        publish_evaluation_attempt(args.run, attempt)
        print(json.dumps(result), flush=True)
        return result
    except BaseException as error:
        write_json_new(args.run / ('evaluation_failure_attempt' + str(args.attempt) + '_' + str(time.time_ns()) + '.json'),
            dict(status='failed', error=repr(error), traceback=traceback.format_exc(),
                 seconds=time.time()-started, official_test_accessed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
