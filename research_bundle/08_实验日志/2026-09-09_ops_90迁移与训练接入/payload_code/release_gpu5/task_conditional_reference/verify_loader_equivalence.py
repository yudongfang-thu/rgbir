"""Independent CPU audit of real native-pair and tracked-pair loader behavior.

No model, optimizer or GPU is created. The report compares every original
sample field, global CPU RNG continuation, and raw annotation projections.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import random
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset, img2label_paths


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def same_rng(a, b):
    return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and
            a[1][2:] == b[1][2:] and torch.equal(a[2], b[2]))


def equal_value(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_value(x, y) for x, y in zip(a, b))
    return a == b


def projected_gt_check(label_file, matrix, original_shape, output_size, boxes, classes):
    # LLVIP image paths resolve into raw/ but its YOLO labels live in processed/.
    # The native dataset's actual label_file is the source of raw annotations.
    label_file = Path(label_file)
    text = label_file.read_text().strip()
    raw = np.asarray([[float(x) for x in row.split()] for row in text.splitlines()], dtype=np.float64).reshape(-1, 5) if text else np.zeros((0, 5))
    h, w = original_shape
    centers = raw[:, 1:3] * np.asarray([w, h])
    half = raw[:, 3:5] * np.asarray([w, h]) / 2
    lower, upper = centers - half, centers + half
    corners = np.stack((lower, np.column_stack((upper[:, 0], lower[:, 1])),
                        upper, np.column_stack((lower[:, 0], upper[:, 1]))), axis=1)
    homogeneous = np.concatenate((corners, np.ones((*corners.shape[:2], 1))), axis=-1)
    warped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    warped = warped[..., :2] / warped[..., 2:]
    projected = np.concatenate((warped.min(1), warped.max(1)), axis=1)
    oh, ow = output_size
    projected[:, [0, 2]] = projected[:, [0, 2]].clip(0, ow)
    projected[:, [1, 3]] = projected[:, [1, 3]].clip(0, oh)
    b = boxes.detach().cpu().numpy().astype(np.float64)
    actual = np.concatenate((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), axis=1) * np.asarray([ow, oh, ow, oh])
    target_classes = classes.detach().cpu().numpy().reshape(-1)
    if len(actual) > len(projected):
        raise AssertionError('Augmentation created more target boxes than raw annotations')
    if len(actual):
        errors = np.abs(actual[:, None, :] - projected[None, :, :]).max(-1)
        allowed = target_classes[:, None] == raw[None, :, 0]
        rows, cols = linear_sum_assignment(np.where(allowed, errors, 1e9))
        maximum = float(errors[rows, cols].max())
        matched_classes = bool(allowed[rows, cols].all())
        if not matched_classes or maximum > .001:
            raise AssertionError(f'Raw GT projection disagrees with augmented GT: max coordinate error={maximum}')
    else:
        maximum = 0.0
    return {'raw_gt_count': len(raw), 'output_gt_count': len(actual),
            'native_filtered_or_cropped_count': len(raw)-len(actual),
            'max_coordinate_error_pixels': maximum, 'tolerance_pixels': .001,
            'projection_matches_all_output_gt': True, 'source_label': str(label_file)}


def make_native_datasets(cfg):
    overrides = {key: cfg[key] for key in ('imgsz', 'batch', 'nbs', 'workers', 'seed')}
    overrides.update(cfg['augmentation'])
    overrides.update(task='detect', mode='train', cache=False, rect=False)
    args = get_cfg(overrides=overrides)
    datasets = []
    for key in ('student_data_yaml', 'privileged_data_yaml'):
        source = cfg['paths'][key]
        if 'test' in yaml.safe_load(Path(source).read_text()):
            raise ValueError('Train/val-only YAML required')
        data = check_det_dataset(source, autodownload=False)
        datasets.append(build_yolo_dataset(args, data['train'], cfg['batch'], data,
                                          mode='train', rect=False, stride=32))
    return datasets


def verify_dataset(cfg, vendor_cls, tracked_cls):
    mapping = json.loads(Path(cfg['paths']['paired_train_mapping']).read_text())
    base, teacher = make_native_datasets(cfg)
    # All native metadata/transforms are copied before any sample is transformed;
    # the old and tracked paths therefore start with identical cache/buffer state.
    tracked_base, tracked_teacher = copy.deepcopy(base), copy.deepcopy(teacher)
    old = vendor_cls(base, teacher, mapping, max_teacher_cache=cfg['teacher_cache_images'])
    new = tracked_cls(tracked_base, tracked_teacher, mapping, max_teacher_cache=cfg['teacher_cache_images'])
    indices = [0, len(old)//2, len(old)-1]
    reports, old_rows, new_rows = [], [], []
    for index, seed in zip(indices, (0, 42, 123)):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        before = rng_state()
        expected = old[index]
        after_old = rng_state()
        restore_rng(before)
        actual = new[index]
        after_new = rng_state()
        fields = {key: equal_value(value, actual.get(key)) for key, value in expected.items() if key != 'pair_info'}
        fields['pair_info_original_fields'] = all(equal_value(value, actual['pair_info'].get(key)) for key, value in expected['pair_info'].items())
        if not all(fields.values()):
            raise AssertionError('Changed original sample fields: ' + str([key for key, value in fields.items() if not value]))
        if not same_rng(after_old, after_new):
            raise AssertionError('Tracked loader changed Python/NumPy/Torch CPU RNG continuation')
        info = actual['pair_info']
        checks = {}
        teacher_index = new._teacher_index[info['strong_source']]
        label_files = {'rgb': new.base.label_files[index],
                       'ir': new.teacher_base.label_files[teacher_index]}
        for modality, source, matrix, box_key, class_key in (
                ('rgb', info['weak_source'], info['rgb_matrix'], 'bboxes', 'cls'),
                ('ir', info['strong_source'], info['ir_matrix'], 'strong_bboxes', 'strong_cls')):
            checks[modality] = projected_gt_check(label_files[modality], matrix, info['original_shape'],
                info['output_shape'], actual[box_key], actual[class_key])
        reports.append({'dataset_index': index, 'augmentation_seed': seed, 'source': info['weak_source'],
            'all_original_fields_equal': fields, 'cpu_rng_continuation_equal': True,
            'rgb_ir_matrices_equal': bool(np.array_equal(info['rgb_matrix'], info['ir_matrix'])),
            'pair_info': info, 'raw_gt_geometry': checks})
        old_rows.append(expected); new_rows.append(actual)
    old_batch = old.collate_fn(old_rows)
    new_batch = new.collate_fn(new_rows)
    collate_checks = {key: equal_value(value, new_batch.get(key)) for key, value in old_batch.items() if key != 'pair_info'}
    if not all(collate_checks.values()):
        raise AssertionError('Collate changed original batch fields')
    if sum(row['raw_gt_geometry'][mod]['output_gt_count'] for row in reports for mod in ('rgb', 'ir')) == 0:
        raise AssertionError('The sample set contains no surviving boxes for geometry validation')
    return {'dataset': cfg['dataset'], 'train_images': len(old), 'samples': reports,
            'collate_all_original_fields_equal': collate_checks, 'status': 'PASSED',
            'cpu_only': True, 'test_accessed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new evidence path')
    sys.path[:0] = [str(args.module_dir), str(args.module_dir/'legacy_oev1')]
    from paired_rgbir_data import DualLabelRGBIRDataset
    from tracked_pair_data import TrackedDualLabelRGBIRDataset
    from prepare_configs import configurations
    torch.set_num_threads(4)
    result = {'status': 'RUNNING', 'started_at': datetime.datetime.now().astimezone().isoformat(),
              'module_dir': str(args.module_dir), 'gpu_visible': os.environ['CUDA_VISIBLE_DEVICES'],
              'datasets': [], 'scope': '3 real train samples per dataset plus collate; no optimizer equivalence claim'}
    try:
        for cfg in configurations().values():
            result['datasets'].append(verify_dataset(cfg, DualLabelRGBIRDataset, TrackedDualLabelRGBIRDataset))
        result['status'] = 'PASSED'
    except Exception as error:
        import traceback
        result.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
    result['finished_at'] = datetime.datetime.now().astimezone().isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'datasets': [(row['dataset'], row['status']) for row in result['datasets']],
                      'error': result.get('error')}, ensure_ascii=False))
    if result['status'] != 'PASSED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
