"""Observe native augmentation parameters without another random draw or image transform."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from paired_rgbir_data import DualLabelRGBIRDataset, PairingContractError


def parameter_matrix(kind, params, image_shape):
    matrix = np.eye(3, dtype=np.float64)
    if kind == 'RandomPerspective':
        matrix = np.asarray(params['M'], dtype=np.float64).copy()
    elif kind == 'LetterBox':
        matrix[0, 0], matrix[1, 1] = params['ratio']
        matrix[0, 2], matrix[1, 2] = params['left'], params['top']
    elif kind == 'RandomFlip':
        if params['flip']:
            # Annotation coordinates are continuous image boundaries, as native Instances.flip*.
            axis = 0 if params['direction'] == 'horizontal' else 1
            extent = image_shape[1] if axis == 0 else image_shape[0]
            matrix[axis, axis] = -1
            matrix[axis, 2] = extent
    else:
        raise PairingContractError('Unsupported geometry parameter tap: ' + kind)
    return matrix


class ParameterTap:
    def __init__(self, original, recorder, kind):
        self.original, self.recorder, self.kind = original, recorder, kind

    def __call__(self, labels):
        shape = tuple(labels['img'].shape[:2])
        source = str(Path(labels['im_file']).resolve())
        original_shape = tuple(labels['ori_shape'])
        params = self.original(labels)
        if source not in self.recorder:
            resize = np.diag([shape[1] / original_shape[1], shape[0] / original_shape[0], 1.0])
            self.recorder[source] = {'matrix': resize, 'original_shape': original_shape, 'events': []}
        row = self.recorder[source]
        matrix = parameter_matrix(self.kind, params, shape)
        row['matrix'] = matrix @ row['matrix']
        row['events'].append({'kind': self.kind, 'matrix': matrix.tolist()})
        return params


def install_taps(transform, recorder, seen=None):
    seen = set() if seen is None else seen
    if id(transform) in seen:
        return
    seen.add(id(transform))
    name = type(transform).__name__
    if name == 'Compose':
        for child in transform.transforms:
            install_taps(child, recorder, seen)
    elif name in ('RandomPerspective', 'LetterBox', 'RandomFlip'):
        if isinstance(transform.get_params, ParameterTap):
            raise PairingContractError('Geometry tap installed twice')
        transform.get_params = ParameterTap(transform.get_params, recorder, name)
    # Disabled mixing transforms may reference the shared pipeline; do not traverse them.


class TrackedDualLabelRGBIRDataset(DualLabelRGBIRDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_recorder = {}
        install_taps(self.base.transforms, self.geometry_recorder)

    def __getitem__(self, index):
        self.geometry_recorder.clear()
        row = super().__getitem__(index)
        info = dict(row['pair_info'])
        weak = self.geometry_recorder.get(info['weak_source'])
        strong = self.geometry_recorder.get(info['strong_source'])
        if weak is None or strong is None:
            raise PairingContractError('Native transform emitted no coordinate trace')
        info.update(rgb_matrix=weak['matrix'].tolist(), ir_matrix=strong['matrix'].tolist(),
                    original_shape=list(weak['original_shape']),
                    output_shape=list(row['img'].shape[-2:]),
                    geometry_trace={'rgb': copy.deepcopy(weak['events']), 'ir': copy.deepcopy(strong['events'])})
        row['pair_info'] = info
        return row
