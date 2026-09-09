"""Syntax/import smoke only; no model loading, forward pass, GPU query or hashing."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def explicit(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true', help='Import pinned runtime on Linux without CUDA visibility.')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sources = list(root.glob('*.py'))
    for folder in ('release_gpu5', 'tools', 'fast_candidate', 'native_fast', 'benchmark_review', 'production_entry'):
        sources.extend((root/folder).rglob('*.py'))
    for path in sources:
        compile(path.read_bytes(), str(path), 'exec')
    result = dict(status='SYNTAX_PASSED', source_files=len(sources), new_hashes_computed=False,
                  models_loaded=False, gpu_queried=False, training_started=False)
    if args.runtime:
        if os.name != 'posix':
            raise SystemExit('--runtime is intended for the pinned Linux 90 environment')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['RGBIR90_PROJECT_ROOT'] = str(root)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(root/'cache/ultralytics'))
        sys.path[:0] = [str(root), str(root/'release_gpu5')]
        import torch
        import ultralytics
        import yaml
        import numpy
        import scipy
        import runtime
        import train_independent
        import independent_criterion
        import selection_adapter
        import classification_logit
        import tools.project_resource_guard as guard
        import dispatch_single_formal
        if str(torch.__version__) != '2.10.0+cu128' or ultralytics.__version__ != '8.4.115':
            raise ValueError('Pinned Torch/Ultralytics version mismatch')
        candidate = explicit(root/'fast_candidate/batched_selection_v1.py', '_port90_fast_candidate')
        native = explicit(root/'native_fast/native_fast.py', '_port90_native_fast')
        api = candidate.make_api(selection_adapter, classification_logit)
        if not issubclass(candidate.make_criterion_type(independent_criterion, api), independent_criterion.IndependentCriterion):
            raise TypeError('C1 factory does not bind the pinned criterion')
        if not issubclass(native.make_criterion_type(independent_criterion), independent_criterion.IndependentCriterion):
            raise TypeError('N factory does not bind the pinned criterion')
        explicit(root/'production_entry/train_c1_fast.py', '_port90_c1_entry')
        explicit(root/'benchmark_review/cadence106.py', '_port90_cadence')
        blocked = []
        for path in sorted((root/'configs').glob('*_s42.template.yaml')):
            cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
            try:
                train_independent.validate_execution(cfg, formal=False)
            except ValueError as error:
                if 'PORT90_BLOCKED' not in str(error):
                    raise
                blocked.append(path.name)
            else:
                raise AssertionError('A template unexpectedly admitted execution: '+path.name)
        if len(blocked) != 3:
            raise AssertionError('Missing blocked C1/N/C0 templates')
        result.update(status='CPU_IMPORT_PASSED_TEMPLATES_BLOCKED', templates_blocked=blocked,
                      versions=dict(torch=str(torch.__version__), ultralytics=ultralytics.__version__,
                                    numpy=numpy.__version__, scipy=scipy.__version__, yaml=yaml.__version__),
                      lease_file=str(guard.DEFAULT_LEASE_FILE),
                      cuda_initialized=torch.cuda.is_initialized())
        if result['cuda_initialized']:
            raise AssertionError('CPU import unexpectedly initialized CUDA')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
