"""Read-only CPU comparison of the two realized student-seed canaries; no hashes."""
import inspect
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import yaml
import ultralytics
from ultralytics.data.build import build_dataloader, seed_worker
from ultralytics.engine.trainer import BaseTrainer

R = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_expand_20260906')
RUNS = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906')


def load_json(path):
    return json.loads(path.read_text())


def tensor_comparison(paths, filename):
    states = [torch.load(path / filename, map_location='cpu', weights_only=True) for path in paths]
    same_keys = set(states[0]) == set(states[1])
    equal_keys, unequal_keys, details = [], [], {}
    for key in sorted(set(states[0]) & set(states[1])):
        first, second = states[0][key], states[1][key]
        equal = torch.equal(first, second)
        (equal_keys if equal else unequal_keys).append(key)
        if not equal:
            details[key] = {'shape': list(first.shape),
                            'differing_elements': int((first != second).sum()),
                            'max_absolute_difference': float((first.float() - second.float()).abs().max())}
    return {'same_keys': same_keys, 'total_keys': len(states[0]),
            'equal_tensor_count': len(equal_keys), 'different_tensor_count': len(unequal_keys),
            'all_equal': same_keys and not unequal_keys, 'different_keys': unequal_keys,
            'equal_keys': equal_keys if filename == 'first_batch.pt' else None,
            'different_tensor_details': details}


result = {'status': 'completed', 'student_seeds': [0, 123], 'teacher_seed': 42, 'reference_seed': 42,
          'torch': str(torch.__version__), 'ultralytics': ultralytics.__version__,
          'cuda_used': False, 'scope': 'canary initialization and first 30 logged batches only',
          'identity': [], 'cross_seed': {}}
torch.set_num_threads(4)
for arm in ('paired', 'weight0'):
    paths = [RUNS / f'canary_{arm}_s{seed}_attempt1' for seed in (0, 123)]
    for seed, path in zip((0, 123), paths):
        manifest = load_json(path / 'launch_manifest.json')
        args = yaml.safe_load((path / 'args.yaml').read_text())
        receipt = load_json(path / 'completion_receipt.json')
        effective = yaml.safe_load((R / f'workers/canary_s{seed}_attempt1/effective_config.yaml').read_text())
        command = manifest['command']
        identity = {'arm': arm, 'path': str(path), 'expected_seed': seed,
                    'args_seed': args['seed'], 'launch_seed': manifest['seed'],
                    'completion_seed': receipt['seed'], 'effective_config_seed': effective['seed'],
                    'command_seed': int(command[command.index('--seed') + 1])}
        identity['all_match'] = all(identity[key] == seed for key in (
            'args_seed', 'launch_seed', 'completion_seed', 'effective_config_seed', 'command_seed'))
        result['identity'].append(identity)
    logged = [[json.loads(line) for line in (path / 'kd_batches.jsonl').read_text().splitlines()] for path in paths]
    comparison = {'initial_student': tensor_comparison(paths, 'initial_student.pt'),
                  'first_batch': tensor_comparison(paths, 'first_batch.pt'),
                  'first_batch_sample_paths_equal': logged[0][0]['student_files'] == logged[1][0]['student_files'],
                  'first_batch_sample_paths': logged[0][0]['student_files'],
                  'logged_batch_counts': [len(records) for records in logged],
                  'all_logged_batch_sample_orders_equal': len(logged[0]) == len(logged[1]) and all(
                      first['student_files'] == second['student_files'] for first, second in zip(*logged))}
    result['cross_seed'][arm] = comparison
sources = {'build_dataloader': inspect.getsource(build_dataloader),
           'seed_worker': inspect.getsource(seed_worker),
           'base_trainer_seed_lines': [line.strip() for line in inspect.getsource(BaseTrainer.__init__).splitlines()
                                       if 'seed' in line]}
source_path = R / 'cross_seed_native_source.txt'
source_path.write_text('\n\n'.join(f'{key}:\n{value}' for key, value in sources.items()))
result['native_pipeline'] = {
    'build_dataloader_path': inspect.getsourcefile(build_dataloader),
    'source_snapshot': str(source_path),
    'dataloader_generator_seed_lines': [line.strip() for line in sources['build_dataloader'].splitlines()
                                       if 'seed' in line or 'generator' in line],
    'seed_worker_source': sources['seed_worker'],
    'base_trainer_seed_lines': sources['base_trainer_seed_lines']}
out = R / 'cross_seed_realization.json'
with out.open('x') as file:
    file.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'identity_all_match': all(r['all_match'] for r in result['identity']),
                  'cross_seed': {arm: {'initial_different_tensors': v['initial_student']['different_tensor_count'],
                                      'first_batch_all_equal': v['first_batch']['all_equal'],
                                      'all_logged_sample_orders_equal': v['all_logged_batch_sample_orders_equal']}
                                 for arm, v in result['cross_seed'].items()},
                  'native_pipeline': result['native_pipeline']}, indent=2))
