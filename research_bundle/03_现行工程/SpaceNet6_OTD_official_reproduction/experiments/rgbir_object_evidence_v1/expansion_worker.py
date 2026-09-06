"""One-GPU, two-arm seed expansion; immutable release_v2 method implementation."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
GUARD = REPO / 'tools/project_resource_guard.py'
RESOURCES = {'expected_vram_mib': 10000, 'expected_rss_mib': 49152, 'free_safety_mib': 2048}


def write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_path(a, arm):
    return a.run_root / f'{a.stage}_{arm}_s{a.seed}_{a.attempt}'


def train_command(a, arm):
    command = [sys.executable, str(a.source / 'train_object_evidence.py'),
               '--config', str(a.source / 'config_drone.yaml'), '--output', str(run_path(a, arm)),
               '--arm', arm, '--seed', str(a.seed)]
    if a.stage == 'canary':
        command += ['--max-steps', '24']
    return command


def guarded(a, job_id, kind, command, log_path, on_status=None):
    guard_command = [sys.executable, str(GUARD), 'run', '--job-id', job_id, '--kind', kind,
                     '--candidate-gpu', str(a.gpu), '--gpu-count', '1',
                     '--cuda-processes-per-gpu', '1', '--expected-vram-mib', '10000',
                     '--expected-rss-mib', '49152', '--free-safety-mib', '2048']
    if a.stage == 'canary' and kind == 'train':
        guard_command.append('--non-formal-train')
    guard_command += ['--', *command]
    # Retry only a guard admission refusal, never a started child's exit code 2.
    with log_path.open('x') as log:
        while True:
            queued = launched = False
            process = subprocess.Popen(guard_command, cwd=REPO, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(record, dict):
                    status = record.get('status')
                    queued = queued or status == 'QUEUED'
                    launched = launched or status == 'LAUNCHED'
                    if on_status is not None and status in ('QUEUED', 'LAUNCHED'):
                        on_status(status)
            code = process.wait()
            if code != 2 or not queued or launched:
                return code
            log.write('Admission queued; retrying the same physical GPU after 30 seconds.\n')
            log.flush()
            time.sleep(30)


def comparison_path(a):
    return a.root / f'canary_comparison_s{a.seed}_{a.attempt}.json'


def validate_canary_resources(resources, gpu):
    key = str(gpu)
    if resources['gpu_ids'] != [gpu]:
        raise AssertionError('Canary GPU identity differs from worker allocation')
    if resources['cuda_pid_counts'][key] > 1:
        raise AssertionError('Canary used more than one CUDA PID')
    if resources['per_gpu_peak_vram_mib'][key] > RESOURCES['expected_vram_mib']:
        raise AssertionError('Canary VRAM peak exceeds the frozen reservation')
    if resources['peak_rss_mib'] > RESOURCES['expected_rss_mib']:
        raise AssertionError('Canary RSS peak exceeds the frozen reservation')


def compare_canaries(a):
    import torch
    paths = [run_path(a, arm) for arm in ('paired', 'weight0')]
    result = {'status': 'failed', 'seed': a.seed, 'teacher_seed': 42, 'reference_seed': 42,
              'physical_gpu': a.gpu, 'source': str(a.source.resolve()),
              'arms': ['paired', 'weight0'], 'cpu_tensor_comparison': True,
              'receipts': [str(p / 'completion_receipt.json') for p in paths], 'checks': {}}
    try:
        records = [json.loads((p / 'completion_receipt.json').read_text()) for p in paths]
        for r, arm in zip(records, ('paired', 'weight0')):
            validate_canary_resources(r['resources'], a.gpu)
            if r['seed'] != a.seed or r['arm'] != arm or r['status'] != 'canary_completed':
                raise AssertionError('Canary receipt identity or completion mismatch')
            if r['optimizer_updates'] < 24 or not r['selected_objects']:
                raise AssertionError('Canary updates or selected objects absent')
            checks = r['gradient_checks']
            if not any(c['kd_score_gradient_l2'] > 0 for c in checks):
                raise AssertionError('No nonzero KD gradient')
            if not all(c['weight0_exact_loss_gradient'] and not c['teacher_has_grad']
                       and not c['reference_has_grad'] for c in checks):
                raise AssertionError('Gradient isolation or native equivalence failed')
        for filename in ('initial_student.pt', 'first_batch.pt'):
            values = [torch.load(p / filename, map_location='cpu', weights_only=True) for p in paths]
            equal = set(values[0]) == set(values[1]) and all(
                torch.equal(value, values[1][key]) for key, value in values[0].items())
            result['checks'][filename] = equal
            if not equal:
                raise AssertionError(f'Paired/weight0 direct tensor comparison failed: {filename}')
        result.update(status='passed',
                      resource_reservation_checks_passed=True,
                      gpu_reserved_peak_mib=max(r['gpu_reserved_peak_mib'] for r in records),
                      gpu_allocated_peak_mib=max(r['gpu_allocated_peak_mib'] for r in records),
                      resources=[r['resources'] for r in records],
                      optimizer_updates=[r['optimizer_updates'] for r in records],
                      amp_skipped_updates=[r['amp_skipped_updates'] for r in records])
    except Exception as error:
        result['error'] = repr(error)
    with comparison_path(a).open('x') as file:
        file.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result['status'] == 'passed'


def run_worker(a):
    import yaml
    a.root.mkdir(parents=True, exist_ok=True)
    a.run_root.mkdir(parents=True, exist_ok=True)
    worker = a.root / 'workers' / f'{a.stage}_s{a.seed}_{a.attempt}'
    worker.mkdir(parents=True, exist_ok=False)
    status = {'stage': a.stage, 'seed': a.seed, 'physical_gpu': a.gpu, 'pid': os.getpid(),
              'teacher_seed': 42, 'reference_seed': 42, 'source': str(a.source.resolve()),
              'status': 'waiting_gpu_lock', 'arm_order': a.arm_order, 'results': []}

    def update(**fields):
        status.update(fields, updated_at=timestamp())
        write(worker / 'status.json', status)

    plan = dict(status, resources=RESOURCES, commands={arm: train_command(a, arm) for arm in a.arm_order},
                config_seed=42, student_seed_override=a.seed,
                fixed_auxiliaries='IR teacher seed42 and RGB reference seed42 for every student seed')
    write(worker / 'plan.json', plan)
    effective = yaml.safe_load((a.source / 'config_drone.yaml').read_text())
    effective['seed'] = a.seed
    (worker / 'effective_config.yaml').write_text(yaml.safe_dump(effective, sort_keys=False))
    for arm in a.arm_order:
        write(worker / f'{arm}_seed_override.json', {
            'arm': arm, 'student_seed': a.seed, 'source_config_seed': 42,
            'teacher_seed': 42, 'reference_seed': 42,
            'command': train_command(a, arm),
            'effective_config': str(worker / 'effective_config.yaml')})
    update()
    with (a.root / f'gpu_{a.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if a.stage == 'full':
            try:
                accepted = json.loads(comparison_path(a).read_text())
                if (accepted['status'] != 'passed' or accepted['seed'] != a.seed
                        or accepted['physical_gpu'] != a.gpu
                        or accepted['source'] != str(a.source.resolve())
                        or not accepted['resource_reservation_checks_passed']):
                    raise ValueError('This seed, GPU and method source require an accepted canary')
            except Exception as error:
                update(status='failed_preflight', error=repr(error))
                return 1
        for index, arm in enumerate(a.arm_order):
            run = run_path(a, arm)
            result = {'arm': arm, 'run': str(run)}
            update(status='training', active_arm=arm, queued_arms=a.arm_order[index+1:])
            try:
                if run.exists():
                    raise FileExistsError(f'Preserve existing attempt: {run}')
                code = guarded(a, f'rgbir_oev1_expand_{a.stage}_{arm}_s{a.seed}_{a.attempt}',
                               'train', train_command(a, arm), worker / f'{arm}_train.log',
                               lambda state: update(status='queued' if state == 'QUEUED' else 'training'))
                result.update(train_exit_code=code, status='training_failed' if code else 'trained')
                if code == 0 and a.stage == 'full':
                    update(status='evaluating')
                    command = [sys.executable, str(a.source / 'evaluate_object_evidence.py'),
                               '--config', str(a.source / 'config_drone.yaml'), '--run', str(run)]
                    code = guarded(a, f'rgbir_oev1_expand_eval_{arm}_s{a.seed}_{a.attempt}', 'eval',
                                   command, worker / f'{arm}_eval.log',
                                   lambda state: update(status='queued_eval' if state == 'QUEUED' else 'evaluating'))
                    result.update(eval_exit_code=code, status='evaluation_failed' if code else 'completed')
            except Exception as error:
                result.update(status='failed', error=repr(error))
            status['results'].append(result)
            write(worker / f'{arm}_result.json', result)
            update()
        successful = all(r['status'] in ('trained', 'completed') for r in status['results'])
        if a.stage == 'canary':
            try:
                successful = compare_canaries(a) and successful
            except Exception as error:
                status['comparison_error'] = repr(error)
                successful = False
        update(status='completed' if successful else 'partial_failed', active_arm=None, queued_arms=[])
        return 0 if successful else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 123), required=True)
    parser.add_argument('--arm-order', nargs=2, choices=('paired', 'weight0'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', choices=('canary', 'full'), required=True)
    parser.add_argument('--attempt', default='attempt1')
    a = parser.parse_args()
    a.arm_order = a.arm_order or (['weight0', 'paired'] if a.seed == 0 else ['paired', 'weight0'])
    if set(a.arm_order) != {'paired', 'weight0'}:
        parser.error('--arm-order must contain each arm exactly once')
    if a.gpu < 0 or Path(a.attempt).name != a.attempt:
        parser.error('GPU must be nonnegative and attempt a simple filename suffix')
    for filename in ('train_object_evidence.py', 'evaluate_object_evidence.py', 'config_drone.yaml'):
        if not (a.source / filename).is_file():
            parser.error(f'Missing frozen source: {a.source / filename}')
    return a


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    sys.exit(run_worker(parse_args()))
