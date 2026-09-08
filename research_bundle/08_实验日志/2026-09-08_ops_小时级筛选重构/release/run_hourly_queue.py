"""Three canaries, then fixed N/C0/C1 FT3+full-dev pairs on the existing globallease."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import yaml

ARMS = ('N', 'C0', 'C1')
BASE = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
PY = Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python')
REFERENCE = BASE / 'rgbir_independent_kd_v2_20260907/release_gpu5'
DISPATCH = BASE / 'rgbir_task_conditional_v1_20260907/release_v8'
NATIVE_BINDING = BASE / 'rgbir_independent_kd_v2_20260907/evaluator_profile_attempt2/evaluation_profile_binding/binding.json'
ENDPOINT = 'HOURLY_SCREEN_FT_E3_LAST_EMA'
VRAM_LIMIT, RSS_LIMIT = 8192, 32768


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def file_stat(path):
    path = Path(path)
    value = path.stat()
    return dict(path=str(path), bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def finite_positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Missing finite positive measured resource: ' + name)
    return value


def training_reservation(canary):
    resources = canary['resources']
    peaks = list(resources['per_gpu_peak_vram_mib'].values())
    if not peaks:
        raise ValueError('Canary has no measured NVML peak')
    nvml = max(finite_positive(x, 'NVML') for x in peaks)
    rss = finite_positive(resources['peak_rss_mib'], 'process-tree RSS')
    allocated = finite_positive(canary['gpu_allocated_peak_mib'], 'allocated')
    reserved = finite_positive(canary['gpu_reserved_peak_mib'], 'reserved')
    peak = max(nvml, allocated, reserved)
    vram_budget = math.ceil((peak + max(256, 0.05 * peak)) / 256) * 256
    rss_budget = math.ceil((rss + max(2048, 0.05 * rss)) / 1024) * 1024
    if vram_budget > VRAM_LIMIT or rss_budget > RSS_LIMIT:
        raise ValueError('Canary plus margin exceeds frozen ceiling; no automatic expansion: '
                         + repr(dict(vram_mib=vram_budget, rss_mib=rss_budget,
                                     vram_limit=VRAM_LIMIT, rss_limit=RSS_LIMIT)))
    return dict(vram_mib=vram_budget, rss_mib=rss_budget,
                measured_nvml_peak_mib=nvml, measured_allocated_peak_mib=allocated,
                measured_reserved_peak_mib=reserved, measured_rss_peak_mib=rss,
                margin='GPU max(256 MiB,5%), round up to256; RSS max(2048 MiB,5%), round up to1024')


def check_canary(path, cfg, config_path):
    result = read(path)
    if (result.get('status') != 'HOURLY_SCREEN_CANARY_COMPLETED'
            or result.get('arm') != cfg['arm'] or result.get('model') != cfg['model']
            or type(result.get('successful_updates')) is not int
            or result['successful_updates'] < 24):
        raise ValueError('Corresponding 24-successful-update canary required: ' + str(path))
    if Path(result['config_copy']).read_bytes() != Path(config_path).read_bytes():
        raise ValueError('Canary configuration bytes differ: ' + cfg['arm'])
    initialization = read(Path(path).parent / 'initialization_check.json')
    if (initialization.get('status') != 'PASS_FULL_STATE_WARM_START'
            or initialization.get('initial_checkpoint', {}).get('path') != str(Path(cfg['model']).resolve())
            or initialization.get('fresh_optimizer') is not True
            or initialization.get('fresh_ema') is not True):
        raise ValueError('Full-state warm-start initialization check failed: ' + cfg['arm'])
    return dict(receipt=file_stat(path), initialization=initialization,
                successful_updates=result['successful_updates'], resources=training_reservation(result))


def first_thirty(path):
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Sample stream record must be a dictionary')
                row.pop('arm', None)
                for field in ('batch', 'im_file', 'cls', 'bboxes', 'teacher_cls', 'teacher_bboxes'):
                    if field not in row:
                        raise ValueError('Sample stream field missing: ' + field)
                if type(row['batch']) is not int or not isinstance(row['im_file'], list) or len(row['im_file']) != 32:
                    raise ValueError('Sample stream must describe a full B32 batch')
                json.dumps(row, allow_nan=False)
                rows.append(row)
                if len(rows) == 30:
                    break
    if len(rows) != 30:
        raise ValueError('Need the first thirty actual batches: ' + str(path))
    first = rows[0]['batch']
    if first not in (0, 1) or [row['batch'] for row in rows] != list(range(first, first + 30)):
        raise ValueError('Sample stream is not the first contiguous thirty batches')
    return rows


def compare_streams(output):
    paths = {arm: output / 'canaries' / arm / 'sample_stream.jsonl' for arm in ARMS}
    records = {arm: first_thirty(path) for arm, path in paths.items()}
    for arm in ARMS[1:]:
        for index, (left, right) in enumerate(zip(records['N'], records[arm])):
            if left != right:
                raise ValueError('Canary sample stream differs N versus ' + arm + ' at record ' + str(index + 1))
    return dict(batch_records=30, batch_size=32, arms=list(ARMS), record_content_exact=True,
                ignored_fields=['arm'], sources={arm: file_stat(path) for arm, path in paths.items()})


def check_terminal(stage, arm, receipt):
    expected = 'HOURLY_SCREEN_TRAINING_COMPLETED' if stage == 'train' else 'HOURLY_SCREEN_EVALUATION_COMPLETED'
    if (receipt.get('status') != expected or receipt.get('scope') != 'HOURLY_SCREEN_FT'
            or receipt.get('arm') != arm or receipt.get('seed') != 42
            or receipt.get('single_seed') is not True or receipt.get('endpoint') != ENDPOINT
            or receipt.get('formal_e200_complete') is not False
            or receipt.get('new_hash_computed') is not False
            or receipt.get('official_test_accessed') is not False):
        raise ValueError('Missing completed independent FT3 identity: ' + arm + ' ' + stage)
    if stage == 'train':
        if receipt.get('epochs_configured') != 3 or receipt.get('last_epoch') != 3 or receipt.get('batches') != 192:
            raise ValueError('Training must complete exactly3 epochs/192 batches')
    elif (receipt.get('epochs') != 3 or receipt.get('full_dev_images') != 1469
          or receipt.get('full_dev_gt_objects') != 22462):
        raise ValueError('Evaluation must cover the fixed FT3 endpoint and complete dev')


def make_job(release, output, arm, stage, budget=None, native_contract=None):
    cfg = release / 'configs' / ('drone_' + arm + '_s42_FT3.yaml')
    canary = output / 'canaries' / arm
    run_dir = output / 'runs' / arm
    name = re.sub(r'[^A-Za-z0-9_-]', '_', output.name)
    job = dict(id='hourly_' + name + '_' + arm + '_' + stage, arm=arm, stage=stage,
               formal=False, kind='eval' if stage == 'eval' else 'train')
    if stage in ('canary', 'train'):
        destination = canary if stage == 'canary' else run_dir
        command = [str(PY), str(release / 'train_hourly.py'), '--reference-dir', str(REFERENCE),
                   '--config', str(cfg), '--output', str(destination)]
        command += ['--canary'] if stage == 'canary' else ['--canary-receipt', str(canary / 'canary.json')]
        receipt = destination / ('canary.json' if stage == 'canary' else 'hourly_training_receipt.json')
        job.update(vram_mib=VRAM_LIMIT if stage == 'canary' else budget['vram_mib'],
                   rss_mib=RSS_LIMIT if stage == 'canary' else budget['rss_mib'])
    elif stage == 'eval':
        destination = output / 'evaluations' / arm
        command = [str(PY), str(release / 'evaluate_hourly.py'), '--reference-dir', str(REFERENCE),
                   '--run', str(run_dir), '--output', str(destination),
                   '--native-profile-binding', str(NATIVE_BINDING),
                   '--native-contract-config', str(native_contract),
                   '--canary-receipt', str(canary / 'canary.json')]
        receipt = destination / 'hourly_evaluation_receipt.json'
        job.update(vram_mib=2048, rss_mib=8192)
    else:
        raise ValueError('Unknown stage: ' + stage)
    job.update(command=command, expected_receipt=str(receipt))
    return job


def run(args):
    release, output = args.release_dir.resolve(), args.output.resolve()
    if Path('/mnt/dataset/yudongfang') not in output.parents:
        raise ValueError('Output must remain under /mnt/dataset/yudongfang')
    if output.exists():
        raise FileExistsError('Use a new output; completed and failed attempts are preserved')
    # Preserve the pinned venv invocation; never resolve its Python symlink.
    for path in (PY, REFERENCE / 'runtime.py', DISPATCH / 'resource_dispatch.py', NATIVE_BINDING,
                 release / 'train_hourly.py', release / 'evaluate_hourly.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    configs = {}
    for arm in ARMS:
        path = release / 'configs' / ('drone_' + arm + '_s42_FT3.yaml')
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        if (cfg['arm'] != arm or cfg['seed'] != 42 or cfg['epochs'] != 3
                or cfg['expected_train_images'] != 2048 or cfg['batch'] != 32):
            raise ValueError('Wrong common hourly configuration: ' + arm)
        if not Path(cfg['native_contract_config']).is_file():
            raise FileNotFoundError(cfg['native_contract_config'])
        configs[arm] = cfg
    for arm in ARMS[1:]:
        for key in ('model', 'teacher', 'reference', 'native_contract_config', 'paths', 'lr0', 'lrf', 'warmup_epochs'):
            if configs[arm][key] != configs['N'][key]:
                raise ValueError('Common initialization/data/LR identity differs: ' + key)
    sys.path.insert(0, str(DISPATCH))
    import resource_dispatch
    if Path(resource_dispatch.__file__).resolve() != DISPATCH / 'resource_dispatch.py':
        raise RuntimeError('Wrong existing globallease dispatcher')
    output.mkdir(parents=True, exist_ok=False)
    queue = output / 'queue'
    queue.mkdir()
    shutil.copyfile(Path(__file__), queue / 'executed_driver.py')
    if (queue / 'executed_driver.py').read_bytes() != Path(__file__).read_bytes():
        raise AssertionError('Driver source copy differs')
    canary_jobs = [make_job(release, output, arm, 'canary') for arm in ARMS]
    write_new(queue / 'manifest.json', dict(scope='HOURLY_SCREEN_FT', canary_jobs=canary_jobs,
        sequence=['N canary', 'C0 canary', 'C1 canary', 'N train/eval', 'C0 train/eval', 'C1 train/eval'],
        common_model=configs['N']['model'], reference_release=str(REFERENCE), python=str(PY),
        dispatcher=file_stat(DISPATCH / 'resource_dispatch.py'),
        native_profile_binding=file_stat(NATIVE_BINDING),
        initial_canary_ceilings=dict(vram_mib=VRAM_LIMIT, rss_mib=RSS_LIMIT),
        new_hash_computed=False, new_scheduler_created=False, ap_adaptive=False, formal_e200_complete=False))
    completed, checks = [], {}
    started = time.time()
    try:
        for job in canary_jobs:
            resource_dispatch.run_job(job, queue)
            arm = job['arm']
            config = release / 'configs' / ('drone_' + arm + '_s42_FT3.yaml')
            checks[arm] = check_canary(Path(job['expected_receipt']), configs[arm], config)
            completed.append(dict(id=job['id'], arm=arm, stage='canary', receipt=job['expected_receipt']))
        streams = compare_streams(output)
        for arm in ARMS[1:]:
            if checks[arm]['initialization']['initial_checkpoint'] != checks['N']['initialization']['initial_checkpoint']:
                raise ValueError('Canary initial checkpoint stat differs across arms')
        write_new(queue / 'canary_checks.json', dict(status='HOURLY_CANARIES_MATCHED', arms=checks,
                  sample_stream=streams, new_hash_computed=False, scope='Warm-start/first30 flow/resources only'))
        jobs = []
        for arm in ARMS:
            jobs += [make_job(release, output, arm, 'train', checks[arm]['resources']),
                     make_job(release, output, arm, 'eval', native_contract=configs[arm]['native_contract_config'])]
        write_new(queue / 'training_evaluation_jobs.json', dict(jobs=jobs, new_hash_computed=False))
        for job in jobs:
            resource_dispatch.run_job(job, queue)
            check_terminal(job['stage'], job['arm'], read(job['expected_receipt']))
            completed.append(dict(id=job['id'], arm=job['arm'], stage=job['stage'], receipt=job['expected_receipt']))
            write_new(queue / (job['id'] + '_completed.json'), completed[-1])
        write_new(queue / 'completion.json', dict(status='HOURLY_SCREEN_MATRIX_COMPLETED', completed=completed,
            seconds=time.time() - started, single_seed=True, seed=42, epochs=3, train_images=2048,
            batches_per_arm=192, new_hash_computed=False, formal_e200_complete=False,
            formal_paper_gain_claim=False, ap_adaptive=False))
    except BaseException as error:
        write_new(queue / 'failure.json', dict(status='HOURLY_SCREEN_MATRIX_STOPPED_ON_FAILURE',
            error=repr(error), traceback=traceback.format_exc(), completed=completed,
            seconds=time.time() - started, new_hash_computed=False, automatic_retry=False,
            prior_attempts_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
