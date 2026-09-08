# coding: utf-8
"""LLVIP L3 three-arm screen on the original global lease only.

Resource reservation and first-thirty validators below are copied unchanged
from the completed hourly run_hourly_queue.py. No GPU is started on import.
"""
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

DATASETS = ('llvip',)
ARMS = {'llvip': ('N', 'L3-DFL', 'L3-GT')}
DEV = {'llvip': (2406, 7879), 'drone': (1469, 22462)}
BASE = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
PY = Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python')
REFERENCE = BASE / 'rgbir_independent_kd_v2_20260907/release_gpu5'
DISPATCH = BASE / 'rgbir_task_conditional_v1_20260907/release_v8'
NATIVE_BINDING = BASE / 'rgbir_independent_kd_v2_20260907/evaluator_profile_attempt2/evaluation_profile_binding/binding.json'
SCOPE = 'OBJECT_DFL_FT3_BNFROZEN'
ENDPOINT = 'OBJECT_DFL_FT3_BNFROZEN_LAST_EMA'
C1_COEFFICIENT = 0.09227393550836771
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


def coefficient_plan(dataset, receipt):
    if receipt.get('status') != 'OBJECT_DFL_CALIBRATION_COMPLETED' or receipt.get('dataset') != dataset:
        raise ValueError('Calibration did not complete for this dataset')
    fixed = dict(scope='OBJECT_DFL_FIXED8_BNFROZEN', batches=8, seed=42,
                 reset_all_parameters_buffers_each_batch=True, optimizer_updates=0, ema_updates=0,
                 bn_running_buffers_unchanged=True, formal64_admission=False,
                 new_hash_computed=False, official_test_accessed=False)
    if any(receipt.get(k) != v for k, v in fixed.items()):
        raise ValueError('Calibration fixed-eight/reset/BN/no-update identity differs')
    values, reasons = receipt['coefficients'], receipt.get('blocked', {})
    if not isinstance(values, dict) or not isinstance(reasons, dict):
        raise ValueError('Calibration coefficients/blocked must be dictionaries')
    if values.get('N', 0) != 0:
        raise ValueError('Calibration changed native zero coefficient')
    accepted, blocked = {'N': 0.}, {}
    for arm in ARMS[dataset][1:]:
        if arm == 'C1':
            if values.get(arm, C1_COEFFICIENT) != C1_COEFFICIENT or arm in reasons:
                raise ValueError('Frozen C1 coefficient changed or blocked')
            accepted[arm] = C1_COEFFICIENT
            continue
        if arm not in values:
            raise ValueError('Missing calibration outcome: ' + arm)
        value = values[arm]
        if value is None:
            if not isinstance(reasons.get(arm), str) or not reasons[arm].strip():
                raise ValueError('Null calibration needs an explicit blocked reason: ' + arm)
            blocked[arm] = reasons[arm]
        else:
            if arm in reasons or not 0 < finite_positive(value, arm) <= 1:
                raise ValueError('Inconsistent/out-of-range calibration: ' + arm)
            accepted[arm] = float(value)
    if dataset == 'llvip':
        if ('L3-DFL' in accepted) != ('L3-GT' in accepted):
            raise ValueError('Shared-mask L2 arms must share calibration validity')
        if 'L3-DFL' in accepted and accepted['L3-DFL'] != accepted['L3-GT']:
            raise ValueError('Shared-mask L3-GT must retain the L3-DFL coefficient')
    return accepted, blocked


def effective_config(cfg, coefficient, calibration_path):
    value = dict(cfg)
    arm = value['arm']
    value['kd_coefficient'] = coefficient
    value['classification_coefficient'] = coefficient if arm in ('C1', 'C2') else 0.
    value['localization_coefficient'] = coefficient if arm.startswith('L3') else 0.
    value['calibration_receipt'] = str(calibration_path)
    return value


def validate_configs(dataset, configs):
    fixed = dict(seed=42, epochs=3, imgsz=640, batch=32, nbs=64, workers=4, amp=True,
                 optimizer='SGD', lr0=.0001, lrf=1., warmup_epochs=0., expected_train_images=2048,
                 freeze_bn_running_statistics=True, scope=SCOPE)
    for arm, cfg in configs.items():
        if cfg['arm'] != arm or cfg['dataset'] != dataset or cfg['expected_val_images'] != DEV[dataset][0]:
            raise ValueError('Dataset/arm/dev identity differs')
        for key, value in fixed.items():
            if cfg.get(key) != value:
                raise ValueError('Frozen direction configuration differs: ' + key)
        if arm == 'N' and cfg.get('kd_coefficient') != 0:
            raise ValueError('N template must have literal zero coefficient')
        if arm == 'C1' and cfg.get('kd_coefficient') != C1_COEFFICIENT:
            raise ValueError('C1 template coefficient precision changed')
    for arm in ARMS[dataset][1:]:
        for key in ('model', 'teacher', 'reference', 'native_contract_config', 'paths',
                    'auxiliary_data_identity', 'augmentation', 'lr0', 'lrf', 'warmup_epochs'):
            if configs[arm][key] != configs['N'][key]:
                raise ValueError('Within-dataset common initialization/data/protocol differs: ' + key)


def check_terminal(stage, dataset, arm, receipt):
    status = {'canary': 'OBJECT_DFL_CANARY_COMPLETED', 'train': 'OBJECT_DFL_TRAINING_COMPLETED',
              'eval': 'OBJECT_DFL_EVALUATION_COMPLETED'}[stage]
    if (receipt.get('status') != status or receipt.get('scope') != SCOPE
            or receipt.get('dataset') != dataset or receipt.get('arm') != arm or receipt.get('seed') != 42
            or receipt.get('single_seed') is not True or receipt.get('endpoint') != ENDPOINT
            or receipt.get('formal_e200_complete') is not False or receipt.get('new_hash_computed') is not False
            or receipt.get('official_test_accessed') is not False):
        raise ValueError('Missing completed direction identity: ' + dataset + '/' + arm + '/' + stage)
    if stage in ('canary', 'train'):
        if (receipt.get('bn_running_buffers_unchanged') is not True
                or receipt.get('bn_affine_trainable') is not True or receipt.get('bn_buffer_count', 0) <= 0):
            raise ValueError('Actual BN freeze evidence missing')
        if type(receipt.get('successful_updates')) is not int or receipt['successful_updates'] < 24:
            raise ValueError('At least 24 actual optimizer updates required')
        if stage == 'train' and (receipt.get('epochs_configured') != 3 or receipt.get('last_epoch') != 3 or receipt.get('batches') != 192):
            raise ValueError('Exactly 3 epochs / 192 batches required')
    elif (receipt.get('epochs') != 3 or receipt.get('full_dev_images') != DEV[dataset][0]
          or receipt.get('full_dev_gt_objects') != DEV[dataset][1]):
        raise ValueError('Complete original dev population required')


def check_canary(path, cfg, config_path):
    result = read(path)
    check_terminal('canary', cfg['dataset'], cfg['arm'], result)
    if result.get('model') != cfg['model'] or Path(result['config_copy']).read_bytes() != Path(config_path).read_bytes():
        raise ValueError('Canary model/config bytes differ')
    for key in ('kd_coefficient', 'classification_coefficient', 'localization_coefficient'):
        if result.get(key) != cfg[key]:
            raise ValueError('Canary effective coefficient differs: ' + key)
    initial = read(Path(path).parent / 'initialization_check.json')
    if (initial.get('status') != 'PASS_FULL_STATE_WARM_START'
            or initial.get('initial_checkpoint', {}).get('path') != str(Path(cfg['model']).resolve())
            or initial.get('fresh_optimizer') is not True or initial.get('fresh_ema') is not True
            or initial.get('teacher_reference_isolated') is not True or initial.get('head_included') is not True):
        raise ValueError('Canary full-state fresh initialization failed')
    return dict(receipt=file_stat(path), initialization=initial, resources=training_reservation(result),
                successful_updates=result['successful_updates'], actual_coefficient=cfg['kd_coefficient'])


def compare_to_native(output, dataset, arms):
    paths = {a: output / 'canaries' / dataset / a / 'sample_stream.jsonl' for a in arms}
    records = {a: first_thirty(p) for a, p in paths.items()}
    for arm in arms:
        if records[arm] != records['N']:
            raise ValueError('Actual first30 canary stream differs: ' + dataset + '/' + arm)
    return dict(batch_records=30, batch_size=32, arms=list(arms), record_content_exact=True,
                ignored_fields=['arm'], sources={a: file_stat(p) for a, p in paths.items()})


def make_job(release, output, dataset, arm, stage, reference_dir=REFERENCE, budget=None, cfg=None):
    template = release / 'configs' / (dataset + '_' + arm + '_s42_FT3.yaml')
    effective = output / 'effective_configs' / template.name
    canary = output / 'canaries' / dataset / arm
    run_dir = output / 'runs' / dataset / arm
    name = re.sub(r'[^A-Za-z0-9_-]', '_', output.name)
    job = dict(id='object_dfl_' + name + '_' + dataset + '_' + arm + '_' + stage,
               dataset=dataset, arm=arm, stage=stage, formal=False, kind='eval' if stage == 'eval' else 'train')
    if stage == 'calibration':
        destination = output / 'calibration' / dataset
        command = [str(PY), str(release / 'calibrate_object_dfl.py'), '--reference-dir', str(reference_dir),
                   '--config', str(template), '--output', str(destination)]
        receipt = destination / 'calibration_receipt.json'
        vram, rss = VRAM_LIMIT, RSS_LIMIT
    elif stage in ('canary', 'train'):
        destination = canary if stage == 'canary' else run_dir
        command = [str(PY), str(release / 'train_object_dfl.py'), '--reference-dir', str(reference_dir),
                   '--config', str(effective), '--output', str(destination)]
        command += ['--canary'] if stage == 'canary' else ['--canary-receipt', str(canary / 'canary.json')]
        receipt = destination / ('canary.json' if stage == 'canary' else 'completion_receipt.json')
        vram, rss = (VRAM_LIMIT, RSS_LIMIT) if stage == 'canary' else (budget['vram_mib'], budget['rss_mib'])
    elif stage == 'eval':
        destination = output / 'evaluations' / dataset / arm
        command = [str(PY), str(release / 'evaluate_object_dfl.py'), '--reference-dir', str(reference_dir),
                   '--config', str(effective), '--checkpoint', str(run_dir / 'weights/last.pt'),
                   '--output', str(destination), '--native-config', str(cfg.get('native_contract_config')
                       or release / 'configs/llvip_native_evaluation.yaml')]
        if dataset == 'drone':
            command += ['--native-profile-binding', str(NATIVE_BINDING)]
        receipt = destination / 'direction_evaluation_receipt.json'
        vram, rss = 2048, 8192
    else:
        raise ValueError('Unsupported stage')
    job.update(command=command, expected_receipt=str(receipt), vram_mib=vram, rss_mib=rss)
    return job


def process_dataset(release, output, dataset, configs, executor, reference_dir=REFERENCE):
    queue = output / 'queue'
    report = dict(dataset=dataset, completed=[], blocked={}, status='RUNNING')
    started = time.time()
    def execute(job):
        write_new(queue / (job['id'] + '_job.json'), job)
        executor(job, queue)
        report['completed'].append(dict(id=job['id'], dataset=dataset, arm=job['arm'],
                                        stage=job['stage'], receipt=job['expected_receipt'], receipt_validated=False))
        return Path(job['expected_receipt'])
    try:
        if configs is None:
            configs = {arm: yaml.safe_load((release / 'configs' / (dataset + '_' + arm + '_s42_FT3.yaml')).read_text(encoding='utf-8')) for arm in ARMS[dataset]}
        validate_configs(dataset, configs)
        if dataset == 'drone' and not NATIVE_BINDING.is_file():
            raise FileNotFoundError(NATIVE_BINDING)
        calibration = execute(make_job(release, output, dataset, 'N', 'calibration', reference_dir))
        coefficients, blocked = coefficient_plan(dataset, read(calibration))
        if read(calibration)['student_initial_checkpoint'] != file_stat(Path(configs['N']['model']).resolve()):
            raise ValueError('Calibration did not use this dataset common student initialization')
        report['completed'][-1]['receipt_validated'] = True
        report['blocked'] = blocked
        if blocked:
            report.update(status='BLOCKED_CALIBRATION',seconds=time.time()-started,new_hash_computed=False)
            write_new(queue / (dataset + '_completion.json'),report)
            return report
        effective, effective_paths, effective_bytes = {}, {}, {}
        for arm, coefficient in coefficients.items():
            cfg = effective_config(configs[arm], coefficient, calibration)
            target = output / 'effective_configs' / (dataset + '_' + arm + '_s42_FT3.yaml')
            with target.open('x', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            effective[arm], effective_paths[arm], effective_bytes[arm] = cfg, target, target.read_bytes()
        write_new(queue / (dataset + '_coefficient_freeze.json'), dict(dataset=dataset, coefficients=coefficients,
                  blocked=blocked, calibration=file_stat(calibration), new_hash_computed=False,
                  effective_configs={a: file_stat(p) for a, p in effective_paths.items()}))
        checks = {}
        for arm in coefficients:
            path = execute(make_job(release, output, dataset, arm, 'canary', reference_dir))
            checks[arm] = check_canary(path, effective[arm], effective_paths[arm])
            report['completed'][-1]['receipt_validated'] = True
        streams = compare_to_native(output, dataset, tuple(coefficients))
        for arm in coefficients:
            if checks[arm]['initialization']['initial_checkpoint'] != checks['N']['initialization']['initial_checkpoint']:
                raise ValueError('Canary initial checkpoint stat differs across arms')
        write_new(queue / (dataset + '_canary_checks.json'), dict(status='OBJECT_DFL_CANARIES_MATCHED',
                  dataset=dataset, arms=checks, sample_stream=streams, new_hash_computed=False))
        for arm in coefficients:
            for stage in ('train', 'eval'):
                if effective_paths[arm].read_bytes() != effective_bytes[arm]:
                    raise ValueError('Frozen effective config was modified')
                job = make_job(release, output, dataset, arm, stage, reference_dir,
                               budget=checks[arm]['resources'], cfg=effective[arm])
                path = execute(job)
                result = read(path)
                check_terminal(stage, dataset, arm, result)
                if stage == 'train':
                    if any(result.get(k) != effective[arm][k] for k in ('kd_coefficient', 'classification_coefficient', 'localization_coefficient')):
                        raise ValueError('Training coefficient differs from frozen calibration')
                    if Path(result['config_copy']).read_bytes() != effective_bytes[arm]:
                        raise ValueError('Training config copy differs from canary effective config')
                    if first_thirty(path.parent / 'sample_stream.jsonl') != first_thirty(output / 'canaries' / dataset / arm / 'sample_stream.jsonl'):
                        raise ValueError('Actual full-training first30 differs from its canary')
                    write_new(queue / (dataset + '_' + arm + '_training_stream_check.json'),
                              dict(batch_records=30, training_equals_canary=True, new_hash_computed=False))
                report['completed'][-1]['receipt_validated'] = True
        report['status'] = 'COMPLETED_WITH_BLOCKED_ARMS' if blocked else 'COMPLETED'
    except Exception as error:
        report.update(status='DATASET_STOPPED_ON_TECHNICAL_FAILURE', error=repr(error), traceback=traceback.format_exc(),
                      automatic_retry=False, unrelated_next_dataset_may_continue=True)
    report['seconds'] = time.time() - started
    report['new_hash_computed'] = False
    write_new(queue / (dataset + '_completion.json'), report)
    return report


def run(args):
    release, output, reference = args.release_dir.resolve(), args.output.resolve(), args.reference_dir.resolve()
    if Path('/mnt/dataset/yudongfang') not in output.parents or output.exists():
        raise ValueError('Output must be a new directory below /mnt/dataset/yudongfang')
    for path in (PY, reference / 'runtime.py', DISPATCH / 'resource_dispatch.py',
                 release / 'calibrate_object_dfl.py', release / 'train_object_dfl.py', release / 'evaluate_object_dfl.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(DISPATCH))
    import resource_dispatch
    if Path(resource_dispatch.__file__).resolve() != (DISPATCH / 'resource_dispatch.py').resolve():
        raise RuntimeError('Wrong existing globallease dispatcher')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'queue').mkdir()
    (output / 'effective_configs').mkdir()
    shutil.copyfile(Path(__file__), output / 'queue/executed_driver.py')
    assert Path(__file__).read_bytes() == (output / 'queue/executed_driver.py').read_bytes()
    write_new(output / 'queue/manifest.json', dict(scope=SCOPE, endpoint=ENDPOINT, dataset_order=list(DATASETS),
              arms=ARMS, reference_dir=str(reference), source=file_stat(Path(__file__)),
              dispatcher=file_stat(DISPATCH / 'resource_dispatch.py'), python=str(PY),
              sequence_per_dataset='calibration8 -> all valid-arm canaries24 -> each arm train/eval',
              initial_calibration_canary_ceilings=dict(vram_mib=VRAM_LIMIT, rss_mib=RSS_LIMIT),
              evaluation_reservation=dict(vram_mib=2048, rss_mib=8192),
              new_hash_computed=False, new_scheduler_created=False, ap_adaptive=False))
    started = time.time()
    reports = [process_dataset(release, output, dataset, None, resource_dispatch.run_job, reference) for dataset in DATASETS]
    failed = any(r['status'] == 'DATASET_STOPPED_ON_TECHNICAL_FAILURE' for r in reports)
    blocked = any(r['blocked'] for r in reports)
    status = 'OBJECT_DFL_MATRIX_PARTIAL_FAILURE' if failed else ('OBJECT_DFL_MATRIX_COMPLETED_WITH_BLOCKED_ARMS' if blocked else 'OBJECT_DFL_MATRIX_COMPLETED')
    result = dict(status=status, datasets=reports, seconds=time.time() - started, single_seed=True, seed=42,
                  epochs=3, train_images_each_dataset=2048, batches_per_full_arm=192,
                  new_hash_computed=False, formal_e200_complete=False, formal_paper_gain_claim=False, ap_adaptive=False)
    write_new(output / 'queue/completion.json', result)
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, default=REFERENCE)
    raise SystemExit(run(parser.parse_args()))
