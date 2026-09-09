"""New experiment entry point. Formal localization requires real geometry and calibration evidence."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import types
from legacy_bridge import legacy, LEGACY, HERE
from tracked_pair_data import TrackedDualLabelRGBIRDataset
from task_criterion import TaskCriterion
import torch
import yaml

C_ONLY = ('n', 'c', 'c_shuffled', 'c_same_modal')


def build_trainer(cfg, config_path, output, arm, max_steps=None):
    original_write_json = legacy.write_json
    def audited_write_json(path, data):
        if Path(path).name == 'runtime_ready.json':
            data = dict(data, teacher_labels_used_by_kd=arm != 'c_same_modal',
                        privileged_ir_labels_used_by_kd=arm != 'c_same_modal',
                        content_control=arm)
        return original_write_json(path, data)
    legacy.write_json = audited_write_json
    legacy.EvidenceCriterion = TaskCriterion
    if arm == 'c_shuffled':
        from shuffled_pair_data import ShuffledContentRGBIRDataset
        def factory(*args, **kwargs):
            return ShuffledContentRGBIRDataset(*args, donor_roster=cfg['derangement_roster'], **kwargs)
        legacy.DualLabelRGBIRDataset = factory
    elif arm == 'c_same_modal':
        def factory(*args, **kwargs):
            return TrackedDualLabelRGBIRDataset(*args, same_modal=True, **kwargs)
        legacy.DualLabelRGBIRDataset = factory
    else:
        legacy.DualLabelRGBIRDataset = TrackedDualLabelRGBIRDataset
    trainer = legacy.build_trainer(cfg, config_path, output, arm, max_steps)
    if arm == 'c_shuffled':
        original_preprocess = trainer.preprocess_batch
        def preprocess(self, batch):
            batch = original_preprocess(batch)
            batch['content_img'] = batch['content_img'].to(self.device, non_blocking=True).float()/255
            return batch
        trainer.preprocess_batch = types.MethodType(preprocess, trainer)
    return trainer


def validate_config(cfg, arm, formal):
    if arm not in (*C_ONLY, 'l', 'cl', 'cgt', 'cl_random'):
        raise ValueError('Unsupported implemented arm')
    for key in ('student_data_yaml', 'privileged_data_yaml'):
        if 'test' in yaml.safe_load(Path(cfg['paths'][key]).read_text()):
            raise ValueError('Train/val-only YAML required')
    if str(torch.__version__) != cfg['torch_version'] or str(legacy.ultralytics.__version__) != cfg['ultralytics_version']:
        raise RuntimeError('Pinned environment changed')
    if arm == 'c_same_modal':
        if cfg['teacher'] == cfg['reference'] or cfg['paths']['student_data_yaml'] != cfg['paths']['privileged_data_yaml']:
            raise ValueError('Same-modal requires an independent RGB teacher and RGB-only data')
    if arm == 'c_shuffled' and not Path(cfg.get('derangement_roster', '')).is_file():
        raise ValueError('Frozen train-only derangement roster required')
    if cfg['localization_coefficient'] is None and arm not in C_ONLY:
        raise ValueError('Localization coefficient must come from calibration')
    if formal:
        if cfg.get('protocol_status') != 'FROZEN' or cfg.get('formal_training_authorized') is not True:
            raise ValueError('Formal run requires frozen protocol')
        if arm in ('c_shuffled', 'c_same_modal'):
            acceptance_path = Path(cfg.get('canary_acceptance') or '')
            if not acceptance_path.is_file():
                raise ValueError('Formal C control requires its accepted real canary')
            acceptance = json.loads(acceptance_path.read_text())
            if (acceptance.get('status') != 'ACCEPTED' or arm not in acceptance.get('arms', [])
                or acceptance.get('minimum_successful_updates', 0) < 24
                or acceptance.get('nonzero_C_gradient') is not True
                or acceptance.get('dataset') != cfg['dataset']
                or acceptance.get('teacher') != cfg['teacher']
                or acceptance.get('reference') != cfg['reference']
                or acceptance.get('legacy_C_N_equivalence') is not True):
                raise ValueError('C control canary is incomplete or uses different models')
            if arm == 'c_shuffled' and acceptance.get('derangement_roster') != cfg['derangement_roster']:
                raise ValueError('Shuffled canary donor roster differs')
        if arm not in C_ONLY:
            for key in ('geometry_contract', 'calibration_receipt', 'd2_receipt', 'canary_acceptance'):
                if not cfg.get(key) or not Path(cfg[key]).is_file():
                    raise ValueError('Missing formal evidence: ' + key)
            from geometry_contract import GeometryContract
            geometry = GeometryContract.load(cfg['geometry_contract'])
            if not geometry.verified or not geometry.entries:
                raise ValueError('Formal geometry must contain independently accepted coverage')
            calibration = json.loads(Path(cfg['calibration_receipt']).read_text())
            if calibration.get('status') != 'CALIBRATED' or calibration['lambda_L'] != cfg['localization_coefficient']:
                raise ValueError('Frozen coefficient differs from accepted calibration')
            if calibration.get('dataset') != cfg['dataset'] or calibration.get('geometry_contract') != cfg['geometry_contract']:
                raise ValueError('Calibration dataset/geometry differ')
            if calibration.get('state_source') != cfg['reference'] or calibration.get('teacher') != cfg['teacher']:
                raise ValueError('Calibration teacher/reference differ')
            d2 = json.loads(Path(cfg['d2_receipt']).read_text())
            if not d2.get('geometry_verified') or d2.get('totals', {}).get('selected_count', 0) <= 0:
                raise ValueError('No geometry-eligible actual-anchor opportunity')
            if d2.get('dataset') != cfg['dataset'] or d2.get('geometry_contract') != cfg['geometry_contract'] or d2.get('split') != 'train':
                raise ValueError('D2 must use matching train dataset and geometry')
            if d2.get('teacher') != cfg['teacher'] or d2.get('reference') != cfg['reference']:
                raise ValueError('D2 model identities differ')
            acceptance = json.loads(Path(cfg['canary_acceptance']).read_text())
            if acceptance.get('status') != 'ACCEPTED' or arm not in acceptance.get('arms', []):
                raise ValueError('No accepted real canary for this arm')
            if acceptance.get('dataset') != cfg['dataset'] or acceptance.get('geometry_contract') != cfg['geometry_contract'] or acceptance.get('lambda_L') != cfg['localization_coefficient']:
                raise ValueError('Canary protocol differs')
            if acceptance.get('minimum_successful_updates', 0) < 24 or acceptance.get('legacy_C_N_equivalence') is not True:
                raise ValueError('Insufficient canary updates or missing historical equivalence')


def run(args):
    cfg = yaml.safe_load(args.config.read_text())
    cfg['seed'] = args.seed
    validate_config(cfg, args.arm, args.max_steps is None)
    lease = legacy.require_bound_lease_from_environment()
    if len(lease['gpus']) != 1:
        raise ValueError('Exactly one bound GPU per task')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    shutil.copy2(args.config, args.output / 'submitted_config.yaml')
    resolved_config = args.output / 'protocol_config.yaml'
    resolved_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    snapshot = args.output / 'implementation_snapshot'
    snapshot.mkdir()
    for p in HERE.glob('*.py'):
        shutil.copy2(p, snapshot / p.name)
    shutil.copytree(LEGACY, snapshot / 'legacy_oev1', ignore=shutil.ignore_patterns('__pycache__'))
    legacy.write_json(args.output / 'launch_manifest.json', {'method_id': cfg['method_id'], 'arm': args.arm,
        'seed': cfg['seed'], 'command': [sys.executable, *sys.argv], 'gpu': lease['gpus'],
        'canary_max_updates': args.max_steps, 'test_accessed': False,
        'teacher_labels_used_by_kd': args.arm != 'c_same_modal', 'geometry_contract': cfg.get('geometry_contract'),
        'formal_localization': args.max_steps is None and args.arm not in C_ONLY})
    trainer = None
    started = time.time()
    try:
        trainer = build_trainer(cfg, resolved_config, args.output, args.arm, args.max_steps)
        trainer.train()
        criterion = trainer.criterion_ref
        if args.max_steps is None and trainer.epoch + 1 != cfg['epochs']:
            raise RuntimeError('Frozen epoch budget not completed')
        if args.max_steps is not None:
            if trainer.real_updates < args.max_steps:
                raise RuntimeError('Fewer than requested successful optimizer updates')
            if not criterion.gradient_checks:
                raise RuntimeError('Missing real gradient validation')
            if args.arm != 'l' and not any(r['kd_score_gradient_l2'] > 0 for r in criterion.gradient_checks):
                raise RuntimeError('No nonzero original C score gradient')
            if args.arm not in C_ONLY and not criterion.loc_gradient_checks:
                raise RuntimeError('No nonzero real localization gradient')
        receipt = {'status': 'canary_completed' if args.max_steps is not None else 'training_completed',
            'method_id': cfg['method_id'], 'arm': args.arm, 'seed': cfg['seed'],
            'epochs_configured': cfg['epochs'], 'last_epoch': trainer.epoch + 1,
            'optimizer_updates': trainer.real_updates, 'optimizer_update_attempts': trainer.update_attempts,
            'amp_skipped_updates': trainer.skipped_amp_updates, 'ema_updates': trainer.ema.updates,
            'batches': criterion.calls, 'selected_objects': criterion.selected_total,
            'selected_localization_objects': criterion.loc_selected_total,
            'gradient_checks': criterion.gradient_checks,
            'gpu_allocated_peak_mib': torch.cuda.max_memory_allocated()/2**20,
            'gpu_reserved_peak_mib': torch.cuda.max_memory_reserved()/2**20,
            'resources': legacy.bound_lease_resource_record_from_environment(), 'seconds': time.time()-started,
            'checkpoint': str(args.output/'weights/last.pt'), 'official_test_accessed': False,
            'single_seed_exploratory': True}
        legacy.write_json(args.output / 'completion_receipt.json', receipt)
        legacy.emit_bound_run_receipt(run_dir=args.output/'run_evidence', method_identity=cfg['method_identity'],
            dataset=cfg['dataset'], data_role='development_train', seed=cfg['seed'], run_kind='train',
            trainers=[Path(__file__), HERE/'tracked_pair_data.py', HERE/'shuffled_pair_data.py', *legacy.implementation_files(legacy.DetectionTrainer)],
            losses=[HERE/'task_criterion.py', HERE/'localization_loss.py', HERE/'content_controls.py', LEGACY/'object_evidence_loss.py',
                    *legacy.implementation_files(criterion.native)],
            configs=[resolved_config, Path(cfg['paths']['student_data_yaml']), Path(cfg['paths']['privileged_data_yaml'])],
            split_rosters=[Path(cfg['paths']['paired_train_mapping'])] +
                ([Path(cfg['derangement_roster'])] if args.arm == 'c_shuffled' else []),
            metric_files=[args.output/'completion_receipt.json'],
            environment={'torch': str(torch.__version__), 'ultralytics': legacy.ultralytics.__version__},
            inputs={'method_id':cfg['method_id'], 'arm':args.arm, 'teacher_labels_used_by_kd':args.arm != 'c_same_modal',
                    'student_native_gt_only': True, 'canary': args.max_steps is not None,
                    'initial_weights':cfg['model'], 'teacher_weights':cfg['teacher'], 'reference_weights':cfg['reference']})
        print(json.dumps({'status': receipt['status'], 'output': str(args.output)}), flush=True)
    except BaseException as error:
        legacy.write_json(args.output/'failure_receipt.json', {'status':'failed','arm':args.arm,
            'error':repr(error),'traceback':traceback.format_exc(),'seconds':time.time()-started,
            'updates':getattr(trainer,'real_updates',0),'official_test_accessed':False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=[*C_ONLY,'l','cl','cgt','cl_random'], required=True)
    p.add_argument('--seed', type=int, choices=[0,42,123], required=True)
    p.add_argument('--max-steps', type=int)
    a = p.parse_args()
    if a.max_steps is not None and a.max_steps < 1:
        p.error('max-steps must be positive')
    run(a)


if __name__ == '__main__':
    main()
