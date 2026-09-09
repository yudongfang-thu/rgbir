"""Train a single independent KD arm through the pinned native trainer."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import torch
import yaml
from runtime import HERE, REFERENCE, LEGACY, legacy, ORIGINAL_CRITERION, ORIGINAL_DATASET, TrackedDualLabelRGBIRDataset
from independent_criterion import IndependentCriterion
from protocol import ARMS, SOURCES, require_valid_config


def build_trainer(cfg, config_path, output, arm=None, max_steps=None, historical=False):
    cfg = dict(cfg)
    cfg['arm'] = arm or cfg['arm']
    cfg['kd_weight'] = .1
    legacy.EvidenceCriterion = ORIGINAL_CRITERION if historical else IndependentCriterion
    legacy.DualLabelRGBIRDataset = ORIGINAL_DATASET if historical else TrackedDualLabelRGBIRDataset
    old_arm = 'weight0' if cfg['arm'] == 'N' else 'paired'
    return legacy.build_trainer(cfg, config_path, output, old_arm, max_steps)


def validate_execution(cfg, formal=False):
    if cfg.get('port90', {}).get('status') != 'READY':
        raise ValueError('PORT90_BLOCKED: bind actual 90 inputs and complete the local readiness work first')
    require_valid_config(cfg, formal=formal)
    if cfg.get('source', 'paired') != 'paired':
        raise ValueError('Only paired source is implemented/eligible in this release')
    for key in ('student_data_yaml', 'privileged_data_yaml'):
        if 'test' in yaml.safe_load(Path(cfg['paths'][key]).read_text()):
            raise ValueError('Train/dev-only YAML required')
    if str(torch.__version__) != cfg['torch_version'] or str(legacy.ultralytics.__version__) != cfg['ultralytics_version']:
        raise RuntimeError('Pinned environment mismatch')
    for key in ('model', 'teacher', 'reference'):
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(cfg[key])
    if not formal:
        return
    if cfg.get('protocol_status') != 'FROZEN':
        raise ValueError('Formal protocol not frozen')
    from admission import check_readiness
    check_readiness(cfg, HERE)


def run(args):
    cfg = yaml.safe_load(args.config.read_text())
    cfg.update(arm=args.arm, source=args.source, seed=args.seed)
    validate_execution(cfg, formal=args.max_steps is None)
    lease = legacy.require_bound_lease_from_environment()
    if len(lease['gpus']) != 1:
        raise ValueError('Exactly one bound GPU required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    (args.output/'protocol_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    snapshot = args.output/'implementation_snapshot'
    shutil.copytree(HERE, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    legacy.write_json(args.output/'launch_manifest.json', dict(method_id=cfg['method_id'],
        arm=cfg['arm'], source=cfg['source'], seed=cfg['seed'], command=[sys.executable, *sys.argv],
        gpu=lease['gpus'], inputs={k:dict(path=cfg[k], bytes=Path(cfg[k]).stat().st_size) for k in ('model','teacher','reference')},
        source_module=str(HERE), test_accessed=False, canary_max_updates=args.max_steps))
    started = time.time()
    trainer = None
    try:
        trainer = build_trainer(cfg, args.config, args.output, max_steps=args.max_steps)
        trainer.train()
        criterion = trainer.criterion_ref
        if args.max_steps is None and trainer.epoch + 1 != cfg['epochs']:
            raise RuntimeError('Run stopped before E200')
        if args.max_steps is not None:
            if trainer.real_updates < args.max_steps:
                raise RuntimeError('Canary did not reach successful-update count')
            active = any(max(r.get('kd_gradient_l2', 0), r.get('kd_score_gradient_l2', 0),
                             r.get('kd_box_gradient_l2', 0)) > 0 for r in criterion.gradient_checks)
            if not active or criterion.selected_total <= 0:
                raise RuntimeError('No actual KD gradient/selected-object evidence')
        receipt = dict(status='canary_completed' if args.max_steps is not None else 'training_completed',
            method_id=cfg['method_id'], arm=cfg['arm'], source=cfg['source'], dataset=cfg['dataset'], seed=cfg['seed'],
            model=cfg['model'], teacher=cfg['teacher'], reference=cfg['reference'],
            classification_coefficient=cfg['classification_coefficient'], localization_coefficient=cfg['localization_coefficient'],
            epochs_configured=cfg['epochs'], last_epoch=trainer.epoch+1,
            optimizer_updates=trainer.real_updates, optimizer_update_attempts=trainer.update_attempts,
            amp_skipped_updates=trainer.skipped_amp_updates, ema_updates=trainer.ema.updates,
            batches=criterion.calls, selected_objects=criterion.selected_total,
            gradient_checks=criterion.gradient_checks, kd_last=criterion.last_stats,
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            resources=legacy.bound_lease_resource_record_from_environment(), seconds=time.time()-started,
            checkpoint=str(args.output/'weights/last.pt'), official_test_accessed=False)
        if args.max_steps is not None:
            from evidence_bindings import capture_execution_binding
            receipt['execution_binding'] = str(capture_execution_binding(args.output, cfg, 'canary', HERE))
        legacy.write_json(args.output/'completion_receipt.json', receipt)
        evidence_configs = [args.output/'protocol_config.yaml', Path(cfg['paths']['student_data_yaml']), Path(cfg['paths']['privileged_data_yaml'])]
        if cfg['arm'] in ('L1','L_GT'):
            evidence_configs.append(Path(cfg['geometry_contract']))
        legacy.emit_bound_run_receipt(run_dir=args.output/'run_evidence', method_identity=cfg['method_identity'],
            dataset=cfg['dataset'], data_role='development_train', seed=cfg['seed'], run_kind='train',
            trainers=[Path(__file__), LEGACY/'train_object_evidence.py', REFERENCE/'tracked_pair_data.py',
                      *legacy.implementation_files(legacy.DetectionTrainer)],
            losses=[HERE/'independent_criterion.py', HERE/'classification_logit.py', HERE/'selection_adapter.py',
                    LEGACY/'object_evidence_loss.py', *legacy.implementation_files(criterion.native)],
            configs=evidence_configs,
            split_rosters=[Path(cfg['paths']['paired_train_mapping'])], metric_files=[args.output/'completion_receipt.json'],
            environment=dict(torch=str(torch.__version__), ultralytics=str(legacy.ultralytics.__version__)),
            inputs=dict(method_id=cfg['method_id'], arm=cfg['arm'], source=cfg['source'],
                teacher_labels_used_by_kd=True, student_native_gt_only=True, initial_weights=cfg['model'],
                teacher_weights=cfg['teacher'], reference_weights=cfg['reference'], canary=args.max_steps is not None))
        print(json.dumps(dict(status=receipt['status'], output=str(args.output), updates=trainer.real_updates)), flush=True)
    except BaseException as error:
        legacy.write_json(args.output/'failure_receipt.json', dict(status='failed', arm=cfg['arm'],
            error=repr(error), traceback=traceback.format_exc(), seconds=time.time()-started,
            updates=getattr(trainer, 'real_updates', 0), official_test_accessed=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--source', choices=SOURCES, default='paired')
    p.add_argument('--seed', type=int, choices=(0,42,123), required=True)
    p.add_argument('--max-steps', type=int)
    args = p.parse_args()
    if args.max_steps is not None and args.max_steps < 24:
        p.error('Real new-path canary requires at least 24 successful updates')
    run(args)


if __name__ == '__main__':
    main()
