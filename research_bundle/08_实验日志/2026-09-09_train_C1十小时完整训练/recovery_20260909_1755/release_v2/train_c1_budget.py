"""Frozen C1 batched loss with original E200 training and epoch wall-clock observation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

from budget_policy import forecast, write_json

OLD = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_c1_fastpath_20260908')
PLAN = OLD/'production_release_v2/production_plan_v2.json'


def clone(function, **bindings):
    result = types.FunctionType(function.__code__, dict(function.__globals__, **bindings),
                                function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def prepare(seed):
    import yaml
    plan = json.loads(PLAN.read_text())
    cell = plan['cells'][str(seed)]
    config = Path(cell['config'])
    cfg = yaml.safe_load(config.read_text())
    if cfg != yaml.safe_load(Path(cell['original_config']).read_text()):
        raise ValueError('Original scientific configuration changed')
    if (cfg['arm'], cfg['source'], cfg['seed'], cfg['epochs'], cfg['batch'], cfg['imgsz']) != ('C1','paired',seed,200,32,640):
        raise ValueError('Wrong frozen run identity')
    if cfg['classification_coefficient'] != 0.09227393550836771 or cfg['localization_coefficient'] != 0:
        raise ValueError('Frozen coefficient changed')
    cadence = json.loads(Path(plan['cadence_receipt']).read_text())
    if cadence['actual_optimizer_calls'] < 24 or not cadence['final_student_ema_finite'] or cadence['candidate_execution']['fallback_batches']:
        raise ValueError('Existing fast-path training evidence is invalid')
    reference = Path(plan['reference_dir'])
    sys.path.insert(0, str(reference))
    import runtime, train_independent as training, independent_criterion as criterion
    import selection_adapter as selection, classification_logit as classification
    training.validate_execution(cfg, formal=True)
    spec = importlib.util.spec_from_file_location('_accepted_c1_batched', plan['candidate_source'])
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate
    spec.loader.exec_module(candidate)
    fast_type = candidate.make_criterion_type(criterion, candidate.make_api(selection, classification))
    return plan, config, cfg, runtime, training, fast_type


def run(args):
    plan, config, cfg, runtime, training, fast_type = prepare(args.seed)
    import torch
    if args.preflight:
        print(json.dumps(dict(status='CPU_PREFLIGHT_PASS', seed=args.seed, cuda_initialized=torch.cuda.is_initialized(),
                              original_configuration_equal=True, original_readiness_passed=True,
                              candidate_source=plan['candidate_source'], new_content_digest=False)), flush=True)
        return
    runtime.legacy.require_bound_lease_from_environment()
    begin = float(os.environ['C1_RUN_STARTED_MONOTONIC'])
    fast_build = clone(training.build_trainer, IndependentCriterion=fast_type)
    holder = {}
    original_emitter = runtime.legacy.emit_bound_run_receipt

    def emit_actual(**kwargs):
        if Path(kwargs['run_dir']).name == 'run_evidence':
            kwargs['trainers'] = [*kwargs['trainers'], Path(__file__), Path(__file__).with_name('budget_policy.py')]
            kwargs['losses'] = [*kwargs['losses'], Path(plan['candidate_source'])]
            kwargs['inputs'] = dict(kwargs.get('inputs', {}), execution_version='C1-BATCHED-E200-BUDGET10H-20260909',
                                   old_checkpoint_resumed=False, budget_seconds=36000)
        return original_emitter(**kwargs)

    def build(bound_cfg, config_path, output, arm=None, max_steps=None, historical=False):
        if max_steps is not None or historical:
            raise ValueError('Fresh full E200 only')
        trainer = fast_build(bound_cfg, config_path, output, arm=arm, max_steps=None, historical=False)
        holder['trainer'] = trainer
        rows = []
        epoch_begin = [None]
        snapshot = Path(output)/'budget_execution_snapshot'
        snapshot.mkdir()
        for i, path in enumerate([Path(__file__), Path(__file__).with_name('budget_policy.py'), Path(plan['candidate_source']), config]):
            shutil.copy2(path, snapshot/f'{i:02d}_{path.name}')

        def start(t):
            if type(t.criterion_ref) is not fast_type or getattr(t.args, 'resume', False):
                raise RuntimeError('Fresh accepted fast path was not used')
            if len(t.train_loader.dataset) != 17990 or len(t.train_loader) != 563:
                raise RuntimeError('Full original training population changed')
            write_json(Path(output)/'fastpath_launch.json', dict(status='RUNNING', seed=args.seed,
                fresh_initialization=cfg['model'], candidate_source=plan['candidate_source'],
                epochs=200, batch=32, workers=4, train_images=17990, test_accessed=False,
                accepted_implementation='RGBIR-C1-BATCHED-SELECTION-v1-E200-FRESH-20260908',
                wrapper_version='C1-BATCHED-E200-BUDGET10H-20260909'))

        def epoch_start(t):
            epoch_begin[0] = time.monotonic()

        def epoch_end(t):
            torch.cuda.synchronize()
            now = time.monotonic()
            row = dict(epoch=int(t.epoch)+1, epoch_seconds=now-epoch_begin[0],
                       run_elapsed_seconds=now-begin, optimizer_updates=t.real_updates,
                       amp_skips=t.skipped_amp_updates, batches=t.criterion_ref.calls,
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            rows.append(row)
            with (Path(output)/'epoch_timing.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            decision = forecast(rows, now-begin, reserve=3600 if len(rows) < 200 else 0)
            write_json(Path(output)/'budget_progress.json', decision)
            print('C1_EPOCH_TIMING '+json.dumps(dict(**row, projected_total_seconds=decision['projected_total_seconds'])), flush=True)
            if decision['stop_for_budget']:
                write_json(Path(output)/'budget_stop.json', dict(status='BUDGET_ABORTED', **decision))
                raise RuntimeError('Engineering budget exceeded; no AP-based stop or method conclusion')

        trainer.add_callback('on_train_start', start)
        trainer.add_callback('on_train_epoch_start', epoch_start)
        trainer.add_callback('on_fit_epoch_end', epoch_end)
        return trainer

    runtime.legacy.emit_bound_run_receipt = emit_actual
    try:
        native_args = argparse.Namespace(config=config, output=args.output, arm='C1', source='paired', seed=args.seed, max_steps=None)
        clone(training.run, build_trainer=build)(native_args)
        trainer = holder['trainer']
        write_json(args.output/'fastpath_completion_receipt.json', dict(status='FAST_C1_E200_COMPLETED',
                   seed=args.seed, last_epoch=int(trainer.epoch)+1, optimizer_updates=trainer.real_updates,
                   checkpoint=str(args.output/'weights/last.pt'), independent_evaluation_pending=True,
                   official_test_accessed=False, new_content_digest=False))
    finally:
        runtime.legacy.emit_bound_run_receipt = original_emitter
        runtime.legacy.EvidenceCriterion = runtime.ORIGINAL_CRITERION
        runtime.legacy.DualLabelRGBIRDataset = runtime.ORIGINAL_DATASET


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0,42,123), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())
