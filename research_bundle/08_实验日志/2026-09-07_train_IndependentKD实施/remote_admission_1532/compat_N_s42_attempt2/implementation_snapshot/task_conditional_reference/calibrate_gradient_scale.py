"""One train-only gradient calibration at the frozen R state, without optimizer updates."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import statistics
import time
import torch
import yaml
from legacy_bridge import legacy, HERE
from train_task_conditional import build_trainer


def gradient_norm(value, parameters):
    values = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    return float(torch.stack([v.detach().float().square().sum() for v in values if v is not None]).sum().sqrt()) if any(v is not None for v in values) else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if not cfg.get('geometry_contract'):
        raise ValueError('Real calibration requires independent geometry evidence')
    from geometry_contract import GeometryContract
    contract = GeometryContract.load(cfg['geometry_contract'])
    if not contract.verified or not contract.entries:
        raise ValueError('Real calibration cannot use unverified/empty geometry')
    for key in ('student_data_yaml', 'privileged_data_yaml'):
        if 'test' in yaml.safe_load(Path(cfg['paths'][key]).read_text()):
            raise ValueError('Train/val-only YAML required')
    lease = legacy.require_bound_lease_from_environment()
    if len(lease['gpus']) != 1:
        raise ValueError('One bound GPU required')
    a.output.mkdir(parents=True, exist_ok=False)
    cfg['seed'] = 20260907
    cfg['localization_coefficient'] = 0.1  # unused, arm c adds zero L; calibration measures unit L
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    trainer = build_trainer(cfg, a.config, a.output, 'c', max_steps=None)
    trainer._setup_train()
    trainer.model.load_state_dict(trainer.criterion_ref.reference.state_dict(), strict=True)
    trainer.model.eval()
    head = trainer.model.model[-1]
    indices = list(head.f)[:2]
    parameters = [p for index in indices for p in trainer.model.model[index].parameters() if p.requires_grad]
    if len(indices) != 2 or not parameters:
        raise ValueError('Missing shared P3/P4 feature modules')
    rows, ratios = [], []
    started = time.time()
    trainer.epoch = 0
    for i, batch in enumerate(itertools.islice(trainer.train_loader, 64)):
        batch = trainer.preprocess_batch(batch)
        trainer.model.zero_grad(set_to_none=True)
        # Float32 gradients avoid GradScaler skip policy contaminating the one-off scale.
        prediction = trainer.model(batch['img'])
        trainer.criterion_ref(prediction, batch)
        native, c, loc = trainer.criterion_ref.last_components
        gn, gl = gradient_norm(native, parameters), gradient_norm(loc, parameters)
        ratio = gn / gl if gl > 0 else None
        if ratio is not None:
            ratios.append(ratio)
        row = {'batch':i, 'files':list(batch['im_file']), 'native_norm':gn, 'unit_L_norm':gl,
               'native_to_unit_L':ratio, 'selected_count':trainer.criterion_ref.last_stats['l']['selected_count']}
        rows.append(row)
        legacy.append_json(a.output/'calibration_batches.jsonl', row)
        del prediction, native, c, loc, batch
        trainer.criterion_ref.last_components = None
    if len(rows) != 64:
        raise RuntimeError('Frozen 64 calibration batches unavailable')
    raw = .1 * statistics.median(ratios) if ratios else None
    coefficient = min(1.0, raw) if raw is not None else None
    result = {'status':'CALIBRATED' if ratios else 'NO_LOCALIZATION_SIGNAL',
        'lambda_L':coefficient, 'uncapped_lambda_L':raw, 'capped_at_one':raw is not None and raw > 1,
        'target_gradient_ratio':.1, 'valid_nonzero_batches':len(ratios), 'total_batches':len(rows),
        'seed':20260907, 'parameter_module_indices':indices, 'parameter_count':sum(p.numel() for p in parameters),
        'dataset':cfg['dataset'], 'teacher':cfg['teacher'],
        'state_source':cfg['reference'], 'formal_student_initialization':cfg['model'],
        'geometry_contract':cfg['geometry_contract'], 'optimizer_updates':0, 'test_accessed':False,
        'achieved_gradient_ratios':[coefficient / ratio for ratio in ratios] if ratios else [],
        'seconds':time.time()-started, 'resources':legacy.bound_lease_resource_record_from_environment(),
        'gpu_allocated_peak_mib':torch.cuda.max_memory_allocated()/2**20,
        'gpu_reserved_peak_mib':torch.cuda.max_memory_reserved()/2**20}
    legacy.write_json(a.output/'calibration_receipt.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
