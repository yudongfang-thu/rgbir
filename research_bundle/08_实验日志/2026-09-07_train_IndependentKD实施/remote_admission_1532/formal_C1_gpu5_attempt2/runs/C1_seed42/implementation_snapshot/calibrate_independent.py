"""Outcome-blind 64-batch calibration, train-mode R-copy reset before every batch."""
import argparse
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import time
import traceback
import torch
import yaml
from runtime import HERE, legacy, to_device
from train_independent import build_trainer, validate_execution
from coverage_probe import build_natural_loader
from selection_adapter import build_classification_selection
from classification_logit import classification_loss_components, classification_loss_from_selection


def gradients(value, parameters):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite calibration loss is a technical failure')
    values = tuple(None if x is None else x.detach().float()
                   for x in torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True))
    if any(not bool(torch.isfinite(x).all()) for x in values if x is not None):
        raise FloatingPointError('Nonfinite float32 calibration gradient')
    return values


def norm(values):
    parts = [x.square().sum() for x in values if x is not None]
    result = float(torch.stack(parts).sum().sqrt()) if parts else 0.0
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite calibration gradient norm')
    return result


def cosine(left, right):
    nl, nr = norm(left), norm(right)
    if not nl or not nr:
        return None
    terms = [(a*b).sum() for a,b in zip(left,right) if a is not None and b is not None]
    return float(torch.stack(terms).sum())/(nl*nr) if terms else 0.0


def parameter_set(model):
    head = model.model[-1]
    indices = list(head.f)[:2]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError('Expected separate P3/P4 feature sources')
    named = [(f'model.{i}.{n}',p) for i in indices for n,p in model.model[i].named_parameters() if p.requires_grad]
    if not named:
        raise ValueError('Empty calibration parameter set')
    return indices, named


def run(args):
    cfg = yaml.safe_load(args.config.read_text())
    family = 'C1' if args.family == 'C1' else 'L1'
    cfg.update(arm=family, source='paired')
    validate_execution(cfg, formal=False)
    if family == 'L1' and not cfg.get('geometry_contract'):
        raise ValueError('No verified geometry for localization calibration')
    lease = legacy.require_bound_lease_from_environment()
    if len(lease['gpus']) != 1:
        raise ValueError('Calibration requires one bound GPU')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, args.output/'protocol_config.yaml')
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        trainer = build_trainer(cfg, args.config, args.output, max_steps=None)
        trainer._setup_train()
        trainer.epoch = 0
        # The formal loader remains unchanged in production; this diagnostic owns
        # a separate explicitly seeded natural loader and closes unused workers.
        for existing in (trainer.train_loader, trainer.test_loader):
            iterator = getattr(existing, 'iterator', None)
            if hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()
        reference = trainer.criterion_ref.reference
        teacher = trainer.criterion_ref.teacher
        state = {k:v.detach().clone() for k,v in reference.state_dict().items()}
        trainer.model.load_state_dict(state, strict=True)
        trainer.model.train()
        indices, named = parameter_set(trainer.model)
        params = [p for _,p in named]
        loader = build_natural_loader(cfg, seed=20260907)
        from diagnose_opportunities import dataset_config, split_images, source_groups as governed_source_groups
        source_data = dataset_config(cfg['paths']['student_data_yaml'])
        source_images = split_images(source_data, 'train')
        group_lookup = governed_source_groups(source_data, source_images, cfg['dataset'], 'train')
        group_lookup.update({str(Path(k).resolve()):v for k,v in list(group_lookup.items())})
        rows, ratios, unique_images, source_groups = [], [], set(), set()
        target_batches = 2 if args.profile_only else 64
        strides = tuple(int(x) for x in trainer.model.stride)
        for i, raw_batch in enumerate(itertools.islice(loader, target_batches)):
            trainer.model.load_state_dict(state, strict=True)
            trainer.model.train()
            trainer.model.zero_grad(set_to_none=True)
            teacher.eval(); reference.eval()
            batch = to_device(raw_batch, trainer.device)
            b = int(batch['img'].shape[0])
            if b != 32:
                raise ValueError('Calibration requires natural full B32 batches')
            prediction = trainer.model(batch['img'])
            student = legacy.raw_prediction(prediction)
            with torch.no_grad():
                t = legacy.raw_prediction(teacher(batch['strong_img']))
                r = legacy.raw_prediction(reference(batch['img']))
            native, _ = trainer.criterion_ref.native(prediction, batch)
            gnative = gradients(native.sum(), params)
            row = dict(batch=i, actual_B=b, files=list(batch['im_file']), pair_info=batch.get('pair_info'),
                native_norm=norm(gnative), state_restored=True, model_training=trainer.model.training)
            if family == 'C1':
                selection = build_classification_selection(student,t,r,batch,strides=strides,
                    config=trainer.criterion_ref.evidence_cfg,selection_seed=cfg['seed']+i+1)
                components = classification_loss_components(selection)
                unit, stats = classification_loss_from_selection(selection)
                gbase = gradients(b*.1*components['c0_loss'], params)
                gunit = gradients(b*unit, params)
                gy = gradients(b*components['target_loss'], params)
                goff = gradients(b*.25*components['off_target_loss_unit'], params)
                row.update(c0_weighted_norm=norm(gbase), target_norm=norm(gy), off_target_norm=norm(goff),
                    target_off_target_cosine=cosine(gy,goff), target_native_cosine=cosine(gy,gnative))
            else:
                unit, stats = trainer.criterion_ref.loc_adapter.compute(student,t,r,batch,strides,task='L1',return_records=True)
                gt, gt_stats = trainer.criterion_ref.loc_adapter.compute(student,t,r,batch,strides,task='L_GT',return_records=True)
                if stats.get('selected_anchors') != gt_stats.get('selected_anchors') or stats.get('base_count') != gt_stats.get('base_count'):
                    raise AssertionError('L1/L_GT selection identity differs')
                gbase, gunit = gnative, gradients(b*unit, params)
                ggt = gradients(b*gt, params)
                row.update(gt_unit_norm=norm(ggt), gt_native_cosine=cosine(ggt,gnative))
                for bi, _, _ in stats.get('selected_anchors', []):
                    unique_images.add(batch['im_file'][bi])
                    image = batch['im_file'][bi]
                    group = group_lookup.get(image, group_lookup.get(str(Path(image).resolve())))
                    if group is None or str(group).startswith('unavailable:'):
                        raise ValueError('Selected image missing governed source group')
                    source_groups.add(group)
            numerator, denominator = norm(gbase), norm(gunit)
            valid = all(math.isfinite(x) and x>0 for x in (numerator,denominator))
            ratio = numerator/denominator if valid else None
            if ratio is not None:
                ratios.append(ratio)
            row.update(unit_norm=denominator, reference_norm=numerator, ratio=ratio,
                kd_native_cosine=cosine(gunit,gnative), selected_count=stats.get('selected_count',0),
                base_count=stats.get('base_count'), stats=stats)
            legacy.append_json(args.output/'calibration_batches.jsonl', row)
            rows.append(row)
            del prediction, student, t, r, native, unit, batch, raw_batch, gnative, gbase, gunit
            if family == 'C1':
                del selection, components, gy, goff
            else:
                del gt, ggt
        if len(rows) != target_batches:
            raise RuntimeError('Incomplete fixed calibration stream')
        if any(p.grad is not None for m in (teacher,reference) for p in m.parameters()):
            raise AssertionError('Frozen teacher/reference gradients')
        raw = (1. if family=='C1' else .1)*statistics.median(ratios) if ratios else None
        coefficient = raw if family=='C1' or raw is None else min(1.,raw)
        eligible = (len(ratios)>=16 and coefficient is not None and 0<coefficient<=1)
        if family=='L1':
            eligible &= len(unique_images)>=16 and len(source_groups)>=2
        status = 'PROFILED' if args.profile_only else 'CALIBRATED' if eligible else 'CALIBRATION_REVIEW_REQUIRED'
        receipt = dict(status=status, family=family, dataset=cfg['dataset'], teacher=cfg['teacher'],
            reference=cfg['reference'], model=cfg['model'], state_source=cfg['reference'], seed=20260907,
            lambda_C1=coefficient if family=='C1' else None, lambda_L1=coefficient if family=='L1' else None,
            uncapped_coefficient=raw, capped_at_one=family=='L1' and raw is not None and raw>1,
            nonzero_batches=len(ratios), total_batches=len(rows), unique_images=len(unique_images),
            source_groups=sorted(source_groups), geometry_contract=cfg.get('geometry_contract'),
            parameter_module_indices=indices, parameter_names=[n for n,_ in named],
            train_mode=True, reset_parameters_and_buffers_each_batch=True, forward_precision='float32',
            loader=getattr(loader,'coverage_metadata',{}), optimizer_updates=0, test_accessed=False,
            achieved_ratios=[coefficient/x for x in ratios] if coefficient is not None else [],
            gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
            gpu_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
            resources=legacy.bound_lease_resource_record_from_environment(), seconds=time.time()-started)
        from evidence_bindings import capture_execution_binding
        receipt['execution_binding'] = str(capture_execution_binding(args.output,cfg,'calibration',HERE))
        legacy.write_json(args.output/'calibration_receipt.json',receipt)
        legacy.write_json(args.output/'completion_receipt.json',receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('parameter_names','achieved_ratios','loader')}),flush=True)
    except BaseException as error:
        legacy.write_json(args.output/'failure_receipt.json',dict(status='failed',error=repr(error),
            traceback=traceback.format_exc(),seconds=time.time()-started,test_accessed=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=('C1','L1'),required=True)
    p.add_argument('--profile-only',action='store_true',help='Two real batches for resource measurement, never calibration acceptance')
    run(p.parse_args())


if __name__=='__main__':
    main()
