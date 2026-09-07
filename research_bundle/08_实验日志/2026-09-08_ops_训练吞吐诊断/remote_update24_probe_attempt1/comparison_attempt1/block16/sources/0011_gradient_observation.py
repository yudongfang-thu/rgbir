"""Fixed-epoch shared-parameter gradients; observation never changes KD dose.

The first training batch at epochs 0/10/50/100/199 is observed even when KD is
zero. autograd.grad reads unscaled, unclipped gradients without accumulating
into Parameter.grad. Returned records contain JSON values only, never tensors.
"""
from __future__ import annotations

import math
import torch

OBSERVATION_EPOCHS = (0, 10, 50, 100, 199)
NEW_ARMS = ('C1', 'C1_y', 'L1', 'L_GT')


def shared_parameter_set(model):
    """Same explicit P3/P4 pre-head parameter set as the accepted calibrator."""
    head = model.model[-1]
    indices = list(head.f)[:2]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError('Expected separate P3/P4 feature sources')
    named = [(f'model.{i}.{name}', parameter) for i in indices
             for name, parameter in model.model[i].named_parameters() if parameter.requires_grad]
    if not named or len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError('Empty or aliased shared feature parameter set')
    return indices, named


def _read_gradients(loss, parameters):
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError('Gradient observation requires a scalar Tensor')
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return tuple(None if value is None else value.detach()
                 for value in torch.autograd.grad(loss, parameters, retain_graph=True,
                     create_graph=False, allow_unused=True))


def _summary(values):
    present = [value for value in values if value is not None]
    nonfinite = sum(not bool(torch.isfinite(value).all()) for value in present)
    norm = None
    if not nonfinite:
        # FP64 reductions avoid overflow of finite FP32 squares. The actual
        # gradients retain their real training precision; this is a readout.
        parts = [value.double().square().sum() for value in present]
        norm = float(torch.stack(parts).sum().sqrt()) if parts else 0.0
        if not math.isfinite(norm):
            norm = None
    return dict(norm=norm, present_parameter_tensors=len(present),
                absent_parameter_tensors=len(values)-len(present),
                nonfinite_parameter_tensors=nonfinite,
                gradient_dtypes=sorted({str(value.dtype) for value in present}))


def _cosine(left, right, left_norm, right_norm):
    if left_norm is None or right_norm is None or left_norm == 0 or right_norm == 0:
        return None
    terms = [(a.double()*b.double()).sum() for a,b in zip(left,right)
             if a is not None and b is not None]
    dot = float(torch.stack(terms).sum()) if terms else 0.0
    result = dot / left_norm / right_norm
    return min(1.0, max(-1.0, result)) if math.isfinite(result) else None


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator/denominator
    return value if math.isfinite(value) else None


class FixedEpochGradientObserver:
    """State contains only observed epoch integers, never computation graphs."""
    def __init__(self):
        self.observed_epochs = set()

    def should_observe(self, epoch):
        return int(epoch) in OBSERVATION_EPOCHS and int(epoch) not in self.observed_epochs

    def observe(self, model, native_total, kd_unit, *, arm, epoch, batch_index,
                actual_batch, coefficient, target_unit=None, off_target_unit=None,
                off_target_weight=None):
        if arm not in NEW_ARMS:
            raise ValueError('Historical N/C0 are outside shared-gradient observation')
        if not self.should_observe(epoch):
            return None
        if type(actual_batch) is not int or actual_batch < 1 or not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError('Invalid actual batch or frozen coefficient')
        if arm.startswith('C') and (target_unit is None or off_target_unit is None):
            raise ValueError('Classification observation needs both actual content components')
        if arm == 'C1' and off_target_weight != .25 or arm == 'C1_y' and off_target_weight != 0.0:
            raise ValueError('Classification observation must use its actual frozen eta')
        indices, named = shared_parameter_set(model)
        parameters = [parameter for _,parameter in named]
        scale = actual_batch*coefficient
        native = _read_gradients(native_total.sum(), parameters)
        kd = _read_gradients(scale*kd_unit, parameters)
        native_summary, kd_summary = _summary(native), _summary(kd)
        row = dict(schema='rgbir-fixed-shared-gradient-v1', arm=arm,
            epoch=int(epoch), batch_index=int(batch_index), epoch_batch_index=0,
            schedule=list(OBSERVATION_EPOCHS), schedule_policy='first_training_batch_in_each_fixed_epoch',
            parameter_module_indices=indices, parameter_names=[name for name,_ in named],
            actual_B=actual_batch, coefficient=float(coefficient),
            native=native_summary, weighted_kd=kd_summary,
            weighted_kd_native_ratio=_ratio(kd_summary['norm'],native_summary['norm']),
            weighted_kd_native_cosine=_cosine(kd,native,kd_summary['norm'],native_summary['norm']),
            gradient_stage='unscaled_unclipped_per_batch_autograd_grad',
            uses_actual_forward_precision=True, optimizer_or_scaler_step_called=False,
            parameter_grad_accumulation_modified=False, loss_or_coefficient_modified=False)
        summaries = [native_summary, kd_summary]
        if arm.startswith('C'):
            target = _read_gradients(scale*target_unit, parameters)
            off_target = _read_gradients(scale*off_target_weight*off_target_unit, parameters)
            target_summary, off_summary = _summary(target), _summary(off_target)
            row.update(off_target_weight=float(off_target_weight),
                weighted_target=target_summary, weighted_off_target=off_summary,
                target_off_target_cosine=_cosine(target,off_target,target_summary['norm'],off_summary['norm']),
                target_native_cosine=_cosine(target,native,target_summary['norm'],native_summary['norm']),
                weighted_target_native_ratio=_ratio(target_summary['norm'],native_summary['norm']),
                weighted_target_full_kd_ratio=_ratio(target_summary['norm'],kd_summary['norm']))
            summaries += [target_summary,off_summary]
        row['status'] = ('NONFINITE_OBSERVATION' if any(s['norm'] is None for s in summaries)
                         else 'ZERO_KD_SIGNAL' if kd_summary['norm'] == 0 else 'OBSERVED')
        # This is deliberately independent of valid-object count and magnitude.
        self.observed_epochs.add(int(epoch))
        return row
