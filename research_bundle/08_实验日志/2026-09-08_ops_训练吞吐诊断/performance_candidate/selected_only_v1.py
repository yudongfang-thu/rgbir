"""Training-only C1 candidate; original selection oracle and selected S/T pool.

No release import/patch occurs on module import. Bind the exact pinned adapter
and classification modules explicitly through make_api(). Placeholder values
are private loss storage, never observations; complete diagnostics use this
same raw batch under no_grad and never another model forward.
"""
from dataclasses import dataclass, replace
import time
import types

import torch


def _clone(fn, **bindings):
    copied = types.FunctionType(fn.__code__, dict(fn.__globals__, **bindings),
                                fn.__name__, fn.__defaults__, fn.__closure__)
    copied.__kwdefaults__ = fn.__kwdefaults__
    return copied


def _deferred_pool(scores, foreground, background, minimum_foreground=1, minimum_background=4):
    """Original validation/counts, but no full-class LME measurement yet.

    Called only after every raw S/T/R score is certified finite FP16. The
    adapter passes its float32 view here. This finite domain cannot overflow
    float32 LME/differences. Original C0 evidence still runs in the first pass.
    """
    if scores.ndim != 2 or scores.shape[0] < 1 or not scores.is_floating_point():
        raise ValueError('scores must be floating point [C,A], C>=1')
    if (foreground.dtype != torch.bool or background.dtype != torch.bool
            or foreground.ndim != 2 or foreground.shape != background.shape
            or foreground.shape[1] != scores.shape[1]):
        raise ValueError('region masks must be matching bool[M,A]')
    if foreground.device != scores.device or background.device != scores.device:
        raise ValueError('region masks must share score device')
    if minimum_foreground < 1 or minimum_background < 1:
        raise ValueError('Minimum region sizes must be positive')
    if bool((foreground & background).any()):
        raise ValueError('Foreground and background must not overlap')
    nfg, nbg = foreground.sum(-1), background.sum(-1)
    valid = (nfg >= minimum_foreground) & (nbg >= minimum_background)
    z = scores.float()
    # A zero-sized score slice keeps the original score tensor in the graph,
    # including empty/no-selected batches, without reading nonselected values.
    zero = z[:, :0].sum() * 0.
    return z.new_zeros((foreground.shape[0], z.shape[0])) + zero, valid


def _timed_diagnostic(fn, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        value = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


@dataclass
class TrainingSelection:
    learning: object
    diagnostics: object
    thin_path_used: bool
    fallback_reason: object
    full_diagnostics_requested: bool
    diagnostics_seconds: float = 0.

    def __getattr__(self, name):
        # Original classification_loss_components and source assertions can
        # operate on the actual learning payload without a different formula.
        return getattr(self.learning, name)


class SelectedOnlyAPI:
    def __init__(self, adapter, classification):
        self.adapter, self.classification = adapter, classification
        self.original_selector = adapter.build_classification_selection
        self.original_pool = adapter.pool_relative_logits
        self.original_loss = classification.classification_loss_from_selection
        self.first_pass = _clone(self.original_selector, pool_relative_logits=_deferred_pool)

    def build(self, student, teacher, reference, batch, *, strides=(8, 16, 32),
              config=None, selection_seed=0, full_diagnostics=False):
        kwargs = dict(strides=strides, config=config, selection_seed=selection_seed)
        raw_scores = [raw['scores'] for raw in (student, teacher, reference)]
        fallback = None
        if any(value.dtype != torch.float16 for value in raw_scores):
            fallback = 'RAW_SCORE_DTYPE_NOT_FP16'
        elif not all(bool(torch.isfinite(value).all()) for value in raw_scores):
            fallback = 'NONFINITE_RAW_SCORE_USE_ORIGINAL_VALIDATION'
        if fallback:
            learning = self.original_selector(student, teacher, reference, batch, **kwargs)
            return TrainingSelection(learning, learning, False, fallback, full_diagnostics)

        # All matching/evidence/correctness/base/q/rho/stable selection, source
        # identities, complete GT exclusion masks and C0 statistics remain the
        # original function bytecode. C1 full-class content never feeds it.
        # Preserve the original single whole-score FP32 cast: every selected
        # pool branch accumulates into one FP32 score gradient before converting
        # it back to raw FP16. Do not cast individual raw slices independently.
        float_student = dict(student, scores=student['scores'].float())
        with torch.no_grad():
            float_teacher = dict(teacher, scores=teacher['scores'].detach().float())
            float_reference = dict(reference, scores=reference['scores'].detach().float())
        learning = self.first_pass(float_student, float_teacher, float_reference, batch, **kwargs)
        sd, td = learning.student_delta, learning.teacher_delta
        level_positions = {level: li for li, level in enumerate(learning.config.levels)}
        for region in learning.regions:
            keep = learning.selected[region.base_rows]
            rows = region.base_rows[keep]
            if not len(rows):
                continue
            bi, start, end = region.image_index, region.anchor_start, region.anchor_end
            li = level_positions[region.level]
            fg, bg = region.rgb_foreground[keep], region.rgb_background[keep]
            tfg, tbg = region.teacher_foreground[keep], region.teacher_background[keep]
            sval, sv = self.original_pool(float_student['scores'][bi, :, start:end], fg, bg,
                learning.config.minimum_foreground, learning.config.minimum_background)
            with torch.no_grad():
                tval, tv = self.original_pool(float_teacher['scores'][bi, :, start:end], tfg, tbg,
                    learning.config.minimum_foreground, learning.config.minimum_background)
                if not torch.equal(sv & tv, learning.valid_levels[rows, li]):
                    raise RuntimeError('Selected pool validity differs from the original common mask')
            # Keep every base row and its original reduction position. Never
            # pass K selected rows to a kernel whose denominator is M_base.
            sd[rows, li] = sval
            td[rows, li] = tval.detach()
        sources = self.adapter._source_tensors(student, teacher, reference, batch)
        learning = replace(learning, student_delta=sd, teacher_delta=td.detach(),
            source_tensors=sources, source_versions=tuple(self.adapter._tensor_version(t) for t in sources))
        payload = TrainingSelection(learning, None, True, None, full_diagnostics)
        if full_diagnostics:
            payload.diagnostics, payload.diagnostics_seconds = _timed_diagnostic(
                lambda: self.original_selector(student, teacher, reference, batch, **kwargs),
                student['scores'].device)
        return payload

    def loss(self, payload, *, temperature=2., raw_teacher_clip=16., off_target_weight=.25,
             return_records=False):
        if not isinstance(payload, TrainingSelection):
            raise TypeError('Use this candidate with its own TrainingSelection API')
        kwargs = dict(temperature=temperature, raw_teacher_clip=raw_teacher_clip,
                      off_target_weight=off_target_weight)
        if not payload.thin_path_used:
            loss, stats = self.original_loss(payload.learning, return_records=return_records, **kwargs)
            stats.update(full_diagnostics_collected=True, thin_path_used=False,
                         thin_path_fallback_reason=payload.fallback_reason,
                         full_diagnostics_seconds=0.)
            return loss, stats
        if return_records and payload.diagnostics is None:
            raise ValueError('Complete records require full_diagnostics=True at selection construction')
        selection = payload.learning
        terms = self.classification._classification_terms(selection.student_delta, selection.teacher_delta,
            selection.valid_levels, selection.selected, selection.labels, **kwargs)
        loss = terms['loss']
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite relative class KD')
        if payload.diagnostics is not None:
            (_, stats), seconds = _timed_diagnostic(lambda: self.original_loss(payload.diagnostics,
                return_records=return_records, **kwargs), selection.student_delta.device)
            payload.diagnostics_seconds += seconds
            # Stats come from true full measurements. The differentiable loss
            # always comes from the same selected-only graph, on log batches too.
            stats['loss_unweighted'] = float(loss.detach())
            stats['target_loss_unweighted'] = float(terms['target_loss'].detach())
            stats['off_target_loss_unit'] = float(terms['off_target_loss_unit'].detach())
            stats['off_target_loss_unweighted'] = float((off_target_weight*terms['off_target_loss_unit']).detach())
        else:
            m, levels, nc = selection.student_delta.shape
            stats = dict(selection.c0_stats)
            # Inherited C0 entropy/clipping have another meaning. Never present
            # those or private zero slots as measured full-class diagnostics.
            missing = ['target_binary_entropy_selected', 'target_gap_to_positive_selected',
                'target_clipped_count', 'target_clipping_population', 'target_clipped_fraction',
                'gt_channel_relative_entropy_selected', 'per_level', 'per_class',
                'rgb_other_gt_foreground_fraction_selected_level_mean',
                'teacher_other_gt_foreground_fraction_selected_level_mean', 'base_records']
            missing += [name+'_delta_class_mean_'+scope for name in ('student','teacher','reference')
                        for scope in ('base_valid','selected_valid')]
            stats.update({name: None for name in missing})
            stats.update(arm='C1_y' if off_target_weight == 0 else 'C1',
                selection_version=self.adapter.SELECTION_VERSION,
                selection_source=str(self.adapter.LEGACY_SOURCE),
                temperature=temperature, raw_teacher_clip=raw_teacher_clip,
                off_target_weight=off_target_weight, class_count=nc,
                supervised_class_channels=nc if off_target_weight > 0 else 1,
                payload_shape=[m, levels, nc], normalizer=max(1,m),
                loss_unweighted=float(loss.detach()), c0_loss_unweighted=float(selection.c0_loss.detach()),
                target_loss_unweighted=float(terms['target_loss'].detach()),
                off_target_loss_unit=float(terms['off_target_loss_unit'].detach()),
                off_target_loss_unweighted=float((off_target_weight*terms['off_target_loss_unit']).detach()),
                selected_valid_level_count=int((selection.valid_levels & selection.selected[:,None]).sum()),
                relative_evidence_not_detector_confidence=True, student_background_has_gradient=True,
                teacher_gradient_enabled=False, target_binary_entropy_semantics='relative_evidence',
                missing_diagnostics=missing)
        stats.update(thin_path_used=True, thin_path_fallback_reason=None,
            full_diagnostics_collected=payload.diagnostics is not None,
            full_diagnostics_seconds=payload.diagnostics_seconds,
            learning_delta_coverage='selected_S_and_T_only; other_slots_private_placeholders',
            raw_score_finite_domain='all_three_raw_scores_finite_float16')
        return loss, stats


def make_api(adapter, classification):
    return SelectedOnlyAPI(adapter, classification)


def full_diagnostics_due(calls, log_every_batches, sanity, observer_due):
    """Original log cadence plus shared-gradient observation; no RNG access."""
    if calls < 1 or log_every_batches < 1:
        raise ValueError('Positive calls and logging interval required')
    return bool(sanity or calls <= 3 or calls % log_every_batches == 0 or observer_due)


def make_criterion_type(criterion_module, api, selection_observer=None):
    """Optional fresh-process trainer binding, never a production monkeypatch.

    Caller must bind this returned class into a private build_trainer globals
    copy. The callback is read-only audit plumbing and is not required for the
    loss. Existing native/aux forward, actual-B/lambda, sanity and shared-gradient
    code execute the original __call__ bytecode. No optimizer update is added.
    """
    original_type = criterion_module.IndependentCriterion
    original_call = original_type.__call__

    def call(self, prediction, batch):
        if self.method_arm not in ('C1', 'C1_y'):
            raise ValueError('Selected-only candidate supports C1/C1_y only')

        def select(*args, **kwargs):
            due = full_diagnostics_due(self.calls, self.cfg['log_every_batches'], self.sanity,
                self.shared_gradient_observer.should_observe(self.trainer.epoch))
            payload = api.build(*args, full_diagnostics=due, **kwargs)
            if selection_observer is not None:
                selection_observer(self, payload)
            return payload

        def loss(payload, **kwargs):
            value, stats = api.loss(payload, **kwargs)
            self.selected_only_diagnostics_seconds_total = (
                getattr(self, 'selected_only_diagnostics_seconds_total', 0.) + payload.diagnostics_seconds)
            for name, increment in (
                    ('thin_learning_batches', int(payload.thin_path_used)),
                    ('full_diagnostics_batches', int(stats['full_diagnostics_collected'])),
                    ('fallback_batches', int(not payload.thin_path_used))):
                setattr(self, name, getattr(self, name, 0) + increment)
                stats[name] = getattr(self, name)
            return value, stats

        bound = _clone(original_call, build_classification_selection=select,
                       classification_loss_from_selection=loss)
        return bound(self, prediction, batch)

    return type('SelectedOnlyDiagnosticC1Criterion', (original_type,), {'__call__': call})
