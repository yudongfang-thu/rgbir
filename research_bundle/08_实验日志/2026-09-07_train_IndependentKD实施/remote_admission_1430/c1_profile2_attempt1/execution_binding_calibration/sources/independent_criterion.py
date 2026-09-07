"""One KD branch per run; historical N/C0 call their original implementation."""
import time
import torch
from runtime import legacy, ORIGINAL_CRITERION
from selection_adapter import build_classification_selection
from classification_logit import classification_loss_from_selection, classification_loss_components
from gradient_observation import FixedEpochGradientObserver


class IndependentCriterion(ORIGINAL_CRITERION):
    def __init__(self, native, teacher, reference, cfg, arm, trainer, sanity):
        self.method_arm = cfg['arm']
        old_arm = 'weight0' if self.method_arm == 'N' else 'paired'
        super().__init__(native, teacher, reference, cfg, old_arm, trainer, sanity)
        self.source = cfg.get('source', 'paired')
        if self.source != 'paired':
            raise ValueError('Content controls require their separately accepted implementation')
        self.loc_adapter = None
        self.native_assignment = None
        if self.method_arm in ('C1', 'C1_y', 'L1', 'L_GT'):
            self.shared_gradient_observer = FixedEpochGradientObserver()
        if self.method_arm in ('L1', 'L_GT'):
            from localization_adapter import LocalizationAdapter
            self.loc_adapter = LocalizationAdapter(cfg['geometry_contract'], cfg.get('localization'))
            native.assigner.register_forward_hook(self._capture_assignment)

    def _capture_assignment(self, module, inputs, output):
        self.native_assignment = output[3].detach(), output[4].detach()

    def __call__(self, prediction, batch):
        if self.method_arm in ('N', 'C0'):
            return super().__call__(prediction, batch)
        native, items = self.native(prediction, batch)
        if 'teacher_batch' not in batch:
            raise ValueError('Independent training requires the paired-label batch contract')
        self.calls += 1
        student = legacy.raw_prediction(prediction)
        self.teacher.eval(); self.reference.eval()
        with torch.no_grad():
            teacher = legacy.raw_prediction(self.teacher(batch['strong_img']))
            reference = legacy.raw_prediction(self.reference(batch['img']))
        strides = tuple(int(x) for x in self.trainer.model.stride)
        if self.method_arm in ('C1', 'C1_y'):
            selection = build_classification_selection(student, teacher, reference, batch,
                strides=strides, config=self.evidence_cfg, selection_seed=self.cfg['seed'] + self.calls)
            kd, stats = classification_loss_from_selection(selection,
                off_target_weight=0.0 if self.method_arm == 'C1_y' else 0.25)
            weight = float(self.cfg['classification_coefficient'])
        else:
            kd, stats = self.loc_adapter.compute(student, teacher, reference, batch,
                strides, task=self.method_arm, return_records=self.sanity)
            weight = float(self.cfg['localization_coefficient'])
        b = int(batch['img'].shape[0])
        total = legacy.combine_loss(native, kd, b, weight)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Nonfinite independent KD loss')
        if self.shared_gradient_observer.should_observe(self.trainer.epoch):
            components = (classification_loss_components(selection,
                off_target_weight=0.0 if self.method_arm == 'C1_y' else 0.25)
                if self.method_arm in ('C1', 'C1_y') else None)
            observation = self.shared_gradient_observer.observe(self.trainer.model, native, kd,
                arm=self.method_arm, epoch=self.trainer.epoch, batch_index=self.calls,
                actual_batch=b, coefficient=weight,
                target_unit=components['target_loss'] if components is not None else None,
                off_target_unit=components['off_target_loss_unit'] if components is not None else None,
                off_target_weight=(0.0 if self.method_arm == 'C1_y' else 0.25) if components is not None else None)
            observation.update(optimizer_updates_before_batch=self.trainer.real_updates,
                optimizer_attempts_before_batch=self.trainer.update_attempts,
                amp_skipped_updates_before_batch=self.trainer.skipped_amp_updates,
                ema_updates_before_batch=self.trainer.ema.updates,
                amp_enabled=bool(self.trainer.amp), grad_scaler_value=float(self.trainer.scaler.get_scale()),
                selected_count=int(stats.get('selected_count', 0)),
                base_count=int(stats.get('base_count', 0)), student_files=list(batch['im_file']))
            legacy.append_json(self.trainer.save_dir/'shared_gradient_observations.jsonl', observation)
            del components, observation
        selected = int(stats.get('selected_count', 0))
        self.selected_total += selected
        check_needed = self.sanity and (self.calls == 1 or
            (selected and not any(x.get('kd_gradient_l2', 0) > 0 for x in self.gradient_checks)))
        if check_needed:
            tensors = (student['scores'], student['boxes'])
            gn = torch.autograd.grad(native.sum(), tensors, retain_graph=True, allow_unused=True)
            gk = torch.autograd.grad(kd, tensors, retain_graph=True, allow_unused=True)
            gt = torch.autograd.grad(total, tensors, retain_graph=True, allow_unused=True)
            errors = []
            for tensor, n, k, actual in zip(tensors, gn, gk, gt):
                n = torch.zeros_like(tensor) if n is None else n
                k = torch.zeros_like(tensor) if k is None else k
                expected = n + b * weight * k
                if actual is None or not torch.allclose(actual, expected, rtol=3e-3, atol=3e-5):
                    raise AssertionError('Actual batch/weight gradient composition differs')
                errors.append(float((actual - expected).abs().max()))
            norms = [float(x.float().norm()) if x is not None else 0. for x in gk]
            forbidden = norms[1] if self.method_arm.startswith('C') else norms[0]
            if forbidden != 0:
                raise AssertionError('KD directly reached the other task output')
            row = dict(batch=self.calls, kd_score_gradient_l2=norms[0],
                kd_box_gradient_l2=norms[1], kd_gradient_l2=sum(norms),
                composition_max_error=max(errors), actual_B=b,
                teacher_has_grad=any(p.grad is not None for p in self.teacher.parameters()),
                reference_has_grad=any(p.grad is not None for p in self.reference.parameters()))
            if row['teacher_has_grad'] or row['reference_has_grad']:
                raise AssertionError('Frozen auxiliary accumulated gradients')
            self.gradient_checks.append(row)
            legacy.append_json(self.trainer.save_dir/'gradient_checks.jsonl', row)
        owner = dict(same_object=0, different_object=0, native_background=0)
        if self.native_assignment is not None:
            fg, local_gt = self.native_assignment
            ids = batch['batch_idx'].reshape(-1).long()
            for bi, ai, gi in stats.get('selected_anchors', []):
                if not bool(fg[bi, ai]):
                    owner['native_background'] += 1
                else:
                    global_ids = torch.nonzero(ids == bi, as_tuple=False).flatten()
                    actual_gt = int(global_ids[int(local_gt[bi, ai])])
                    owner['same_object' if actual_gt == gi else 'different_object'] += 1
        self.last_stats = dict(stats, arm=self.method_arm, source=self.source, batch=self.calls,
            epoch=int(self.trainer.epoch), actual_B=b, coefficient=weight,
            loss_unweighted=float(kd.detach()), weighted_kd_total=float(kd.detach())*b*weight,
            native_total=float(native.detach().sum()), total_loss=float(total.detach()),
            native_owner=owner, optimizer_updates=self.trainer.real_updates,
            seconds=time.time()-self.trainer.wall_started)
        if self.sanity or self.calls <= 3 or self.calls % self.cfg['log_every_batches'] == 0:
            self.last_stats['student_files'] = list(batch['im_file'])
            legacy.append_json(self.trainer.save_dir/'kd_batches.jsonl', self.last_stats)
        return total, items
