"""One object-coordinate DFL scalar; private training criterion only."""
import torch
from runtime import legacy, ORIGINAL_CRITERION
from gradient_observation import shared_parameter_set


def make_type(api):
    # Keep the accepted builder signature; C1 API is not evaluated in this scope.
    class ObjectDFLCriterion(ORIGINAL_CRITERION):
        def losses(self, student, teacher, reference, batch, arms):
            from l3_distribution_loss import compute
            result = {}
            strides = tuple(int(x) for x in self.trainer.model.stride)
            for arm in arms:
                if arm not in ('L3-DFL', 'L3-GT'):
                    raise ValueError('Only the frozen L3 loss variants are supported')
                result[arm] = compute(student, teacher, reference, batch, strides, variant=arm)
            if len(result) == 2:
                a, b = result['L3-DFL'][1], result['L3-GT'][1]
                for key in ('base_count', 'normalizer', 'selected_count', 'selected_anchors', 'selected_object_ids'):
                    if key not in a or key not in b or a[key] != b[key]:
                        raise AssertionError('DFL/GT selection contract differs: ' + key)
            return result

        def __call__(self, prediction, batch):
            native, items = self.native(prediction, batch)
            self.calls += 1
            student = legacy.raw_prediction(prediction)
            self.teacher.eval(); self.reference.eval()
            with torch.no_grad():
                teacher = legacy.raw_prediction(self.teacher(batch['strong_img']))
                reference = legacy.raw_prediction(self.reference(batch['img']))
            arm = self.cfg['arm']
            request = 'L3-DFL' if arm == 'N' else arm
            kd, stats = self.losses(student, teacher, reference, batch, [request])[request]
            weight = float(self.cfg['kd_coefficient'])
            b = int(batch['img'].shape[0])
            total = native.sum() + b * weight * kd
            if not bool(torch.isfinite(total)):
                raise FloatingPointError('Nonfinite object DFL loss')
            if arm == 'N' and not torch.equal(total, native.sum()):
                raise AssertionError('N changed native loss')
            selected = int(stats['selected_count']); self.selected_total += selected
            if self.calls == 1 or (self.cfg.get('canary_execution', False) and selected
                                  and not any(x['kd_gradient_l2'] > 0 for x in self.gradient_checks)):
                _, named = shared_parameter_set(self.trainer.model)
                names = [n for n, p in named]; params = [p for n, p in named]
                gradients = torch.autograd.grad(kd, params, retain_graph=True, allow_unused=True)
                if any(not bool(torch.isfinite(v).all()) for v in gradients if v is not None):
                    raise FloatingPointError('Nonfinite KD shared-parameter gradient')
                norm = float(sum((v.detach().double().square().sum() for v in gradients if v is not None),
                                 torch.zeros((), device=kd.device)).sqrt())
                if not torch.isfinite(torch.tensor(norm)):
                    raise FloatingPointError('Nonfinite KD gradient norm')
                row = dict(batch=self.calls, kd_gradient_l2=norm, parameter_names=names,
                    gradient_scope='P3_P4_pre_head_source_module_parameters', actual_B=b,
                    coefficient=weight, teacher_has_grad=any(p.grad is not None for p in self.teacher.parameters()),
                    reference_has_grad=any(p.grad is not None for p in self.reference.parameters()))
                if row['teacher_has_grad'] or row['reference_has_grad']:
                    raise AssertionError('Auxiliary gradient leak')
                if norm > 0 or self.calls == 1:
                    self.gradient_checks.append(row)
                legacy.append_json(self.trainer.save_dir/'gradient_checks.jsonl', row)
            self.last_stats = dict(stats, arm=arm, batch=self.calls, epoch=int(self.trainer.epoch),
                method_identity=self.cfg['method_identity'], actual_B=b, coefficient=weight,
                loss_unweighted=float(kd.detach()), weighted_kd_total=float(kd.detach())*b*weight,
                native_total=float(native.detach().sum()), total_loss=float(total.detach()))
            if self.calls <= 3 or self.calls % 16 == 0:
                legacy.append_json(self.trainer.save_dir/'kd_batches.jsonl', self.last_stats)
            return total, items
    return ObjectDFLCriterion
