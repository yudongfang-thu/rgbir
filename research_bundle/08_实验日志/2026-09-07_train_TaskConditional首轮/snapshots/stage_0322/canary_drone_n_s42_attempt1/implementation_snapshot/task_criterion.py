"""Compose unchanged OEv1 evidence with a separately selected localization target."""
from __future__ import annotations
import torch
from legacy_bridge import legacy, object_evidence_loss, EvidenceConfig
from localization_loss import localization_loss, LocalizationConfig
from geometry_contract import GeometryContract


def xyxy_labels(batch, size):
    boxes = batch['bboxes'].detach().float()
    scale = boxes.new_tensor([size[1], size[0]])
    return torch.cat(((boxes[:, :2] - boxes[:, 2:] / 2) * scale,
                      (boxes[:, :2] + boxes[:, 2:] / 2) * scale), 1)


def geometry_mask(contract, batch, prediction, strides):
    if contract is None:
        return None
    size = tuple(batch['img'].shape[-2:])
    xyxy = xyxy_labels(batch, size)
    paths = batch['im_file']
    chunks = []
    for feature, stride in zip(prediction['feats'], strides):
        mask = contract.object_mask(paths, xyxy, batch['batch_idx'], batch['pair_info'], stride)
        chunks.append(mask[:, None].expand(-1, feature.shape[-2] * feature.shape[-1]))
    return torch.cat(chunks, 1)


class TaskCriterion(legacy.EvidenceCriterion):
    def __init__(self, native, teacher, reference, cfg, arm, trainer, sanity):
        super().__init__(native, teacher, reference, cfg, arm, trainer, sanity)
        self.loc_cfg = LocalizationConfig(**cfg.get('localization', {}))
        self.geometry = GeometryContract.load(cfg['geometry_contract']) if cfg.get('geometry_contract') else None
        self.loc_selected_total = 0
        self.native_assignment = None
        self.last_components = None
        self.loc_gradient_checks = []
        if hasattr(native.assigner, 'register_forward_hook'):
            native.assigner.register_forward_hook(self._capture_native_assignment)

    def _capture_native_assignment(self, module, inputs, output):
        # Pinned assigner returns labels, boxes, scores, fg_mask, local_gt_index.
        self.native_assignment = (output[3].detach(), output[4].detach())

    def __call__(self, prediction, batch):
        native_total, items = self.native(prediction, batch)
        if 'teacher_batch' not in batch:
            return native_total, items
        self.calls += 1
        student = legacy.raw_prediction(prediction)
        self.teacher.eval(); self.reference.eval()
        with torch.no_grad():
            teacher = legacy.raw_prediction(self.teacher(batch['strong_img']))
            reference = legacy.raw_prediction(self.reference(batch['img']))
        strides = tuple(int(s) for s in self.trainer.model.stride)
        c_loss, c_stats = object_evidence_loss(student, teacher, reference, batch, strides=strides,
            config=self.evidence_cfg, arm='paired', seed=self.cfg['seed'] + self.calls)
        mode = 'gt' if self.arm == 'cgt' else ('random' if self.arm == 'cl_random' else 'teacher')
        l_loss, l_stats = localization_loss(student, teacher, reference, batch, strides=strides,
            config=self.loc_cfg, mode=mode, seed=self.cfg['seed'] + self.calls,
            geometry_eligible=geometry_mask(self.geometry, batch, student, strides),
            geometry_verified=self.geometry is not None and self.geometry.verified, return_records=self.sanity)
        wc = 0.0 if self.arm in ('n', 'l') else float(self.cfg['kd_weight'])
        wl = 0.0 if self.arm in ('n', 'c') else float(self.cfg['localization_coefficient'])
        b = int(batch['img'].shape[0])
        # Preserve the exact historical C/N multiplication order for endpoint reuse.
        c_total = legacy.combine_loss(native_total, c_loss, b, wc)
        total = c_total if wl == 0.0 else c_total + float(b) * wl * l_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Nonfinite task-conditional loss')
        self.selected_total += int(c_stats['selected_count'])
        self.loc_selected_total += int(l_stats['selected_count'])
        owner = {'same_object': 0, 'different_object': 0, 'native_background': 0}
        if self.native_assignment is not None:
            fg, local_gt = self.native_assignment
            ids = batch['batch_idx'].reshape(-1).long()
            for bi, ai, gi in l_stats.get('selected_anchors', []):
                if not bool(fg[bi, ai]):
                    owner['native_background'] += 1
                else:
                    globals_for_image = torch.nonzero(ids == bi, as_tuple=False).flatten()
                    native_gi = int(globals_for_image[int(local_gt[bi, ai])])
                    owner['same_object' if native_gi == gi else 'different_object'] += 1
        if owner['different_object']:
            raise AssertionError('Static unique owner disagrees with native GT owner')
        need_c_check = c_stats['selected_count'] and not any(
            row['kd_score_gradient_l2'] > 0 for row in self.gradient_checks)
        if self.sanity and (self.calls == 1 or need_c_check or
                            (l_stats['selected_count'] and not self.loc_gradient_checks)):
            tensors = (student['scores'], student['boxes'])
            gn = torch.autograd.grad(native_total.sum(), tensors, retain_graph=True, allow_unused=True)
            gc = torch.autograd.grad(c_loss, tensors, retain_graph=True, allow_unused=True)
            gl = torch.autograd.grad(l_loss, tensors, retain_graph=True, allow_unused=True)
            gt = torch.autograd.grad(total, tensors, retain_graph=True, allow_unused=True)
            zero = native_total.sum() + b * (0.0 * c_loss + 0.0 * l_loss)
            gz = torch.autograd.grad(zero, tensors, retain_graph=True, allow_unused=True)
            errors = []
            for t, n, c, l, z, target in zip(tensors, gn, gc, gl, gz, gt):
                n = torch.zeros_like(t) if n is None else n
                c = torch.zeros_like(t) if c is None else c
                l = torch.zeros_like(t) if l is None else l
                expected = n + b * (wc*c + wl*l)
                errors.append(float((target - expected).abs().max()))
                if not torch.allclose(target, expected, rtol=3e-3, atol=3e-5):
                    raise AssertionError('B/lambda gradient composition mismatch')
                if not torch.equal(z, n):
                    raise AssertionError('weight0 is not exactly native gradient')
            c_norm = float(gc[0].float().norm()) if gc[0] is not None else 0.0
            l_norm = float(gl[1].float().norm()) if gl[1] is not None else 0.0
            row = {'batch': self.calls, 'kd_score_gradient_l2': c_norm, 'kd_box_gradient_l2': l_norm,
                   'composition_max_error': max(errors), 'weight0_exact_loss_gradient': bool(torch.equal(zero, native_total.sum())),
                   'teacher_has_grad': any(p.grad is not None for p in self.teacher.parameters()),
                   'reference_has_grad': any(p.grad is not None for p in self.reference.parameters())}
            self.gradient_checks.append(row)
            if l_norm > 0:
                self.loc_gradient_checks.append(row)
            legacy.append_json(self.trainer.save_dir / 'gradient_checks.jsonl', row)
        self.last_components = (native_total.sum(), b*c_loss, b*l_loss)
        self.last_stats = {'batch': self.calls, 'epoch': int(self.trainer.epoch), 'arm': self.arm,
            'loss_unweighted': float(c_loss.detach()), 'localization_unweighted': float(l_loss.detach()),
            'native_total': float(native_total.detach().sum()), 'total_loss': float(total.detach()),
            'lambda_C': wc, 'lambda_L': wl, 'actual_batch': b, 'c': c_stats, 'l': l_stats,
            'native_owner': owner, 'optimizer_updates': self.trainer.real_updates}
        if self.sanity or self.calls <= 3 or self.calls % self.cfg['log_every_batches'] == 0:
            self.last_stats['student_files'] = list(batch['im_file'])
            legacy.append_json(self.trainer.save_dir / 'kd_batches.jsonl', self.last_stats)
        return total, items
