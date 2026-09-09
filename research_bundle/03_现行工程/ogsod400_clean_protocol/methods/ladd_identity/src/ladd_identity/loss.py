from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss


class IdentityLADDLoss(v8DetectionLoss):
    def __init__(self, model, teacher_model=None, lambda_z=1.0, lambda_orth=0.01):
        super().__init__(model)
        self.teacher_model = teacher_model
        self.model = model
        self.lambda_z = float(lambda_z)
        self.lambda_orth = float(lambda_orth)

    @staticmethod
    def _dict(prediction):
        if isinstance(prediction, dict):
            return prediction
        if isinstance(prediction, tuple):
            for item in reversed(prediction):
                if isinstance(item, dict):
                    return item
        raise TypeError(type(prediction))

    def loss(self, preds, batch):
        detector_total, detector_items = super().loss(preds, batch)
        zero = detector_total.new_zeros(())
        if self.teacher_model is None or "teacher_img" not in batch or "identity_z_feats" not in preds:
            extras = torch.stack((zero, zero))
            return torch.cat((detector_total, extras)), torch.cat((detector_items, extras.detach()))
        with torch.no_grad():
            teacher = self._dict(self.teacher_model(batch["teacher_img"]))
        z_losses = []
        orth_losses = []
        for block, teacher_feat, student_z in zip(
            self.model.identity_blocks, teacher["feats"], preds["identity_z_feats"]
        ):
            teacher_z = block.projector.encode(teacher_feat.detach())
            z_losses.append(F.smooth_l1_loss(student_z, teacher_z.detach()))
            orth_losses.append(block.projector.orthogonality_loss())
        z_loss = torch.stack(z_losses).mean() * self.lambda_z
        orth_loss = torch.stack(orth_losses).mean() * self.lambda_orth
        bsz = int(batch["img"].shape[0])
        extras = torch.stack((z_loss, orth_loss))
        return torch.cat((detector_total, extras * bsz)), torch.cat((detector_items, extras.detach()))
