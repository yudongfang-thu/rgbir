"""Independent expression of the author BCDL per-anchor classification term."""
import torch
import torch.nn.functional as F


def bcdl_per_anchor(student_logits, teacher_logits, beta=1.0):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('student and teacher logits must have identical shapes')
    q = torch.sigmoid(teacher_logits.detach())
    p = torch.sigmoid(student_logits)
    # Stable Bernoulli cross-entropy, expressed without BCEWithLogits.
    cross_entropy = q * F.softplus(-student_logits) + (1.0 - q) * F.softplus(student_logits)
    return ((q - p).abs().pow(beta) * cross_entropy).sum(dim=1)
