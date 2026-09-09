def novel_kd_loss(pred,
                  soft_label,
                  detach_target=True,
                  beta=1.0):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = soft_label.sigmoid()
    score = pred.sigmoid()

    if detach_target:
        target = target.detach()

    scale_factor = target - score
    kd_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * scale_factor.abs().pow(beta)
    kd_loss = kd_loss.sum(dim=1, keepdim=False)
    return kd_loss
