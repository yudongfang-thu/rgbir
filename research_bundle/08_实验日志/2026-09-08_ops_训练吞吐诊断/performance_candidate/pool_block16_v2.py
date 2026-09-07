"""Independent rank-2 reduction candidate: fixed 16-object chunks.

No production module is edited or monkeypatched. Floating reduction grouping in
backward can differ; exact equality and numeric differences must be reported.
"""
import torch

BLOCK_SIZE = 16


def pool_relative_logits_block16(scores, foreground, background,
                                  minimum_foreground=1, minimum_background=4):
    if scores.ndim != 2 or scores.shape[0] < 1 or not scores.is_floating_point():
        raise ValueError("scores must be floating point [C,A], C>=1")
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Nonfinite pooling scores")
    if (foreground.dtype != torch.bool or background.dtype != torch.bool
            or foreground.ndim != 2 or foreground.shape != background.shape
            or foreground.shape[1] != scores.shape[1]):
        raise ValueError("region masks must be matching bool[M,A]")
    if foreground.device != scores.device or background.device != scores.device:
        raise ValueError("region masks must share score device")
    if minimum_foreground < 1 or minimum_background < 1:
        raise ValueError("Minimum region sizes must be positive")
    if bool((foreground & background).any()):
        raise ValueError("Foreground and background must not overlap")
    nfg, nbg = foreground.sum(-1), background.sum(-1)
    valid = (nfg >= minimum_foreground) & (nbg >= minimum_background)
    z = scores.float()
    if foreground.shape[0] == 0:
        return z[:, :0].T, valid
    values = []
    for start in range(0, foreground.shape[0], BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, foreground.shape[0])
        fg_value = (z[None].expand(end-start, -1, -1)
                    .masked_fill(~foreground[start:end, None], -1e30)
                    .reshape((end-start)*z.shape[0], z.shape[1]).logsumexp(1)
                    .reshape(end-start, z.shape[0])
                    - nfg[start:end].clamp_min(1).float().log()[:, None])
        bg_value = (z[None].expand(end-start, -1, -1)
                    .masked_fill(~background[start:end, None], -1e30)
                    .reshape((end-start)*z.shape[0], z.shape[1]).logsumexp(1)
                    .reshape(end-start, z.shape[0])
                    - nbg[start:end].clamp_min(1).float().log()[:, None])
        values.append(torch.where(valid[start:end, None], fg_value-bg_value,
                                  torch.zeros_like(fg_value)))
    return torch.cat(values, 0), valid
