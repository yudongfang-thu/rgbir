from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityProjector(nn.Module):
    def __init__(self, channels: int, common_channels: int):
        super().__init__()
        self.channels = channels
        self.common_channels = common_channels
        q, _ = torch.linalg.qr(torch.randn(channels, common_channels), mode="reduced")
        self.weight = nn.Parameter(q.T.contiguous().view(common_channels, channels, 1, 1))

    def encode(self, x):
        return F.conv2d(x, self.weight)

    def decode(self, z):
        return F.conv_transpose2d(z, self.weight)

    def orthogonality_loss(self):
        w = self.weight.flatten(1)
        eye = torch.eye(self.common_channels, device=w.device, dtype=w.dtype)
        return (w @ w.T - eye).square().mean()


class IdentityResidualBlock(nn.Module):
    """Single learned z projection with exact private identity residual."""

    def __init__(self, channels: int, common_ratio: float = 0.25):
        super().__init__()
        common_channels = max(1, min(channels, round(channels * common_ratio)))
        self.projector = IdentityProjector(channels, common_channels)

    def forward(self, x):
        z = self.projector.encode(x)
        shared = self.projector.decode(z)
        private = x - shared
        corrected = private + shared
        return {
            "base": x,
            "z": z,
            "shared": shared,
            "private": private,
            "corrected": corrected,
        }

