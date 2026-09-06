"""BYOL, LARS, and the fixed YOLO-OS-SSL optimizer schedule."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class MLP(nn.Module):
    """The frozen 256/4096/256 projector or predictor architecture."""

    def __init__(self, input_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _last_feature(value: Any) -> torch.Tensor:
    if isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError("Backbone returned an empty feature sequence.")
        value = value[-1]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Backbone returned {type(value).__name__}, not a tensor feature map.")
    if value.ndim == 4:
        value = value.mean(dim=(-2, -1))
    if value.ndim != 2 or value.shape[1] != 256:
        raise RuntimeError(f"YOLO-OS-SSL requires GAP features [B,256], got {tuple(value.shape)}.")
    return value


class BYOLModel(nn.Module):
    """Online encoder/projector/predictor plus a no-gradient EMA target."""

    def __init__(self, online_encoder: nn.Module) -> None:
        super().__init__()
        self.online_encoder = online_encoder
        self.online_projector = MLP(256)
        self.online_predictor = MLP(256)
        self.target_encoder = copy.deepcopy(online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self._freeze_target()

    def _freeze_target(self) -> None:
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_projector.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()
        self.target_projector.eval()

    def train(self, mode: bool = True) -> "BYOLModel":
        super().train(mode)
        # Target BN statistics must be its EMA state, never a target forward update.
        self.target_encoder.eval()
        self.target_projector.eval()
        return self

    def _online(self, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = _last_feature(self.online_encoder(view))
        projection = self.online_projector(representation)
        prediction = self.online_predictor(projection)
        return projection, prediction

    @torch.no_grad()
    def _target(self, view: torch.Tensor) -> torch.Tensor:
        representation = _last_feature(self.target_encoder(view))
        return self.target_projector(representation)

    @staticmethod
    def cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = torch.nn.functional.normalize(prediction, dim=-1)
        target = torch.nn.functional.normalize(target.detach(), dim=-1)
        return 2.0 - 2.0 * (prediction * target).sum(dim=-1)

    def forward(self, first_view: torch.Tensor, second_view: torch.Tensor) -> torch.Tensor:
        _, first_prediction = self._online(first_view)
        _, second_prediction = self._online(second_view)
        first_target = self._target(first_view)
        second_target = self._target(second_view)
        return 0.5 * (
            self.cosine_distance(first_prediction, second_target).mean()
            + self.cosine_distance(second_prediction, first_target).mean()
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.99 <= momentum <= 1.0:
            raise ValueError(f"EMA momentum must be in [0.99, 1.0], got {momentum}.")
        for online, target in (
            (self.online_encoder, self.target_encoder),
            (self.online_projector, self.target_projector),
        ):
            online_state = online.state_dict()
            target_state = target.state_dict()
            if online_state.keys() != target_state.keys():
                raise RuntimeError("Online and target state keys no longer match.")
            for key, target_value in target_state.items():
                online_value = online_state[key].detach()
                if torch.is_floating_point(target_value):
                    target_value.mul_(momentum).add_(online_value, alpha=1.0 - momentum)
                else:
                    target_value.copy_(online_value)

    def online_backbone_state(self) -> dict[str, torch.Tensor]:
        """Return cloned, full-key encoder state (``model.0.*`` ... for YOLO)."""

        return {key: value.detach().cpu().clone() for key, value in self.online_encoder.state_dict().items()}

    def recovery_state(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "online_encoder_state": {
                key: value.detach().cpu().clone() for key, value in self.online_encoder.state_dict().items()
            },
            "online_projector_state": {
                key: value.detach().cpu().clone() for key, value in self.online_projector.state_dict().items()
            },
            "online_predictor_state": {
                key: value.detach().cpu().clone() for key, value in self.online_predictor.state_dict().items()
            },
            "target_encoder_state": {
                key: value.detach().cpu().clone() for key, value in self.target_encoder.state_dict().items()
            },
            "target_projector_state": {
                key: value.detach().cpu().clone() for key, value in self.target_projector.state_dict().items()
            },
        }

    def load_recovery_state(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        for module, key in (
            (self.online_encoder, "online_encoder_state"),
            (self.online_projector, "online_projector_state"),
            (self.online_predictor, "online_predictor_state"),
            (self.target_encoder, "target_encoder_state"),
            (self.target_projector, "target_projector_state"),
        ):
            if key not in state:
                raise RuntimeError(f"Checkpoint lacks {key}.")
            module.load_state_dict(state[key], strict=True)
        self._freeze_target()


def ema_momentum(step: int, total_steps: int, *, base_momentum: float = 0.99) -> float:
    """Cosine monotonic schedule from 0.99 at step zero to 1.0 at the end."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if total_steps == 1:
        progress = 1.0
    else:
        progress = min(max(step, 0), total_steps - 1) / (total_steps - 1)
    return 1.0 - (1.0 - base_momentum) * (math.cos(math.pi * progress) + 1.0) / 2.0


class LARS(Optimizer):
    """LARS with frozen adaptation exclusions encoded in parameter groups."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | list[dict[str, Any]],
        *,
        lr: float = 0.8,
        momentum: float = 0.9,
        weight_decay: float = 1e-6,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("LARS lr, momentum, and weight_decay must be non-negative.")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            eps=eps,
            lars_adaptation=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            adapt = group["lars_adaptation"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients.")
                update = gradient
                if weight_decay:
                    update = update.add(parameter, alpha=weight_decay)
                if adapt:
                    parameter_norm = torch.norm(parameter)
                    update_norm = torch.norm(update)
                    if parameter_norm > 0 and update_norm > 0:
                        trust_ratio = (
                            group["trust_coefficient"]
                            * parameter_norm
                            / (update_norm + group["eps"])
                        )
                        update = update.mul(trust_ratio)
                state = self.state[parameter]
                buffer = state.get("momentum_buffer")
                if buffer is None:
                    buffer = state["momentum_buffer"] = update.clone()
                else:
                    buffer.mul_(momentum).add_(update)
                parameter.add_(buffer, alpha=-lr)
        return loss


_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
)


def lars_parameter_groups(module: nn.Module, *, weight_decay: float = 1e-6) -> list[dict[str, Any]]:
    """Split norm/bias tensors from weight decay and LARS adaptation."""

    norm_parameter_ids = {
        id(parameter) for submodule in module.modules() if isinstance(submodule, _NORM_TYPES) for parameter in submodule.parameters(recurse=False)
    }
    regular: list[nn.Parameter] = []
    excluded: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "bias" or name.endswith(".bias") or id(parameter) in norm_parameter_ids:
            excluded.append(parameter)
        else:
            regular.append(parameter)
    if not regular or not excluded:
        raise RuntimeError("LARS parameter grouping expected both regular and bias/norm parameter sets.")
    return [
        {"params": regular, "weight_decay": weight_decay, "lars_adaptation": True},
        {"params": excluded, "weight_decay": 0.0, "lars_adaptation": False},
    ]


class WarmupCosineSchedule:
    """1000-step linear warm-up from 1e-4 followed by cosine decay to zero."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int = 10_000,
        warmup_steps: int = 1_000,
        start_factor: float = 1e-4,
    ) -> None:
        if total_steps <= warmup_steps or warmup_steps <= 0:
            raise ValueError("total_steps must exceed positive warmup_steps.")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.start_factor = start_factor
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_index = 0
        self.set_step(0)

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            return self.start_factor + (1.0 - self.start_factor) * alpha
        decay_updates = self.total_steps - self.warmup_steps
        # The first cosine update is at factor 1 and the final (10,000th in
        # the formal run) is exactly zero, rather than merely approaching it.
        progress = min(step - self.warmup_steps, decay_updates - 1) / max(decay_updates - 1, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("Scheduler step cannot be negative.")
        self.step_index = step
        factor = self.factor(step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "start_factor": self.start_factor,
            "base_lrs": self.base_lrs,
            "step_index": self.step_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (self.total_steps, self.warmup_steps, self.start_factor, self.base_lrs)
        observed = (state.get("total_steps"), state.get("warmup_steps"), state.get("start_factor"), state.get("base_lrs"))
        if observed != expected:
            raise RuntimeError("Checkpoint scheduler protocol does not match this run.")
        self.set_step(int(state["step_index"]))
