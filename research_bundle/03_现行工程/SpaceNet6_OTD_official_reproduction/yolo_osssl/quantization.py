"""Small, backend-neutral state controls for frozen Stage-II fake quantization.

The training entrypoint owns AIMET construction.  This module deliberately owns
only the invariant that the q/f counterfactual differs in fake quantization,
not in BatchNorm, observer, encoding, or module-mode state.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import torch
from torch import nn
from ultralytics.nn.modules.block import Attention, Bottleneck, PSABlock


class QuantizationStateError(RuntimeError):
    """Raised when a supposedly frozen quantization counterfactual mutates state."""


class YoloQATWrapper(nn.Module):
    """Trace-safe raw-head façade for a pinned Ultralytics DetectionModel.

    Ultralytics evaluation returns ``(decoded, {boxes, scores, feats})``.  That
    heterogeneous dictionary cannot be traced by AIMET's ConnectedGraph.  The
    wrapper preserves the exact raw tensors needed by the native criterion in
    a fixed, five-tensor tuple.  It never decodes, reorders, or adapts heads.
    """

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector
        # Detect concatenates its three output levels functionally.  Explicit
        # identities give the concatenated tensors their own QDQ boundary so
        # the complete detector body can be compiled as one HTP partition.
        self.box_output = nn.Identity()
        self.score_output = nn.Identity()
        self.criterion: Any | None = None

    @property
    def stride(self) -> torch.Tensor:
        stride = getattr(self.detector, "stride", None)
        if not isinstance(stride, torch.Tensor):
            raise QuantizationStateError("Ultralytics DetectionModel has no tensor stride")
        return stride

    def _raw_tuple(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        modules = getattr(self.detector, "model", None)
        if not isinstance(modules, (nn.ModuleList, nn.Sequential)) or not modules:
            raise QuantizationStateError("Ultralytics DetectionModel has no Detect module list")
        head = modules[-1]
        previous_training = bool(head.training)
        # Only the Detect branch flag is changed: all QAT BN modules retain
        # their current q/f mode.  This forces the raw dict path in both
        # tracing/evaluation and normal QAT training.
        head.training = True
        try:
            raw = self.detector(image)
        finally:
            head.training = previous_training
        if not isinstance(raw, Mapping):
            raise QuantizationStateError("pinned Detect head did not return the raw prediction mapping")
        boxes, scores, feats = raw.get("boxes"), raw.get("scores"), raw.get("feats")
        if (
            not isinstance(boxes, torch.Tensor)
            or not isinstance(scores, torch.Tensor)
            or not isinstance(feats, (list, tuple))
            or len(feats) != 3
            or not all(isinstance(item, torch.Tensor) for item in feats)
        ):
            raise QuantizationStateError("pinned Detect raw prediction ABI drifted")
        return self.box_output(boxes), self.score_output(scores), feats[0], feats[1], feats[2]

    def forward(self, value: torch.Tensor | Mapping[str, Any]) -> Any:
        if isinstance(value, Mapping):
            if self.criterion is None:
                raise QuantizationStateError("Stage-II criterion was not attached to YoloQATWrapper")
            prediction = self._raw_tuple(value["img"])
            return self.criterion(prediction, dict(value))
        return self._raw_tuple(value)

    def loss(self, batch: Mapping[str, Any], prediction: Any | None = None) -> Any:
        if self.criterion is None:
            raise QuantizationStateError("Stage-II criterion was not attached to YoloQATWrapper")
        return self.criterion(self._raw_tuple(batch["img"]) if prediction is None else prediction, dict(batch))


class _QnnSiLU(nn.Module):
    """SiLU decomposed into separately quantizable sigmoid and multiply ops."""

    def __init__(self, multiply: nn.Module) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.multiply = multiply

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.multiply(value, self.sigmoid(value))


class _QnnOutputBoundary(nn.Module):
    """Numerically neutral multiply that carries an explicit output quantizer."""

    def __init__(self, multiply: nn.Module) -> None:
        super().__init__()
        self.multiply = multiply
        self.register_buffer("one", torch.tensor(1.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.multiply(value, self.one)


class _QnnBottleneck(Bottleneck):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.cv2(self.cv1(value))
        return self.add_op(value, residual) if self.add else residual


class _QnnPSABlock(PSABlock):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.attn_add(value, self.attn(value)) if self.add else self.attn(value)
        return self.ffn_add(value, self.ffn(value)) if self.add else self.ffn(value)


class _QnnAttention(Attention):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        positions = height * width
        qkv = self.qkv(value)
        q, k, v = qkv.view(batch, self.num_heads, self.key_dim * 2 + self.head_dim, positions).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attention = self.qk_matmul((q * self.scale).transpose(-2, -1), k)
        attention = self.attention_softmax(attention)
        attended = self.va_matmul(v, attention.transpose(-2, -1)).view(batch, channels, height, width)
        return self.proj(self.pe_add(attended, self.pe(v.reshape(batch, channels, height, width))))


def _make_elementwise_ops_explicit(model: nn.Module) -> None:
    """Expose mathematically unchanged SiLU/residual ops to AIMET quantization."""

    from aimet_torch._base.nn.modules.custom import Add, MatMul, Multiply

    if isinstance(model, YoloQATWrapper) and not isinstance(model.box_output, _QnnOutputBoundary):
        model.box_output = _QnnOutputBoundary(Multiply())
        model.score_output = _QnnOutputBoundary(Multiply())

    for name, child in tuple(model.named_children()):
        if isinstance(child, nn.SiLU):
            setattr(model, name, _QnnSiLU(Multiply()))
            continue
        _make_elementwise_ops_explicit(child)

    for module in model.modules():
        if isinstance(module, Bottleneck) and module.add and not isinstance(module, _QnnBottleneck):
            module.__class__ = _QnnBottleneck
            module.add_module("add_op", Add())
        elif isinstance(module, PSABlock) and module.add and not isinstance(module, _QnnPSABlock):
            module.__class__ = _QnnPSABlock
            module.add_module("attn_add", Add())
            module.add_module("ffn_add", Add())
        elif isinstance(module, Attention) and not isinstance(module, _QnnAttention):
            module.__class__ = _QnnAttention
            module.add_module("qk_matmul", MatMul())
            module.add_module("attention_softmax", nn.Softmax(dim=-1))
            module.add_module("va_matmul", MatMul())
            module.add_module("pe_add", Add())


def _quantizer_modules(model: nn.Module) -> list[nn.Module]:
    """Return modules exposing an AIMET-like quantizer or observer control."""

    names = ("enabled", "observer_enabled", "set_quantizers_enabled", "input_quantizers", "output_quantizers")
    return [module for module in model.modules() if any(hasattr(module, name) for name in names)]


def _set_flag(module: nn.Module, name: str, value: bool) -> tuple[bool, Any] | None:
    if not hasattr(module, name):
        return None
    previous = getattr(module, name)
    if isinstance(previous, bool):
        setattr(module, name, value)
        return True, previous
    return None


def _set_quantizer_enabled(quantizer: Any, enabled: bool) -> None:
    if quantizer is None:
        return
    if hasattr(quantizer, "enabled") and isinstance(quantizer.enabled, bool):
        quantizer.enabled = enabled
        return
    method = "enable" if enabled else "disable"
    candidate = getattr(quantizer, method, None)
    if callable(candidate):
        candidate()


def _set_module_quantizers(module: nn.Module, enabled: bool) -> None:
    method = getattr(module, "set_quantizers_enabled", None)
    if callable(method):
        method(enabled)
    for name in ("input_quantizers", "output_quantizers", "param_quantizers"):
        quantizers = getattr(module, name, None)
        if isinstance(quantizers, (dict, nn.ModuleDict)):
            values = quantizers.values()
        elif isinstance(quantizers, (list, tuple, nn.ModuleList)):
            values = quantizers
        else:
            continue
        for quantizer in values:
            _set_quantizer_enabled(quantizer, enabled)


def _has_aimet_quantizer_removers(module: nn.Module) -> bool:
    return any(
        callable(getattr(module, name, None))
        for name in ("_remove_input_quantizers", "_remove_output_quantizers", "_remove_param_quantizers")
    )


def _remove_aimet_quantizers(stack: ExitStack, model: nn.Module) -> bool:
    """Enter AIMET-v2 removal contexts; they restore identical quantizer objects."""

    entered = False
    for module in model.modules():
        for name in ("_remove_input_quantizers", "_remove_output_quantizers", "_remove_param_quantizers"):
            remover = getattr(module, name, None)
            if callable(remover):
                stack.enter_context(remover())
                entered = True
    return entered


def _iter_quantizers(module: nn.Module) -> Iterable[Any]:
    for name in ("input_quantizers", "output_quantizers", "param_quantizers"):
        quantizers = getattr(module, name, None)
        if isinstance(quantizers, (dict, nn.ModuleDict)):
            yield from quantizers.values()
        elif isinstance(quantizers, (list, tuple, nn.ModuleList)):
            yield from quantizers


def _quantizer_values(module: nn.Module, name: str) -> list[Any]:
    quantizers = getattr(module, name, None)
    if isinstance(quantizers, (dict, nn.ModuleDict)):
        return list(quantizers.values())
    if isinstance(quantizers, (list, tuple, nn.ModuleList)):
        return list(quantizers)
    return []


class QuantizationModeGuard(AbstractContextManager["QuantizationModeGuard"]):
    """Read-only diagnostic modes over one calibrated QuantSim model."""

    _COLLECTIONS = {
        "weights": {"param_quantizers"},
        "activations": {"input_quantizers", "output_quantizers"},
        "w8a8": {"input_quantizers", "output_quantizers", "param_quantizers"},
        "off": set(),
    }

    def __init__(self, model: nn.Module, mode: Literal["off", "weights", "activations", "w8a8"]) -> None:
        if mode not in self._COLLECTIONS:
            raise QuantizationStateError(f"unsupported quantization mode {mode!r}")
        self.model = model
        self.mode = mode
        self._modes: dict[str, bool] = {}
        self._states: list[tuple[Any, str, Any]] = []
        self._stack = ExitStack()

    def __enter__(self) -> "QuantizationModeGuard":
        self._modes = {name: bool(module.training) for name, module in self.model.named_modules()}
        enabled = self._COLLECTIONS[self.mode]
        self.model.eval()
        self._stack.enter_context(torch.no_grad())
        for module in self.model.modules():
            for collection in ("input_quantizers", "output_quantizers", "param_quantizers"):
                values = _quantizer_values(module, collection)
                for quantizer in values:
                    for flag in ("enabled", "observer_enabled"):
                        value = getattr(quantizer, flag, None)
                        if isinstance(value, bool):
                            self._states.append((quantizer, flag, value))
                if collection in enabled:
                    continue
                remover = getattr(module, f"_remove_{collection}", None)
                if callable(remover):
                    self._stack.enter_context(remover())
                else:
                    for quantizer in values:
                        _set_quantizer_enabled(quantizer, False)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._stack.close()
        for name, module in self.model.named_modules():
            module.training = self._modes[name]
        for quantizer, flag, value in self._states:
            setattr(quantizer, flag, value)
        return False


def quantization_mode(
    model: nn.Module,
    mode: Literal["off", "weights", "activations", "w8a8"],
) -> QuantizationModeGuard:
    return QuantizationModeGuard(model, mode)


@dataclass
class QuantizerStateGuard(AbstractContextManager["QuantizerStateGuard"]):
    """Temporarily select q or f behavior and restore modes and quantizers.

    The stage lifecycle owns encoding calibration and BN freezing at epoch
    boundaries. This per-forward guard only switches the state needed by the
    q/f counterfactual and restores it afterwards.
    """

    model: nn.Module
    enabled: bool
    freeze_bn: bool = True
    observers_off: bool = True
    enforce_no_grad: bool = True
    _modes: dict[str, bool] | None = None
    _flags: list[tuple[nn.Module, str, Any]] | None = None
    _quantizer_flags: list[tuple[Any, str, Any]] | None = None
    _aimet_stack: ExitStack | None = None

    def __enter__(self) -> "QuantizerStateGuard":
        self._modes = {name: bool(module.training) for name, module in self.model.named_modules()}
        self._flags = []
        self._quantizer_flags = []
        self._aimet_stack = ExitStack()
        if self.enforce_no_grad:
            self._aimet_stack.enter_context(torch.no_grad())
        if not self.enabled and self.freeze_bn:
            # f/o must be full eval counterfactuals, not merely BN-frozen q.
            self.model.eval()
        using_aimet_contexts = not self.enabled and _remove_aimet_quantizers(self._aimet_stack, self.model)
        for module in self.model.modules():
            if self.freeze_bn and isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
            for flag in ("enabled", "observer_enabled"):
                saved = _set_flag(module, flag, self.enabled if flag == "enabled" else not self.observers_off)
                if saved is not None:
                    _, previous = saved
                    self._flags.append((module, flag, previous))
            for quantizer in _iter_quantizers(module):
                for flag in ("enabled", "observer_enabled"):
                    value = getattr(quantizer, flag, None)
                    if isinstance(value, bool):
                        self._quantizer_flags.append((quantizer, flag, value))
            if not using_aimet_contexts:
                _set_module_quantizers(module, self.enabled)
        if self.enabled:
            assert_live_quantizers_enabled(self.model)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        assert self._modes is not None and self._flags is not None and self._quantizer_flags is not None
        # Restore AIMET's exact original quantizer objects before comparing.
        self._aimet_stack.close()
        # Restore mode/flags before comparing the observable invariant.
        for name, module in self.model.named_modules():
            # Direct assignment preserves each nested module's original mode;
            # Module.train() would recursively overwrite child modes.
            module.training = self._modes[name]
        for module, flag, previous in self._flags:
            setattr(module, flag, previous)
        for quantizer, flag, previous in self._quantizer_flags:
            setattr(quantizer, flag, previous)
        return False


def quantizers_on(model: nn.Module) -> QuantizerStateGuard:
    # q is the normal QAT training path: BN/observers and autograd remain live.
    return QuantizerStateGuard(
        model, enabled=True, freeze_bn=False, observers_off=False, enforce_no_grad=False
    )


def quantizers_off(model: nn.Module) -> QuantizerStateGuard:
    # f is a frozen no-grad counterfactual.  Optical teacher follows the same
    # eval/no-grad policy at the Stage-II caller.
    return QuantizerStateGuard(model, enabled=False)


def assert_live_quantizers_enabled(model: nn.Module) -> int:
    """Confirm that the q forward actually contains enabled quantizers."""

    inventory = 0
    disabled = 0
    for module in model.modules():
        for quantizer in _iter_quantizers(module):
            if quantizer is None:
                continue
            inventory += 1
            if getattr(quantizer, "enabled", True) is False:
                disabled += 1
    if inventory == 0:
        raise QuantizationStateError("q forward has no live AIMET quantizer inventory")
    if disabled:
        raise QuantizationStateError(f"q forward has {disabled} disabled live quantizers")
    return inventory


def aimet_default_per_channel_config() -> str:
    """Locate AIMET's bundled W8-per-channel/A8-per-tensor configuration."""

    try:
        import aimet_torch
    except ImportError as exc:  # pragma: no cover - pinned remote runtime
        raise QuantizationStateError("AIMET 2.37.0 is required for QAT") from exc
    root = Path(aimet_torch.__file__).resolve().parent
    matches = sorted(root.rglob("default_config_per_channel.json"))
    if len(matches) != 1:
        raise QuantizationStateError("cannot resolve AIMET bundled default_config_per_channel.json")
    return str(matches[0])


_ULTRALYTICS_STRUCTURAL_IGNORES = {"ultralytics.nn.modules.conv.Concat"}


def _register_ultralytics_structural_ignores(model: nn.Module) -> None:
    """Register YOLO's parameter-free Concat with AIMET 2.37."""

    try:
        from aimet_torch.nn import QuantizationMixin
    except ImportError as exc:  # pragma: no cover - pinned remote runtime
        raise QuantizationStateError("AIMET QuantizationMixin is unavailable") from exc
    for module in model.modules():
        cls = type(module)
        qualified = f"{cls.__module__}.{cls.__qualname__}"
        if qualified in _ULTRALYTICS_STRUCTURAL_IGNORES:
            QuantizationMixin.ignore(cls)


def build_aimet_quantsim(
    model: nn.Module,
    dummy_input: torch.Tensor,
    *,
    weight_bitwidth: int = 8,
    activation_bitwidth: int = 8,
) -> Any:
    """Construct an AIMET QuantSim; W8A8 remains the training default."""

    try:
        from aimet_torch.batch_norm_fold import fold_all_batch_norms
        from aimet_torch.common.defs import QuantScheme, QuantizationDataType
        from aimet_torch.quantsim import QuantizationSimModel
    except ImportError as exc:  # pragma: no cover - pinned remote runtime
        raise QuantizationStateError("AIMET QuantizationSimModel is unavailable") from exc
    wrapper = model if isinstance(model, YoloQATWrapper) else YoloQATWrapper(model)
    _make_elementwise_ops_explicit(wrapper)
    # HTP requires a fully quantized graph.  Fold Conv-BN pairs before QuantSim
    # construction so QAT learns the exact graph that is later exported,
    # instead of changing quantizer placement after training.
    wrapper.eval()
    fold_all_batch_norms(wrapper, tuple(dummy_input.shape), dummy_input=dummy_input)
    if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in wrapper.modules()):
        raise QuantizationStateError("QAT graph still contains an unfused BatchNorm")
    _register_ultralytics_structural_ignores(wrapper.detector)
    sim = QuantizationSimModel(
        model=wrapper,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.min_max,
        default_data_type=QuantizationDataType.int,
        default_output_bw=activation_bitwidth,
        default_param_bw=weight_bitwidth,
        config_file=aimet_default_per_channel_config(),
    )
    return sim


def compute_aimet_encodings(sim: Any, forward_pass_callback: Any, callback_args: Any) -> None:
    """Run AIMET encoding collection through the supplied real-data callback."""

    compute = getattr(sim, "compute_encodings", None)
    if not callable(compute):
        raise QuantizationStateError("QuantSim lacks compute_encodings")
    compute(forward_pass_callback=forward_pass_callback, forward_pass_callback_args=callback_args)


def freeze_batch_norm(model: nn.Module) -> int:
    """Freeze BN affine parameters and running statistics after common QAT."""

    frozen = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(False)
            frozen += 1
    return frozen


def freeze_aimet_quantizers(model: nn.Module) -> None:
    """Freeze encodings and disallow range learning after calibration."""

    for module in model.modules():
        for quantizer in _iter_quantizers(module):
            allow_overwrite = getattr(quantizer, "allow_overwrite", None)
            if callable(allow_overwrite):
                allow_overwrite(False)
            requires_grad = getattr(quantizer, "requires_grad_", None)
            if callable(requires_grad):
                requires_grad(False)


def disable_aimet_range_learning(model: nn.Module) -> None:
    """Keep calibrated min/max fixed while allowing a later encoding refresh."""

    for module in model.modules():
        for quantizer in _iter_quantizers(module):
            requires_grad = getattr(quantizer, "requires_grad_", None)
            if callable(requires_grad):
                requires_grad(False)
