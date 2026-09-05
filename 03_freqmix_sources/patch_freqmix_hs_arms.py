"""Add M-FM-HS-v1 arms to an isolated pemt_v5 direct code root.

The student receives an energy-matched mixed image while the canonical H_S
criterion keeps the original clean SAR tensor for its frozen SAR teacher.
"""

from __future__ import annotations

import pathlib
import sys


CODE_ROOT = pathlib.Path(sys.argv[1]).resolve()
CORE = CODE_ROOT / "src" / "ogsod400_core" / "pemt_v5_direct.py"
RUNNER = CODE_ROOT / "tools" / "run_pemt_v5_direct.py"

HS_ARMS = (
    "h_s_freqmix_gray_shuffled",
    "h_s_freqmix_sar_shuffled",
    "h_s_freqmix_gray_sign_randomized",
)


def replace_once(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"expected one marker in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


if HS_ARMS[0] in CORE.read_text(encoding="utf-8"):
    print(f"already patched: {CODE_ROOT}")
    raise SystemExit(0)

replace_once(
    CORE,
    'FREQMIX_ARMS = frozenset({"freqmix_paired", "freqmix_shuffled"})\nFM_KERNEL = 5',
    '''NATIVE_FREQMIX_ARMS = frozenset({"freqmix_paired", "freqmix_shuffled"})
H_S_FREQMIX_ARMS = frozenset(
    {
        "h_s_freqmix_gray_shuffled",
        "h_s_freqmix_sar_shuffled",
        "h_s_freqmix_gray_sign_randomized",
    }
)
FREQMIX_ARMS = NATIVE_FREQMIX_ARMS | H_S_FREQMIX_ARMS
FM_KERNEL = 5
FM_RANDOM_SEED_OFFSET = 20260819
# Rebuild the route sets after declaring the H_S frequency arms. These arms
# keep the canonical H_S criterion and add only a mixed student input.
H_S_ARMS = H_S_ARMS | H_S_FREQMIX_ARMS
HYBRID_CMD_ARMS = H_S_ARMS | H_F_ARMS''',
)

replace_once(
    CORE,
    "NATIVE_ARMS = NATIVE_ARMS | SIDECAR_ARMS | CONTRAST_ARMS | FREQMIX_ARMS | GT_WEIGHT_ARMS",
    "NATIVE_ARMS = NATIVE_ARMS | SIDECAR_ARMS | CONTRAST_ARMS | NATIVE_FREQMIX_ARMS | GT_WEIGHT_ARMS",
)

core_text = CORE.read_text(encoding="utf-8")
class_start = core_text.index("class _FreqMixer:")
class_end = core_text.index("\ndef _tw_keep", class_start)
new_class = '''class _FreqMixer:
    """Build the frozen native or H_S frequency-mixed student input."""

    def __init__(self, platform):
        self.arm = platform.arm
        self.torch = platform.torch
        self.kernel = int(FM_KERNEL)
        self.seed = int(platform.config.seed)
        self.random_generator = None
        self.reported = False

    def _low(self, image):
        return self.torch.nn.functional.avg_pool2d(
            image, kernel_size=self.kernel, stride=1, padding=self.kernel // 2
        )

    @staticmethod
    def _rms(image):
        return image.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()

    def _private_generator(self, image):
        if self.random_generator is None:
            self.random_generator = self.torch.Generator(device=image.device)
            self.random_generator.manual_seed(FM_RANDOM_SEED_OFFSET + self.seed)
        return self.random_generator

    def __call__(self, batch):
        torch = self.torch
        image = batch["img"]
        gray = batch.get("teacher_img")
        if gray is None:
            raise PemtV5DirectError("freqmix arm is missing the paired gray batch")

        if self.arm in NATIVE_FREQMIX_ARMS:
            donor = gray.roll(1, dims=0) if self.arm == "freqmix_shuffled" else gray
            mixed = self._low(image) + (donor - self._low(donor))
            merged = dict(batch)
            merged["img"] = mixed.clamp(0.0, 1.0)
            return merged

        gray_donor = gray.roll(1, dims=0)
        gray_high = gray_donor - self._low(gray_donor)
        target_rms = self._rms(gray_high)

        if self.arm == "h_s_freqmix_gray_shuffled":
            donor_high = gray_high
        elif self.arm == "h_s_freqmix_sar_shuffled":
            sar_donor = image.roll(1, dims=0)
            sar_high = sar_donor - self._low(sar_donor)
            donor_high = sar_high * (target_rms / self._rms(sar_high).clamp_min(1e-12))
        elif self.arm == "h_s_freqmix_gray_sign_randomized":
            signs = torch.where(
                torch.rand(
                    (int(image.shape[0]), 1, int(image.shape[2]), int(image.shape[3])),
                    device=image.device,
                    generator=self._private_generator(image),
                )
                < 0.5,
                -1.0,
                1.0,
            )
            donor_high = gray_high * signs
        else:
            raise PemtV5DirectError(f"unknown H_S freqmix arm: {self.arm}")

        pre_clamp = self._low(image) + donor_high
        if not self.reported:
            rms_error = float((self._rms(donor_high) - target_rms).abs().max().detach().cpu())
            clamp_fraction = float(
                ((pre_clamp < 0.0) | (pre_clamp > 1.0)).float().mean().detach().cpu()
            )
            print(
                f"[freqmix-hs] arm={self.arm} rms_error={rms_error:.8g} "
                f"clamp_fraction={clamp_fraction:.8g}"
            )
            self.reported = True
        merged = dict(batch)
        merged["img"] = pre_clamp.clamp(0.0, 1.0)
        return merged

'''
CORE.write_text(core_text[:class_start] + new_class + core_text[class_end + 1 :], encoding="utf-8")

replace_once(
    CORE,
    '''            def loss_with_freqmix(_model, batch, predictions=None):
                if not bool(_model.training):
                    return original(batch, predictions)
                return original(mixer(batch), predictions)''',
    '''            def loss_with_freqmix(_model, batch, predictions=None):
                if not bool(_model.training):
                    return original(batch, predictions)
                mixed_batch = mixer(batch)
                if self.arm in H_S_FREQMIX_ARMS:
                    if predictions is not None:
                        raise PemtV5DirectError(
                            "H_S freqmix production loss expects to own the student forward"
                        )
                    mixed_predictions = _model.predict(mixed_batch["img"])
                    return original(batch, mixed_predictions)
                return original(mixed_batch, predictions)''',
)

replace_once(
    RUNNER,
    '            "freqmix_shuffled",',
    '''            "freqmix_shuffled",
            "h_s_freqmix_gray_shuffled",
            "h_s_freqmix_sar_shuffled",
            "h_s_freqmix_gray_sign_randomized",''',
)

print(f"patched {CORE}")
print(f"patched {RUNNER}")
