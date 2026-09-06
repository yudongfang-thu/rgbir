"""Dataset adapter for paired OS-SSL images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .manifest import Arm, PairRecord, select_arm_pairs
from .transforms import ProductionPairTransform


class PairManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[PairRecord], *, arm: Arm, transform: ProductionPairTransform | None = None) -> None:
        self.arm = arm
        self.pairs = select_arm_pairs(records, arm)
        self.transform = transform or ProductionPairTransform()

    @staticmethod
    def _open_rgb(path: str):
        from PIL import Image

        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image is missing: {image_path}")
        with Image.open(image_path) as image:
            return image.convert("RGB").copy()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        first = self._open_rgb(pair.source.sar_path)
        second_path = pair.source.sar_path if self.arm == "sar_only" else pair.second.rgb_path
        second = self._open_rgb(second_path)
        first_view, second_view = self.transform(first, second)
        return {"first": first_view, "second": second_view, "pair_id": pair.source.pair_id, "arm": self.arm}
