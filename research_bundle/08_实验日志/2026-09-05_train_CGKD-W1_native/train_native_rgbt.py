# -*- coding: utf-8 -*-
"""Native (weight0) RGB student trainer for CGA-KD W1.

Reuses the frozen protocol yaml so the recipe matches cmdistill arms exactly
(same hyperparams/augmentation/data), minus all KD losses.
"""
import argparse
import json
import os

import yaml
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    aug = cfg["augmentation"]
    model = YOLO(cfg["model"])
    overrides = dict(
        data=cfg["paths"]["student_data_yaml"],
        epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        workers=cfg["workers"], optimizer=cfg["optimizer"],
        lr0=cfg["lr0"], lrf=cfg["lrf"], momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"], warmup_momentum=cfg["warmup_momentum"],
        warmup_bias_lr=cfg["warmup_bias_lr"],
        cos_lr=cfg["cos_lr"], close_mosaic=cfg["close_mosaic"],
        patience=cfg["patience"], amp=cfg["amp"], deterministic=cfg["deterministic"],
        seed=a.seed,
        mosaic=aug["mosaic"], mixup=aug["mixup"], cutmix=aug.get("cutmix", 0.0),
        degrees=aug["degrees"], perspective=aug["perspective"],
        translate=aug["translate"], scale=aug["scale"],
        fliplr=aug["fliplr"], flipud=aug["flipud"],
        hsv_h=aug["hsv_h"], hsv_s=aug["hsv_s"], hsv_v=aug["hsv_v"],
        erasing=aug["erasing"],
        project=os.path.dirname(a.output), name=os.path.basename(a.output),
        exist_ok=True, verbose=True,
    )
    model.train(**overrides)

    receipt = {"arm": "native_weight0", "method_identity": "CGA-KD-W1",
               "config": a.config, "seed": a.seed, "output": a.output,
               "recipe_frozen_from": "cmdistill_protocol_drone.yaml"}
    with open(os.path.join(a.output, "completion_receipt.json"), "w") as f:
        json.dump(receipt, f, indent=1)
    print("NATIVE_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
