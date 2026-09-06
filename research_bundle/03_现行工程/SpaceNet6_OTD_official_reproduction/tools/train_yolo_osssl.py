#!/usr/bin/env python3
"""Train YOLO11n BYOL OS-SSL with a small, direct training loop."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolo_osssl.data import PairManifestDataset
from yolo_osssl.manifest import read_pair_manifest
from yolo_osssl.model import BYOLModel, LARS, WarmupCosineSchedule, ema_momentum, lars_parameter_groups
from yolo_osssl.yolo import YOLO11nBackbone, build_yolo11n_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("paired", "sar_only", "shuffled"))
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--nc", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: BYOLModel, optimizer: torch.optim.Optimizer, scheduler: WarmupCosineSchedule, step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "online_backbone_state": model.online_backbone_state(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "seed": args.seed,
            "arm": args.arm,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if args.steps <= 1 or args.batch_size <= 1:
        raise ValueError("steps and batch-size must be greater than one")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    set_seed(args.seed)
    records = read_pair_manifest(args.manifest)
    dataset = PairManifestDataset(records, arm=args.arm)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        generator=generator,
    )
    if len(loader) == 0:
        raise RuntimeError("Dataset is smaller than one training batch")

    template, _, version = build_yolo11n_template(args.weights, nc=args.nc)
    model = BYOLModel(YOLO11nBackbone(template)).to(device)
    optimizer = LARS(lars_parameter_groups(model), lr=0.8, momentum=0.9, weight_decay=1e-6)
    warmup = args.warmup_steps if args.warmup_steps is not None else min(1_000, max(1, args.steps // 10))
    warmup = min(warmup, args.steps - 1)
    scheduler = WarmupCosineSchedule(optimizer, total_steps=args.steps, warmup_steps=warmup)
    start_step = 0
    latest = args.output_dir / "latest.pt"
    if args.resume and latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({"steps": args.steps, "batch_size": args.batch_size, "seed": args.seed, "arm": args.arm, "ultralytics": version}, indent=2) + "\n",
        encoding="utf-8",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(loader)
    model.train()
    for step in range(start_step, args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        first = batch["first"].to(device, non_blocking=True)
        second = batch["second"].to(device, non_blocking=True)
        scheduler.set_step(step)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = model(first, second)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite SSL loss at step {step + 1}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.update_target(ema_momentum(step, args.steps))
        completed = step + 1
        if completed == 1 or completed % 20 == 0:
            print(json.dumps({"step": completed, "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"]}), flush=True)
        if completed % args.save_every == 0:
            save_checkpoint(latest, model, optimizer, scheduler, completed, args)

    save_checkpoint(latest, model, optimizer, scheduler, args.steps, args)
    save_checkpoint(args.output_dir / "final.pt", model, optimizer, scheduler, args.steps, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
