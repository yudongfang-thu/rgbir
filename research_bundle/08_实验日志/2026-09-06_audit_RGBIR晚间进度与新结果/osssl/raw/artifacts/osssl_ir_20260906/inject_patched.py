#!/usr/bin/env python3
"""Strictly inject a YOLO OS-SSL final backbone into one saved full template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hashlib as _hl
from yolo_osssl.injection import InjectionError, extract_online_backbone_state, strict_inject_online_backbone

def stable_state_sha256(state):
    h = _hl.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()
from yolo_osssl.yolo import (
    YOLOContractError,
    backbone_keys,
    build_yolo11n_template,
    load_template_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="Frozen SHA-locked yolo11n.pt.")
    parser.add_argument("--nc", required=True, type=int)
    parser.add_argument("--template-state", required=True, type=Path, help="One pre-saved full template shared by all arms.")
    parser.add_argument("--ssl-final-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-local-ultralytics-api-drift",
        action="store_true",
        help="Engineering-only runtime acknowledgement; production stays locked to 8.4.115.",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - legacy torch only
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise InjectionError("SSL checkpoint is not a dictionary.")
    if payload.get("checkpoint_kind") != "final":
        raise InjectionError("Downstream injection only accepts the immutable final SSL checkpoint.")
    return payload


def main() -> int:
    args = parse_args()
    try:
        _load_checkpoint(args.ssl_final_checkpoint)
        template, weights_hash, version = build_yolo11n_template(
            args.weights, nc=args.nc, allow_api_drift=args.allow_local_ultralytics_api_drift
        )
        load_template_state(args.template_state, template)
        online_backbone_state = extract_online_backbone_state(args.ssl_final_checkpoint)
        non_backbone_before = {
            key: value for key, value in template.state_dict().items() if key not in backbone_keys(template.state_dict())
        }
        non_backbone_hash = stable_state_sha256(non_backbone_before)
        strict_inject_online_backbone(template, online_backbone_state)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "yolo-osssl-injected-template-v1",
                "model_state": {key: value.detach().cpu().clone() for key, value in template.state_dict().items()},
                "weights_sha256": weights_hash,
                "ultralytics_version": version,
                "nc": args.nc,
                "non_backbone_template_sha256": non_backbone_hash,
            },
            args.output,
        )
        print(
            json.dumps(
                {
                    "engineering_status": "ok",
                    "output": str(args.output),
                    "strict_backbone_tensors": len(online_backbone_state),
                    "non_backbone_template_sha256": non_backbone_hash,
                },
                sort_keys=True,
            )
        )
        return 0
    except (InjectionError, YOLOContractError, RuntimeError, OSError) as exc:
        print(json.dumps({"engineering_status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
