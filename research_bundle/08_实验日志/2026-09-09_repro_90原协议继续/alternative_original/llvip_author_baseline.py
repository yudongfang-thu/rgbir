"""Author LLVIP YOLOv5 checkpoint identity / original val.run wrapper.

No scheduler, installation, digest calculation, or alternate model implementation.
Identity mode is CPU-only and does not run forward. Evaluation requires a concrete
protocol JSON prepared after identity inspection; caller supplies the project lease.
Run with python -B to avoid writing bytecode inside the original source tree.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_sizes(source):
    return {name: (source / name).stat().st_size for name in
            ("val.py", "models/yolo.py", "models/common.py", "models/experimental.py",
             "utils/datasets.py", "utils/metrics.py", "utils/general.py")}


class Tee(io.StringIO):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def write(self, value):
        self.parent.write(value)
        self.parent.flush()
        return super().write(value)


class CanaryComplete(Exception):
    """Exit the original evaluation loop before aggregate AP is reached."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("identity", "canary", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path("/mnt/dataX/ydf/projects/RGBT_campaign_90"))
    parser.add_argument("--source", type=Path, required=True, help="Complete author's LLVIP/yolov5 directory")
    parser.add_argument("--output", type=Path, required=True, help="New attempt directory; must not exist")
    parser.add_argument("--weights", type=Path, help="Identity mode: one extracted author .pt file")
    parser.add_argument("--protocol", type=Path, help="Evaluation mode: already prepared concrete protocol")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserve earlier attempt; choose a new output directory: {output}")
    if args.mode == "identity":
        if args.weights is None:
            parser.error("identity requires --weights")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.protocol is None:
        parser.error("canary/evaluate requires --protocol")
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / "executed_wrapper.py")
    root = args.root.resolve()
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                      OPENBLAS_NUM_THREADS="1", MPLCONFIGDIR=str(root / "cache/llvip_author_matplotlib"),
                      TORCH_HOME=str(root / "cache/torch"))
    lease = None
    resource_record = None
    if args.mode != "identity":
        sys.path.insert(0, str(root))
        from tools.project_resource_guard import require_bound_lease_from_environment, bound_lease_resource_record_from_environment
        lease = require_bound_lease_from_environment()
        resource_record = bound_lease_resource_record_from_environment
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
            raise RuntimeError("This evaluator requires one bound GPU, identified by the existing project lease")
    sys.path.insert(0, str(source))
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import numpy as np
    if "int" not in np.__dict__:
        np.int = int
    if "float" not in np.__dict__:
        np.float = float
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.mode != "identity":
        torch.cuda.set_per_process_memory_fraction(.68, 0)
    started = time.perf_counter()
    source_before = source_sizes(source)
    base = {"started": dt.datetime.now(dt.timezone.utc).isoformat(), "mode": args.mode,
            "author_repo": "https://github.com/bupt-ai-cz/LLVIP", "source_root": str(source),
            "source_file_sizes": source_before, "torch": torch.__version__, "numpy": np.__version__,
            "new_hash_computed": False, "training_started": False,
            "compatibility": ["known author legacy pickle loading", "NumPy int/float aliases"],
            "source_content_identity_proved": False, "lease": lease,
            "identity": "PAPER-RECONSTRUCTED", "release_description": "author-released single-modality LLVIP baseline",
            "allocator_memory_fraction": .68 if args.mode != "identity" else None}
    try:
        if args.mode == "identity":
            weights = args.weights.resolve()
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                raise TypeError("Expected author YOLOv5 checkpoint dict; inspect separately instead of guessing")
            model = checkpoint.get("ema")
            selected = "ema"
            if model is None:
                model = checkpoint.get("model")
                selected = "model"
            if not isinstance(model, torch.nn.Module):
                raise TypeError("Checkpoint has no stored nn.Module; do not build a random replacement")
            info = dict(base, status="IDENTITY_READ_CPU", weights=str(weights),
                        weights_bytes=weights.stat().st_size, checkpoint_keys=list(checkpoint),
                        selected=selected, epoch=checkpoint.get("epoch"),
                        model_class=f"{type(model).__module__}.{type(model).__name__}",
                        parameters=sum(p.numel() for p in model.parameters()),
                        names=getattr(model, "names", None), yaml=getattr(model, "yaml", None),
                        stride=getattr(model, "stride", torch.tensor([])).tolist(),
                        cuda_initialized=torch.cuda.is_initialized(), new_forward=False,
                        modality="not inferred from weights; verify archive filename and author metadata")
            if info["cuda_initialized"]:
                raise RuntimeError("Identity mode unexpectedly initialized CUDA")
            write_json(output / "identity.json", info)
            print(json.dumps(info, ensure_ascii=False, default=str))
            return

        protocol = load_json(args.protocol)
        if protocol.get("identity") != "PAPER-RECONSTRUCTED":
            raise ValueError("This disclosed reconstruction must retain PAPER-RECONSTRUCTED identity")
        if protocol.get("device") != "0":
            raise ValueError("Use logical device 0 mapped by the bound project lease, not a fixed physical GPU")
        frozen = {"imgsz": 1280, "batch_size": 32, "conf_thres": .001, "iou_thres": .6,
                  "half": False, "augment": False, "save_hybrid": False, "label_version": "previous",
                  "expected_images": 3463, "expected_gt": 7931}
        for key, value in frozen.items():
            if protocol.get(key) != value:
                raise ValueError(f"This wrapper targets the disclosed author CLI protocol: {key} must be {value!r}")
        if protocol.get("modality") not in ("visible", "infrared"):
            raise ValueError("Explicit visible/infrared identity is required")
        identity = load_json(protocol["identity_receipt"])
        weights = Path(protocol["weights"]).resolve()
        if identity["status"] != "IDENTITY_READ_CPU" or identity["cuda_initialized"]:
            raise ValueError("Missing completed CPU identity receipt")
        if str(weights) != identity["weights"] or weights.stat().st_size != identity["weights_bytes"]:
            raise ValueError("Weight path/size differ from inspected identity")
        if identity["source_file_sizes"] != source_before:
            raise ValueError("Source file sizes changed after identity inspection")
        expected = load_json(protocol["expected_roster"])["stems"]
        if len(expected) != 3463 or len(set(expected)) != 3463:
            raise ValueError("Expected official test roster must have 3463 unique stems")
        write_json(output / "frozen_protocol.json", protocol)
        import yaml
        data = yaml.safe_load(Path(protocol["data"]).read_text(encoding="utf-8"))
        if data.get("nc") != 1 or data.get("names") != ["person"] or "download" in data:
            raise ValueError("Require local person-only dataset YAML without automatic download directives")
        shutil.copyfile(protocol["data"], output / "effective_data.yaml")
        for name in source_before:
            dest = output / "source_snapshot" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, dest)
        import utils.datasets as author_datasets
        def path_size_cache_key(paths):
            # The original method calls hashlib.md5 on metadata. Explicitly replace
            # only cache identity: no digest, image loading/labels/augmentation changes.
            return ("plain_path_size_v1", tuple((str(p), Path(p).stat().st_size if Path(p).is_file() else None)
                                              for p in paths))
        author_datasets.get_hash = path_size_cache_key
        base["cache_identity_compatibility"] = "author get_hash metadata MD5 replaced by plain path/size tuple; isolated baseline data view required"
        base["device_selection"] = "original select_device default inside one lease-visible GPU; logical cuda:0 without overwriting physical CUDA_VISIBLE_DEVICES"
        import val as author_val
        original_loader = author_val.create_dataloader
        original_metric = author_val.ap_per_class
        original_nms = author_val.non_max_suppression
        captured = {}
        canary_batches = []

        class CanaryLoader:
            def __init__(self, loader):
                self.loader = loader
                self.dataset = loader.dataset
            def __len__(self):
                return len(self.loader)
            def __getattr__(self, name):
                return getattr(self.loader, name)
            def __iter__(self):
                for i, batch in enumerate(self.loader):
                    torch.cuda.synchronize()
                    before_batch = time.perf_counter()
                    yield batch
                    torch.cuda.synchronize()
                    canary_batches.append({"batch": i, "shape": list(batch[0].shape),
                                           "stems": [Path(p).stem for p in batch[2]],
                                           "seconds": time.perf_counter() - before_batch})
                    write_json(output / "canary_batches.json", canary_batches)
                    if len(canary_batches) == 2:
                        raise CanaryComplete()
                raise RuntimeError("Canary loader ended before two complete batches")

        def checked_loader(*positional, **keywords):
            result = original_loader(*positional, **keywords)
            dataset = result[0].dataset
            images = getattr(dataset, "img_files", getattr(dataset, "im_files", []))
            stems = [Path(p).stem for p in images]
            if len(stems) != 3463 or len(set(stems)) != 3463 or set(stems) != set(expected):
                raise RuntimeError("Original loader differs from the frozen official test roster")
            gt = sum(len(x) for x in dataset.labels)
            if gt != 7931:
                raise RuntimeError(f"Original loader GT count {gt} differs from previous-annotation 7931")
            captured["roster"] = {"images": len(stems), "gt": gt, "stems": stems,
                                  "modality": protocol["modality"]}
            write_json(output / "evaluated_roster.json", captured["roster"])
            if args.mode == "canary":
                return CanaryLoader(result[0]), result[1]
            return result

        def checked_nms(*positional, **keywords):
            tee = Tee(sys.stdout)
            with contextlib.redirect_stdout(tee):
                result = original_nms(*positional, **keywords)
            if "NMS time limit" in tee.getvalue():
                raise RuntimeError("Original NMS timeout: incomplete evaluation rejected")
            return result

        def capture_metric(tp, conf, pred_cls, target_cls, *positional, **keywords):
            if args.mode == "canary":
                raise RuntimeError("Canary must stop before aggregate AP computation")
            if len(target_cls) != 7931:
                raise RuntimeError("Final metric GT denominator differs from frozen 7931")
            np.savez(output / "author_metric_inputs.npz", tp=tp, conf=conf,
                     pred_cls=pred_cls, target_cls=target_cls)
            result = original_metric(tp, conf, pred_cls, target_cls, *positional, **keywords)
            ap = result[2]
            captured["metrics"] = {"AP50": float(ap[:, 0].mean()), "AP75": float(ap[:, 5].mean()),
                                   "AP50_95": float(ap.mean()), "ap_at_each_iou": ap.tolist(),
                                   "predictions": len(conf), "gt": len(target_cls)}
            return result

        author_val.create_dataloader = checked_loader
        author_val.non_max_suppression = checked_nms
        author_val.ap_per_class = capture_metric
        if args.mode == "evaluate":
            canary = load_json(protocol["canary_receipt"])
            if canary.get("status") != "CANARY_PASS" or canary.get("ap_computed") is not False:
                raise ValueError("Full evaluation requires its completed two-batch no-AP canary receipt")
            for key in (*frozen, "weights", "modality", "data"):
                if canary["protocol"].get(key) != protocol.get(key):
                    raise ValueError(f"Canary protocol differs on {key}; measure this exact configuration first")
        try:
            result = author_val.run(data=protocol["data"], weights=str(weights), batch_size=32,
                                imgsz=1280, conf_thres=.001, iou_thres=.6, task="val",
                                # Original select_device('0') rewrites CUDA_VISIBLE_DEVICES.
                                # Its CLI-default empty string preserves the already-bound
                                # single GPU and returns the same logical cuda:0.
                                device="", half=False, augment=False, single_cls=False,
                                save_txt=args.mode == "evaluate", save_conf=True, save_hybrid=False, save_json=False,
                                plots=False, project=output, name="author_output", exist_ok=False)
        except CanaryComplete:
            if args.mode != "canary" or len(canary_batches) != 2 or any(x["shape"][0] != 32 for x in canary_batches):
                raise RuntimeError("Canary stop did not represent two complete B32 batches")
            write_json(output / "receipt.json", dict(base, status="CANARY_PASS", protocol=protocol,
                       batches=2, images=64, ap_computed=False, new_forward=True,
                       torch_max_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
                       torch_max_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20,
                       observed_lease_resources=resource_record(),
                       elapsed_seconds=time.perf_counter()-started, canary_batches=canary_batches))
            print("CANARY_PASS: original B32/1280/FP32 loop completed two batches; no aggregate AP")
            return
        if "metrics" not in captured or "roster" not in captured:
            raise RuntimeError("Author evaluation did not produce complete captured metrics")
        if source_sizes(source) != source_before:
            raise RuntimeError("Original source sizes changed during evaluation")
        write_json(output / "receipt.json", dict(base, status="COMPLETED", protocol=protocol,
                   metrics=captured["metrics"], author_return=result, elapsed_seconds=time.perf_counter()-started,
                   source_sizes_unchanged=True, ap_computed=True,
                   torch_max_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
                   torch_max_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20,
                   observed_lease_resources=resource_record()))
        print(json.dumps({"status": "COMPLETED", "metrics": captured["metrics"]}, ensure_ascii=False))
    except Exception as exc:
        detail = dict(base, status="FAILED", error=repr(exc),
                      elapsed_seconds=time.perf_counter()-started, cuda_initialized=torch.cuda.is_initialized())
        if args.mode != "identity":
            detail.update(torch_max_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,
                          torch_max_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20,
                          observed_lease_resources=resource_record())
        write_json(output / "failure.json", detail)
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
