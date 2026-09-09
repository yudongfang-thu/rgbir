#!/usr/bin/env python3
"""Create an immutable, self-contained receipt for one JSTARS experiment run."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPOSURE_LEDGER = REPO_ROOT / "configs" / "research" / "jstars_dataset_exposure_v1.json"
METHOD_IDENTITIES = {
    "AUTHOR-EXACT",
    "PAPER-RECONSTRUCTED",
    "THIRD-PARTY-RECONSTRUCTED",
    "PROTOCOL-ADAPTED",
    "PAPER_FAITHFUL",
    "PAPER_FAITHFUL_PATCHED",
    "ADAPTED",
    "ADAPTED-PARTIAL",
    "HNEWA-INSPIRED",
    "NATIVE",
    "PROPOSED",
    "MATCHED-NULL",
}


class ReceiptError(ValueError):
    """Raised when a run receipt would be incomplete or ambiguous."""


def _load_json(path: Path, expected: type, label: str) -> Any:
    if not path.is_file():
        raise ReceiptError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, expected):
        raise ReceiptError(f"{label} must contain {expected.__name__}: {path}")
    return value


def _require_files(paths: Iterable[Path], label: str) -> list[Path]:
    values = [Path(path).resolve() for path in paths]
    if not values:
        raise ReceiptError(f"at least one {label} file is required")
    missing = [str(path) for path in values if not path.is_file()]
    if missing:
        raise ReceiptError(f"missing {label} files: {missing}")
    return values


def _require_existing_optional(paths: Iterable[Path], label: str) -> list[Path]:
    values = [Path(path).resolve() for path in paths]
    missing = [str(path) for path in values if not path.is_file()]
    if missing:
        raise ReceiptError(f"missing {label} files: {missing}")
    return values


def _copy_group(paths: list[Path], destination: Path, *, relative_to: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=False)
    copied = []
    for index, source in enumerate(paths, start=1):
        target = destination / f"{index:02d}_{source.name}"
        shutil.copy2(source, target)
        copied.append(str(target.relative_to(relative_to)))
    return copied


def implementation_files(*values: Any) -> list[Path]:
    """Resolve the Python files that actually implement supplied objects."""

    paths: list[Path] = []
    for value in values:
        if value is None:
            continue
        target = value if inspect.isclass(value) or inspect.isfunction(value) else type(value)
        raw = inspect.getsourcefile(target)
        if raw is None:
            continue
        path = Path(raw).resolve()
        if path.is_file() and path not in paths:
            paths.append(path)
    if not paths:
        raise ReceiptError("could not resolve any implementation source files")
    return paths


def canonical_dataset(dataset: str) -> str:
    """Map run-specific dataset labels to the frozen exposure-ledger identity."""

    value = dataset.strip().lower()
    for prefix, canonical in (
        ("spacenet6", "spacenet6"),
        ("vedai", "vedai"),
        ("dronevehicle", "dronevehicle"),
        ("llvip", "llvip"),
        ("stf", "stf"),
    ):
        if value.startswith(prefix):
            return canonical
    return value


def load_test_exposure(dataset: str, ledger_path: Path = DEFAULT_EXPOSURE_LEDGER) -> dict[str, Any]:
    """Return one validated frozen exposure row for a run dataset."""

    from tools.validate_jstars_data_contract import validate_exposure_ledger

    ledger = _load_json(ledger_path, dict, "test exposure ledger")
    report = validate_exposure_ledger(ledger)
    if report.get("status") != "PASS":
        raise ReceiptError(f"test exposure ledger failed validation: {report.get('issue_counts')}")
    expected = canonical_dataset(dataset)
    matches = [
        dict(row)
        for row in ledger.get("datasets", [])
        if canonical_dataset(str(row.get("dataset", ""))) == expected
    ]
    if len(matches) != 1:
        raise ReceiptError(f"need exactly one exposure row for {expected}, found {len(matches)}")
    return matches[0]


def bound_resource_record() -> tuple[str, dict[str, Any]]:
    """Read the live bound lease and expose its observed peak resource record."""

    from tools.project_resource_guard import (
        LEASE_ID_ENV,
        bound_lease_resource_record_from_environment,
        require_bound_lease_from_environment,
    )

    lease = require_bound_lease_from_environment()
    resources = bound_lease_resource_record_from_environment()
    resources["lease_id"] = os.environ.get(LEASE_ID_ENV, "")
    return str(lease["job_id"]), resources


def emit_bound_run_receipt(
    *,
    run_dir: Path,
    method_identity: str,
    dataset: str,
    data_role: str,
    seed: int,
    run_kind: str,
    trainers: list[Path],
    losses: list[Path],
    configs: list[Path],
    split_rosters: list[Path],
    metric_files: list[Path],
    argv: list[str] | None = None,
    environment: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    exposure_ledger: Path = DEFAULT_EXPOSURE_LEDGER,
) -> dict[str, Any]:
    """Emit the full receipt required for a successful lease-owned run."""

    job_id, resources = bound_resource_record()
    return build_receipt(
        run_dir=run_dir,
        job_id=job_id,
        method_identity=method_identity,
        dataset=dataset,
        data_role=data_role,
        seed=seed,
        status="COMPLETED",
        trainers=trainers,
        losses=losses,
        configs=configs,
        split_rosters=split_rosters,
        metric_files=metric_files,
        argv=argv or [sys.executable, *sys.argv],
        environment=environment or {},
        resources=resources,
        test_exposure=load_test_exposure(dataset, exposure_ledger),
        run_kind=run_kind,
        inputs=inputs or {},
    )


def validate_persisted_receipt(
    receipt_path: Path,
    *,
    job_id: str,
    dataset: str,
    seed: int,
    run_kind: str,
) -> dict[str, Any]:
    """Validate the full on-disk receipt before a queue marks a job complete."""

    receipt = _load_json(receipt_path, dict, "run receipt")
    expected = {
        "schema": "jstars-run-receipt-v1",
        "terminal_status": "COMPLETED",
        "job_id": job_id,
        "run_kind": run_kind,
    }
    drift = {key: receipt.get(key) for key, value in expected.items() if receipt.get(key) != value}
    if drift:
        raise ReceiptError(f"run receipt identity/status mismatch: {drift}")
    if canonical_dataset(str(receipt.get("dataset", ""))) != canonical_dataset(dataset):
        raise ReceiptError("run receipt dataset mismatch")
    if int(receipt.get("seed", -1)) != int(seed):
        raise ReceiptError("run receipt seed mismatch")
    if receipt.get("method_identity") not in METHOD_IDENTITIES:
        raise ReceiptError("run receipt method identity is missing or unsupported")
    command = receipt.get("command", {}).get("argv")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
        raise ReceiptError("run receipt command is incomplete")
    if not isinstance(receipt.get("environment"), dict):
        raise ReceiptError("run receipt environment is missing")
    resources = receipt.get("resources")
    required_resources = {"gpu_ids", "cuda_pid_counts", "per_gpu_peak_vram_mib", "peak_rss_mib"}
    if not isinstance(resources, dict) or required_resources - set(resources):
        raise ReceiptError("run receipt resource record is incomplete")
    exposure = receipt.get("test_exposure")
    if (
        not isinstance(exposure, dict)
        or canonical_dataset(str(exposure.get("dataset", ""))) != canonical_dataset(dataset)
        or "test_status" not in exposure
        or "confirmatory" not in exposure
    ):
        raise ReceiptError("run receipt exposure binding is incomplete")
    snapshots = receipt.get("source_snapshots")
    if not isinstance(snapshots, dict):
        raise ReceiptError("run receipt source snapshots are missing")
    for group in ("trainer", "config", "split_roster"):
        if not isinstance(snapshots.get(group), list) or not snapshots[group]:
            raise ReceiptError(f"run receipt {group} snapshots are missing")
    if run_kind == "train" and (not isinstance(snapshots.get("loss"), list) or not snapshots["loss"]):
        raise ReceiptError("training run receipt loss snapshots are missing")
    metrics = receipt.get("metric_snapshots")
    if not isinstance(metrics, list) or not metrics:
        raise ReceiptError("completed run receipt metric snapshots are missing")
    relative_paths = [value for values in snapshots.values() for value in values] + metrics
    if not all(isinstance(value, str) and (receipt_path.parent / value).is_file() for value in relative_paths):
        raise ReceiptError("run receipt references a missing evidence snapshot")
    return receipt


def build_receipt(
    *,
    run_dir: Path,
    job_id: str,
    method_identity: str,
    dataset: str,
    data_role: str,
    seed: int,
    status: str,
    trainers: list[Path],
    losses: list[Path],
    configs: list[Path],
    split_rosters: list[Path],
    metric_files: list[Path],
    argv: list[str],
    environment: dict[str, Any],
    resources: dict[str, Any],
    test_exposure: dict[str, Any],
    run_kind: str = "train",
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and snapshot the evidence required by the frozen run contract."""

    if run_dir.exists():
        raise ReceiptError(f"refusing to overwrite run receipt directory: {run_dir}")
    if not job_id.strip() or not dataset.strip() or not data_role.strip():
        raise ReceiptError("job_id, dataset, and data_role must be non-empty")
    if method_identity not in METHOD_IDENTITIES:
        raise ReceiptError(f"unsupported method identity: {method_identity}")
    if status not in {"COMPLETED", "FAILED", "INTERRUPTED"}:
        raise ReceiptError(f"unsupported terminal status: {status}")
    if run_kind not in {"train", "eval", "feature"}:
        raise ReceiptError(f"unsupported run kind: {run_kind}")
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise ReceiptError("argv must be a non-empty list of non-empty strings")

    required_resource_fields = {
        "gpu_ids",
        "cuda_pid_counts",
        "per_gpu_peak_vram_mib",
        "peak_rss_mib",
    }
    missing_resources = sorted(required_resource_fields - set(resources))
    if missing_resources:
        raise ReceiptError(f"resource record is missing fields: {missing_resources}")
    if canonical_dataset(str(test_exposure.get("dataset", ""))) != canonical_dataset(dataset):
        raise ReceiptError("test exposure row does not match the run dataset")
    if "test_status" not in test_exposure or "confirmatory" not in test_exposure:
        raise ReceiptError("test exposure row lacks test_status or confirmatory")

    groups = {
        "trainer": _require_files(trainers, "trainer"),
        "loss": _require_files(losses, "loss") if run_kind == "train" else _require_existing_optional(losses, "loss"),
        "config": _require_files(configs, "config"),
        "split_roster": _require_files(split_rosters, "split roster"),
    }
    metrics = [Path(path).resolve() for path in metric_files]
    if status == "COMPLETED":
        metrics = _require_files(metrics, "metric")
    elif any(not path.is_file() for path in metrics):
        raise ReceiptError("every supplied metric file must exist")

    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        snapshots = {}
        for name, paths in groups.items():
            snapshots[name] = (
                _copy_group(paths, run_dir / "source_snapshot" / name, relative_to=run_dir)
                if paths
                else []
            )
        metric_snapshots = (
            _copy_group(metrics, run_dir / "metric_snapshot", relative_to=run_dir)
            if metrics
            else []
        )
        receipt = {
            "schema": "jstars-run-receipt-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "run_kind": run_kind,
            "method_identity": method_identity,
            "dataset": dataset.lower(),
            "data_role": data_role,
            "seed": int(seed),
            "terminal_status": status,
            "command": {"argv": argv},
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                **environment,
            },
            "source_snapshots": snapshots,
            "metric_snapshots": metric_snapshots,
            "resources": resources,
            "test_exposure": test_exposure,
            "inputs": inputs or {},
        }
        receipt_path = run_dir / "run_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--method-identity", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-role", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--status", required=True)
    parser.add_argument("--run-kind", choices=("train", "eval", "feature"), default="train")
    parser.add_argument("--trainer", required=True, action="append", type=Path)
    parser.add_argument("--loss", action="append", default=[], type=Path)
    parser.add_argument("--config", required=True, action="append", type=Path)
    parser.add_argument("--split-roster", required=True, action="append", type=Path)
    parser.add_argument("--metric", action="append", default=[], type=Path)
    parser.add_argument("--argv-json", required=True, type=Path)
    parser.add_argument("--environment-json", required=True, type=Path)
    parser.add_argument("--resource-json", required=True, type=Path)
    parser.add_argument("--test-exposure-json", required=True, type=Path)
    parser.add_argument("--inputs-json", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = build_receipt(
            run_dir=args.run_dir,
            job_id=args.job_id,
            method_identity=args.method_identity,
            dataset=args.dataset,
            data_role=args.data_role,
            seed=args.seed,
            status=args.status,
            trainers=args.trainer,
            losses=args.loss,
            configs=args.config,
            split_rosters=args.split_roster,
            metric_files=args.metric,
            argv=_load_json(args.argv_json, list, "argv JSON"),
            environment=_load_json(args.environment_json, dict, "environment JSON"),
            resources=_load_json(args.resource_json, dict, "resource JSON"),
            test_exposure=_load_json(args.test_exposure_json, dict, "test exposure JSON"),
            run_kind=args.run_kind,
            inputs=(
                _load_json(args.inputs_json, dict, "inputs JSON")
                if args.inputs_json is not None
                else {}
            ),
        )
    except (ReceiptError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "receipt": str((args.run_dir / "run_receipt.json").resolve()),
                "terminal_status": receipt["terminal_status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
