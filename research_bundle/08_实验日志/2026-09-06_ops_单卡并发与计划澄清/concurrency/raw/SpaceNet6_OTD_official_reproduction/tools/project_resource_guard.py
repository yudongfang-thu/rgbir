#!/usr/bin/env python3
"""Project-wide GPU/host-memory lease guard.

Every launcher in this project should acquire a lease before starting CUDA
work.  The lease file is shared by queues and agents on the same checkout, so
the limits apply to the project rather than to one terminal.
"""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEASE_FILE = REPO_ROOT / "runs" / ".project_resource_leases.json"

MAX_ACTIVE_GPUS = 4  # 2026-09-06 放宽: 服务器空闲且占用后仍剩>=2空卡时最多4卡(AGENTS.md §2.1)
MAX_CUDA_PROCESSES_PER_GPU = 2
PROJECT_VRAM_FRACTION = 0.70
HOST_RSS_ADMISSION_MIB = 240 * 1024
HOST_RSS_HARD_MIB = 300 * 1024
DEFAULT_FREE_SAFETY_MIB = 4096
DEFAULT_TRAIN_RSS_MIB = 32 * 1024
DEFAULT_LIGHT_RSS_MIB = 8 * 1024
PENDING_LEASE_SECONDS = 120
LEASE_ID_ENV = "JSTARS_RESOURCE_LEASE_ID"
LEASE_FILE_ENV = "JSTARS_RESOURCE_LEASE_FILE"
LEASE_GPUS_ENV = "JSTARS_RESOURCE_GPU_IDS"


class ResourceGuardError(RuntimeError):
    pass


class ResourceUnavailable(ResourceGuardError):
    def __init__(self, reasons: Sequence[str]):
        self.reasons = tuple(str(reason) for reason in reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class ResourceRequest:
    job_id: str
    kind: str
    candidate_gpus: tuple[int, ...]
    expected_vram_mib: int
    expected_rss_mib: int
    gpu_count: int = 1
    cuda_processes_per_gpu: int = 1
    formal_train: bool | None = None
    profiled_second_train: bool = False
    free_safety_mib: int = DEFAULT_FREE_SAFETY_MIB

    def validate(self) -> None:
        if not self.job_id:
            raise ResourceGuardError("job_id must be non-empty")
        if self.kind not in {"train", "eval", "feature", "ddp", "cpu", "other"}:
            raise ResourceGuardError(f"unsupported resource kind: {self.kind}")
        if len(set(self.candidate_gpus)) != len(self.candidate_gpus):
            raise ResourceGuardError("candidate_gpus must contain distinct GPU indexes")
        if any(gpu < 0 for gpu in self.candidate_gpus):
            raise ResourceGuardError("GPU indexes must be non-negative")
        if self.gpu_count < 0 or self.gpu_count > len(self.candidate_gpus):
            raise ResourceGuardError("gpu_count must fit within candidate_gpus")
        if self.gpu_count == 0:
            if self.candidate_gpus or self.cuda_processes_per_gpu != 0 or self.expected_vram_mib != 0:
                raise ResourceGuardError("CPU-only leases require no GPU candidates, CUDA processes, or VRAM")
            if self.kind in {"train", "eval", "ddp"}:
                raise ResourceGuardError("train/eval/ddp work requires at least one GPU")
        elif not self.candidate_gpus or self.cuda_processes_per_gpu <= 0 or self.expected_vram_mib <= 0:
            raise ResourceGuardError("GPU leases require candidates, positive CUDA processes, and positive VRAM")
        if self.expected_rss_mib <= 0:
            raise ResourceGuardError("expected RSS must be positive")
        if self.free_safety_mib < 0:
            raise ResourceGuardError("free_safety_mib cannot be negative")


def gpu_snapshot() -> dict[int, dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    rows: dict[int, dict[str, int]] = {}
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4 or not all(value.isdecimal() for value in values):
            continue
        index, free, used, total = (int(value) for value in values)
        rows[index] = {
            "memory_free_mib": free,
            "memory_used_mib": used,
            "memory_total_mib": total,
        }
    if not rows:
        raise ResourceGuardError("nvidia-smi returned no usable GPU rows")
    return rows


def gpu_process_snapshot() -> list[dict[str, int]]:
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    gpu_for_uuid = {
        pieces[1]: int(pieces[0])
        for line in gpu_rows.splitlines()
        if len(pieces := [piece.strip() for piece in line.split(",", 1)]) == 2 and pieces[0].isdecimal()
    }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
    except subprocess.CalledProcessError as exc:
        if "No running processes found" in str(exc.output):
            return []
        raise ResourceGuardError(f"nvidia-smi process query failed: {exc.output}") from exc
    rows: list[dict[str, int]] = []
    for line in output.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 3 or pieces[0] not in gpu_for_uuid or not pieces[1].isdecimal():
            continue
        memory = pieces[2].split()[0]
        if not memory.isdecimal():
            continue
        rows.append({"gpu": gpu_for_uuid[pieces[0]], "pid": int(pieces[1]), "used_mib": int(memory)})
    return rows


def process_snapshot() -> dict[int, dict[str, int]]:
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,rss=,state="],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    rows: dict[int, dict[str, int]] = {}
    for line in output.splitlines():
        pieces = line.split()
        if len(pieces) != 4 or not all(piece.isdecimal() for piece in pieces[:3]) or pieces[3].startswith("Z"):
            continue
        pid, ppid, rss_kib = (int(piece) for piece in pieces[:3])
        rows[pid] = {"ppid": ppid, "rss_mib": (rss_kib + 1023) // 1024}
    return rows


def _normalize_gpu_rows(rows: Mapping[int | str, Mapping[str, Any]]) -> dict[int, dict[str, int]]:
    normalized: dict[int, dict[str, int]] = {}
    for raw_index, row in rows.items():
        index = int(raw_index)
        free = row.get("memory_free_mib", row.get("free_mib"))
        total = row.get("memory_total_mib", row.get("total_mib"))
        if free is None or total is None:
            raise ResourceGuardError(f"GPU {index} snapshot lacks free/total memory")
        normalized[index] = {"memory_free_mib": int(free), "memory_total_mib": int(total)}
    return normalized


def _descendants(roots: set[int], processes: Mapping[int, Mapping[str, int]]) -> set[int]:
    members = {pid for pid in roots if pid in processes}
    changed = True
    while changed:
        changed = False
        for pid, row in processes.items():
            if pid not in members and int(row["ppid"]) in members:
                members.add(pid)
                changed = True
    return members


def _materialized_job_tree(
    lease: Mapping[str, Any], processes: Mapping[int, Mapping[str, int]]
) -> set[int]:
    job_pid = lease.get("job_pid")
    if job_pid is None:
        return set()
    roots = {int(job_pid)} | {int(pid) for pid in lease.get("observed_cuda_pids", [])}
    return _descendants(roots, processes)


class ProjectResourceGuard:
    def __init__(
        self,
        lease_file: Path,
        *,
        process_sampler: Callable[[], dict[int, dict[str, int]]] = process_snapshot,
        gpu_process_sampler: Callable[[], list[dict[str, int]]] = gpu_process_snapshot,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.lease_file = Path(lease_file)
        self.process_sampler = process_sampler
        self.gpu_process_sampler = gpu_process_sampler
        self.clock = clock

    @staticmethod
    def _new_state() -> dict[str, Any]:
        return {
            "schema": "jstars-project-resource-leases-v1",
            "policy": {
                "max_active_gpus": MAX_ACTIVE_GPUS,
                "max_cuda_processes_per_gpu": MAX_CUDA_PROCESSES_PER_GPU,
                "project_vram_fraction_lt": PROJECT_VRAM_FRACTION,
                "host_rss_admission_mib": HOST_RSS_ADMISSION_MIB,
                "host_rss_hard_mib": HOST_RSS_HARD_MIB,
                "formal_trains_per_gpu_default": 1,
            },
            "leases": {},
        }

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        self.lease_file.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.lease_file.with_suffix(self.lease_file.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.lease_file.is_file():
                state = json.loads(self.lease_file.read_text(encoding="utf-8"))
                if state.get("schema") != "jstars-project-resource-leases-v1":
                    raise ResourceGuardError(f"unsupported lease schema: {self.lease_file}")
            else:
                state = self._new_state()
            yield state
            temporary = self.lease_file.with_name(f".{self.lease_file.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(self.lease_file)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _refresh_leases(
        self,
        state: dict[str, Any],
        processes: Mapping[int, Mapping[str, int]],
        gpu_processes: Sequence[Mapping[str, int]],
    ) -> None:
        now = self.clock()
        live_gpu_pids = {int(row["pid"]) for row in gpu_processes}
        stale: list[str] = []
        for lease_id, lease in state["leases"].items():
            peak_vram = {
                str(gpu): int(value)
                for gpu, value in lease.get("per_gpu_peak_vram_mib", {}).items()
            }
            peak_cuda_pids = {
                str(gpu): int(value)
                for gpu, value in lease.get("peak_cuda_pid_counts", {}).items()
            }
            for gpu in lease["gpus"]:
                peak_vram.setdefault(str(gpu), 0)
                peak_cuda_pids.setdefault(str(gpu), 0)
            lease["per_gpu_peak_vram_mib"] = peak_vram
            lease["peak_cuda_pid_counts"] = peak_cuda_pids
            lease["peak_rss_mib"] = int(lease.get("peak_rss_mib", 0))
            job_pid = lease.get("job_pid")
            if job_pid is None:
                owner_alive = int(lease["owner_pid"]) in processes
                if not owner_alive or now - float(lease["created_at"]) >= PENDING_LEASE_SECONDS:
                    stale.append(lease_id)
                continue
            tree = _materialized_job_tree(lease, processes)
            observed = {
                int(row["pid"])
                for row in gpu_processes
                if int(row["pid"]) in tree and int(row["gpu"]) in {int(gpu) for gpu in lease["gpus"]}
            }
            observed |= set(lease.get("observed_cuda_pids", [])) & live_gpu_pids
            lease["observed_cuda_pids"] = sorted(observed)
            tree = _materialized_job_tree(lease, processes)
            tracked_pids = tree | observed
            lease["peak_rss_mib"] = max(
                int(lease["peak_rss_mib"]),
                sum(int(processes[pid]["rss_mib"]) for pid in tree),
            )
            current_vram: dict[int, int] = {}
            current_cuda_pids: dict[int, set[int]] = {}
            declared_gpus = {int(gpu) for gpu in lease["gpus"]}
            for row in gpu_processes:
                pid = int(row["pid"])
                gpu = int(row["gpu"])
                if pid not in tracked_pids or gpu not in declared_gpus:
                    continue
                current_vram[gpu] = current_vram.get(gpu, 0) + int(row["used_mib"])
                current_cuda_pids.setdefault(gpu, set()).add(pid)
            for gpu in declared_gpus:
                key = str(gpu)
                peak_vram[key] = max(peak_vram[key], current_vram.get(gpu, 0))
                peak_cuda_pids[key] = max(peak_cuda_pids[key], len(current_cuda_pids.get(gpu, set())))
            if int(job_pid) not in processes and not observed:
                stale.append(lease_id)
        for lease_id in stale:
            state["leases"].pop(lease_id, None)

    @staticmethod
    def _usage(
        state: Mapping[str, Any],
        processes: Mapping[int, Mapping[str, int]],
        gpu_processes: Sequence[Mapping[str, int]],
    ) -> dict[str, Any]:
        leases = list(state["leases"].values())
        reserved_vram: dict[int, int] = {}
        reserved_cuda_pids: dict[int, int] = {}
        effective_vram: dict[int, int] = {}
        effective_cuda_pids: dict[int, int] = {}
        formal_trains: dict[int, int] = {}
        lease_trees: list[tuple[Mapping[str, Any], set[int], set[int]]] = []
        for lease in leases:
            job_pid = lease.get("job_pid")
            if job_pid is None:
                owner = int(lease["owner_pid"])
                materialized_tree: set[int] = set()
                tree = {owner} if owner in processes else set()
            else:
                materialized_tree = _materialized_job_tree(lease, processes)
                tree = set(materialized_tree)
                owner = int(lease["owner_pid"])
                if owner in processes:
                    tree.add(owner)
            lease_trees.append((lease, tree, materialized_tree))
            for gpu in lease["gpus"]:
                index = int(gpu)
                reserved_vram[index] = reserved_vram.get(index, 0) + int(lease["expected_vram_mib"])
                reserved_cuda_pids[index] = reserved_cuda_pids.get(index, 0) + int(
                    lease["cuda_processes_per_gpu"]
                )
                if lease.get("formal_train"):
                    formal_trains[index] = formal_trains.get(index, 0) + 1

        project_processes = {pid for _lease, tree, _materialized in lease_trees for pid in tree}
        actual_vram: dict[int, int] = {}
        actual_cuda_pids: dict[int, set[int]] = {}
        for row in gpu_processes:
            pid = int(row["pid"])
            if pid not in project_processes:
                continue
            gpu = int(row["gpu"])
            actual_vram[gpu] = actual_vram.get(gpu, 0) + int(row["used_mib"])
            actual_cuda_pids.setdefault(gpu, set()).add(pid)

        for lease, tree, _materialized in lease_trees:
            lease_vram: dict[int, int] = {}
            lease_cuda_pids: dict[int, set[int]] = {}
            for row in gpu_processes:
                pid = int(row["pid"])
                if pid not in tree:
                    continue
                gpu = int(row["gpu"])
                lease_vram[gpu] = lease_vram.get(gpu, 0) + int(row["used_mib"])
                lease_cuda_pids.setdefault(gpu, set()).add(pid)
            declared_gpus = {int(gpu) for gpu in lease["gpus"]}
            for gpu in declared_gpus | set(lease_vram) | set(lease_cuda_pids):
                reserved_memory = int(lease["expected_vram_mib"]) if gpu in declared_gpus else 0
                reserved_processes = int(lease["cuda_processes_per_gpu"]) if gpu in declared_gpus else 0
                effective_vram[gpu] = effective_vram.get(gpu, 0) + max(
                    reserved_memory, lease_vram.get(gpu, 0)
                )
                effective_cuda_pids[gpu] = effective_cuda_pids.get(gpu, 0) + max(
                    reserved_processes, len(lease_cuda_pids.get(gpu, set()))
                )
        project_rss_mib = sum(int(processes[pid]["rss_mib"]) for pid in project_processes)
        reserved_rss_mib = sum(int(lease["expected_rss_mib"]) for lease in leases)
        unmaterialized_rss_mib = sum(
            max(
                0,
                int(lease["expected_rss_mib"])
                - sum(int(processes[pid]["rss_mib"]) for pid in materialized_tree),
            )
            for lease, _tree, materialized_tree in lease_trees
        )
        effective_rss_mib = project_rss_mib + unmaterialized_rss_mib
        leased_gpus = {int(gpu) for lease in leases for gpu in lease["gpus"]}
        return {
            "active_gpus": sorted(leased_gpus | set(actual_vram) | set(actual_cuda_pids)),
            "reserved_vram_mib": reserved_vram,
            "reserved_cuda_processes": reserved_cuda_pids,
            "effective_vram_mib": effective_vram,
            "effective_cuda_processes": effective_cuda_pids,
            "formal_trains": formal_trains,
            "actual_vram_mib": actual_vram,
            "actual_cuda_pids": {gpu: len(pids) for gpu, pids in actual_cuda_pids.items()},
            "project_rss_mib": project_rss_mib,
            "reserved_rss_mib": reserved_rss_mib,
            "unmaterialized_rss_mib": unmaterialized_rss_mib,
            "effective_rss_mib": effective_rss_mib,
        }

    def acquire(
        self,
        request: ResourceRequest,
        *,
        gpus: Mapping[int | str, Mapping[str, Any]],
        owner_pid: int | None = None,
    ) -> dict[str, Any]:
        request.validate()
        formal_train = request.formal_train
        if formal_train is None:
            formal_train = request.kind in {"train", "ddp"}
        gpu_rows = _normalize_gpu_rows(gpus)
        owner = os.getpid() if owner_pid is None else int(owner_pid)
        missing = sorted(set(request.candidate_gpus) - set(gpu_rows))
        if missing:
            raise ResourceGuardError(f"GPU snapshot lacks candidates: {missing}")

        with self._locked_state() as state:
            # Sample after taking the state lock. Otherwise another launcher can
            # create or bind a lease between the snapshot and this refresh, and
            # the stale snapshot can incorrectly reap that new live lease.
            processes = self.process_sampler()
            gpu_processes = self.gpu_process_sampler()
            if owner not in processes:
                raise ResourceGuardError(f"owner PID {owner} is not live")
            self._refresh_leases(state, processes, gpu_processes)
            if any(lease["job_id"] == request.job_id for lease in state["leases"].values()):
                raise ResourceGuardError(f"job already owns a lease: {request.job_id}")
            usage = self._usage(state, processes, gpu_processes)
            effective_rss = int(usage["effective_rss_mib"])
            if effective_rss >= HOST_RSS_HARD_MIB:
                raise ResourceUnavailable([f"project RSS {effective_rss} MiB reached the 300 GiB hard limit"])
            if effective_rss + request.expected_rss_mib >= HOST_RSS_ADMISSION_MIB:
                raise ResourceUnavailable(
                    [
                        f"project RSS reservation would reach {effective_rss + request.expected_rss_mib} MiB; "
                        "new work stops at 240 GiB"
                    ]
                )

            active_gpus = set(usage["active_gpus"])
            eligible: list[int] = []
            rejected: list[str] = []
            for gpu in request.candidate_gpus:
                row = gpu_rows[gpu]
                total = int(row["memory_total_mib"])
                free = int(row["memory_free_mib"])
                actual_vram = int(usage["actual_vram_mib"].get(gpu, 0))
                effective_vram = int(usage["effective_vram_mib"].get(gpu, 0))
                effective_slots = int(usage["effective_cuda_processes"].get(gpu, 0))
                unmaterialized_reservation = max(0, effective_vram - actual_vram)
                effective_free = free - unmaterialized_reservation
                reasons = []
                if effective_slots + request.cuda_processes_per_gpu > MAX_CUDA_PROCESSES_PER_GPU:
                    reasons.append("CUDA PID limit")
                if effective_vram + request.expected_vram_mib >= total * PROJECT_VRAM_FRACTION:
                    reasons.append("project VRAM would reach 70%")
                if effective_free <= request.expected_vram_mib + request.free_safety_mib:
                    reasons.append("actual free VRAM lacks requested memory plus safety margin")
                if (
                    formal_train
                    and int(usage["formal_trains"].get(gpu, 0)) >= 1
                    and not request.profiled_second_train
                ):
                    reasons.append("second formal train lacks an explicit passed profile")
                if reasons:
                    rejected.append(f"GPU {gpu}: {', '.join(reasons)}")
                else:
                    eligible.append(gpu)

            choices = [
                choice
                for choice in itertools.combinations(eligible, request.gpu_count)
                if len(active_gpus | set(choice)) <= MAX_ACTIVE_GPUS
            ]
            if not choices:
                if eligible:
                    rejected.append("selecting the requested GPUs would exceed 3 active physical GPUs")
                raise ResourceUnavailable(rejected or ["no eligible GPU"])

            def score(choice: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
                headroom = sum(
                    int(gpu_rows[gpu]["memory_free_mib"])
                    - int(usage["effective_vram_mib"].get(gpu, 0))
                    for gpu in choice
                )
                return headroom, tuple(-gpu for gpu in choice)

            selected = max(choices, key=score)
            lease_id = f"{request.job_id}-{uuid.uuid4().hex[:12]}"
            lease = {
                "lease_id": lease_id,
                "job_id": request.job_id,
                "kind": request.kind,
                "gpus": list(selected),
                "expected_vram_mib": request.expected_vram_mib,
                "expected_rss_mib": request.expected_rss_mib,
                "cuda_processes_per_gpu": request.cuda_processes_per_gpu,
                "formal_train": formal_train,
                "profiled_second_train": request.profiled_second_train,
                "free_safety_mib": request.free_safety_mib,
                "owner_pid": owner,
                "job_pid": None,
                "observed_cuda_pids": [],
                "peak_rss_mib": 0,
                "per_gpu_peak_vram_mib": {str(gpu): 0 for gpu in selected},
                "peak_cuda_pid_counts": {str(gpu): 0 for gpu in selected},
                "created_at": self.clock(),
                "admission": {
                    "active_gpus_after": sorted(active_gpus | set(selected)),
                    "project_rss_reserved_after_mib": effective_rss + request.expected_rss_mib,
                    "per_gpu": {
                        str(gpu): {
                            "project_cuda_processes_after": int(
                                usage["effective_cuda_processes"].get(gpu, 0)
                            ) + request.cuda_processes_per_gpu,
                            "project_vram_reserved_after_mib": int(
                                usage["effective_vram_mib"].get(gpu, 0)
                            ) + request.expected_vram_mib,
                        }
                        for gpu in selected
                    },
                },
            }
            state["leases"][lease_id] = lease
            return dict(lease)

    def bind(self, lease_id: str, job_pid: int) -> dict[str, Any]:
        with self._locked_state() as state:
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise ResourceGuardError(f"unknown lease: {lease_id}")
            lease["job_pid"] = int(job_pid)
            lease["bound_at"] = self.clock()
            return dict(lease)

    def release(self, lease_id: str) -> bool:
        with self._locked_state() as state:
            return state["leases"].pop(lease_id, None) is not None

    def validate_bound(self, lease_id: str, pid: int, expected_gpus: Sequence[int]) -> dict[str, Any]:
        processes = self.process_sampler()
        with self._locked_state() as state:
            lease = state["leases"].get(lease_id)
            if lease is None or lease.get("job_pid") is None:
                raise ResourceGuardError(f"lease is not bound: {lease_id}")
            tree = _materialized_job_tree(lease, processes)
            if int(pid) not in tree:
                raise ResourceGuardError(f"PID {pid} is not covered by lease {lease_id}")
            if [int(gpu) for gpu in lease["gpus"]] != [int(gpu) for gpu in expected_gpus]:
                raise ResourceGuardError(f"CUDA_VISIBLE_DEVICES does not match lease {lease_id}")
            return dict(lease)

    def bound_resource_record(
        self, lease_id: str, pid: int, expected_gpus: Sequence[int]
    ) -> dict[str, Any]:
        with self._locked_state() as state:
            processes = self.process_sampler()
            gpu_processes = self.gpu_process_sampler()
            self._refresh_leases(state, processes, gpu_processes)
            lease = state["leases"].get(lease_id)
            if lease is None or lease.get("job_pid") is None:
                raise ResourceGuardError(f"lease is not bound: {lease_id}")
            tree = _materialized_job_tree(lease, processes)
            if int(pid) not in tree:
                raise ResourceGuardError(f"PID {pid} is not covered by lease {lease_id}")
            if [int(gpu) for gpu in lease["gpus"]] != [int(gpu) for gpu in expected_gpus]:
                raise ResourceGuardError(f"CUDA_VISIBLE_DEVICES does not match lease {lease_id}")
            return {
                "gpu_ids": [int(gpu) for gpu in lease["gpus"]],
                "cuda_pid_counts": {
                    str(gpu): int(count)
                    for gpu, count in lease.get("peak_cuda_pid_counts", {}).items()
                },
                "per_gpu_peak_vram_mib": {
                    str(gpu): int(value)
                    for gpu, value in lease.get("per_gpu_peak_vram_mib", {}).items()
                },
                "peak_rss_mib": int(lease.get("peak_rss_mib", 0)),
            }

    def inspect(self, *, gpus: Mapping[int | str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        with self._locked_state() as state:
            processes = self.process_sampler()
            gpu_processes = self.gpu_process_sampler()
            self._refresh_leases(state, processes, gpu_processes)
            usage = self._usage(state, processes, gpu_processes)
            return {
                "policy": dict(state["policy"]),
                "leases": list(state["leases"].values()),
                "usage": usage,
                "gpus": _normalize_gpu_rows(gpus) if gpus is not None else None,
            }


def _bound_lease_environment() -> tuple[str, Path, tuple[int, ...]]:
    lease_id = os.environ.get(LEASE_ID_ENV)
    lease_file = os.environ.get(LEASE_FILE_ENV)
    gpu_text = os.environ.get(LEASE_GPUS_ENV)
    if not lease_id or not lease_file or gpu_text is None:
        raise ResourceGuardError(
            f"GPU work must run through project_resource_guard.py run or a lease-aware queue; "
            f"missing {LEASE_ID_ENV}/{LEASE_FILE_ENV}/{LEASE_GPUS_ENV}"
        )
    try:
        expected_gpus = tuple(int(value) for value in gpu_text.split(",") if value != "")
    except ValueError as exc:
        raise ResourceGuardError(f"invalid {LEASE_GPUS_ENV}: {gpu_text}") from exc
    return lease_id, Path(lease_file), expected_gpus


def require_bound_lease_from_environment(*, wait_seconds: float = 10.0) -> dict[str, Any]:
    lease_id, lease_file, expected_gpus = _bound_lease_environment()
    guard = ProjectResourceGuard(lease_file)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return guard.validate_bound(lease_id, os.getpid(), expected_gpus)
        except ResourceGuardError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def bound_lease_resource_record_from_environment(*, wait_seconds: float = 10.0) -> dict[str, Any]:
    """Return receipt-compatible peak resources for this process's bound lease."""

    lease_id, lease_file, expected_gpus = _bound_lease_environment()
    guard = ProjectResourceGuard(
        lease_file,
        process_sampler=process_snapshot,
        gpu_process_sampler=gpu_process_snapshot,
    )
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            return guard.bound_resource_record(lease_id, os.getpid(), expected_gpus)
        except ResourceGuardError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--kind", choices=("train", "eval", "feature", "ddp", "cpu", "other"), required=True)
    parser.add_argument("--candidate-gpu", type=int, action="append")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--expected-vram-mib", type=int, required=True)
    parser.add_argument("--expected-rss-mib", type=int, required=True)
    parser.add_argument("--cuda-processes-per-gpu", type=int, default=1)
    train_mode = parser.add_mutually_exclusive_group()
    train_mode.add_argument("--formal-train", action="store_true", dest="formal_train")
    train_mode.add_argument("--non-formal-train", action="store_false", dest="formal_train")
    parser.set_defaults(formal_train=None)
    parser.add_argument("--profiled-second-train", action="store_true")
    parser.add_argument("--free-safety-mib", type=int, default=DEFAULT_FREE_SAFETY_MIB)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease-file", type=Path, default=DEFAULT_LEASE_FILE)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect")
    acquire = commands.add_parser("acquire")
    _add_request_arguments(acquire)
    acquire.add_argument(
        "--owner-pid",
        type=int,
        required=True,
        help="live launcher PID that remains alive until this lease is bound or released",
    )
    bind = commands.add_parser("bind")
    bind.add_argument("--lease-id", required=True)
    bind.add_argument("--pid", required=True, type=int)
    release = commands.add_parser("release")
    release.add_argument("--lease-id", required=True)
    run = commands.add_parser("run")
    _add_request_arguments(run)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def _request_from_args(args: argparse.Namespace, rows: Mapping[int, Mapping[str, int]]) -> ResourceRequest:
    candidates = tuple(args.candidate_gpu if args.candidate_gpu is not None else sorted(rows))
    return ResourceRequest(
        job_id=args.job_id,
        kind=args.kind,
        candidate_gpus=candidates,
        expected_vram_mib=args.expected_vram_mib,
        expected_rss_mib=args.expected_rss_mib,
        gpu_count=args.gpu_count,
        cuda_processes_per_gpu=args.cuda_processes_per_gpu,
        formal_train=args.formal_train,
        profiled_second_train=args.profiled_second_train,
        free_safety_mib=args.free_safety_mib,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    guard = ProjectResourceGuard(args.lease_file)
    if args.command == "inspect":
        print(json.dumps(guard.inspect(gpus=gpu_snapshot()), indent=2, sort_keys=True))
        return 0
    if args.command == "bind":
        print(json.dumps({"status": "BOUND", "lease": guard.bind(args.lease_id, args.pid)}, sort_keys=True))
        return 0
    if args.command == "release":
        print(json.dumps({"status": "RELEASED", "released": guard.release(args.lease_id)}, sort_keys=True))
        return 0

    rows = {} if args.gpu_count == 0 else gpu_snapshot()
    try:
        lease = guard.acquire(
            _request_from_args(args, rows),
            gpus=rows,
            owner_pid=args.owner_pid if args.command == "acquire" else None,
        )
    except ResourceUnavailable as exc:
        print(json.dumps({"status": "QUEUED", "reasons": exc.reasons}, sort_keys=True))
        return 2
    if args.command == "acquire":
        print(json.dumps({"status": "ACQUIRED", "lease": lease}, sort_keys=True))
        return 0

    command = list(args.argv)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        guard.release(lease["lease_id"])
        raise ResourceGuardError("run requires a command after --")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in lease["gpus"])
    environment[LEASE_ID_ENV] = str(lease["lease_id"])
    environment[LEASE_FILE_ENV] = str(args.lease_file.resolve())
    environment[LEASE_GPUS_ENV] = ",".join(str(gpu) for gpu in lease["gpus"])
    process: subprocess.Popen[Any] | None = None

    def lease_has_cuda_children() -> bool:
        leases = {row["lease_id"]: row for row in guard.inspect()["leases"]}
        current = leases.get(lease["lease_id"])
        return bool(current and current.get("observed_cuda_pids"))

    try:
        # Bind to the already-live launcher before spawning the workload. The
        # workload is then covered as a descendant, eliminating the interval
        # where a concurrent inspector could reap an unbound lease.
        guard.bind(lease["lease_id"], os.getpid())
        process = subprocess.Popen(command, env=environment)
        print(
            json.dumps(
                {
                    "status": "LAUNCHED",
                    "lease_id": lease["lease_id"],
                    "gpu_ids": lease["gpus"],
                    "cuda_processes_per_gpu": lease["cuda_processes_per_gpu"],
                    "expected_vram_mib": lease["expected_vram_mib"],
                    "expected_rss_mib": lease["expected_rss_mib"],
                    "pid": process.pid,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        while process.poll() is None:
            guard.inspect()
            time.sleep(1.0)
        return int(process.returncode)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        try:
            cuda_children_live = lease_has_cuda_children()
        except ResourceGuardError:
            cuda_children_live = True
        if not cuda_children_live:
            guard.release(lease["lease_id"])
        else:
            print(
                json.dumps(
                    {
                        "status": "LEASE_RETAINED_FOR_CUDA_CHILDREN",
                        "lease_id": lease["lease_id"],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
