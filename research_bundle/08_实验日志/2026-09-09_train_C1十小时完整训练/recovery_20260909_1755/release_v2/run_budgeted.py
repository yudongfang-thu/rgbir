"""One C1 train+independent-evaluation run under the existing shared resource lease."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from budget_policy import BUDGET_SECONDS, require_e200, require_full_evaluation, write_json

REPO = Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction')
PYTHON = str(REPO/'environments/sn6-int8-kd/bin/python')
HERE = Path(__file__).resolve().parent


class RunStopped(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def stop_child(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def run_process(command, logfile, deadline, monitor, poll_seconds=1):
    started = time.monotonic()
    process = None
    try:
        if started >= deadline:
            raise RunStopped('BUDGET_ABORTED', 'No remaining time to start the next stage')
        with Path(logfile).open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=os.environ.copy())
            write_json(Path(logfile).with_suffix('.launch.json'), dict(command=command, pid=process.pid,
                       started_at=datetime.datetime.now().astimezone().isoformat(), deadline_monotonic=deadline))
            last_monitor = 0
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise RunStopped('BUDGET_ABORTED', 'Run wall-clock deadline reached')
                if time.monotonic()-last_monitor >= 5:
                    monitor()
                    last_monitor = time.monotonic()
                time.sleep(poll_seconds)
            elapsed = time.monotonic()-started
            if time.monotonic() >= deadline:
                raise RunStopped('BUDGET_ABORTED', 'Stage completed after its run deadline')
            monitor()
            return dict(command=command, exit_code=process.returncode, elapsed_seconds=elapsed, pid=process.pid)
    except BaseException:
        if process is not None:
            stop_child(process)
        raise


def utilization():
    result = subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu','--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=15)
    return {int(a): int(b) for a,b in (line.split(',') for line in result.stdout.splitlines())}


def main(args):
    sys.path.insert(0, str(REPO))
    from tools import project_resource_guard as resource
    # The shared file stays unchanged. Our own atomic requests obey max3.
    resource.MAX_ACTIVE_GPUS = 3
    guard = resource.ProjectResourceGuard(resource.DEFAULT_LEASE_FILE)
    args.output.mkdir(parents=True, exist_ok=False)
    queued = time.monotonic()
    lease = None
    started = None
    final = dict(status='PREPARING', seed=args.seed, budget_seconds=BUDGET_SECONDS,
                 run_kind='train_then_independent_dev', official_test_accessed=False, stages=[])
    write_json(args.output/'run_status.json', final)
    try:
        while lease is None:
            rows = resource.gpu_snapshot()
            state = guard.inspect(gpus=rows)
            busy = utilization()
            active = set(state['usage']['active_gpus'])
            candidates = [g for g in rows if len(active | {g}) <= 3 and
                          state['usage']['effective_cuda_processes'].get(g, 0) == 0 and
                          rows[g]['memory_used_mib'] < 100]
            candidates.sort(key=lambda g: (rows[g]['memory_used_mib'] >= 100, busy.get(g, 100),
                                           -rows[g]['memory_free_mib'], g))
            reasons = []
            for gpu in candidates:
                request = resource.ResourceRequest(job_id=f'c1budget_s{args.seed}_{args.output.name}', kind='train',
                    candidate_gpus=(gpu,), expected_vram_mib=8192, expected_rss_mib=32768,
                    formal_train=True, free_safety_mib=2048)
                try:
                    lease = guard.acquire(request, gpus=rows)
                    break
                except resource.ResourceUnavailable as error:
                    reasons.append(str(error))
            if lease is None:
                write_json(args.output/'run_status.json', dict(final, status='QUEUED',
                           queue_seconds=time.monotonic()-queued, reasons=reasons or ['No unused project GPU slot']))
                time.sleep(30)
        lease = guard.bind(lease['lease_id'], os.getpid())
        started = time.monotonic()
        deadline = started+BUDGET_SECONDS-10  # reserve bounded termination/receipt cleanup
        os.environ.update({resource.LEASE_ID_ENV: lease['lease_id'],
                           resource.LEASE_FILE_ENV: str(resource.DEFAULT_LEASE_FILE),
                           resource.LEASE_GPUS_ENV: ','.join(map(str, lease['gpus'])),
                           'CUDA_VISIBLE_DEVICES': ','.join(map(str, lease['gpus'])),
                           'C1_RUN_STARTED_MONOTONIC': str(started),
                           'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
        final.update(status='TRAINING', queue_seconds=started-queued, physical_gpu_ids=lease['gpus'],
                     started_at=datetime.datetime.now().astimezone().isoformat())
        write_json(args.output/'admission.json', dict(lease=lease, gpu_snapshot=rows, utilization=busy,
                   max_project_gpus=3, admission_mode='empty_card_after_shared_card_budget_failure',
                   measured_c1_nvml_peak_mib=7630, measured_c1_tree_rss_mib=28814))
        write_json(args.output/'run_status.json', final)

        def monitor():
            gpu_rows = resource.gpu_snapshot()
            snapshot = guard.inspect(gpus=gpu_rows)
            usage = snapshot['usage']
            violations = []
            if usage['project_rss_mib']*2**20 >= 300_000_000_000:
                violations.append('Project RSS reached300GB')
            if len(usage['active_gpus']) > 3:
                violations.append('Project active GPU count exceeded3')
            for gpu in lease['gpus']:
                row = gpu_rows[gpu]
                if row['memory_free_mib'] < 2048:
                    violations.append(f'GPU{gpu} free memory below2GiB')
                if usage['actual_vram_mib'].get(gpu, 0) >= .7*row['memory_total_mib']:
                    violations.append(f'GPU{gpu} project VRAM reached70%')
                if usage['actual_cuda_pids'].get(gpu, 0) > 2:
                    violations.append(f'GPU{gpu} project CUDA processes exceeded2')
            record = dict(elapsed_seconds=time.monotonic()-started, gpus={g:gpu_rows[g] for g in lease['gpus']},
                          project_rss_mib=usage['project_rss_mib'], violations=violations)
            with (args.output/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            if violations:
                raise RunStopped('RESOURCE_ABORTED', '; '.join(violations))

        train = args.output/'training'
        command = [PYTHON,'-B',str(HERE/'train_c1_budget.py'),'--seed',str(args.seed),'--output',str(train)]
        result = run_process(command, args.output/'training.log', deadline, monitor)
        final['stages'].append(dict(kind='train', **result))
        if result['exit_code']:
            if (train/'budget_stop.json').exists():
                raise RunStopped('BUDGET_ABORTED', 'Epoch-based total runtime projection exceeded10hours')
            raise RunStopped('TRAINING_FAILED', f'Training process exited{result["exit_code"]}')
        require_e200(train)
        final['status'] = 'EVALUATING'
        write_json(args.output/'run_status.json', final)
        command = [PYTHON,'-B',str(HERE/'evaluate_c1_budget.py'),'--run',str(train)]
        result = run_process(command, args.output/'evaluation.log', deadline, monitor)
        final['stages'].append(dict(kind='eval', **result))
        if result['exit_code']:
            raise RunStopped('EVALUATION_FAILED', f'Evaluation process exited{result["exit_code"]}')
        final.update(status='COMPLETED', evaluation=require_full_evaluation(train))
    except BaseException as error:
        final.update(status=getattr(error, 'status', 'TECHNICAL_FAILURE'), error=repr(error), traceback=traceback.format_exc())
    finally:
        if lease is not None:
            try:
                final['resources'] = guard.bound_resource_record(lease['lease_id'], os.getpid(), lease['gpus'])
            except Exception as error:
                final['resource_receipt_error'] = repr(error)
                if final['status'] == 'COMPLETED':
                    final['status'] = 'RESOURCE_RECEIPT_FAILED'
            finally:
                try:
                    guard.release(lease['lease_id'])
                except Exception as error:
                    final['lease_release_error'] = repr(error)
        final['run_seconds'] = time.monotonic()-started if started is not None else None
        final['finished_at'] = datetime.datetime.now().astimezone().isoformat()
        if final['status'] == 'COMPLETED' and final['run_seconds'] > BUDGET_SECONDS:
            final['status'] = 'COMPLETED_OVER_BUDGET'
        write_json(args.output/'run_status.json', final)
        write_json(args.output/'run_final.json', final)
        print(json.dumps(final), flush=True)
    return 0 if final['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0,42,123), required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
