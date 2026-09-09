"""Single-GPU stages under the existing project lease and one atomic admission.

This module never writes a lease schema itself. It calls the installed guard's
acquire while holding that guard's lock, using a view of the already locked state.
First measurements are short bootstrap stages without another project task on
the selected GPU. Other users may share it when the full memory margin fits.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = Path('/mnt/dataX/ydf/projects/RGBT_campaign_90')
PROFILE_SCHEMA = 'rgbir-independent-resource-profile-v1'
POLL_SECONDS = 30
MAX_SLOTS = 2
VRAM_FRACTION = .70
FREE_MIB = 2048
ADMISSION_RSS_MIB = 240 * 1024
HARD_RSS_BYTES = 300_000_000_000
BOOTSTRAP_CAP_MIB = 16000
BOOTSTRAP_STAGES = ('calibration', 'canary', 'compatibility', 'evaluation_profile')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def append_event(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def load_guard(repo):
    """Load the server's actual shared guard; no copied substitute or new pool."""
    path = Path(repo) / 'tools/project_resource_guard.py'
    spec = importlib.util.spec_from_file_location('_independent_actual_resource_guard', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ('ProjectResourceGuard', 'ResourceRequest', 'ResourceUnavailable',
                 'DEFAULT_LEASE_FILE', 'MAX_ACTIVE_GPUS', 'LEASE_ID_ENV',
                 'LEASE_FILE_ENV', 'LEASE_GPUS_ENV'):
        if not hasattr(module, name):
            raise RuntimeError('Shared guard API is missing ' + name)
    return module


def _mib(text):
    pieces = str(text).split()
    if not pieces or not pieces[0].isdecimal():
        raise ValueError('Unusable nvidia-smi memory field: ' + str(text))
    return int(pieces[0])


def parse_gpu_xml(text):
    """One XML snapshot includes compute AND graphics processes, not only CUDA."""
    root = ET.fromstring(text)
    rows = {}
    for index, node in enumerate(root.findall('gpu')):
        # minor_number is the physical CUDA index on the pinned 94 server.
        minor = node.findtext('minor_number')
        if minor is None or not minor.isdecimal():
            raise ValueError('Cannot resolve physical GPU index')
        gpu = int(minor)
        mem = node.find('fb_memory_usage')
        processes_node = node.find('processes')
        if mem is None or processes_node is None:
            raise ValueError('GPU snapshot lacks memory or all-process inventory')
        pids = []
        for proc in processes_node.findall('process_info'):
            pid = proc.findtext('pid')
            if pid is None or not pid.isdecimal():
                raise ValueError('Unusable GPU process PID')
            pids.append(int(pid))
        rows[gpu] = dict(memory_total_mib=_mib(mem.findtext('total')),
                         memory_free_mib=_mib(mem.findtext('free')),
                         memory_used_mib=_mib(mem.findtext('used')),
                         all_process_pids=sorted(set(pids)))
    if not rows:
        raise ValueError('No usable GPUs in nvidia-smi XML')
    return rows


def full_gpu_snapshot():
    return parse_gpu_xml(subprocess.check_output(
        ['nvidia-smi', '-q', '-x'], text=True, stderr=subprocess.STDOUT, timeout=20))


def compute_path(stage):
    if stage in ('evaluation_profile', 'evaluation'):
        return 'evaluation'
    return 'training' if stage in ('canary', 'train') else stage


def validate_job(job):
    for key in ('id', 'kind', 'stage', 'command', 'profile_key', 'vram_mib', 'rss_mib', 'result_receipt'):
        if key not in job:
            raise ValueError('Missing job field: ' + key)
    if not isinstance(job['command'], list) or not job['command']:
        raise ValueError('command must be a nonempty argv list')
    if job['kind'] not in ('train', 'eval', 'feature', 'other'):
        raise ValueError('This dispatcher supports only single-GPU jobs')
    if job['stage'] not in ('calibration', 'canary', 'compatibility', 'train', 'evaluation', 'evaluation_profile', 'feature'):
        raise ValueError('Unsupported measured stage')
    if not isinstance(job['profile_key'], str) or not job['profile_key']:
        raise ValueError('A readable computation profile key is required')
    for key in ('vram_mib', 'rss_mib'):
        value = job[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('Positive integer reservation required: ' + key)
    if job.get('bootstrap_profile'):
        if job['stage'] not in BOOTSTRAP_STAGES or job.get('formal'):
            raise ValueError('Only a short measured stage may bootstrap')
        if job['vram_mib'] != BOOTSTRAP_CAP_MIB:
            raise ValueError('Unmeasured bootstrap reserves and caps exactly 16000 MiB')
        if job.get('requires_profile'):
            raise ValueError('Choose bootstrap or a measured profile, not both')
    elif not job.get('requires_profile'):
        raise ValueError('Measured profile required for this computation path')
    if job.get('formal') and job['stage'] != 'train':
        raise ValueError('Only stage=train can be formal')
    if job.get('calibration_profile_only') and job['stage'] != 'calibration':
        raise ValueError('Two-batch profiling is only a calibration computation path')
    if job.get('require_project_companion') and (job['stage'] != 'canary' or job.get('bootstrap_profile')):
        raise ValueError('Companion measurement requires a profiled non-bootstrap canary')
    if job['stage'] in ('calibration', 'canary', 'compatibility', 'train', 'evaluation_profile', 'evaluation') and not job.get('config'):
        raise ValueError('Training path needs its actual config for profile binding')
    # Experiment JSON never selects a GPU. Selection is derived inside the lock.
    if any(k in job for k in ('gpu', 'gpus', 'device', 'candidate_gpus')):
        raise ValueError('Do not hardcode GPU placement in jobs')
    if job.get('queue_owner', 'independent_v2') != 'independent_v2':
        raise ValueError('External queue retains this task; do not dispatch it here')


def _profile_binding(job, receipt):
    """Reuse execution binding, including actual loaded code and effective config."""
    import yaml
    cfg = yaml.safe_load(Path(job['config']).read_text(encoding='utf-8'))
    stage = receipt['stage']
    if stage not in ('canary', 'calibration', 'compatibility'):
        raise ValueError('This profile has no supported actual execution binding')
    from evidence_bindings import validate_execution_binding
    return validate_execution_binding(receipt['execution_binding'], cfg, stage, HERE)


def measured_reservation(job):
    validate_job(job)
    if job.get('bootstrap_profile'):
        return dict(bootstrap=True, measured=False, vram_mib=job['vram_mib'],
                    rss_mib=job['rss_mib'], profiled_second_train=False)
    profile = read_json(job['requires_profile'])
    if (profile.get('schema') != PROFILE_SCHEMA or profile.get('status') != 'COMPLETED'
            or profile.get('measurement_valid') is not True):
        raise ValueError('Prerequisite is not a valid measured resource profile')
    if (profile.get('profile_key') != job['profile_key'] or
            compute_path(profile.get('stage')) != compute_path(job['stage'])):
        raise ValueError('Memory profile belongs to a different computation path')
    if profile.get('stage') in ('calibration', 'canary', 'compatibility'):
        _profile_binding(job, profile)
    elif profile.get('stage') == 'evaluation_profile':
        from evaluator_profile import validate_evaluation_profile_binding
        import yaml
        cfg = yaml.safe_load(Path(job['config']).read_text(encoding='utf-8'))
        identity = validate_evaluation_profile_binding(profile['evaluation_profile_binding'], cfg, HERE)
        if identity != profile.get('profile_identity'):
            raise ValueError('Evaluation resource identity differs from actual byte binding')
    elif profile.get('stage') == 'feature':
        # External evaluation/feature runners must provide an explicitly reviewed
        # matching command/config identity; they cannot borrow a training peak.
        if not job.get('profile_identity') or profile.get('profile_identity') != job['profile_identity']:
            raise ValueError('Missing matching evaluation/feature profile identity')
    else:
        raise ValueError('Use a matched actual short-stage execution binding for profile reuse')
    resources = profile['resources']
    vram = max(resources.get('per_gpu_peak_vram_mib', {}).values(), default=0)
    rss = resources.get('peak_rss_mib', 0)
    if not all(math.isfinite(float(x)) and x > 0 for x in (vram, rss)):
        raise ValueError('Profile lacks measured NVML or complete process-tree RSS')
    if vram > job['vram_mib'] or rss > job['rss_mib']:
        raise ValueError('Reservation is below a measured peak')
    companion = profile.get('companion_measurement', {}).get('verified') is True
    return dict(bootstrap=False, measured=True, vram_mib=job['vram_mib'],
                rss_mib=job['rss_mib'], profiled_second_train=companion,
                prerequisite=str(Path(job['requires_profile']).resolve()))


def admission_candidates(usage, rows, job, reservation, guard_max_gpus=3):
    """Pure policy selection; invoke only under the shared lease lock."""
    active = {int(x) for x in usage['active_gpus']}
    empty = {g for g, r in rows.items()
             if not r['all_process_pids'] and r['memory_used_mib'] < 100 and g not in active}
    reasons, candidates = [], []
    if int(usage['project_rss_mib']) * 2**20 >= HARD_RSS_BYTES:
        return [], dict(reasons=['Project actual RSS reached 300 GB'], active=sorted(active), empty=sorted(empty))
    if int(usage['effective_rss_mib']) + reservation['rss_mib'] >= ADMISSION_RSS_MIB:
        return [], dict(reasons=['Project RSS reservation reached 240 GiB'], active=sorted(active), empty=sorted(empty))
    for gpu in sorted(rows):
        row = rows[gpu]
        after = active | {gpu}
        # The installed guard may be stricter. Never override its maximum.
        allowed = len(after) <= min(3, guard_max_gpus)
        if guard_max_gpus >= 4 and len(after) == 4 and len(empty - after) >= 2:
            allowed = True
        why = []
        if not allowed:
            why.append('physical-GPU limit / fourth GPU needs two fully empty GPUs')
        if reservation['bootstrap'] and gpu in active:
            why.append('first measurement requires no other project task or lease on this GPU')
        effective = int(usage['effective_vram_mib'].get(gpu, 0))
        actual = int(usage['actual_vram_mib'].get(gpu, 0))
        slots = int(usage['effective_cuda_processes'].get(gpu, 0))
        if job.get('require_project_companion') and (slots != 1 or int(usage['actual_cuda_pids'].get(gpu, 0)) != 1):
            why.append('companion canary requires exactly one existing project CUDA task')
        if job.get('formal') and gpu in active and not reservation['profiled_second_train']:
            why.append('formal sharing requires a measured companion canary; single-card-task fallback remains allowed')
        if slots + 1 > MAX_SLOTS:
            why.append('two project CUDA tasks already reserved')
        if effective + reservation['vram_mib'] >= VRAM_FRACTION * row['memory_total_mib']:
            why.append('project reserved VRAM would reach 70%')
        if row['memory_free_mib'] - max(0, effective - actual) <= reservation['vram_mib'] + FREE_MIB:
            why.append('whole-GPU free memory lacks peak plus 2 GiB margin')
        if job.get('formal') and usage['formal_trains'].get(gpu, 0) and not reservation['measured']:
            why.append('second formal train has no measured profile')
        if why:
            reasons.append(dict(gpu=gpu, reasons=why))
        else:
            candidates.append(gpu)
    return candidates, dict(active=sorted(active), empty=sorted(empty), reasons=reasons,
                            candidate_gpus=candidates, atomic_shared_lock=True)


class _AlreadyLockedView:
    """Delegate acquire to the original guard, reusing its currently held state."""
    def __init__(self, guard, state):
        self.guard, self.state = guard, state

    def __getattr__(self, name):
        return getattr(self.guard, name)

    @contextmanager
    def _locked_state(self):
        yield self.state


def atomic_acquire(guard, guard_module, job, reservation, snapshot=full_gpu_snapshot):
    # No nvidia-smi snapshot or candidate list is carried across this lock.
    with guard._locked_state() as state:
        processes = guard.process_sampler()
        gpu_processes = guard.gpu_process_sampler()
        guard._refresh_leases(state, processes, gpu_processes)
        usage = guard._usage(state, processes, gpu_processes)
        rows = snapshot()
        candidates, audit = admission_candidates(usage, rows, job, reservation,
                                                 guard_module.MAX_ACTIVE_GPUS)
        if not candidates:
            raise guard_module.ResourceUnavailable([json.dumps(audit, ensure_ascii=False)])
        request = guard_module.ResourceRequest(
            job_id=job['id'], kind=job['kind'], candidate_gpus=tuple(candidates),
            expected_vram_mib=reservation['vram_mib'], expected_rss_mib=reservation['rss_mib'],
            gpu_count=1, cuda_processes_per_gpu=1, formal_train=bool(job.get('formal')),
            profiled_second_train=reservation['profiled_second_train'], free_safety_mib=FREE_MIB)
        # Calls the installed implementation; it constructs the real lease.
        lease = type(guard).acquire(_AlreadyLockedView(guard, state), request, gpus=rows)
        after = set(lease['admission']['active_gpus_after'])
        audit.update(selected_gpus=lease['gpus'], active_gpus_after=sorted(after),
                     fully_empty_gpus_after=sorted(set(audit['empty']) - after),
                     four_gpu_exception=len(after) == 4,
                     exception_basis='AGENTS.md section 2.1; snapshot under shared guard lock',
                     installed_guard_max_gpus=guard_module.MAX_ACTIVE_GPUS)
        return lease, audit


def stop_owned_tree(pid):
    """Only terminate the child workload and descendants launched by this runner."""
    import psutil
    try:
        parent = psutil.Process(pid)
        owned = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for proc in reversed(owned):
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def monitor_violations(usage, rows, lease, reservation):
    errors = []
    if int(usage['project_rss_mib']) * 2**20 >= HARD_RSS_BYTES:
        errors.append('Project process-tree RSS reached 300 GB')
    for gpu in lease['gpus']:
        row = rows[gpu]
        if row['memory_free_mib'] < FREE_MIB:
            errors.append('Whole GPU has less than 2 GiB free')
        if usage['actual_vram_mib'].get(gpu, 0) >= row['memory_total_mib'] * VRAM_FRACTION:
            errors.append('Project actual GPU memory reached 70%')
        if usage['actual_cuda_pids'].get(gpu, 0) > MAX_SLOTS:
            errors.append('Project GPU task count exceeded two')
    return errors


def stage_measurement(job, resources):
    """Successful technical receipt is separate from process exit code."""
    receipt = read_json(job['result_receipt'])
    stage = job['stage']
    calibration_ok = (receipt.get('status') == 'PROFILED' and receipt.get('total_batches') == 2
                      and receipt.get('optimizer_updates') == 0) if job.get('calibration_profile_only') else (
                      receipt.get('status') == 'CALIBRATED' and receipt.get('total_batches') == 64)
    checks = {
        'calibration': calibration_ok,
        'canary': receipt.get('status') == 'canary_completed' and receipt.get('optimizer_updates', 0) >= 24,
        'compatibility': receipt.get('status') == 'ACCEPTED' and receipt.get('trajectory_exact') is True
                         and receipt.get('successful_updates', 0) >= 24 and receipt.get('loader_batches', 0) >= 30,
        'train': receipt.get('status') in ('completed', 'train_completed', 'training_completed'),
        'evaluation': receipt.get('status') in ('completed', 'COMPLETED'),
        'feature': receipt.get('status') in ('completed', 'COMPLETED'),
        'evaluation_profile': receipt.get('status') == 'evaluation_profile_completed'
            and receipt.get('native_evidence_metrics_exact') is True
            and receipt.get('native_seen') == 1469 and receipt.get('evidence_seen') == 1469
            and receipt.get('baseline_training_receipt_created') is False,
    }
    if not checks[stage]:
        raise ValueError('Stage receipt does not establish complete measured exposure')
    binding = receipt.get('execution_binding')
    if stage in ('calibration', 'canary', 'compatibility'):
        if not binding:
            raise ValueError('Measured stage lacks actual execution binding')
        # This also checks source bytes and config against the actual job.
        _profile_binding(job, dict(stage=stage, execution_binding=binding))
    elif stage == 'evaluation_profile':
        from evaluator_profile import validate_evaluation_profile_binding
        import yaml
        cfg=yaml.safe_load(Path(job['config']).read_text(encoding='utf-8'))
        identity=validate_evaluation_profile_binding(receipt['evaluation_profile_binding'],cfg,HERE)
        if identity!=receipt.get('profile_identity'):
            raise ValueError('Measured evaluator profile identity differs')
    if max(resources['per_gpu_peak_vram_mib'].values(), default=0) <= 0 or resources['peak_rss_mib'] <= 0:
        raise ValueError('No nonzero actual NVML/process-tree measurement')
    return receipt, binding


def run_job(job, output, repo, guard_module, *, sleep=time.sleep):
    reservation = measured_reservation(job)
    if Path(job['result_receipt']).exists():
        raise FileExistsError('Do not reuse a prior stage receipt: ' + job['result_receipt'])
    if job.get('config') and not Path(job['config']).is_file():
        raise FileNotFoundError('Actual computation config is missing: ' + job['config'])
    guard = guard_module.ProjectResourceGuard(guard_module.DEFAULT_LEASE_FILE)
    events = output / (job['id'] + '_events.jsonl')
    with (output / (job['id'] + '.log')).open('x', encoding='utf-8') as log:
        while True:
            try:
                lease, admission = atomic_acquire(guard, guard_module, job, reservation)
                break
            except guard_module.ResourceUnavailable as exc:
                append_event(events, dict(status='QUEUED', time=time.time(), reasons=exc.reasons))
                sleep(POLL_SECONDS)
        write_json(output / (job['id'] + '_admission.json'), admission)
        environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        visible = ','.join(str(x) for x in lease['gpus'])
        environment.update(CUDA_VISIBLE_DEVICES=visible)
        environment[guard_module.LEASE_ID_ENV] = lease['lease_id']
        environment[guard_module.LEASE_FILE_ENV] = str(guard_module.DEFAULT_LEASE_FILE.resolve())
        environment[guard_module.LEASE_GPUS_ENV] = visible
        proc, errors = None, []
        minimum_free = None
        companion_samples, companion_all = 0, True
        resources = dict(per_gpu_peak_vram_mib={str(g): 0 for g in lease['gpus']}, peak_rss_mib=0,
                         cuda_pid_counts={str(g): 0 for g in lease['gpus']})
        try:
            guard.bind(lease['lease_id'], os.getpid())
            proc = subprocess.Popen(job['command'], cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=environment)
            append_event(events, dict(status='LAUNCHED', time=time.time(), pid=proc.pid,
                                     lease_id=lease['lease_id'], gpu_ids=lease['gpus']))
            while proc.poll() is None:
                state = guard.inspect()
                rows = full_gpu_snapshot()
                current = next((x for x in state['leases'] if x['lease_id'] == lease['lease_id']), None)
                if current is None:
                    raise RuntimeError('Running job lost its shared lease')
                errors.extend(monitor_violations(state['usage'], rows, lease, reservation))
                for gpu in lease['gpus']:
                    key = str(gpu)
                    resources['per_gpu_peak_vram_mib'][key] = max(resources['per_gpu_peak_vram_mib'][key],
                        int(current['per_gpu_peak_vram_mib'].get(key, 0)))
                    resources['cuda_pid_counts'][key] = max(resources['cuda_pid_counts'][key],
                        int(current['peak_cuda_pid_counts'].get(key, 0)))
                    free = rows[gpu]['memory_free_mib']
                    minimum_free = free if minimum_free is None else min(free, minimum_free)
                resources['peak_rss_mib'] = max(resources['peak_rss_mib'], current['peak_rss_mib'])
                if max(resources['per_gpu_peak_vram_mib'].values()) > reservation['vram_mib']:
                    errors.append('Workload exceeded its measured reservation / bootstrap 16000 MiB cap')
                if resources['peak_rss_mib'] > reservation['rss_mib']:
                    errors.append('Workload process-tree RSS exceeded its reservation')
                whole_gpu={str(g):dict(whole_gpu_used_mib=rows[g]['memory_used_mib'],
                    whole_gpu_free_mib=rows[g]['memory_free_mib'],
                    project_actual_vram_mib=state['usage']['actual_vram_mib'].get(g,0),
                    project_actual_cuda_tasks=state['usage']['actual_cuda_pids'].get(g,0)) for g in lease['gpus']}
                if job.get('require_project_companion') and current.get('observed_cuda_pids'):
                    companion_samples += 1
                    companion_all &= all(state['usage']['actual_cuda_pids'].get(g,0)==2 for g in lease['gpus'])
                append_event(events, dict(status='SAMPLE', time=time.time(), resources=resources,
                                         whole_gpu=whole_gpu,
                                         whole_gpu_minimum_free_mib=minimum_free,
                                         project_rss_mib=state['usage']['project_rss_mib']))
                if errors:
                    stop_owned_tree(proc.pid)
                    raise RuntimeError('; '.join(errors))
                sleep(1)
            if proc.returncode:
                raise RuntimeError('Workload returned ' + str(proc.returncode))
            receipt, binding = stage_measurement(job, resources)
            # The workload's guard peak call can capture a peak between polls.
            inner = receipt.get('resources', {})
            resources['peak_rss_mib'] = max(resources['peak_rss_mib'], inner.get('peak_rss_mib', 0))
            for key, value in inner.get('per_gpu_peak_vram_mib', {}).items():
                resources['per_gpu_peak_vram_mib'][str(key)] = max(
                    resources['per_gpu_peak_vram_mib'].get(str(key), 0), value)
            if max(resources['per_gpu_peak_vram_mib'].values()) > reservation['vram_mib'] or resources['peak_rss_mib'] > reservation['rss_mib']:
                raise RuntimeError('Receipt peak exceeds reservation')
            report = dict(schema=PROFILE_SCHEMA, status='COMPLETED', measurement_valid=True,
                          stage=job['stage'], profile_key=job['profile_key'], profile_identity=receipt.get('profile_identity',job.get('profile_identity')),
                          execution_binding=binding, result_receipt=job['result_receipt'], resources=resources,
                          evaluation_profile_binding=receipt.get('evaluation_profile_binding'),
                          companion_measurement=dict(requested=bool(job.get('require_project_companion')),
                              sampled_workload_active_intervals=companion_samples,
                              all_sampled_active_intervals_had_companion=companion_all if companion_samples else None,
                              verified=bool(job.get('require_project_companion')) and companion_samples>0 and companion_all
                                  and job['stage']=='canary' and receipt.get('optimizer_updates',0)>=24),
                          technical_stage_status=receipt.get('status'),
                          resource_measurement_only=bool(job.get('calibration_profile_only')),
                          minimum_free_mib=minimum_free, admission=admission, reservation=reservation,
                          lease_id=lease['lease_id'], exit_code=proc.returncode, time=time.time())
            write_json(output / (job['id'] + '_resource_profile.json'), report)
            return report
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                stop_owned_tree(proc.pid)
                proc.wait(timeout=10)
            write_json(output / (job['id'] + '_failure.json'),
                       dict(status='FAILED', time=time.time(), error=repr(exc), monitor_errors=errors,
                            resources=resources, lease_id=lease['lease_id']))
            raise
        finally:
            # Retain lease when a CUDA descendant is still alive; never hide it.
            state = guard.inspect()
            current = next((x for x in state['leases'] if x['lease_id'] == lease['lease_id']), None)
            if current and current.get('observed_cuda_pids'):
                append_event(events, dict(status='LEASE_RETAINED_FOR_CUDA_CHILDREN',
                                         lease_id=lease['lease_id'], time=time.time()))
            else:
                guard.release(lease['lease_id'])


def ordered_jobs(jobs):
    """Independent evaluation before training; never duplicate external queues."""
    for job in jobs:
        validate_job(job)
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job id in manifest')
    return sorted(jobs, key=lambda j: (0 if j['stage'] in ('evaluation','evaluation_profile') else 1,
                                      j.get('priority', 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    jobs = ordered_jobs(manifest['jobs'])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'manifest.json', manifest)
    module = load_guard(args.repo)
    # Keep exact read-only guard source alongside this attempt, not a substitute.
    (args.output / 'actual_project_resource_guard.py').write_bytes(
        (args.repo / 'tools/project_resource_guard.py').read_bytes())
    for job in jobs:
        run_job(job, args.output, args.repo, module)
    write_json(args.output / 'completion.json', dict(status='COMPLETED',
               time=time.time(), jobs=[j['id'] for j in jobs]))


if __name__ == '__main__':
    main()
