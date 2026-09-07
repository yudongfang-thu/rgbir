"""Materialize only technically validated C1 stages from real measured receipts.

This CPU orchestration script is outside the frozen training release. It never
starts a workload and never chooses lambda, an AP endpoint, or a new seed.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import yaml


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path,value,yaml_output=False):
    with Path(path).open('x') as stream:
        if yaml_output:
            yaml.safe_dump(value,stream,sort_keys=False)
        else:
            json.dump(value,stream,indent=2,allow_nan=False)


def validate_cpu_preflight(tests,release,cfg):
    """Use the real same-release test runs; old summary-only receipts cannot transfer."""
    release=Path(release).resolve()
    measured=Path(tests.get('release',''))
    if not measured.is_absolute() or measured.resolve()!=release:
        raise ValueError('Pinned CPU tests were not run for this release')
    environment=tests.get('environment',{})
    if any(environment.get(key)!=cfg[key+'_version'] for key in ('torch','ultralytics')):
        raise ValueError('CPU test environment differs from the frozen training environment')
    stages=tests.get('stages',[])
    if (len(stages)!=2 or {row.get('name') for row in stages}!={'operator_tests','reference_package'}
            or any(row.get('exit_code')!=0 for row in stages)):
        raise ValueError('Actual pinned CPU tests did not pass both required stages')
    for row in stages:
        command=row.get('command')
        if not isinstance(command,list) or not command or not all(isinstance(x,str) and x for x in command):
            raise ValueError('Actual CPU test command is missing')
        points_here=False
        for argument in command:
            candidate=Path(argument)
            if candidate.is_absolute():
                try:
                    candidate.resolve().relative_to(release)
                    points_here=True
                except ValueError:
                    pass
        if not points_here:
            raise ValueError('CPU test command does not point to this release')
        log=Path(row.get('log',''))
        if not log.is_absolute() or not log.is_file():
            raise ValueError('Actual CPU test log is missing')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('canary','formal'),required=True)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--draft',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--canary',type=Path)
    p.add_argument('--review',type=Path)
    p.add_argument('--tests',type=Path)
    p.add_argument('--compatibility',type=Path,nargs='*')
    a=p.parse_args();sys.path.insert(0,str(a.release))
    from evidence_bindings import validate_execution_binding
    cfg=yaml.safe_load(a.draft.read_text());cal=read(a.calibration)
    if (cal.get('family')!='C1' or cal.get('status')!='CALIBRATED' or cal.get('total_batches')!=64 or
        cal.get('nonzero_batches',0)<16 or cal.get('seed')!=20260907 or
        cal.get('train_mode') is not True or cal.get('reset_parameters_and_buffers_each_batch') is not True or
        cal.get('test_accessed') is not False or not 0<cal.get('lambda_C1',0)<=1):
        raise ValueError('Actual fixed 64-batch C1 calibration did not qualify')
    validate_execution_binding(cal['execution_binding'],cfg,'calibration',a.release)
    cfg.update(classification_coefficient=cal['lambda_C1'],calibration_receipt=str(a.calibration),seed=42)
    a.output.mkdir(parents=True,exist_ok=False)
    if a.stage=='canary':
        for arm in ('C1','C1_y'):
            current=copy.deepcopy(cfg)
            current.update(arm=arm,method_id='RGBIR-INDEPENDENT-KD-v2-'+arm)
            current['classification']['off_target_weight']=.25 if arm=='C1' else 0.
            write_new(a.output/(arm+'.yaml'),current,True)
        write_new(a.output/'materialization_receipt.json',dict(status='CANARY_CONFIGS_READY',
            calibrated_coefficient=cal['lambda_C1'],calibration=str(a.calibration),formal_authorization=False))
        return
    if not all((a.canary,a.review,a.tests,a.compatibility)):
        raise ValueError('Formal materialization needs all actual technical evidence')
    tests=read(a.tests)
    validate_cpu_preflight(tests,a.release,cfg)
    import admission
    review=read(a.review)
    cfg.update(protocol_status='FROZEN',formal_training_authorized=True,
        canary_acceptance=str(a.canary),readiness_receipt=str(a.output/'readiness.json'))
    ready=dict(status='ACCEPTED',arm='C1',seeds=[0,42,123],
        effective_configuration=admission.configuration_identity(cfg),source_files=review['source_files'],
        compatibility_receipts=[str(x) for x in a.compatibility],canary_receipt=str(a.canary),
        calibration_receipt=str(a.calibration),review_receipt=str(a.review),cpu_tests_receipt=str(a.tests))
    # Validate the proposed receipt in memory before any accepted file exists.
    # All other reads remain the exact admission implementation's filesystem reads.
    original_read=admission._read
    admission._read=lambda path:ready if str(path)==cfg['readiness_receipt'] else original_read(path)
    try:
        admission.check_readiness(cfg,a.release)
    finally:
        admission._read=original_read
    write_new(a.output/'readiness.json',ready)
    for seed in (42,0,123):
        current=dict(cfg,seed=seed)
        write_new(a.output/('C1_s'+str(seed)+'.yaml'),current,True)
    write_new(a.output/'materialization_receipt.json',dict(status='FORMAL_CONFIGS_READY',
        seed_start_order=[42,0,123],calibrated_coefficient=cal['lambda_C1'],
        readiness=str(a.output/'readiness.json'),training_started=False))


if __name__=='__main__':main()
