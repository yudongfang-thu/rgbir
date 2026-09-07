"""Bind real technical evidence to the exact executable configuration."""
import json
from pathlib import Path


def configuration_identity(cfg):
    return {k:v for k,v in cfg.items() if k not in ('seed','readiness_receipt')}


def _read(path):
    return json.loads(Path(path).read_text())


def _same_models(receipt, cfg):
    for key in ('dataset','model','teacher','reference'):
        if receipt.get(key) != cfg[key]:
            raise ValueError('Evidence identity differs: '+key)


def check_readiness(cfg, module_root):
    from evidence_bindings import validate_execution_binding
    module_root = Path(module_root)
    ready = _read(cfg['readiness_receipt'])
    if ready.get('status') != 'ACCEPTED' or ready.get('arm') != cfg['arm']:
        raise ValueError('Readiness is not accepted for this arm')
    if ready.get('effective_configuration') != configuration_identity(cfg):
        raise ValueError('Configuration differs from accepted effective configuration')
    if ready.get('seeds') != [0,42,123]:
        raise ValueError('Readiness lacks frozen seed set')
    sources = ready.get('source_files', [])
    required = {str(p.relative_to(module_root)).replace('\\','/') for p in module_root.rglob('*.py')}
    recorded = {r['relative'] for r in sources}
    if required != recorded:
        raise ValueError('Readiness source set differs from executable release')
    for record in sources:
        if (module_root/record['relative']).read_bytes() != Path(record['accepted_copy']).read_bytes():
            raise ValueError('Source differs from accepted copy: '+record['relative'])
    # All six trajectories are required for reuse of Drone N/C0. A new LLVIP N
    # family must provide the same implementation test identities on its data.
    tested = set()
    for path in ready.get('compatibility_receipts', []):
        r = _read(path); _same_models(r,cfg)
        binding = validate_execution_binding(r['execution_binding'],cfg,'compatibility',module_root)
        actual = binding['effective_configuration']
        if r.get('seed') != actual.get('seed') or r.get('arm') != actual.get('arm'):
            raise ValueError('Compatibility receipt relabels measured seed/arm')
        if r.get('status')!='ACCEPTED' or not r.get('trajectory_exact'):
            raise ValueError('Invalid compatibility trajectory')
        if r.get('successful_updates',0)<24 or r.get('loader_batches',0)<30:
            raise ValueError('Incomplete compatibility exposure')
        tested.add((r['seed'],r['arm']))
    if tested != {(seed,arm) for seed in (0,42,123) for arm in ('N','C0')}:
        raise ValueError('Missing seed/arm compatibility trajectory')
    canary = _read(ready['canary_receipt']); _same_models(canary,cfg)
    canary_binding = validate_execution_binding(canary['execution_binding'],cfg,'canary',module_root)
    if canary.get('seed') != canary_binding['effective_configuration'].get('seed'):
        raise ValueError('Canary receipt relabels measured seed')
    if (canary.get('status')!='canary_completed' or canary.get('arm')!=cfg['arm'] or
        canary.get('source')!=cfg.get('source','paired') or canary.get('optimizer_updates',0)<24 or
        canary.get('official_test_accessed') is not False):
        raise ValueError('Invalid real canary identity/updates')
    for k in ('classification_coefficient','localization_coefficient'):
        if canary.get(k)!=cfg.get(k) or canary_binding['effective_configuration'].get(k)!=cfg.get(k):
            raise ValueError('Canary coefficient differs')
    if cfg['arm'] not in ('N','C0'):
        if not any(max(x.get('kd_gradient_l2',0),x.get('kd_score_gradient_l2',0),
                       x.get('kd_box_gradient_l2',0))>0 for x in canary.get('gradient_checks',[])):
            raise ValueError('No canary KD gradient')
        cal = _read(ready['calibration_receipt']); _same_models(cal,cfg)
        validate_execution_binding(cal['execution_binding'],cfg,'calibration',module_root)
        family='C1' if cfg['arm'] in ('C1','C1_y') else 'L1'
        if (cal.get('status')!='CALIBRATED' or cal.get('family')!=family or
            cal.get('total_batches')!=64 or cal.get('nonzero_batches',0)<16 or
            cal.get('seed')!=20260907 or cal.get('train_mode') is not True or
            cal.get('reset_parameters_and_buffers_each_batch') is not True or cal.get('test_accessed') is not False):
            raise ValueError('Invalid calibration protocol')
        key='classification_coefficient' if family=='C1' else 'localization_coefficient'
        if cal['lambda_'+family]!=cfg[key]:
            raise ValueError('Calibration coefficient differs')
        if family=='L1':
            if (cal.get('unique_images',0)<16 or len(cal.get('source_groups',[]))<2 or
                cal.get('geometry_contract')!=cfg.get('geometry_contract')):
                raise ValueError('Localization coverage missing')
            for key in ('geometry_contract','d2_receipt'):
                current=Path(cfg[key])
                if current.read_bytes()!=Path(ready['localization_evidence_copies'][key]).read_bytes():
                    raise ValueError('Localization evidence changed')
            d2=_read(cfg['d2_receipt'])
            if d2.get('geometry_verified') is not True or d2.get('split')!='train':
                raise ValueError('Invalid verified training D2')
    review=_read(ready['review_receipt'])
    if review.get('status')!='ACCEPTED' or review.get('blocking_issues_remaining')!=0:
        raise ValueError('Implementation review not accepted')
    review_sources={r['relative']:r['accepted_copy'] for r in review.get('source_files',[])}
    if set(review_sources)!=required:
        raise ValueError('Review source scope differs')
    for relative in required:
        if (module_root/relative).read_bytes()!=Path(review_sources[relative]).read_bytes():
            raise ValueError('Review source changed')
    return ready
