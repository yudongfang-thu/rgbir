"""LOG-only draft for six legacy N/C0 evaluation bridges; never signs ACCEPTED.

Uses read-only legacy snapshots and an executed full-N42 parity probe. No Torch,
GPU, hashing, model loading, result mutation or sealed test access. Canonical
source copies follow the frozen formal evaluator's receipt emission order.
"""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import yaml

METRICS = ('AP50', 'AP75', 'mAP50_95', 'precision', 'recall')
OBJECT_SOURCES = {
    'DetectionValidator': 'ultralytics/models/yolo/detect/val.py',
    'BaseValidator': 'ultralytics/engine/validator.py',
    'YOLO.val': 'ultralytics/engine/model.py',
    'check_det_dataset': 'ultralytics/data/utils.py',
    'DetMetrics': 'ultralytics/utils/metrics.py',
    'Metric': 'ultralytics/utils/metrics.py',
    'ap_per_class': 'ultralytics/utils/metrics.py',
    'non_max_suppression': 'ultralytics/utils/nms.py',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def copy_new(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(Path(source).read_bytes())


def load_analyzer(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('_bridge_candidate_analyzer', root/'analyze_independent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + '.' + node.attr
    raise ValueError('Unexpected formal implementation expression')


def formal_source_order(formal_source, records):
    """Derive requested objects from actual frozen emit code, not profile order."""
    tree = ast.parse(Path(formal_source).read_text(encoding='utf-8-sig'))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'implementations' for t in n.targets)]
    if len(assignments) != 1 or dotted(assignments[0].value.func) != 'legacy.implementation_files':
        raise ValueError('Formal evaluator implementation collection changed')
    arguments = [dotted(n) for n in assignments[0].value.args]
    if arguments != list(OBJECT_SOURCES):
        raise ValueError('Formal evaluator implementation object order changed')
    emits = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'emit_bound_run_receipt']
    trainers = [k.value for call in emits for k in call.keywords if k.arg == 'trainers']
    expected = ast.parse('[Path(__file__), *implementations]', mode='eval').body
    if len(trainers) != 1 or ast.dump(trainers[0]) != ast.dump(expected):
        raise ValueError('Formal evaluator trainer source order changed')
    suffixes = ['release_gpu5/evaluate_independent.py']
    for argument in arguments:
        suffix = OBJECT_SOURCES[argument]
        if suffix not in suffixes:
            suffixes.append(suffix)
    result = []
    for index, suffix in enumerate(suffixes, 1):
        hits = [r for r in records if r['original'].endswith('/'+suffix)]
        if len(hits) != 1:
            raise ValueError('Probe missing unique actual source: ' + suffix)
        row = dict(hits[0])
        row['expected_formal_receipt_relative'] = 'source_snapshot/trainer/%02d_%s' % (
            index, PurePosixPath(row['original']).name)
        result.append(row)
    return result


def validate_probe(probe):
    probe = Path(probe)
    receipt, native, evidence = [read(probe/name) for name in (
        'evaluator_profile_receipt.json', 'native_metrics.json', 'evidence_metrics.json')]
    nc, ec = [read(probe/name) for name in ('native_contract.json', 'evidence_contract.json')]
    if (receipt.get('status') != 'evaluation_profile_completed'
        or receipt.get('native_evidence_metrics_exact') is not True
        or receipt.get('baseline_training_receipt_created') is not False
        or receipt.get('official_test_accessed') is not False
        or receipt.get('native_seen') != 1469 or receipt.get('evidence_seen') != 1469):
        raise ValueError('Actual completed full-N42 evaluation probe required')
    if any(native.get(k) != evidence.get(k) or receipt.get('metric_differences', {}).get(k) != 0.0 for k in METRICS):
        raise ValueError('Actual native/evidence five-metric parity failed')
    if native.get('per_class') != evidence.get('per_class') or not native.get('per_class'):
        raise ValueError('Actual native/evidence class parity missing')
    for value in (native, evidence):
        if (value.get('status') != 'completed' or value.get('observed_images') != 1469
            or value.get('checkpoint') != receipt['checkpoint'] or value.get('official_test_accessed') is not False
            or value.get('metric_units') != 'fraction_0_to_1'):
            raise ValueError('Actual probe endpoint identity differs')
    for key in ('roster', 'actual_loader_roster', 'effective_kwargs', 'observed_images'):
        if nc.get(key) != ec.get(key):
            raise ValueError('Actual native/evidence contract differs: ' + key)
    if len(ec['roster']) != 1469 or len(set(ec['roster'])) != 1469 or ec['observed_images'] != 1469:
        raise ValueError('Probe population is incomplete')
    binding = read(probe/'evaluation_profile_binding/binding.json')
    if (binding['roster'] != ec['roster'] or binding['actual_effective_kwargs'] != ec['effective_kwargs']
        or binding['observed_images'] != 1469):
        raise ValueError('Probe bound evaluator contract differs')
    remote_root = PurePosixPath(receipt['evaluation_profile_binding']).parent
    records = []
    for row in binding['source_files']:
        relative = PurePosixPath(row['copy']).relative_to(remote_root)
        path = probe/'evaluation_profile_binding'/str(relative)
        if not path.is_file():
            raise ValueError('Actual bound probe source missing: ' + str(path))
        records.append(dict(row, local_copy=str(path.resolve())))
    return receipt, native, ec, binding, records


def legacy_val_arguments(source):
    tree = ast.parse(Path(source).read_text(encoding='utf-8-sig'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'val'
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'model']
    if len(calls) != 1:
        raise ValueError('Legacy source has ambiguous val invocation')
    expected = ast.parse("model.val(data=str(data),split='val',imgsz=cfg['imgsz'],batch=cfg['batch'],"
        "workers=cfg['workers'],device='0',plots=False,save_json=False,verbose=False,"
        "project=str(a.run),name='eval_val',exist_ok=False)", mode='eval').body
    actual = {k.arg: ast.dump(k.value) for k in calls[0].keywords}
    wanted = {k.arg: ast.dump(k.value) for k in expected.keywords}
    if calls[0].args or actual != wanted:
        raise ValueError('Legacy evaluation kwargs differ from reviewed native call')
    return sorted(actual)


def derived_defaults(default_path, source_rows):
    defaults = yaml.safe_load(Path(default_path).read_text(encoding='utf-8-sig'))
    model = Path(next(r['local_copy'] for r in source_rows if r['original'].endswith('/engine/model.py'))).read_text(encoding='utf-8')
    validator = Path(next(r['local_copy'] for r in source_rows if r['original'].endswith('/engine/validator.py'))).read_text(encoding='utf-8')
    if 'custom = {"rect": True}' not in model or '0.01 if self.args.task == "obb" else 0.001' not in validator:
        raise ValueError('Pinned evaluation rect/conf defaults need fresh review')
    if 'include = {"imgsz", "data", "task", "single_cls"}' not in model:
        raise ValueError('Checkpoint override whitelist changed')
    result = {key: defaults[key] for key in ('quantize','iou','max_det','agnostic_nms','single_cls','augment')}
    result.update(rect=True, conf=.001, half=defaults['quantize'] == 16)
    return result


def prepare(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Keep prior candidate attempts; select a new output')
    analyzer = load_analyzer(args.analyzer_root)
    if not analyzer.verify_analyzer_acceptance(args.accepted_analyzer):
        raise ValueError('Actual analyzer source acceptance is missing/stale')
    old_manifest = read(args.old_manifest)
    probe_receipt, probe_metrics, contract, binding, source_rows = validate_probe(args.probe)
    formal = next(r for r in source_rows if r['original'].endswith('/release_gpu5/evaluate_independent.py'))
    if Path(formal['local_copy']).read_bytes() != (args.analyzer_root/'evaluate_independent.py').read_bytes():
        raise ValueError('Frozen new evaluator differs from actual successful probe source')
    ordered = formal_source_order(formal['local_copy'], source_rows)
    defaults = derived_defaults(args.probe/'pinned_default.yaml', source_rows)
    writer_source = (args.probe/'actual_write_jstars_run_receipt.py').read_text(encoding='utf-8-sig')
    if ('enumerate(paths, start=1)' not in writer_source or 'path not in paths' not in writer_source
        or 'target = destination / f"{index:02d}_{source.name}"' not in writer_source):
        raise ValueError('Actual receipt writer copy order needs fresh review')
    output.mkdir(parents=True, exist_ok=False)
    inventory, entries, manifest_rows, failures = [], [], [], []
    canonical_copies = []
    for row in ordered:
        target = output/'canonical_sources'/PurePosixPath(row['expected_formal_receipt_relative']).name
        copy_new(row['local_copy'], target)
        canonical_copies.append(str(target.relative_to(output)).replace('\\','/'))
    copy_new(args.probe/'actual_write_jstars_run_receipt.py', output/'source_order_evidence/actual_write_jstars_run_receipt.py')
    for spec in old_manifest['runs']:
        original_spec = dict(spec)
        raw = analyzer._legacy_loader().load_endpoint(original_spec, args.old_manifest.parent)
        name = Path(raw['path']).name
        inventory.append(dict(name=name, old_display_arm=spec['arm'], actual_arm=raw.get('source_arm'),
            seed=spec['seed'], raw_receipt_status=raw['status'], issues=raw['issues']))
        if raw['status'] != 'complete':
            failures.append(name+': legacy receipt verification failed')
            continue
        if spec['arm'] not in ('N', 'C', 'C0'):
            continue  # random is inventoried under original identity, no new arm relabel.
        path = Path(raw['path'])
        metric = read(path/'evaluation_val.json')
        train_receipt = read(path/'run_evidence/run_receipt.json')
        eval_receipt = read(path/'eval_evidence/run_receipt.json')
        train_cfg, _ = analyzer._bound_configs(path, train_receipt, 'run_evidence')
        eval_cfg, _ = analyzer._bound_configs(path, eval_receipt, 'eval_evidence')
        issues = []
        effective = dict(defaults, **{k:eval_cfg[k] for k in ('imgsz','batch','workers')})
        if effective != contract['effective_kwargs']:
            issues.append('Derived effective old kwargs differ from actual new probe')
        if raw['roster'] != contract['roster']:
            issues.append('Actual old bound full roster differs from probe')
        env = eval_receipt.get('environment', {})
        if any(env.get(k) != contract['evaluator_identity'].get(k) for k in ('torch','ultralytics')):
            issues.append('Actual old evaluation environment differs from probe')
        if any(eval_cfg.get(k) != train_cfg.get(k) for k in ('model','imgsz','batch','workers','torch_version','ultralytics_version')):
            issues.append('Bound old train/eval configuration identity differs')
        old_data = [path/'eval_evidence'/r for r in eval_receipt['source_snapshots']['config']
                    if 'rgb.data.yaml' in r]
        probe_data = args.probe/'evaluation_profile_binding/student_data.yaml'
        if len(old_data) != 1 or old_data[0].read_bytes() != probe_data.read_bytes():
            issues.append('Actual old bound RGB data YAML differs from probe')
        actual_sources = eval_receipt['source_snapshots']['trainer']
        source_copies = {}
        for relative in actual_sources:
            source = path/'eval_evidence'/relative
            legacy_val_arguments(source)
            target = output/'entries'/name/'evaluation_sources'/Path(relative).name
            copy_new(source,target)
            source_copies[relative] = str(target.relative_to(output)).replace('\\','/')
        # Corroborative historical sidecar, never represented as receipt-bound.
        sidecar = yaml.safe_load((path/'args.yaml').read_text(encoding='utf-8-sig'))
        if sidecar.get('single_cls') is not False or sidecar.get('task') != 'detect':
            issues.append('Historical task/single_cls sidecar differs from scoped native inference')
        historical_probe_match = None
        if metric['checkpoint'] == probe_receipt['checkpoint']:
            historical_probe_match = all(metric[k] == probe_metrics[k] for k in METRICS)
            if historical_probe_match is not True:
                issues.append('Historical N42 endpoint differs from actual repeat native probe')
        candidate_contract = {k:copy.deepcopy(contract[k]) for k in ('schema','expected_val_images',
            'roster','endpoint','official_test_accessed','evaluator_identity','effective_kwargs')}
        entry = dict(checkpoint=metric['checkpoint'],seed=spec['seed'],actual_source_arm=metric['arm'],
            bound_training_config=train_cfg,bound_evaluation_config=eval_cfg,
            evaluation_source_copies=source_copies,evaluation_contract=candidate_contract,
            original_snapshot=str(path.resolve()),historical_N42_vs_probe_exact=historical_probe_match,
            actual_old_loader_order_recorded=False,actual_old_seen_recorded=False,
            old_per_class_recorded=bool(metric.get('per_class')),old_objects_recorded=False,
            bridge_scope='semantic equivalence candidate for five aggregate dev metrics only',
            old_single_cls_sidecar=dict(path=str(path/'args.yaml'),receipt_bound=False,value=sidecar.get('single_cls')),
            static_issues=issues)
        entries.append(entry)
        for relative in ('run_evidence/run_receipt.json','eval_evidence/run_receipt.json',
                         'evaluation_val.json','completion_receipt.json','args.yaml'):
            copy_new(path/relative,output/'entries'/name/'original_evidence'/relative)
        failures.extend(name+': '+issue for issue in issues)
        mapped = dict(original_spec,arm='N' if spec['arm']=='N' else 'C0',source='paired',source_arm=metric['arm'])
        mapped['evaluation_compatibility_receipt'] = str(output/'evaluation_compatibility_candidate.json')
        manifest_rows.append(mapped)
    if len(entries) != 6 or {(r['actual_source_arm'],r['seed']) for r in entries} != {
            (arm,seed) for arm in ('weight0','paired') for seed in (0,42,123)}:
        failures.append('Expected exactly six old paired/weight0 seed0/42/123 endpoints')
    bridge = dict(schema='rgbir-legacy-evaluation-bridge-candidate-v1',status='DRAFT_AWAITING_INDEPENDENT_REVIEW',
        reviewer=None,entries=entries,canonical_evaluator_source_copies=canonical_copies,
        canonical_source_order=ordered,probe_evidence=str(args.probe.resolve()),
        source_scope='Formal evaluate_independent emit list in receipt snapshot order; excludes profile-only sources',
        static_failures=failures,training_admission_blocked_by_this_bridge=False,
        pending=['Independent review of source/version/default inference for the six historical endpoints',
                 'Future formal evaluation must match canonical source bytes/order and effective kwargs',
                 'Old C0/N0/N123 per-class AP and object diagnostics unavailable in these receipts'],
        historical_native_library_source_bytes='not saved by old eval receipts; version + common entry + N42 repeat evidence only',
        official_test_accessed=False,old_results_modified=False)
    write_new(output/'evaluation_compatibility_candidate.json',bridge)
    write_new(output/'manifest_candidate.json',dict(runs=manifest_rows,implementation_checks_passed=False))
    candidate_records=[analyzer.load_endpoint(spec,output) for spec in manifest_rows]
    if any(r.get('evaluation_contract_validation')=='passed' for r in candidate_records):
        raise ValueError('Draft bridge unexpectedly accepted by analyzer')
    summary=dict(status='PREPARED_NOT_ACCEPTED',inventory=inventory,entries=len(entries),
        canonical_source_order=[r['expected_formal_receipt_relative'] for r in ordered],
        static_failures=failures,probe_native_evidence_exact=True,
        historical_N42_vs_probe_exact=next((e['historical_N42_vs_probe_exact'] for e in entries if e['historical_N42_vs_probe_exact'] is not None),None),
        draft_rejected_by_analyzer_for_all_six=all(r['evaluation_contract_validation']=='failed' for r in candidate_records),
        frozen_protocol_ids=sorted({r['protocol_id'] for r in manifest_rows}),
        official_test_accessed=False)
    write_new(output/'candidate_summary.json',summary)
    return summary


def verify_future_sources(candidate, evaluation_evidence):
    candidate = Path(candidate)
    bridge = read(candidate)
    evidence = Path(evaluation_evidence)
    receipt = read(evidence/'run_receipt.json')
    actual = sorted(receipt['source_snapshots']['trainer'])
    expected = [r['expected_formal_receipt_relative'] for r in bridge['canonical_source_order']]
    if actual != expected:
        raise ValueError('Actual formal evaluator source set/order differs from reviewed candidate')
    for relative, copy_path in zip(actual,bridge['canonical_evaluator_source_copies']):
        if (evidence/relative).read_bytes() != (candidate.parent/copy_path).read_bytes():
            raise ValueError('Actual formal evaluator source bytes differ: '+relative)
    return dict(status='FORMAL_SOURCE_SET_ORDER_BYTES_MATCH',evaluation_evidence=str(evidence),sources=actual,
                bridge_acceptance_created=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare')
    for name in ('old-manifest','analyzer-root','accepted-analyzer','probe','output'):
        prep.add_argument('--'+name,type=Path,required=True)
    check=commands.add_parser('verify-formal-sources')
    check.add_argument('--candidate',type=Path,required=True)
    check.add_argument('--evaluation-evidence',type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args) if args.command=='prepare' else verify_future_sources(args.candidate,args.evaluation_evidence)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
