"""CPU-only DRAFT bridge from six executed legacy checkpoint reevaluations.

No ACCEPTED option exists. Old training/evaluation files remain unchanged. The
canonical seven sources are copied from the new actual receipts; all nine real
sources and the wrapper-review evidence remain explicitly attached.
"""
import argparse
import copy
import gzip
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import yaml

HERE = Path(__file__).resolve().parent
LOG = HERE.parent
METHODS = {'N': 'weight0', 'C0': 'paired'}
SEEDS = (42, 0, 123)
EXPECTED = {(arm, seed) for arm in METHODS for seed in SEEDS}
METRICS = ('AP50', 'AP75', 'mAP50_95', 'precision', 'recall')
EXTRA_SOURCES = ('source_snapshot/trainer/08_evaluator_profile.py',
                 'source_snapshot/trainer/09_legacy_checkpoint_evaluate.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_module(path, name):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def equal_bytes(first, second, reason):
    require(Path(first).read_bytes() == Path(second).read_bytes(), reason)


def bound_file(path, receipt_folder, relatives, role):
    content = Path(path).read_bytes()
    require(any((Path(receipt_folder)/r).is_file() and
                (Path(receipt_folder)/r).read_bytes() == content for r in relatives),
            role + ' differs from actual receipt byte snapshots')


def local_remote(remote_path, remote_attempt, local_attempt):
    relative = PurePosixPath(remote_path).relative_to(PurePosixPath(remote_attempt))
    return Path(local_attempt)/str(relative)


def validate_population(contract, objects, roster):
    require(len(roster) == 1469 and len(set(roster)) == 1469, 'Require complete unique old dev1469 roster')
    actual = contract.get('actual_loader_roster', [])
    require(contract.get('roster') == roster and contract.get('expected_val_images') == 1469 and
            contract.get('observed_images') == 1469 and len(actual) == 1469 and
            len(set(actual)) == 1469 and set(actual) == set(roster), 'Incomplete actual dev1469 contract/loader')
    geometry = {}
    with gzip.open(objects, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            image = row['image']
            require(isinstance(image, str) and bool(image), 'Missing actual object image identity')
            for key in ('canvas_shape', 'original_shape'):
                shape = row.get(key)
                require(isinstance(shape, list) and len(shape) == 2 and
                        all(type(v) in (int, float) and v > 0 and int(v) == v for v in shape),
                        'Invalid actual image geometry: '+key)
            require(isinstance(row.get('gt_boxes'), list) and isinstance(row.get('gt_classes'), list)
                    and len(row['gt_boxes']) == len(row['gt_classes']), 'Actual GT arrays differ in length')
            require(image not in geometry, 'Duplicate actual object image')
            geometry[image] = {key: row[key] for key in
                               ('canvas_shape', 'original_shape', 'gt_boxes', 'gt_classes')}
            # This bridge compares observed geometry; it does not recompute AP,
            # classify object errors, or assume requested imgsz equals canvas.
            json.dumps(geometry[image], allow_nan=False)
    require(set(geometry) == set(roster) and len(geometry) == 1469, 'Actual objects population differs')
    return geometry


def validate_observed(old, spec, folder, remote_attempt, analyzer, wrapper,
                      canonical_order, candidate_root, candidate, wrapper_review, analyzer_root):
    """Read actual endpoint files; returned paths are inputs, not new receipts."""
    folder = Path(folder)
    metric, complete, receipt, contract, origin, comparison = [read(folder/name) for name in (
        'evaluation_val.json', 'reevaluation_receipt.json', 'eval_evidence/run_receipt.json',
        'evaluation_contract.json', 'origin_manifest.json', 'historical_metric_comparison.json')]
    cfg = yaml.safe_load((folder/'evaluation_config.yaml').read_text(encoding='utf-8'))
    metadata = wrapper.load_legacy_metadata(old, spec['arm'], spec['seed'], cfg, analyzer)
    old_metric, old_cfg = metadata['metric'], metadata['train_config']
    checkpoint = old_metric['checkpoint']
    arm = METHODS[spec['arm']]
    remote_eval = remote_attempt + '/evaluation_val.json'
    require(complete.get('schema') == 'rgbir-legacy-reevaluation-v1' and complete.get('status') == 'completed'
            and complete.get('full_dev_images') == 1469 and complete.get('all_class_metrics') == 5
            and complete.get('historical_five_metrics_exact') is True, 'Incomplete actual reevaluation receipt')
    for value in (complete, metric):
        require(value.get('seed') == spec['seed'] and value.get('arm') == arm and
                value.get('normalized_method_arm') == spec['arm'] and value.get('checkpoint') == checkpoint
                and value.get('official_test_accessed') is False, 'Actual reevaluation identity differs')
    require(complete.get('old_results_modified') is False and complete.get('new_training_receipt_created') is False,
            'Reevaluation must not create/replace old training evidence')
    require(complete.get('evaluation_val') == remote_eval and complete.get('evaluation_receipt') ==
            remote_attempt + '/eval_evidence/run_receipt.json', 'Reevaluation output origin differs')
    require(metric.get('status') == 'completed' and metric.get('metric_units') == 'fraction_0_to_1' and
            metric.get('endpoint') == 'fixed_budget_last_ema' and metric.get('split') == 'val' and
            metric.get('source') == 'paired' and metric.get('method_id') == old_metric['method_id'] and
            metric.get('dataset') == old_cfg['dataset'], 'New metric identity/units differ')
    require(metric.get('evaluation_kind') == 'posthoc_legacy_checkpoint_diagnostics', 'Missing post-hoc execution identity')
    expected_comparison = wrapper.historical_comparison(old_metric, metric)
    require(expected_comparison['status'] == 'EXACT' and comparison == expected_comparison,
            'Actual old/new five metrics are not exactly equal')
    wrapper.require_all_classes(metric, 5)
    require(receipt.get('terminal_status') == 'COMPLETED' and receipt.get('run_kind') == 'eval' and
            receipt.get('data_role') == 'development_val' and receipt.get('seed') == spec['seed'] and
            receipt.get('dataset') == old_cfg['dataset'] and receipt.get('resources', {}).get('lease_id'),
            'New eval receipt execution identity is incomplete')
    for key in ('method_id', 'arm', 'source', 'checkpoint', 'endpoint', 'evaluation_kind'):
        require(receipt.get('inputs', {}).get(key) == metric.get(key), 'New receipt input differs: '+key)
    require(receipt['inputs'].get('new_training_receipt_created') is False and
            receipt['inputs'].get('original_training_run') == complete.get('old_run') ==
            str(PurePosixPath(checkpoint).parent.parent), 'New receipt lost original training origin')
    sources = receipt['source_snapshots']
    source_names = sources['trainer']
    require(source_names == canonical_order + list(EXTRA_SOURCES), 'Actual evaluator source set/order differs')
    evidence = folder/'eval_evidence'
    for group in sources.values():
        require(all((evidence/name).is_file() for name in group), 'Missing actual new source/config/roster snapshot')
    objects = local_remote(metric['objects'], remote_attempt, folder)
    require(objects == folder/'predictions/objects.jsonl.gz' and metric['evaluation_contract'] ==
            remote_attempt+'/evaluation_contract.json', 'Metric paths do not identify its actual attempt')
    for path in (folder/'evaluation_val.json', objects, folder/'historical_metric_comparison.json'):
        bound_file(path, evidence, receipt['metric_snapshots'], path.name)
    for name in ('evaluation_config.yaml', 'evaluation_contract.json', 'origin_manifest.json'):
        bound_file(folder/name, evidence, sources['config'], name)
    bound_file(folder/'evaluation_val_roster.txt', evidence, sources['split_roster'], 'roster')
    require((folder/'evaluation_val_roster.txt').read_text(encoding='utf-8').splitlines() == metadata['record']['roster'],
            'New bound roster differs from actual old roster')
    bound_cfg, _ = analyzer._bound_configs(folder, receipt, 'eval_evidence')
    require(bound_cfg == cfg, 'New configuration is not its receipt-bound configuration')
    require(origin.get('original_checkpoint') == checkpoint and origin.get('original_run') == complete['old_run']
            and origin.get('seed') == spec['seed'] and origin.get('original_arm') == arm and
            origin.get('new_training_receipt_created') is False and origin.get('original_results_modified') is False,
            'Observed original-run mapping differs')
    require(origin.get('original_training_config') == old_cfg and
            origin.get('original_evaluation_config') == metadata['eval_config'], 'Observed original bound configs differ')
    # Verify old sources/configs/metrics against the newly captured origin copies.
    # Large historical train-roster files absent locally remain documented remote
    # inputs; every actual old evaluator source/config/roster is required here.
    old_files = [Path(old)/n for n in ('completion_receipt.json', 'evaluation_val.json', 'evaluation_val_roster.txt')]
    for subfolder, key in (('run_evidence', 'train_receipt'), ('eval_evidence', 'eval_receipt')):
        old_files.append(Path(old)/subfolder/'run_receipt.json')
        old_receipt = metadata[key]
        for group, names in old_receipt['source_snapshots'].items():
            if subfolder == 'run_evidence' and group == 'split_roster':
                continue
            old_files.extend(Path(old)/subfolder/name for name in names)
        old_files.extend(Path(old)/subfolder/name for name in old_receipt['metric_snapshots'])
    old_files = list(dict.fromkeys(old_files))
    for path in old_files:
        origin_path = folder/'origin_evidence'/path.relative_to(old)
        equal_bytes(path, origin_path, 'Old input differs from actual newly preserved origin: '+str(path))
    for relative, expected in zip(canonical_order, candidate['canonical_evaluator_source_copies']):
        equal_bytes(evidence/relative, candidate_root/expected, 'New actual canonical source differs: '+relative)
    equal_bytes(evidence/EXTRA_SOURCES[0], Path(analyzer_root)/'evaluator_profile.py', 'Actual profile helper changed')
    reviewed = {row['relative']: row['accepted_copy'] for row in wrapper_review['source_files']}
    accepted_wrapper = Path(reviewed['legacy_checkpoint_evaluate.py'])
    if not accepted_wrapper.is_absolute():
        accepted_wrapper = Path(wrapper_review['_receipt_path']).parent/accepted_wrapper
    equal_bytes(evidence/EXTRA_SOURCES[1], accepted_wrapper, 'Actual wrapper differs from independently reviewed source')
    old_data = [Path(old)/'eval_evidence'/r for r in metadata['eval_receipt']['source_snapshots']['config'] if 'rgb.data.yaml' in r]
    new_data = [evidence/r for r in sources['config'] if 'rgb.data.yaml' in r]
    require(len(old_data) == len(new_data) == 1, 'Require unique bound RGB data YAML')
    equal_bytes(old_data[0], new_data[0], 'Old/new actual data YAML differs')
    canonical = analyzer._validate_evaluation_contract(contract, metadata['record']['roster'], metric)
    for key in ('torch', 'ultralytics'):
        require(receipt['environment'].get(key) == contract['evaluator_identity'].get(key) ==
                metadata['eval_receipt']['environment'].get(key), 'Actual evaluator environment differs')
    for key in ('imgsz', 'batch', 'workers'):
        require(contract['effective_kwargs'].get(key) == cfg[key], 'Actual evaluator recipe differs: '+key)
    geometry = validate_population(contract, objects, metadata['record']['roster'])
    return dict(metadata=metadata, metric=metric, complete=complete, receipt=receipt, contract=contract,
                canonical=canonical, geometry=geometry, old_files=old_files, objects=objects,
                sources=source_names, comparison=comparison)


def prepare(args):
    require(not args.output.exists(), 'Preserve previous bridge attempts; select a new output')
    helper = load_module(LOG/'prepare_evaluation_bridge.py', '_observed_bridge_preparation_helpers')
    wrapper = load_module(LOG/'legacy_endpoint_eval_prepare_v1/legacy_checkpoint_evaluate.py', '_observed_bridge_wrapper')
    analyzer = helper.load_analyzer(args.analyzer_root)
    review = dict(wrapper.require_review(args.wrapper_review), _receipt_path=str(args.wrapper_review.resolve()))
    candidate = read(args.candidate)
    require(candidate.get('status') == 'DRAFT_AWAITING_INDEPENDENT_REVIEW' and not candidate.get('static_failures'),
            'Require the unchanged original six-entry draft candidate')
    completion = read(args.diagnostics/'dispatch_attempt1/completion.json')
    require(completion.get('status') == 'COMPLETED', 'Actual diagnostic dispatch is incomplete')
    source_manifest = read(args.diagnostics/'source_manifest.json')
    remote_root = source_manifest['remote_root']
    require(remote_root.endswith('/legacy_diagnostics_v1'), 'Actual diagnostics source mapping differs')
    old_manifest = read(args.old_manifest)
    specs = []
    for item in old_manifest['runs']:
        if item['arm'] not in ('N', 'C', 'C0'):
            continue
        arm = 'N' if item['arm'] == 'N' else 'C0'
        specs.append(dict(item, arm=arm, source='paired', source_arm=METHODS[arm]))
    require(len(specs) == 6 and {(r['arm'], r['seed']) for r in specs} == EXPECTED,
            'Require exactly six unique N/C0 seed0/42/123 entries')
    expected_jobs = ['ikdv2_legacy_diag_'+arm+'_s'+str(seed)+'_a1' for seed in SEEDS for arm in METHODS]
    require(completion.get('jobs') == expected_jobs, 'Actual completed dispatch does not cover the six selected endpoints')
    canonical_order = [r['expected_formal_receipt_relative'] for r in candidate['canonical_source_order']]
    require(len(canonical_order) == 7, 'Require the actual formal seven-source collection')
    formal = args.diagnostics/'N_s42_attempt1/eval_evidence'/canonical_order[0]
    derived_order = helper.formal_source_order(formal, candidate['canonical_source_order'])
    require([r['expected_formal_receipt_relative'] for r in derived_order] == canonical_order,
            'Actual formal evaluator emission order changed')
    equal_bytes(formal, args.analyzer_root/'evaluate_independent.py', 'Current frozen canonical evaluator changed')
    validated = []
    common_contract, common_geometry = None, None
    for spec in sorted(specs, key=lambda row: (SEEDS.index(row['seed']), list(METHODS).index(row['arm']))):
        old = Path(spec['path'])
        if not old.is_absolute():
            old = args.old_manifest.parent/old
        label = spec['arm']+'_s'+str(spec['seed'])+'_attempt1'
        folder, remote_attempt = args.diagnostics/label, remote_root+'/'+label
        result = validate_observed(old, spec, folder, remote_attempt, analyzer, wrapper,
            canonical_order, args.candidate.parent, candidate, review, args.analyzer_root)
        matching = [e for e in candidate['entries'] if e['checkpoint'] == result['metric']['checkpoint'] and e['seed'] == spec['seed']]
        require(len(matching) == 1 and matching[0]['bound_training_config'] == result['metadata']['train_config'] and
                matching[0]['bound_evaluation_config'] == result['metadata']['eval_config'], 'Original candidate run/config mapping differs')
        old_sources = result['metadata']['eval_receipt']['source_snapshots']['trainer']
        require(set(matching[0]['evaluation_source_copies']) == set(old_sources), 'Old evaluator source set differs from candidate')
        for relative in old_sources:
            source = old/'eval_evidence'/relative
            helper.legacy_val_arguments(source)
            equal_bytes(source, args.candidate.parent/matching[0]['evaluation_source_copies'][relative],
                        'Old evaluator source bytes differ from original candidate')
        job_id = 'ikdv2_legacy_diag_'+spec['arm']+'_s'+str(spec['seed'])+'_a1'
        resource_path = args.diagnostics/'dispatch_attempt1'/(job_id+'_resource_profile.json')
        resource = read(resource_path)
        require(resource.get('status') == 'COMPLETED' and resource.get('measurement_valid') is True and
                resource.get('stage') == 'evaluation' and resource.get('result_receipt') == remote_attempt+'/reevaluation_receipt.json',
                'Actual dispatcher did not accept this endpoint evaluation')
        if common_contract is None:
            common_contract, common_geometry = result['canonical'], result['geometry']
        else:
            require(result['canonical'] == common_contract, 'The six actual evaluator contracts differ')
            require(result['geometry'] == common_geometry, 'The six actual GT/canvas identities differ')
        validated.append((spec, old, folder, remote_attempt, result, resource_path))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    mappings = []
    def save(source, relative, remote=None):
        source, destination = Path(source), output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(source.read_bytes())
        mappings.append(dict(local_source=str(source.resolve()), source_remote=remote,
                             independent_copy=relative.replace('\\', '/'), bytes=destination.stat().st_size))
        return relative.replace('\\', '/')
    canonical_copies = [save(validated[0][2]/'eval_evidence'/name,
        'canonical_sources/'+PurePosixPath(name).name, validated[0][3]+'/eval_evidence/'+name) for name in canonical_order]
    entries, manifest_rows = [], []
    for spec, old, folder, remote, result, resource_path in validated:
        label = spec['arm']+'_s'+str(spec['seed'])
        root = 'entries/'+label
        old_sources = {}
        for path in result['old_files']:
            relative = path.relative_to(old).as_posix()
            saved = save(path, root+'/old_evidence/'+relative,
                         result['complete']['old_run']+'/'+relative)
            old_relative = relative[len('eval_evidence/'):] if relative.startswith('eval_evidence/') else None
            if old_relative in result['metadata']['eval_receipt']['source_snapshots']['trainer']:
                old_sources[old_relative] = saved
        actual_files = ['evaluation_val.json', 'evaluation_observed.json', 'reevaluation_receipt.json',
                        'evaluation_contract.json', 'evaluation_config.yaml', 'evaluation_val_roster.txt',
                        'historical_metric_comparison.json', 'origin_manifest.json', 'wrapper_review_receipt.json',
                        'eval_evidence/run_receipt.json']
        actual_files += ['eval_evidence/'+name for group in result['receipt']['source_snapshots'].values() for name in group]
        # Keep one independent objects copy; the original receipt's duplicate is
        # checked byte-for-byte above and its original location remains mapped.
        actual_files += ['predictions/objects.jsonl.gz']
        for relative in dict.fromkeys(actual_files):
            save(folder/relative, root+'/observed/'+relative, remote+'/'+relative)
        save(resource_path, root+'/observed/dispatcher_resource_profile.json',
             remote_root+'/dispatch_attempt1/'+resource_path.name)
        entry = dict(checkpoint=result['metric']['checkpoint'], seed=spec['seed'], actual_source_arm=METHODS[spec['arm']],
            bound_training_config=result['metadata']['train_config'], bound_evaluation_config=result['metadata']['eval_config'],
            evaluation_source_copies=old_sources, evaluation_contract=result['contract'],
            evaluation_contract_origin='actual post-hoc evaluation of this same fixed checkpoint, not a historical contract record',
            old_loader_order_recorded=False, old_seen_recorded=False, old_per_class_recorded=False, old_objects_recorded=False,
            observed_reevaluation=dict(remote_attempt=remote, local_snapshot=str(folder.resolve()),
                metric_copy=root+'/observed/evaluation_val.json', receipt_copy=root+'/observed/eval_evidence/run_receipt.json',
                completion_copy=root+'/observed/reevaluation_receipt.json', contract_copy=root+'/observed/evaluation_contract.json',
                objects_copy=root+'/observed/predictions/objects.jsonl.gz', source_copies={
                    name:root+'/observed/eval_evidence/'+name for name in result['sources']},
                historical_five_metric_comparison=result['comparison'], observed_images=1469, per_class_metrics_available=True),
            bridge_scope='Observed same-checkpoint equivalence of five aggregate development metrics; actual new objects/class metrics are separate post-hoc evidence')
        entries.append(entry)
        manifest_rows.append(dict(spec, evaluation_compatibility_receipt=str(output/'evaluation_compatibility_candidate.json')))
    save(args.candidate, 'input_evidence/prior_candidate.json')
    save(args.old_manifest, 'input_evidence/old_endpoint_manifest.json')
    save(args.diagnostics/'dispatch_attempt1/completion.json', 'input_evidence/dispatch_completion.json', remote_root+'/dispatch_attempt1/completion.json')
    save(args.diagnostics/'source_manifest.json', 'input_evidence/collection_source_manifest.json')
    save(args.wrapper_review, 'wrapper_review/review_receipt.json')
    for row in review['source_files']:
        path = Path(row['accepted_copy'])
        if not path.is_absolute():
            path = args.wrapper_review.parent/path
        save(path, 'wrapper_review/reviewed_sources/'+row['relative'])
    for name in ('build_observed_bridge.py', 'test_observed_bridge.py', 'README.md'):
        save(HERE/name, 'generator_sources/'+name)
    bridge = dict(schema='rgbir-observed-legacy-evaluation-bridge-v1', status='DRAFT_AWAITING_INDEPENDENT_REVIEW', reviewer=None,
        entries=entries, canonical_evaluator_source_copies=canonical_copies,
        canonical_source_order=[{'expected_formal_receipt_relative':name} for name in canonical_order],
        additional_executed_sources=list(EXTRA_SOURCES),
        additional_source_scope='Both actual helper and wrapper are retained in each new receipt and independent source copy; reviewed as data/provenance and metric serialization wrappers around the unchanged native evaluator',
        historical_native_library_source_bytes='Not saved by the old receipts; this bridge relies on six actually repeated same-checkpoint evaluations and current bound native sources',
        observation_scope='Full dev1469 and five exact aggregate metrics for each of six selected fixed last checkpoints only',
        training_admission_blocked_by_this_bridge=False, full_analysis_or_harm_accepted=False,
        official_test_accessed=False, old_results_modified=False, source_mappings=mappings,
        generator_acceptance='Awaiting root independent code/evidence review; this generator never signs ACCEPTED')
    write_new(output/'evaluation_compatibility_candidate.json', bridge)
    write_new(output/'manifest_candidate.json', dict(runs=manifest_rows, implementation_checks_passed=False))
    draft_records = [analyzer.load_endpoint(spec, output) for spec in manifest_rows]
    require(all(r['evaluation_contract_validation'] == 'failed' for r in draft_records), 'Draft unexpectedly accepted by analyzer')
    summary = dict(status='OBSERVED_DRAFT_NOT_ACCEPTED', entries=6, actual_observed_images_per_entry=1469,
        same_checkpoint_five_metrics_exact=True, actual_nine_sources_preserved=True, canonical_sources=len(canonical_copies),
        all_six_gt_canvas_equal=True, draft_rejected_by_analyzer_for_all_six=True,
        source_diagnostics=str(args.diagnostics.resolve()), official_test_accessed=False, old_results_modified=False,
        pending='Independent root review and separately accepted bridge; future formal C1 sources must still match')
    write_new(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-manifest', 'candidate', 'diagnostics', 'analyzer-root', 'wrapper-review', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
