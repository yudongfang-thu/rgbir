"""Independent local CPU verification of the explicit post-hoc class adapter.

No GPU, SSH, network, source edits, old-result edits, or ACCEPTED analysis output.
"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LOG = HERE.parent
SOURCE = LOG/'posthoc_class_adapter_v1'
NEW = LOG.parents[1]/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'


def write_new(path, text):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


manifest_path = SOURCE/'actual_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
originals = []
for row in manifest['runs']:
    folder = Path(row['path'])
    originals += [folder/name for name in ('completion_receipt.json', 'evaluation_val.json',
                  'run_evidence/run_receipt.json', 'eval_evidence/run_receipt.json', 'evaluation_val_roster.txt')]
before = {path: path.read_bytes() for path in originals}
source_names = ('posthoc_class_adapter.py', 'test_posthoc_class_adapter.py', 'prepare_actual_manifest.py',
                'README.md', 'DEVELOPMENT_FAILURES.md')
source_bytes = {name: (SOURCE/name).read_bytes() for name in source_names}

test = subprocess.run([sys.executable, '-m', 'unittest', 'test_posthoc_class_adapter', '-v'],
                      cwd=str(SOURCE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
write_new(HERE/'cpu_tests.log', test.stdout)
assert test.returncode == 0, test.stdout
assert 'Ran 17 tests' in test.stdout

command = [sys.executable, str(SOURCE/'posthoc_class_adapter.py'), '--manifest', str(manifest_path),
           '--analyzer', str(NEW/'analyze_independent.py'), '--output', str(HERE/'actual_analysis_independent.json')]
run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
write_new(HERE/'actual_execution.log', json.dumps(command, ensure_ascii=False)+'\n\n'+run.stdout)
assert run.returncode == 0, run.stdout
actual_path = HERE/'actual_analysis_independent.json'
actual = json.loads(actual_path.read_text(encoding='utf-8'))
reference = json.loads((SOURCE/'actual_analysis_attempt2.json').read_text(encoding='utf-8'))
assert actual == reference, 'Independent actual six-endpoint result differs from the submitted artifact'
records = actual['records']
assert len(records) == 6
assert {(r['arm'], r['seed']) for r in records} == {(a,s) for a in ('N','C0') for s in (0,42,123)}
assert actual['posthoc_adapter_status'] == 'DRAFT_AWAITING_INDEPENDENT_REVIEW'
assert actual['analyzer_acceptance'] == 'NOT_ACCEPTED' and actual['implementation_checks_passed'] is False

spec = importlib.util.spec_from_file_location('_independent_class_adapter', SOURCE/'posthoc_class_adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
analyzer = adapter.module(NEW/'analyze_independent.py')
for record in records:
    assert analyzer.validate_endpoint(record)['valid']
    assert set(record['per_class_percent']) == {'0','1','2','3','4'}
    assert record['per_class_provenance']['kind'] == 'separately_executed_posthoc_legacy_checkpoint_diagnostics'
    assert record['per_class_provenance']['old_per_class_recorded'] is False
    assert record['per_class_provenance']['old_files_modified'] is False
    observed = adapter.read(record['per_class_provenance']['metric'])
    for key in ('mAP50_95','AP50','AP75'):
        wanted = {str(r['class_id']): float(r[key])*100 for r in observed['per_class']}
        assert record['posthoc_class_metrics_percent'][key] == wanted
    old = adapter.read(Path(record['path'])/'evaluation_val.json')
    assert not old.get('per_class')

comparison = analyzer.paired_comparison(records, 'C0', 'N')
assert comparison['complete_three_seed_pairing'] and not comparison['issues']
harm = analyzer.harm_review(records, 'C0')
assert harm['status'] == 'REVIEW_REQUIRED' and harm['missing'] == []
assert 'Background FP/image rises on all three seeds' in harm['triggers']
background = {}
for seed in (0,42,123):
    treatment = next(r for r in records if r['arm']=='C0' and r['seed']==seed)
    baseline = next(r for r in records if r['arm']=='N' and r['seed']==seed)
    background[seed] = treatment['error_analysis']['background_fp']-baseline['error_analysis']['background_fp']
assert sorted(background.values()) == [3,5,46]
assert harm['stop_training'] is False and actual['stop_running_experiments'] is False
for dataset in actual['datasets'].values():
    assert dataset['classification']['proposed_decision'] == 'AWAIT_C1_ENDPOINTS'
    assert dataset['classification']['auto_expansion_eligible'] is False
    assert dataset['localization']['auto_expansion_eligible'] is False

# Opt-out returns only the existing original record; opted-in enrichment cannot
# mutate a caller-owned original record or replace its aggregate numbers.
first = manifest['runs'][0]
original = analyzer.load_endpoint(first, manifest_path.parent)
original_copy = copy.deepcopy(original)
with patch.object(analyzer, 'load_endpoint', return_value=original):
    enriched = adapter.load_with_posthoc(first, manifest_path.parent, analyzer)
assert original == original_copy
assert enriched['metrics_percent'] == original['metrics_percent']
optout = adapter.load_with_posthoc(dict(first,posthoc_class_metrics=False), manifest_path.parent, analyzer)
assert not optout.get('per_class_provenance') and not optout.get('per_class_percent')

written = actual_path.read_bytes()
retry = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
write_new(HERE/'existing_output_rejection.log', retry.stdout)
assert retry.returncode != 0 and 'FileExistsError' in retry.stdout
assert actual_path.read_bytes() == written
assert all(path.read_bytes() == payload for path,payload in before.items())
assert all((SOURCE/name).read_bytes() == payload for name,payload in source_bytes.items())

summary = dict(status='INDEPENDENT_CPU_CHECKS_PASSED', unit_tests=17, actual_records=6,
    actual_result_equals_submitted_attempt2=True, classes_per_record=5, metric_conversion='actual fraction x100 checked for AP/AP50/AP75',
    seed_pairs=[r['seed'] for r in comparison['pairs']], complete_three_seed_pairing=True,
    c0_minus_n_mAP_pp=comparison['stats_pp']['mAP50_95'],
    c0_harm=harm, background_fp_increase_by_seed=background,
    original_file_count_checked=len(before), original_files_unchanged=True,
    input_record_unchanged=True, optout_preserves_original_only=True, existing_output_rejected=True,
    cli_status=actual['posthoc_adapter_status'], automatic_extension=False,
    gpu_operations=0, ssh_or_network_operations=0)
write_new(HERE/'independent_execution_checks.json', json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=True))
