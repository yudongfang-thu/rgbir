"""Repeat existing CPU suite and seal reviewed sources by stat/bytes; no hashes."""
import importlib.util, io, json, sys, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
TASK=HERE.parent
WORKSPACE=HERE.parents[3]
RELEASE=WORKSPACE/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'

def write(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    spec=importlib.util.spec_from_file_location('repeat_natural_independent',HERE/'test_natural_flow_independent.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(module.IndependentNaturalTests))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),python=sys.version,torch=module.torch.__version__,
        cpu_only=True,cuda_initialized=module.torch.cuda.is_initialized(),new_hashes_computed=False,output=stream.getvalue())
    print(stream.getvalue());write(HERE/'cpu_test_attempt2_result.json',report)
    if not result.wasSuccessful():raise SystemExit(1)
    snapshot=HERE/'attempt2_reviewed_sources';snapshot.mkdir()
    sources=[TASK/p for p in ['diagnose_natural_flow.py','run_campaign.py','remote_ops.py','PROTOCOL.md']]
    sources += [HERE/p for p in ['test_attempt2_import.py','test_natural_flow_independent.py','seal_attempt2_review.py']]
    sources += [RELEASE/p for p in ['runtime.py','prepare_configs.py','coverage_probe.py','selection_adapter.py',
        'task_conditional_reference/localization_loss.py','task_conditional_reference/prepare_configs.py']]
    manifest=[]
    for source in sources:
        key=source.relative_to(WORKSPACE);target=snapshot/key;target.parent.mkdir(parents=True,exist_ok=True)
        before=source.stat();data=source.read_bytes();target.write_bytes(data);after=source.stat()
        assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
        assert source.read_bytes()==target.read_bytes()
        manifest.append(dict(source=str(source),snapshot=str(target),bytes=after.st_size,mtime_ns=after.st_mtime_ns,byte_identity=True))
    write(HERE/'attempt2_source_stat_manifest.json',dict(new_hash_computed=False,sources=manifest))
    receipt=dict(date='2026-09-07',auditor='/root/ap_error',executor='/root/baseline_feature_analysis',
        repair_author='/root',overall_verdict='pass',integrity_status='pass',status='PASS_FOR_REAL_CANARY',
        reason_code='EXPLICIT_RELEASE_CONFIG_IMPORT_AND_19_CPU_TESTS_PASS',
        supersedes_execution_clearance_only='EXPERIMENT_AUDIT.json retained; attempt1 failed before batch0',
        tests=['attempt2_import_test_result_v2.json','cpu_test_attempt2_result.json'],test_count=19,
        source_manifest='attempt2_source_stat_manifest.json',audited_input_hashes=[],new_hash_computed=False,
        hash_omission_reason='Explicit inherited user no-hash instruction; path/size/mtime/byte identity instead.',
        cuda_initialized=False,gpu_or_ssh_used_by_auditor=False,real_64_batches_accepted=False,
        training_admitted=False,calibration_admitted=False,geometry_verified=False,
        checks={
            'gt_provenance':dict(status='pass',details='Original trace and dual-label enforcement unchanged; corruption regression retained.'),
            'score_normalization':dict(status='pass',details='C whole-batch and L primary/replay semantics unchanged; 14 original independent checks repeated.'),
            'result_existence':dict(status='pass',details='5 collision tests plus 14 selection/summary checks pass; no successful real execution inferred.'),
            'dead_code':dict(status='pass',details='Executed actual current config-binding AST under original runtime sys.path prefix; wrong prepare_configs module cached and ignored.'),
            'scope':dict(status='pass',details='Source delta from retained failure is only importlib.util and explicit generator import; no new config, flow or selection change.'),
            'eval_type':dict(status='pass',details='CPU regression and synthetic_proxy fixtures only. Runtime prefix executes actual source; full heavy runtime not imported by collision fixture.')},
        claims=[dict(id='import_collision_repaired',impact='supported',evidence='attempt2_import_test_result_v2.json',
                     rationale='Actual wrong file selection and missing llvip_C1/drone_C1 reproduced; new binding returns all12 unchanged independent configs.'),
                dict(id='read_only_canary_clearance',impact='supported',evidence='19 CPU tests',rationale='Proceed to authorized real canary; later failure remains possible.'),
                dict(id='real_stream_or_KD_gain',impact='unsupported',evidence='No successful real run within review',rationale='No geometry, calibration or learning admission.')])
    write(HERE/'ATTEMPT2_IMPORT_REVIEW.json',receipt)

if __name__=='__main__':main()
