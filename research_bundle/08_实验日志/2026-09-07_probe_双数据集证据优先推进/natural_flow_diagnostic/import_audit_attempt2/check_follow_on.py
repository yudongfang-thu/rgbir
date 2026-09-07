"""Read-only follow-on import/config audit with actual runtime source order."""
import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
RELEASE = WORKSPACE / '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
ENTRY = HERE.parent / 'diagnose_natural_flow.py'


def blocked(*args, **kwargs):
    raise AssertionError('External runtime operation must not execute in CPU import audit')


def package(name):
    value = types.ModuleType(name)
    value.__path__ = []
    sys.modules[name] = value
    return value


def main():
    # Only unavailable external framework/lease interfaces are substituted.
    # runtime, legacy, tracked data, loss, selection, coverage are real sources.
    framework = package('ultralytics'); framework.__version__ = 'STUB_NOT_A_RUNTIME_VERSION_CHECK'
    package('ultralytics.models'); package('ultralytics.models.yolo')
    detect = package('ultralytics.models.yolo.detect')
    detect.DetectionTrainer = type('ExternalDetectionTrainerStub', (), {})
    data = package('ultralytics.data'); data.build_yolo_dataset = blocked
    data_utils = package('ultralytics.data.utils'); data_utils.check_det_dataset = blocked
    package('tools')
    guard = package('tools.project_resource_guard')
    guard.require_bound_lease_from_environment = blocked
    guard.bound_lease_resource_record_from_environment = blocked
    receipt = package('tools.write_jstars_run_receipt')
    receipt.emit_bound_run_receipt = blocked; receipt.implementation_files = blocked

    sys.path.insert(0, str(RELEASE))
    import torch
    import yaml
    import coverage_probe
    import runtime
    from selection_adapter import build_classification_selection
    from localization_loss import LocalizationConfig, build_localization_selection
    from diagnose_opportunities import dataset_config, split_images, source_groups
    bindings = {}
    expected = {'coverage_probe': 'coverage_probe.py', 'runtime': 'runtime.py',
                'selection_adapter': 'selection_adapter.py', 'localization_loss': 'task_conditional_reference/localization_loss.py',
                'diagnose_opportunities': 'task_conditional_reference/diagnose_opportunities.py',
                'train_object_evidence': 'task_conditional_reference/legacy_oev1/train_object_evidence.py',
                'tracked_pair_data': 'task_conditional_reference/tracked_pair_data.py',
                'object_evidence_loss': 'task_conditional_reference/legacy_oev1/object_evidence_loss.py',
                'paired_rgbir_data': 'task_conditional_reference/legacy_oev1/paired_rgbir_data.py',
                'geometry_contract': 'task_conditional_reference/geometry_contract.py'}
    for name, relative in expected.items():
        path = Path(sys.modules[name].__file__).resolve()
        assert path == (RELEASE / relative).resolve(), (name, path)
        bindings[name] = str(path)
    ambiguous = importlib.import_module('prepare_configs')
    assert Path(ambiguous.__file__).resolve() == RELEASE / 'task_conditional_reference/prepare_configs.py'
    assert 'llvip_C1' not in ambiguous.configurations()
    tree = ast.parse(ENTRY.read_text(encoding='utf-8'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    spec_statement = next(n for n in run.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'config_spec' for t in n.targets))
    begin = run.body.index(spec_statement)
    last = next(i for i,n in enumerate(run.body[begin:], begin) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='configurations' for t in n.targets))
    ns = {'importlib': importlib, 'release': RELEASE}
    exec(compile(ast.Module(body=run.body[begin:last+1],type_ignores=[]),str(ENTRY),'exec'),ns)
    configurations = ns['configurations']()
    assert Path(ns['config_module'].__file__).resolve() == RELEASE / 'prepare_configs.py'
    used_keys = sorted({n.slice.value for n in ast.walk(run) if isinstance(n, ast.Subscript) and isinstance(n.value,ast.Name)
                        and n.value.id=='cfg' and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str)})
    config_result = {}
    for name, nc, population in [('llvip',1,9619),('drone',5,17990)]:
        cfg = configurations[name+'_C1']
        missing = [key for key in used_keys if key not in cfg]
        assert not missing, missing
        assert cfg['expected_nc'] == nc and cfg['batch']==32 and cfg['workers']==4 and cfg['imgsz']==640
        assert cfg['expected_train_images'] == population
        lc = LocalizationConfig(**cfg['localization'])
        from selection_adapter import evidence_config
        cc = evidence_config(cfg['evidence'])
        prior_path = WORKSPACE / '08_实验日志/2026-09-07_train_IndependentKD实施/remote_cpu1' / ('coverage_'+name+'_attempt1') / 'coverage_receipt.json'
        prior = json.loads(prior_path.read_text(encoding='utf-8'))
        assert cfg['dataset']==prior['dataset'] and cfg['augmentation']==prior['augmentation']
        assert cfg['workers']==prior['workers'] and cfg['batch']==prior['batch_size']
        assert cfg['paths']['student_data_yaml']==prior['yaml_sources'][0]
        assert cfg['paths']['privileged_data_yaml']==prior['yaml_sources'][1]
        assert cfg['paths']['paired_train_mapping']==prior['paired_mapping']
        config_result[name] = {'required_cfg_keys_complete':used_keys,'expected_nc':nc,'expected_train_images':population,
            'old_coverage_recipe_paths_match':True,'LocalizationConfig_constructed':True,'EvidenceConfig_constructed':True}
    source_assign = next(n for n in ast.walk(run) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source_files' for t in n.targets))
    copied = eval(compile(ast.Expression(body=source_assign.value),str(ENTRY),'eval'),{'Path':Path,'__file__':str(ENTRY),'release':RELEASE})
    assert len(copied)==11 and all(path.is_file() for path in copied)
    assert not torch.cuda.is_initialized()
    result = {'status':'PASSED_FOLLOW_ON_IMPORT_CONFIG_SOURCE_CHECKS',
        'actual_internal_runtime_import_order_executed':True,'internal_module_bindings':bindings,
        'old_ambiguous_generator_reproduced':str(ambiguous.__file__),
        'explicit_generator_bound':str(ns['config_module'].__file__),'configs':config_result,
        'all_11_source_snapshot_paths_exist':[str(p) for p in copied],
        'external_stubs':['ultralytics DetectionTrainer/data APIs','project_resource_guard','write_jstars_run_receipt'],
        'limitations':['Not actual pinned Ultralytics/lease runtime','Remote configs/weights existence not rechecked locally','No real data loader or model forward'],
        'cuda_initialized':False,'entry_unchanged':True,'hash_computed':False,
        'entry_stat':{'bytes':ENTRY.stat().st_size,'mtime_ns':ENTRY.stat().st_mtime_ns}}
    with (HERE/'follow_on_receipt.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ['status','actual_internal_runtime_import_order_executed','cuda_initialized','entry_stat']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
