"""Materialize NOT_ADMITTED LLVIP configs and read-only provenance inspection."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import yaml

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
MODULE = WORKSPACE / '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
LOG = WORKSPACE / '08_实验日志'
SNAPSHOT = LOG / '2026-09-07_train_IndependentKD实施/remote_admission_1532/compat_C0_s0_attempt1/implementation_snapshot'
MODEL_INFO = LOG / '2026-09-07_probe_Baseline蒸馏机会重诊断/remote_exports/llvip_full_attempt1/model_identity.json'
KEY_FILES = ('prepare_configs.py', 'protocol.py', 'train_independent.py', 'independent_criterion.py',
             'classification_logit.py', 'localization_adapter.py', 'geometry_support.py', 'admission.py',
             'calibrate_independent.py', 'coverage_probe.py', 'evaluate_independent.py', 'evaluator_profile.py',
             'resource_dispatch.py', 'runtime.py', 'task_conditional_reference/configs/llvip_draft.yaml',
             'task_conditional_reference/legacy_oev1/train_object_evidence.py',
             'task_conditional_reference/legacy_oev1/paired_rgbir_data.py',
             'task_conditional_reference/tracked_pair_data.py')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=HERE / 'prepared_v1')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    base = load_module('_llvip_draft_preparer', MODULE / 'prepare_configs.py').configurations()
    protocol = load_module('_llvip_protocol', MODULE / 'protocol.py')
    model_identity = json.loads(MODEL_INFO.read_text(encoding='utf-8'))
    manifest = {'status': 'NOT_ADMITTED', 'configs': [], 'new_training_started': False,
                'lambda_invented': False, 'readiness_generated': False, 'hash_computed': False,
                'source_module_local': str(MODULE), 'bound_snapshot_local': str(SNAPSHOT),
                'bound_release_remote': '/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5',
                'sources': []}
    for relative in KEY_FILES:
        current, bound = MODULE / relative, SNAPSHOT / relative
        copy_to = a.out / 'source_copies' / relative
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, copy_to)
        manifest['sources'].append({'relative': relative, 'current': str(current), 'snapshot': str(bound),
                                    'current_size': current.stat().st_size,
                                    'snapshot_exists': bound.is_file(),
                                    'byte_equal_to_bound_snapshot': current.read_bytes() == bound.read_bytes() if bound.is_file() else None})
    for arm in ('N', 'L1', 'L_GT'):
        for seed in (42, 0, 123):
            cfg = copy.deepcopy(base['llvip_' + arm])
            cfg.update(seed=seed, protocol_status='NOT_ADMITTED', formal_training_authorized=False,
                       preparation_status='NOT_ADMITTED', readiness_receipt=None,
                       calibration_receipt=None, canary_acceptance=None, geometry_contract=None, d2_receipt=None)
            cfg['run_policy'] = {'allow_joint_kd': False, 'allow_new_routing': False}
            cfg['preparation_notes'] = {'not_a_launch_authorization': True,
                'N_endpoint_reuse': 'old_visible42_is_reference_only_workers8_not_new_N_workers4',
                'lambda_source_required': 'actual_llvip_train_mode_64_batch_calibration' if arm != 'N' else 'not_applicable_zero_KD',
                'evaluator_profile': 'NOT_AVAILABLE_LL VIP_requires_2406_generalization_and_real_profile'.replace('LL VIP', 'LLVIP')}
            assert cfg['teacher'] == model_identity['T42']['path']
            assert cfg['reference'] == model_identity['N42']['path']
            assert cfg['expected_nc'] == 1 and cfg['expected_train_images'] == 9619 and cfg['expected_val_images'] == 2406
            assert cfg['classification_coefficient'] == 0
            assert cfg['localization_coefficient'] == (0 if arm == 'N' else None)
            draft_check = protocol.validate_config(cfg, formal=False)
            assert draft_check['valid'], draft_check
            path = a.out / ('llvip_' + arm + '_seed' + str(seed) + '_NOT_ADMITTED.yaml')
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
            manifest['configs'].append({'path': str(path), 'arm': arm, 'seed': seed,
                                        'draft_logic_valid': True, 'formal_training_admitted': False,
                                        'configuration_only_formal_check': protocol.validate_config(cfg, formal=True)})
    template = base['llvip_N']
    compare_keys = ('model', 'epochs', 'batch', 'nbs', 'imgsz', 'workers', 'optimizer', 'lr0', 'lrf',
                    'momentum', 'weight_decay', 'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
                    'cos_lr', 'close_mosaic', 'patience', 'amp', 'deterministic')
    recipe = {'new_N_family': {k: template[k] for k in compare_keys},
              'old_models': {}, 'new_dataset_paths': template['paths'], 'expected_nc': 1,
              'new_native_implementation': 'legacy ObjectEvidenceTrainer + TrackedDualLabelRGBIRDataset + original weight0 auxiliary path',
              'all_training_transform_values': template['augmentation']}
    for key, identity in model_identity.items():
        args = identity['args']
        comparison = {k: {'old': args.get(k), 'new': template[k], 'equal': args.get(k) == template[k]} for k in compare_keys}
        comparison.update({k: {'old': args.get(k), 'new': v, 'equal': args.get(k) == v} for k, v in template['augmentation'].items()})
        recipe['old_models'][key] = {'checkpoint': identity['path'], 'names': identity['names'],
                                      'seed': args['seed'], 'data': args['data'], 'comparison': comparison,
                                      'differences': [k for k, v in comparison.items() if not v['equal']]}
    save(a.out / 'recipe_and_model_comparison.json', recipe)
    split_record = {}
    for role in ('train', 'val'):
        folder = LOG / ('2026-09-07_probe_TaskConditional机会诊断/llvip_' + role)
        roster = json.loads((folder / 'frozen_roster.json').read_text(encoding='utf-8'))
        split_record[role] = {'source': str(folder / 'frozen_roster.json'),
                              'population_images': roster['population_images'], 'diagnostic_subset_n': roster['count'],
                              'source_groups': sorted({r['source_group'] for r in roster['roster']}),
                              'image_path_prefixes': sorted({str(Path(r['rgb_path']).parent) for r in roster['roster']}),
                              'sample_stems': [r['stem'] for r in roster['roster']]}
    split_record['sample_train_val_stem_overlap'] = sorted(set(split_record['train']['sample_stems']) & set(split_record['val']['sample_stems']))
    split_record['sample_source_group_overlap'] = sorted(set(split_record['train']['source_groups']) & set(split_record['val']['source_groups']))
    split_record['scope'] = 'Saved full-population counts and diagnostic subsets; not a new full dataset file audit'
    for name in ('02_visible.data.yaml', '03_infrared.data.yaml'):
        source = LOG / '2026-09-07_probe_TaskConditional机会诊断/llvip_train/run_evidence/source_snapshot/config' / name
        value = yaml.safe_load(source.read_text(encoding='utf-8'))
        assert 'test' not in value and value['names'] == {0: 'person'} and value['train'] == 'images/fit' and value['val'] == 'images/dev'
        copy_to = a.out / 'data_yaml_source_copies' / name
        copy_to.parent.mkdir(exist_ok=True)
        shutil.copy2(source, copy_to)
        split_record[name] = value
    save(a.out / 'split_identity.json', split_record)
    save(a.out / 'preparation_manifest.json', manifest)
    print(json.dumps({'status': 'NOT_ADMITTED', 'config_count': len(manifest['configs']),
                      'source_differences': [x['relative'] for x in manifest['sources'] if x['byte_equal_to_bound_snapshot'] is not True],
                      'old_recipe_differences': {k: v['differences'] for k, v in recipe['old_models'].items()},
                      'sample_source_group_overlap': split_record['sample_source_group_overlap']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
