"""CPU evidence-binding regression tests using explicit synthetic disk fixtures.

No test fabricates a successful GPU execution or declares formal readiness.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from evidence_bindings import (BASE_SOURCES, DATA_KEYS, STAGE_ENTRY,
    capture_execution_binding, normalized_effective_configuration,
    validate_execution_binding)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.root = self.folder / 'synthetic_release'
        self.root.mkdir()
        self.modules = {}
        for i, relative in enumerate(sorted(BASE_SOURCES | set(STAGE_ENTRY.values()) | {'optional_loaded.py'})):
            source = self.root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text('# synthetic binding fixture ' + relative + '\n', encoding='utf-8')
            module = types.ModuleType('_binding_fixture_' + str(i))
            module.__file__ = str(source)
            self.modules[module.__name__] = module
        # This source is deliberately not loaded and must not invalidate measured
        # training compatibility when its evaluation implementation is prepared.
        (self.root / 'unused_evaluation.py').write_text('# unused fixture\n')
        self.patch = patch.dict(sys.modules, self.modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        data = self.folder / 'data'
        data.mkdir()
        paths = {}
        for key in DATA_KEYS:
            source = data / (key + '.txt')
            if key.endswith('yaml'):
                source.write_bytes(b'train: images/train\r\nval: images/val\r\nnames: [synthetic]\r\n')
            else:
                source.write_bytes(b'{"synthetic": "fixture"}\r\n')
            paths[key] = str(source)
        self.cfg = dict(arm='C1', source='paired', seed=42, method_id='synthetic-test',
            dataset='dronevehicle', model='/synthetic/student.pt', teacher='/synthetic/IR.pt',
            reference='/synthetic/R.pt', epochs=200, batch=32, workers=4, expected_nc=5,
            paths=paths, augmentation=dict(scale=.5, fliplr=.5),
            evidence=dict(reference_conf=.05, teacher_conf=.25),
            classification=dict(temperature=2., off_target_weight=.25, raw_clip=16.),
            localization=dict(pair_iou=.8, temperature=2.),
            classification_coefficient=.12, localization_coefficient=0.,
            geometry_contract=None, d2_receipt=None, protocol_status='DRAFT')

    def capture(self, cfg=None, stage='canary'):
        return capture_execution_binding(self.folder / 'outputs', cfg or self.cfg, stage, self.root)

    def validate(self, path, cfg=None, stage='canary'):
        return validate_execution_binding(path, cfg or self.cfg, stage, self.root)

    def rewrite(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_real_byte_copies_and_original_paths(self):
        path = self.capture()
        result = self.validate(path)
        self.assertEqual(result['acceptance'], 'NOT_ASSERTED')
        self.assertEqual(result['actual_arm'], 'C1')
        for row in result['data_files']:
            self.assertEqual(row['original'], self.cfg['paths'][row['key']])
            self.assertEqual(Path(row['original']).read_bytes(), Path(row['copy']).read_bytes())

    def test_changed_augmentation_rejected(self):
        path = self.capture()
        cfg = copy.deepcopy(self.cfg)
        cfg['augmentation']['scale'] = .7
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            self.validate(path, cfg)

    def test_changed_classification_selection_rejected(self):
        path = self.capture()
        cfg = copy.deepcopy(self.cfg)
        cfg['evidence']['teacher_conf'] = .15
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            self.validate(path, cfg)

    def test_canary_source_and_arm_must_match(self):
        path = self.capture()
        for changes in (dict(source='shuffled'), dict(arm='C1_y')):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'configuration differs'):
                self.validate(path, dict(self.cfg, **changes))

    def test_seed_and_administrative_fields_can_change(self):
        path = self.capture()
        cfg = dict(self.cfg, seed=123, method_id='synthetic-new-seed', protocol_status='FROZEN',
                   readiness_receipt='/synthetic/ready.json', formal_training_authorized=True)
        self.validate(path, cfg)

    def test_lambda_is_preserved_for_separate_admission_check(self):
        path = self.capture()
        cfg = dict(self.cfg, classification_coefficient=.7)
        result = self.validate(path, cfg)
        self.assertEqual(result['effective_configuration']['classification_coefficient'], .12)
        self.assertNotEqual(result['effective_configuration']['classification_coefficient'],
                            cfg['classification_coefficient'])
        self.assertEqual(result['coefficient_verification'], 'admission_required')

    def test_calibration_C1_y_allows_only_predefined_payload_difference(self):
        path = self.capture(stage='calibration')
        target = copy.deepcopy(self.cfg)
        target.update(arm='C1_y', classification_coefficient=.4)
        target['classification']['off_target_weight'] = 0.
        self.validate(path, target, stage='calibration')
        for field, value in (('temperature', 3.), ('raw_clip', 8.)):
            bad = copy.deepcopy(target)
            bad['classification'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'configuration differs'):
                self.validate(path, bad, stage='calibration')

    def test_calibration_wrong_source_or_family_rejected(self):
        path = self.capture(stage='calibration')
        for changes in (dict(source='same_modal'), dict(arm='L1')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(path, dict(self.cfg, **changes), stage='calibration')

    def test_localization_GT_shares_unchanged_family_calibration(self):
        captured = dict(self.cfg, arm='L1', classification_coefficient=0., localization_coefficient=None)
        path = self.capture(captured, 'calibration')
        self.validate(path, dict(captured, arm='L_GT', localization_coefficient=.2), 'calibration')
        altered = copy.deepcopy(captured)
        altered['localization']['pair_iou'] = .7
        with self.assertRaises(ValueError):
            self.validate(path, altered, 'calibration')

    def test_compatibility_shares_native_evidence_but_not_new_carrier(self):
        captured = dict(self.cfg, arm='N', classification_coefficient=0.)
        path = self.capture(captured, 'compatibility')
        requested = copy.deepcopy(self.cfg)
        requested.update(geometry_contract='/synthetic/geometry.json', d2_receipt='/synthetic/d2.json',
                         evaluation_contract={'id': 'prepared-later'}, calibration={'batches': 64})
        requested['classification']['raw_clip'] = 12.
        self.validate(path, requested, 'compatibility')
        requested['evidence']['reference_conf'] = .10
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            self.validate(path, requested, 'compatibility')

    def test_compatibility_wrong_native_augmentation_is_not_ignored(self):
        captured = dict(self.cfg, arm='C0', classification_coefficient=.1)
        path = self.capture(captured, 'compatibility')
        bad = copy.deepcopy(self.cfg)
        bad['augmentation']['fliplr'] = 0.
        with self.assertRaises(ValueError):
            self.validate(path, bad, 'compatibility')

    def test_same_path_changed_data_bytes_rejected(self):
        path = self.capture()
        for key in DATA_KEYS:
            data = Path(self.cfg['paths'][key])
            original = data.read_bytes()
            data.write_bytes(original + b'changed')
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Data source bytes changed'):
                self.validate(path)
            data.write_bytes(original)

    def test_calibration_and_canary_bind_measured_geometry_and_d2(self):
        for stage in ('calibration', 'canary'):
            cfg = dict(self.cfg, arm='L1')
            for key in ('geometry_contract', 'd2_receipt', 'source_geometry_scope'):
                path = self.folder / (stage + '_' + key + '.json')
                path.write_text('{"synthetic": "original"}')
                cfg[key] = str(path)
            binding = self.capture(cfg, stage)
            self.validate(binding, cfg, stage)
            for key in ('geometry_contract', 'd2_receipt', 'source_geometry_scope'):
                path = Path(cfg[key])
                original = path.read_bytes()
                path.write_text('{"synthetic": "changed_selection"}')
                with self.subTest(stage=stage, key=key), self.assertRaisesRegex(ValueError, 'auxiliary input bytes changed'):
                    self.validate(binding, cfg, stage)
                path.write_bytes(original)

    def test_missing_geometry_cannot_be_bound_or_omitted(self):
        cfg = dict(self.cfg, geometry_contract=str(self.folder / 'missing.json'))
        with self.assertRaisesRegex(ValueError, 'Missing auxiliary input'):
            self.capture(cfg)
        Path(cfg['geometry_contract']).write_text('{"synthetic": true}')
        path = self.capture(cfg)
        receipt = json.loads(path.read_text())
        receipt['auxiliary_files'] = []
        self.rewrite(path, receipt)
        with self.assertRaisesRegex(ValueError, 'auxiliary input bindings'):
            self.validate(path, cfg)

    def test_referenced_train_and_val_list_files_are_byte_bound(self):
        lists = []
        for key in ('student_data_yaml', 'privileged_data_yaml'):
            yaml_file = Path(self.cfg['paths'][key])
            train_list = yaml_file.parent / (key + '_train.txt')
            val_list = yaml_file.parent / (key + '_val.txt')
            for path in (train_list, val_list):
                path.write_text('/synthetic/images/a.jpg\n')
                lists.append(path)
            yaml_file.write_text('train: ' + train_list.name + '\nval: ' + val_list.name + '\nnames: [synthetic]\n')
        binding = self.capture()
        self.assertEqual(len(self.validate(binding)['rosters_files']), 4)
        for path in lists:
            original = path.read_bytes()
            path.write_text('/synthetic/images/b.jpg\n')
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, 'rosters input bytes changed'):
                self.validate(binding)
            path.write_bytes(original)

    def test_loaded_module_edit_rejected_but_unused_eval_edit_allowed(self):
        path = self.capture()
        (self.root / 'unused_evaluation.py').write_text('# modified unused fixture\n')
        self.validate(path)
        (self.root / 'optional_loaded.py').write_text('# modified actual dependency\n')
        with self.assertRaisesRegex(ValueError, 'source bytes changed'):
            self.validate(path)

    def test_main_entry_is_captured_even_without_named_import(self):
        main = types.ModuleType('__main__')
        main.__file__ = str(self.root / STAGE_ENTRY['canary'])
        module_name = next(name for name, module in self.modules.items() if module.__file__ == main.__file__)
        with patch.dict(sys.modules, {'__main__': main}):
            del sys.modules[module_name]
            try:
                result = self.validate(self.capture())
                row = next(row for row in result['source_files'] if row['relative'] == STAGE_ENTRY['canary'])
                self.assertIn('__main__', row['module_names'])
            finally:
                sys.modules[module_name] = self.modules[module_name]

    def test_missing_actual_stage_or_runtime_module_cannot_capture(self):
        for required in (STAGE_ENTRY['canary'], 'runtime.py'):
            name = next(name for name, module in self.modules.items() if module.__file__ == str(self.root / required))
            saved = sys.modules.pop(name)
            try:
                with self.subTest(required=required), self.assertRaisesRegex(ValueError, 'not loaded'):
                    self.capture()
            finally:
                sys.modules[name] = saved

    def test_missing_record_or_stage_relabel_rejected(self):
        path = self.capture()
        original = json.loads(path.read_text())
        for change in ('source', 'data', 'stage', 'arm'):
            receipt = copy.deepcopy(original)
            if change == 'source':
                receipt['source_files'] = [r for r in receipt['source_files'] if r['relative'] != 'runtime.py']
            elif change == 'data':
                receipt['data_files'].pop()
            elif change == 'stage':
                receipt['stage'] = 'compatibility'
            else:
                receipt['actual_arm'] = 'C1_y'
            self.rewrite(path, receipt)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(path)

    def test_original_copy_cannot_point_to_itself(self):
        path = self.capture()
        receipt = json.loads(path.read_text())
        receipt['data_files'][0]['copy'] = receipt['data_files'][0]['original']
        self.rewrite(path, receipt)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.validate(path)

    def test_capture_never_overwrites_prior_evidence(self):
        path = self.capture()
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.capture()
        self.assertEqual(path.read_bytes(), original)

    def test_normalization_does_not_mutate_cfg_and_rejects_nonfinite(self):
        original = copy.deepcopy(self.cfg)
        normalized_effective_configuration(self.cfg, 'calibration')
        self.assertEqual(self.cfg, original)
        bad = dict(self.cfg, classification_coefficient=float('nan'))
        with self.assertRaises(ValueError):
            normalized_effective_configuration(bad, 'canary')


if __name__ == '__main__':
    unittest.main()
