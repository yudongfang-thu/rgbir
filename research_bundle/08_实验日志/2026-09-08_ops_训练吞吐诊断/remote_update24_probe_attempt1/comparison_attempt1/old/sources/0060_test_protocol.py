import copy
import unittest

from protocol import validate_config, core_budget, identity_errors


def config(arm='C1'):
    return {'arm': arm, 'source': 'paired', 'seed': 42, 'classification_coefficient': 0.03,
            'localization_coefficient': 0, 'method_id': 'frozen-test', 'model': 'generic.pt',
            'teacher': 'ir.pt', 'reference': 'rgb.pt', 'epochs': 200, 'imgsz': 640,
            'batch': 32, 'nbs': 64, 'workers': 4, 'paths': {'student_data_yaml': 'rgb.yaml'}}


class ProtocolTests(unittest.TestCase):
    def test_formal_logic_does_not_claim_receipt_acceptance(self):
        cfg = config()
        original = copy.deepcopy(cfg)
        result = validate_config(cfg, formal=True)
        self.assertTrue(result['valid'])
        self.assertFalse(result['formal_receipts_verified'])
        self.assertEqual(cfg, original)

    def test_joint_and_unknown_arm_rejected(self):
        self.assertFalse(validate_config(config('CL'))['valid'])
        cfg = config(); cfg['localization_coefficient'] = 0.1
        self.assertFalse(validate_config(cfg)['valid'])

    def test_single_task_coefficients(self):
        cfg = config('L1'); cfg.update(classification_coefficient=0, localization_coefficient=0.2)
        self.assertTrue(validate_config(cfg)['valid'])
        cfg['classification_coefficient'] = 0.1
        self.assertFalse(validate_config(cfg)['valid'])

    def test_native_zero_and_c0_unchanged(self):
        cfg = config('N'); cfg['classification_coefficient'] = 0
        self.assertTrue(validate_config(cfg)['valid'])
        cfg['source'] = 'same_modal'
        self.assertFalse(validate_config(cfg)['valid'])
        cfg = config('C0'); cfg['classification_coefficient'] = 0.1
        self.assertTrue(validate_config(cfg)['valid'])
        cfg['classification_coefficient'] = 0.1001
        self.assertFalse(validate_config(cfg)['valid'])

    def test_calibration_not_invented(self):
        cfg = config(); cfg['classification_coefficient'] = None
        self.assertTrue(validate_config(cfg)['valid'])
        self.assertFalse(validate_config(cfg, formal=True)['valid'])
        for value in (float('nan'), float('inf'), -1, 0, 1.01, True):
            cfg['classification_coefficient'] = value
            self.assertFalse(validate_config(cfg)['valid'])

    def test_same_modal_teacher_not_reference_or_ir_mask(self):
        cfg = config(); cfg.update(source='same_modal', teacher='rgb.pt')
        self.assertFalse(validate_config(cfg)['valid'])
        cfg['teacher'] = 'independent_rgb.pt'; cfg['teacher_labels_used_by_kd'] = True
        self.assertFalse(validate_config(cfg)['valid'])
        cfg['teacher_labels_used_by_kd'] = False
        self.assertTrue(validate_config(cfg)['valid'])

    def test_content_and_seed_are_frozen(self):
        self.assertTrue(identity_errors('C1', 'random', 42))
        self.assertTrue(identity_errors('C1', 'paired', True))
        cfg = config('C1_y'); cfg['classification'] = {'off_target_weight': 0.25}
        self.assertFalse(validate_config(cfg)['valid'])
        cfg['classification']['off_target_weight'] = 0
        self.assertTrue(validate_config(cfg)['valid'])

    def test_authorized_fallback_budget(self):
        self.assertEqual(core_budget()['max_core'], 12)
        self.assertEqual(core_budget()['both_new_winners_with_four_arms'], 24)
        llvip = core_budget('llvip')
        self.assertEqual(llvip['max_core'], 15)
        self.assertEqual(llvip['first_two_line_stage'], 10)
        self.assertEqual(llvip['both_new_winners_with_four_arms'], 27)
        self.assertFalse(llvip['is_launch_authorization'])
        self.assertEqual(core_budget(localization_enabled=False, classification_ablation=False)['max_core'], 3)


if __name__ == '__main__':
    unittest.main()
