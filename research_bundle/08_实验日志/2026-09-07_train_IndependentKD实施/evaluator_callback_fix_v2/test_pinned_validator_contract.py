"""Static contracts from actual read-only 94 Ultralytics 8.4.115 source copies.

This checks installed-source interfaces, not AP equivalence or GPU execution.
"""
import ast
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[2] / '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'


def method(filename, class_name, method_name):
    tree = ast.parse((HERE / 'pinned_8_4_115' / filename).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


class InstalledValidatorContractTests(unittest.TestCase):
    def test_model_val_passes_actual_loaded_network_to_validator(self):
        node = method('engine_model.py', 'Model', 'val')
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'validator']
        self.assertTrue(any(k.arg == 'model' and isinstance(k.value, ast.Attribute)
            and isinstance(k.value.value, ast.Name) and k.value.value.id == 'self'
            and k.value.attr == 'model' for c in calls for k in c.keywords))

    def test_validator_has_local_backend_loader_before_start_and_metrics_after(self):
        node = method('engine_validator.py', 'BaseValidator', '__call__')
        callbacks = [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'run_callbacks'
            and n.args and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == 'on_val_start']
        self.assertEqual(len(callbacks), 1)
        start = callbacks[0].lineno
        attributes = [n for n in ast.walk(node) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == 'self']
        self.assertFalse(any(n.attr == 'model' and isinstance(n.ctx, ast.Store) for n in attributes))
        self.assertTrue(any(n.attr == 'dataloader' and isinstance(n.ctx, ast.Store)
                            and n.lineno < start for n in attributes))
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == 'AutoBackend' and n.lineno < start for n in ast.walk(node)))
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                            and n.func.attr == 'init_metrics' and n.lineno > start for n in ast.walk(node)))

    def test_installed_detection_batch_and_predictions_match_evidence_reader(self):
        node = method('detection_val.py', 'DetectionValidator', '_prepare_batch')
        keys = {key.value for n in ast.walk(node) if isinstance(n, ast.Return)
                and isinstance(n.value, ast.Dict) for key in n.value.keys
                if isinstance(key, ast.Constant)}
        self.assertTrue({'cls', 'bboxes', 'ori_shape', 'imgsz', 'ratio_pad', 'im_file'} <= keys)
        post = method('detection_val.py', 'DetectionValidator', 'postprocess')
        pred_keys = {key.value for n in ast.walk(post) if isinstance(n, ast.Dict)
                     for key in n.keys if isinstance(key, ast.Constant)}
        self.assertTrue({'bboxes', 'conf', 'cls'} <= pred_keys)
        metrics = method('detection_val.py', 'DetectionValidator', 'update_metrics')
        self.assertTrue(any(isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute)
                            and n.target.attr == 'seen' for n in ast.walk(metrics)))

    def test_evaluators_callbacks_do_not_require_validator_model(self):
        for filename in ('evaluator_profile.py', 'evaluate_independent.py'):
            tree = ast.parse((MODULE / filename).read_text(encoding='utf-8'))
            self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'model'
                and isinstance(n.value, ast.Name) and n.value.id in ('v', 'validator')
                for n in ast.walk(tree)), filename)
        source = (MODULE / 'evaluator_profile.py').read_text(encoding='utf-8')
        self.assertIn('BaseValidator,AutoBackend,YOLO.val', source)
        self.assertIn('capture_runtime_sources(v,model,sources)', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
