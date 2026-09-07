"""CPU-only regression of real frozen path collisions; no GPU/SSH/hash."""
import ast, difflib, importlib, importlib.util, io, json, sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
WORKSPACE = HERE.parents[3]
RELEASE = WORKSPACE / '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
RUNTIME = RELEASE / 'runtime.py'
CURRENT = TASK / 'diagnose_natural_flow.py'
FAILED = TASK / 'remote_failed_attempt1/diagnose_natural_flow.py'


def frozen_runtime_path_prefix():
    """Execute original runtime statements preceding heavy train/model imports."""
    tree = ast.parse(RUNTIME.read_text(encoding='utf-8'))
    end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Import)
               and any(alias.name == 'train_object_evidence' for alias in node.names))
    ns = {'__file__': str(RUNTIME)}
    exec(compile(ast.Module(body=tree.body[:end], type_ignores=[]), str(RUNTIME), 'exec'), ns)
    return ns


def executed_current_config_binding():
    tree = ast.parse(CURRENT.read_text(encoding='utf-8'))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    start = next(i for i, node in enumerate(run.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'config_spec' for t in node.targets))
    end = next(i for i, node in enumerate(run.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'configurations' for t in node.targets))
    ns = dict(importlib=importlib, release=RELEASE.resolve())
    exec(compile(ast.Module(body=run.body[start:end+1], type_ignores=[]), str(CURRENT), 'exec'), ns)
    return ns


class ImportRegression(unittest.TestCase):
    def setUp(self):
        self.old_path = list(sys.path)
        self.previous = sys.modules.pop('prepare_configs', None)
        sys.path.insert(0, str(RELEASE))
        self.runtime = frozen_runtime_path_prefix()

    def tearDown(self):
        sys.path[:] = self.old_path
        sys.modules.pop('prepare_configs', None)
        if self.previous is not None:
            sys.modules['prepare_configs'] = self.previous

    def test_01_actual_runtime_prefix_is_collision_order(self):
        self.assertEqual([Path(p).resolve() for p in sys.path[:3]], [
            RELEASE/'task_conditional_reference/legacy_oev1',
            RELEASE/'task_conditional_reference', RELEASE])

    def test_02_old_import_reproduces_actual_missing_C1(self):
        old = importlib.import_module('prepare_configs')
        self.assertEqual(Path(old.__file__).resolve(), RELEASE/'task_conditional_reference/prepare_configs.py')
        for key in ('llvip_C1', 'drone_C1'):
            with self.assertRaises(KeyError):
                old.configurations()[key]

    def test_03_new_binding_ignores_real_cached_wrong_module(self):
        wrong = importlib.import_module('prepare_configs')
        bound = executed_current_config_binding()
        self.assertEqual(Path(bound['config_module'].__file__).resolve(), RELEASE/'prepare_configs.py')
        self.assertIs(sys.modules['prepare_configs'], wrong)
        generated = bound['configurations']()
        self.assertEqual(generated['llvip_C1']['dataset'], 'llvip')
        self.assertEqual(generated['drone_C1']['dataset'], 'dronevehicle')
        self.assertEqual(generated['llvip_C1']['arm'], 'C1')
        self.assertEqual(generated['drone_C1']['arm'], 'C1')

    def test_04_all_generated_configs_equal_unambiguous_generator(self):
        bound = executed_current_config_binding()['configurations']()
        sys.path.insert(0, str(RELEASE))
        sys.modules.pop('prepare_configs', None)
        normal = importlib.import_module('prepare_configs').configurations()
        self.assertEqual(bound, normal)
        self.assertEqual(len(bound), 12)

    def test_05_source_delta_is_only_generator_import_binding(self):
        old = FAILED.read_text(encoding='utf-8')
        now = CURRENT.read_text(encoding='utf-8')
        previous = '    from prepare_configs import configurations\n'
        replacement = (
            '    # runtime prepends legacy/reference dirs that also contain prepare_configs.py.\n'
            '    # Bind the independent-v2 generator by its actual file, not the ambiguous name.\n'
            "    config_spec = importlib.util.spec_from_file_location('natural_flow_frozen_prepare_configs', release / 'prepare_configs.py')\n"
            '    config_module = importlib.util.module_from_spec(config_spec)\n'
            '    config_spec.loader.exec_module(config_module)\n'
            '    configurations = config_module.configurations\n')
        self.assertEqual(old.count(previous), 1)
        self.assertEqual(now, old.replace('import importlib\n', 'import importlib.util\n').replace(previous, replacement))


def main():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ImportRegression))
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), python=sys.version,
        runtime_prefix_executed_from_original_ast=True, full_runtime_imported=False,
        actual_source_collision_tested=True, gpu_or_ssh_used=False, new_hash_computed=False,
        output=stream.getvalue())
    print(stream.getvalue())
    out = HERE/'attempt2_import_test_result_v2.json'
    with out.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__': main()
