"""CPU tests for fresh-process compatibility orchestration, not GPU acceptance."""
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import verify_compatibility as verifier


class IsolationTests(unittest.TestCase):
    def test_real_fresh_python_processes_preserve_one_time_import_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            module=Path(folder)/'fixture_one_time_random.py'
            module.write_text('import random\nvalue=random.random()\n')
            script=('import random,json,sys,importlib;sys.path.insert(0,sys.argv[1]);'
                    'random.seed(42);importlib.import_module("fixture_one_time_random");'
                    'random.sample(range(17990),5);random.sample(range(17990),5);'
                    'random.sample(range(1469),5);print(json.dumps(random.getstate()));')
            first=subprocess.check_output([sys.executable,'-c',script,folder],text=True)
            second=subprocess.check_output([sys.executable,'-c',script,folder],text=True)
            self.assertEqual(first,second)
            cold=json.loads(first)
            fresh=random.Random(42)
            for count in (17990,17990,1469):
                fresh.sample(range(count),5)
            self.assertNotEqual(cold,json.loads(json.dumps(fresh.getstate())))

    def test_coordinator_dispatches_one_blocking_child_and_validates_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'new'; output.mkdir()
            cfg=dict(seed=42,arm='N',source='paired')
            receipt=dict(status='COMPLETED',worker_stage='new',configuration=cfg,
                         fresh_interpreter=True,execution_binding='synthetic_binding.json')
            (output/'worker_receipt.json').write_text(json.dumps(receipt))
            binding=types.ModuleType('evidence_bindings')
            calls=[]
            binding.validate_execution_binding=lambda *args:calls.append(args)
            with patch.dict(sys.modules,{'evidence_bindings':binding}), \
                    patch.object(verifier.torch.cuda,'is_initialized',return_value=False), \
                    patch.object(verifier.subprocess,'run') as execute:
                got=verifier.run_isolated_phase(cfg,Path('cfg.yaml'),output,'new',Path('historical'))
            execute.assert_called_once()
            argv=execute.call_args[0][0]
            self.assertEqual(argv[0],sys.executable)
            self.assertEqual(argv[-4:],['--worker-stage','new','--historical-output','historical'])
            self.assertEqual(execute.call_args[1],{'check':True})
            self.assertEqual(got,receipt)
            self.assertEqual(len(calls),1)

    def test_cuda_parent_is_rejected_before_launch(self):
        with patch.object(verifier.torch.cuda,'is_initialized',return_value=True), \
                patch.object(verifier.subprocess,'run') as execute:
            with self.assertRaisesRegex(RuntimeError,'CUDA context'):
                verifier.run_isolated_phase({'seed':42,'arm':'N'},Path('cfg'),Path('new'),'new')
            execute.assert_not_called()

    def test_child_failure_is_not_relabelled_as_success(self):
        with patch.object(verifier.torch.cuda,'is_initialized',return_value=False), \
                patch.object(verifier.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['synthetic'])):
            with self.assertRaises(subprocess.CalledProcessError):
                verifier.run_isolated_phase({'seed':42,'arm':'N'},Path('cfg'),Path('new'),'new')

    def test_wrong_phase_or_configuration_worker_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            cfg=dict(seed=42,arm='N',source='paired')
            receipt=dict(status='COMPLETED',worker_stage='historical',configuration=cfg,
                         fresh_interpreter=True,execution_binding='fixture')
            (output/'worker_receipt.json').write_text(json.dumps(receipt))
            with patch.object(verifier.torch.cuda,'is_initialized',return_value=False), \
                    patch.object(verifier.subprocess,'run'):
                with self.assertRaisesRegex(ValueError,'identify this phase/config'):
                    verifier.run_isolated_phase(cfg,Path('cfg'),output,'new')

    def test_failed_rng_is_preserved_and_exact_comparison_still_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            old,new=Path(folder)/'old',Path(folder)/'new'
            old.mkdir();new.mkdir()
            expected={'rng':{'python':random.Random(42).getstate()},'not_rng':verifier.torch.ones(10)}
            actual={'rng':{'python':random.Random(43).getstate()},'not_rng':verifier.torch.ones(10)}
            verifier.torch.save(expected,old/'initial_trace.pt')
            observer=verifier.TrajectoryObserver(None,new,old,False,'N')
            with self.assertRaisesRegex(AssertionError,'Exact compatibility failed'):
                observer.check_or_save('initial_trace.pt',actual)
            saved=verifier.load_trace(new/'mismatch_rng_initial_trace.pt')
            self.assertEqual(saved['expected'],{'rng':expected['rng']})
            self.assertEqual(saved['actual'],{'rng':actual['rng']})
            self.assertNotIn('not_rng',saved['actual'])


if __name__=='__main__':
    unittest.main(verbosity=2)
