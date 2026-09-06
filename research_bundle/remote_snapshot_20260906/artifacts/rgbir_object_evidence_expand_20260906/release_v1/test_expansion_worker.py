"""CPU-only queue control-flow checks; no trainer/loss duplication or GPU work."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import expansion_worker as worker


class FakeProcess:
    def __init__(self, code, statuses):
        self.code = code
        self.stdout = io.StringIO(''.join(json.dumps({'status': status}) + '\n' for status in statuses))

    def wait(self):
        return self.code


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(root=self.root / 'artifacts', run_root=self.root / 'runs',
                                    source=self.root / 'source', stage='full', gpu=5, seed=0,
                                    attempt='attempt1', arm_order=['weight0', 'paired'])
        self.args.root.mkdir()
        self.args.source.mkdir()
        (self.args.source / 'config_drone.yaml').write_text('seed: 42\nteacher: fixed42.pt\n')
        worker.write(worker.comparison_path(self.args), {
            'status': 'passed', 'seed': 0, 'physical_gpu': 5,
            'resource_reservation_checks_passed': True,
            'source': str(self.args.source.resolve())})

    def test_canary_resource_reservation_bounds(self):
        valid = {'gpu_ids': [5], 'cuda_pid_counts': {'5': 1},
                 'per_gpu_peak_vram_mib': {'5': 6304}, 'peak_rss_mib': 28895}
        worker.validate_canary_resources(valid, 5)
        for field, value in [('gpu_ids', [6]), ('cuda_pid_counts', {'5': 2}),
                             ('per_gpu_peak_vram_mib', {'5': 10001}), ('peak_rss_mib', 49153)]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                worker.validate_canary_resources(dict(valid, **{field: value}), 5)

    def test_admission_code2_only_retries_before_launched(self):
        processes = [FakeProcess(2, ['QUEUED']), FakeProcess(0, ['LAUNCHED'])]
        with patch.object(worker.subprocess, 'Popen', side_effect=processes) as popen, \
                patch.object(worker.time, 'sleep') as sleep:
            self.assertEqual(worker.guarded(self.args, 'test', 'train', ['python', 'train.py'],
                                            self.root / 'admission.log'), 0)
        self.assertEqual(popen.call_count, 2)
        sleep.assert_called_once_with(30)
        for call in popen.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index('--candidate-gpu') + 1], '5')
            self.assertEqual(command[command.index('--expected-rss-mib') + 1], '49152')
            self.assertNotIn('shell', call.kwargs)

    def test_child_code2_never_retried(self):
        for statuses in (['LAUNCHED'], ['QUEUED', 'LAUNCHED'], []):
            with self.subTest(statuses=statuses), \
                    patch.object(worker.subprocess, 'Popen', return_value=FakeProcess(2, statuses)) as popen, \
                    patch.object(worker.time, 'sleep') as sleep:
                path = self.root / f'child_{len(statuses)}.log'
                self.assertEqual(worker.guarded(self.args, 'test', 'train', ['train'], path), 2)
                popen.assert_called_once()
                sleep.assert_not_called()

    def test_training_failure_preserved_and_other_arm_runs(self):
        with patch.object(worker, 'guarded', side_effect=[1, 0, 0]) as guard:
            self.assertEqual(worker.run_worker(self.args), 1)
        self.assertEqual([c.args[2] for c in guard.call_args_list], ['train', 'train', 'eval'])
        output = self.args.root / 'workers/full_s0_attempt1'
        result = json.loads((output / 'status.json').read_text())
        self.assertEqual(result['status'], 'partial_failed')
        self.assertEqual([r['status'] for r in result['results']], ['training_failed', 'completed'])
        self.assertEqual(json.loads((output / 'weight0_seed_override.json').read_text())['student_seed'], 0)
        self.assertIn('seed: 0', (output / 'effective_config.yaml').read_text())
        self.assertEqual((self.args.source / 'config_drone.yaml').read_text(), 'seed: 42\nteacher: fixed42.pt\n')

    def test_eval_failure_does_not_stop_other_training(self):
        with patch.object(worker, 'guarded', side_effect=[0, 1, 0, 0]) as guard:
            self.assertEqual(worker.run_worker(self.args), 1)
        self.assertEqual([c.args[2] for c in guard.call_args_list], ['train', 'eval', 'train', 'eval'])
        result = json.loads((self.args.root / 'workers/full_s0_attempt1/status.json').read_text())
        self.assertEqual([r['status'] for r in result['results']], ['evaluation_failed', 'completed'])

    def test_existing_attempt_never_reused_and_other_arm_runs(self):
        existing = worker.run_path(self.args, 'weight0')
        existing.mkdir(parents=True)
        (existing / 'failure_receipt.json').write_text('preserve\n')
        with patch.object(worker, 'guarded', side_effect=[0, 0]) as guard:
            self.assertEqual(worker.run_worker(self.args), 1)
        self.assertEqual(guard.call_count, 2)
        self.assertEqual((existing / 'failure_receipt.json').read_text(), 'preserve\n')
        with self.assertRaises(FileExistsError), patch.object(worker, 'guarded') as guard:
            worker.run_worker(self.args)
        guard.assert_not_called()

    def test_full_requires_seed_specific_accepted_canary(self):
        self.args.seed = 123
        with patch.object(worker, 'guarded') as guard:
            self.assertEqual(worker.run_worker(self.args), 1)
        guard.assert_not_called()
        result = json.loads((self.args.root / 'workers/full_s123_attempt1/status.json').read_text())
        self.assertEqual(result['status'], 'failed_preflight')

    def test_seed_and_stage_command_names(self):
        for seed in (0, 123):
            for stage in ('canary', 'full'):
                self.args.seed, self.args.stage = seed, stage
                command = worker.train_command(self.args, 'paired')
                self.assertEqual(command[command.index('--seed') + 1], str(seed))
                self.assertEqual(worker.run_path(self.args, 'paired').name,
                                 f'{stage}_paired_s{seed}_attempt1')
                self.assertEqual('--max-steps' in command, stage == 'canary')
                if stage == 'canary':
                    self.assertEqual(command[-2:], ['--max-steps', '24'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
