"""Known-outcome checks for the new orchestration, without training or GPU use."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from budget_policy import forecast, require_e200, require_full_evaluation, write_json
from run_budgeted import RunStopped, run_process


class BudgetTests(unittest.TestCase):
    def test_five_epoch_gate_and_fixed_reserve(self):
        self.assertFalse(forecast([{'epoch_seconds': 125}]*4, 550)['eligible'])
        result = forecast([{'epoch_seconds': 125}]*5, 700)
        self.assertEqual(result['projected_total_seconds'], 28675)
        self.assertTrue(result['eligible'])
        self.assertFalse(result['reads_ap'])

    def test_slow_run_and_recent_timing(self):
        rows = [{'epoch_seconds': 20}]*2+[{'epoch_seconds': 170}]*3
        self.assertTrue(forecast(rows, 700)['stop_for_budget'])
        self.assertEqual(forecast(rows, 700)['recent_mean_epoch_seconds'], 170)

    def test_partial_endpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            write_json(p/'completion_receipt.json', dict(status='training_completed', last_epoch=199,
                       epochs_configured=200, official_test_accessed=False))
            with self.assertRaises(RuntimeError):
                require_e200(p)
            write_json(p/'completion_receipt.json', dict(status='training_completed', last_epoch=200,
                       epochs_configured=200, official_test_accessed=False))
            self.assertEqual(require_e200(p)['last_epoch'], 200)

    def test_full_dev_and_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            write_json(p/'contract.json', dict(observed_images=1468, expected_val_images=1469))
            write_json(p/'evaluation_val.json', dict(status='completed', endpoint='fixed_budget_last_ema',
                       split='val', official_test_accessed=False, metric_units='fraction_0_to_1',
                       evaluation_contract=str(p/'contract.json')))
            with self.assertRaises(RuntimeError):
                require_full_evaluation(p)
            write_json(p/'contract.json', dict(observed_images=1469, expected_val_images=1469))
            self.assertEqual(require_full_evaluation(p)['split'], 'val')

    def test_successful_child_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            for code in (0, 3):
                result = run_process([sys.executable,'-c',f'raise SystemExit({code})'],
                    Path(directory)/f'{code}.log', time.monotonic()+10, lambda: None, .02)
                self.assertEqual(result['exit_code'], code)

    def test_deadline_does_not_launch_late_child(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'late.log'
            with self.assertRaises(RunStopped) as caught:
                run_process([sys.executable,'-c','raise SystemExit(99)'], p, time.monotonic()-1, lambda: None)
            self.assertEqual(caught.exception.status, 'BUDGET_ABORTED')
            self.assertFalse(p.exists())

    @unittest.skipIf(sys.platform == 'win32', 'POSIX process-group cancellation is tested on94')
    def test_deadline_terminates_running_child(self):
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            with self.assertRaises(RunStopped) as caught:
                run_process([sys.executable,'-c','import time; time.sleep(60)'],
                    Path(directory)/'limit.log', started+.25, lambda: None, .02)
            self.assertEqual(caught.exception.status, 'BUDGET_ABORTED')
            self.assertLess(time.monotonic()-started, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
