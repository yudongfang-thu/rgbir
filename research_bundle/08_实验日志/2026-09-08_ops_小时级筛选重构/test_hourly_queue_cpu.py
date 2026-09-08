"""CPU-only truth cases for hourly queue identity, actual resources and sample flow."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location('hourly_queue_under_test', ROOT / 'release/run_hourly_queue.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def resource():
    return dict(resources=dict(per_gpu_peak_vram_mib={'4': 7726}, peak_rss_mib=28884),
                gpu_allocated_peak_mib=6200, gpu_reserved_peak_mib=7200)


def records(arm='N', n=30):
    return [dict(arm=arm, batch=i+1, im_file=['image'+str(j) for j in range(32)],
                 cls=[[0]], bboxes=[[.5, .5, .1, .1]], teacher_cls=[[0]],
                 teacher_bboxes=[[.5, .5, .1, .1]]) for i in range(n)]


class Tests(unittest.TestCase):
    def test_resource_rounding(self):
        r = q.training_reservation(resource())
        self.assertEqual((r['vram_mib'], r['rss_mib']), (8192, 31744))

    def test_nonfinite_resources_rejected(self):
        for value in (float('nan'), float('inf'), -1, 0, True):
            x = resource(); x['gpu_allocated_peak_mib'] = value
            with self.assertRaises(ValueError): q.training_reservation(x)

    def test_missing_nvml_rejected(self):
        x = resource(); x['resources']['per_gpu_peak_vram_mib'] = {}
        with self.assertRaises(ValueError): q.training_reservation(x)

    def test_framework_peak_cannot_be_ignored(self):
        x = resource(); x['gpu_reserved_peak_mib'] = 8200
        with self.assertRaises(ValueError): q.training_reservation(x)

    def test_margin_ceiling_blocks_expansion(self):
        x = resource(); x['resources']['peak_rss_mib'] = 32000
        with self.assertRaises(ValueError): q.training_reservation(x)

    def streams(self, root, change=None):
        for arm in q.ARMS:
            p = root / 'canaries' / arm; p.mkdir(parents=True)
            rows = records(arm)
            if change and arm == 'C1': change(rows)
            (p / 'sample_stream.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')

    def test_record_equality_ignores_only_arm(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); self.streams(root)
            self.assertTrue(q.compare_streams(root)['record_content_exact'])

    def test_teacher_gt_drift_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.streams(root, lambda rows: rows[17]['teacher_bboxes'][0].__setitem__(0, .6))
            with self.assertRaises(ValueError): q.compare_streams(root)

    def test_short_stream_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / 'stream.jsonl'
            p.write_text(''.join(json.dumps(row)+'\n' for row in records(n=29)))
            with self.assertRaises(ValueError): q.first_thirty(p)

    def test_noninitial_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / 'stream.jsonl'; rows = records()
            for row in rows: row['batch'] += 10
            p.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(ValueError): q.first_thirty(p)

    def test_canary_requires_config_bytes_and_initialization(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); cfg = {'arm':'C1', 'model':'original_N42'}
            config = root/'cfg.yaml'; config.write_text('arm: C1\n')
            copied = root/'copy.yaml'; copied.write_bytes(config.read_bytes())
            x = resource(); x.update(status='HOURLY_SCREEN_CANARY_COMPLETED', arm='C1',
                 model='original_N42', successful_updates=24, config_copy=str(copied))
            receipt = root/'canary.json'; receipt.write_text(json.dumps(x))
            init = root/'initialization_check.json'; init.write_text(json.dumps(dict(status='PASS_FULL_STATE_WARM_START',
                initial_checkpoint=dict(path=str(Path(cfg['model']).resolve()), bytes=100, mtime_ns=1),
                fresh_optimizer=True, fresh_ema=True)))
            self.assertEqual(q.check_canary(receipt, cfg, config)['successful_updates'], 24)
            copied.write_text('arm: N\n')
            with self.assertRaises(ValueError): q.check_canary(receipt, cfg, config)

    def test_eval_terminal_rejects_E8(self):
        x = dict(status='HOURLY_SCREEN_EVALUATION_COMPLETED', scope='HOURLY_SCREEN_FT', arm='C1',
            seed=42, single_seed=True, endpoint=q.ENDPOINT, formal_e200_complete=False,
            new_hash_computed=False, official_test_accessed=False, epochs=3,
            full_dev_images=1469, full_dev_gt_objects=22462)
        q.check_terminal('eval', 'C1', x)
        x['epochs'] = 8
        with self.assertRaises(ValueError): q.check_terminal('eval', 'C1', x)

    def test_job_arguments_and_pinned_python(self):
        release, output = Path('/release'), Path('/campaign')
        for stage in ('canary', 'train', 'eval'):
            job = q.make_job(release, output, 'C1', stage,
                             budget={'vram_mib':8192,'rss_mib':31744}, native_contract='/original_N.yaml')
            self.assertIs(job['formal'], False)
            self.assertEqual(job['command'][0], str(q.PY))
            self.assertIn('--reference-dir', job['command'])
            if stage == 'canary': self.assertIn('--canary', job['command'])
            else: self.assertIn('--canary-receipt', job['command'])
            if stage == 'eval':
                self.assertIn('--native-contract-config', job['command'])
                self.assertEqual(job['vram_mib'], 2048)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    import sys
    receipt = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), new_hash_computed=False,
        gpu_or_ssh=False, torch_imported='torch' in sys.modules, runtime_imported='runtime' in sys.modules,
        source=q.file_stat(ROOT/'release/run_hourly_queue.py'))
    with (ROOT/'hourly_queue_cpu_checks_v2.json').open('x', encoding='utf-8') as f: json.dump(receipt, f, indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
