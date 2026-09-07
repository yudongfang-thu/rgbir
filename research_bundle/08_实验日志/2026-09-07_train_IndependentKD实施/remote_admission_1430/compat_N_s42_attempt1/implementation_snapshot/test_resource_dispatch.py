"""CPU only: real guard policy with a test lock and synthetic process/GPU data."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import resource_dispatch as dispatch


def rows(count=8):
    return {g: dict(memory_total_mib=24564, memory_free_mib=24564,
                    memory_used_mib=0, all_process_pids=[]) for g in range(count)}


def usage(active=()):
    return dict(active_gpus=list(active), project_rss_mib=0, effective_rss_mib=0,
                effective_vram_mib={}, actual_vram_mib={}, effective_cuda_processes={},
                formal_trains={}, actual_cuda_pids={})


def job(stage='canary', bootstrap=True):
    return dict(id='synthetic_' + stage, kind='train', stage=stage,
                command=['python', 'synthetic.py'], profile_key='synthetic-C1-b32',
                vram_mib=16000 if bootstrap else 8000, rss_mib=32768,
                bootstrap_profile=bootstrap, formal=False,
                config='/synthetic/config.yaml', result_receipt='/synthetic/new_receipt.json')


def reservation(bootstrap=True):
    return dict(bootstrap=bootstrap, measured=not bootstrap,
                vram_mib=16000 if bootstrap else 8000, rss_mib=32768,
                profiled_second_train=not bootstrap)


class PolicyTests(unittest.TestCase):
    def test_bootstrap_may_share_external_process_with_full_margin(self):
        state = usage()
        g = rows(2)
        g[0].update(all_process_pids=[42], memory_used_mib=4285, memory_free_mib=20279)
        selected, audit = dispatch.admission_candidates(state, g, job(), reservation(), 4)
        self.assertEqual(selected, [0, 1])
        self.assertEqual(audit['empty'], [1])  # shared is still NOT completely empty
        g[0]['memory_free_mib'] = 18048
        selected, _ = dispatch.admission_candidates(state, g, job(), reservation(), 4)
        self.assertEqual(selected, [1])

    def test_pending_lease_is_not_an_empty_gpu(self):
        selected, audit = dispatch.admission_candidates(usage([0]), rows(2), job(), reservation(), 4)
        self.assertEqual(selected, [1])
        self.assertEqual(audit['empty'], [1])

    def test_fourth_gpu_requires_two_remaining_truly_empty(self):
        state = usage([0, 1, 2])
        selected, _ = dispatch.admission_candidates(state, rows(5), job(), reservation(), 4)
        self.assertEqual(selected, [])
        selected, audit = dispatch.admission_candidates(state, rows(6), job(), reservation(), 4)
        self.assertEqual(selected, [3, 4, 5])
        self.assertEqual(audit['empty'], [3, 4, 5])

    def test_installed_three_gpu_guard_is_not_overridden(self):
        selected, _ = dispatch.admission_candidates(usage([0, 1, 2]), rows(), job(), reservation(), 3)
        self.assertEqual(selected, [])

    def test_fifth_gpu_never_allowed(self):
        selected, _ = dispatch.admission_candidates(usage([0, 1, 2, 3]), rows(), job(), reservation(), 4)
        self.assertEqual(selected, [])

    def test_shared_bootstrap_never_relaxes_fourth_gpu_empty_card_rule(self):
        g=rows(6)
        g[3].update(all_process_pids=[42],memory_used_mib=4285,memory_free_mib=20279)
        selected,audit=dispatch.admission_candidates(usage([0,1,2]),g,job(),reservation(),4)
        self.assertEqual(selected,[3])  # joining external task preserves two true empty GPUs
        self.assertEqual(audit['empty'],[4,5])
        g[4].update(all_process_pids=[43],memory_used_mib=1000,memory_free_mib=23564)
        self.assertEqual(dispatch.admission_candidates(usage([0,1,2]),g,job(),reservation(),4)[0],[])

    def test_double_open_can_share_but_not_triple(self):
        state = usage([0])
        state['effective_vram_mib'][0] = 8000
        state['actual_vram_mib'][0] = 8000
        state['effective_cuda_processes'][0] = 1
        state['formal_trains'][0] = 1
        g = rows(1)
        g[0].update(memory_free_mib=16564, memory_used_mib=8000, all_process_pids=[9])
        selected, _ = dispatch.admission_candidates(state, g, job(bootstrap=False), reservation(False), 4)
        self.assertEqual(selected, [0])
        state['effective_cuda_processes'][0] = 2
        selected, _ = dispatch.admission_candidates(state, g, job(bootstrap=False), reservation(False), 4)
        self.assertEqual(selected, [])

    def test_strict_project_70_percent_and_pending_memory(self):
        state = usage([0])
        g = rows(1)
        state['effective_vram_mib'][0] = 10000
        selected, _ = dispatch.admission_candidates(state, g, job(bootstrap=False), reservation(False), 4)
        self.assertEqual(selected, [])
        state['effective_vram_mib'][0] = 7000
        g[0]['memory_free_mib'] = 16500
        selected, _ = dispatch.admission_candidates(state, g, job(bootstrap=False), reservation(False), 4)
        self.assertEqual(selected, [])  # subtract pending 7000, then reserve 8000+2048

    def test_whole_gpu_margin_and_decimal_host_hard_limit(self):
        g = rows(1)
        g[0]['memory_free_mib'] = 8000 + 2048
        self.assertEqual(dispatch.admission_candidates(usage(), g, job(bootstrap=False),
                                                      reservation(False), 4)[0], [])
        state = usage()
        state['project_rss_mib'] = (300_000_000_000 + 2**20 - 1) // 2**20
        self.assertEqual(dispatch.admission_candidates(state, rows(), job(), reservation(), 4)[0], [])

    def test_240_gib_reservation_queues(self):
        state = usage()
        state['effective_rss_mib'] = 240 * 1024 - 32768
        self.assertEqual(dispatch.admission_candidates(state, rows(), job(), reservation(), 4)[0], [])

    def test_new_short_paths_can_bootstrap_but_formal_cannot(self):
        for stage in ('calibration', 'compatibility', 'canary'):
            self.assertTrue(dispatch.measured_reservation(job(stage))['bootstrap'])
        with self.assertRaisesRegex(ValueError, 'short measured'):
            dispatch.measured_reservation(job('train'))
        bad = job()
        bad['vram_mib'] = 15999
        with self.assertRaisesRegex(ValueError, 'exactly 16000'):
            dispatch.measured_reservation(bad)

    def test_dynamic_only_and_external_queue_owned_task_rejected(self):
        for extra in ({'gpu': 0}, {'queue_owner': 'old_random_queue'}):
            with self.assertRaises(ValueError):
                dispatch.validate_job(dict(job(), **extra))

    def test_evaluation_priority(self):
        jobs = [dict(job(), id='c'), dict(job('evaluation', False), id='e', kind='eval',
                                        requires_profile='/synthetic/profile.json')]
        self.assertEqual([j['id'] for j in dispatch.ordered_jobs(jobs)], ['e', 'c'])

    def test_calibration_peak_cannot_become_training_peak(self):
        profile = dict(schema=dispatch.PROFILE_SCHEMA, status='COMPLETED', measurement_valid=True,
                       profile_key='synthetic-C1-b32', stage='calibration')
        requested = dict(job('canary', False), requires_profile='/synthetic/profile.json')
        with patch.object(dispatch, 'read_json', return_value=profile):
            with self.assertRaisesRegex(ValueError, 'different computation path'):
                dispatch.measured_reservation(requested)

    def test_same_computation_profile_requires_actual_binding_and_sufficient_peak(self):
        profile = dict(schema=dispatch.PROFILE_SCHEMA, status='COMPLETED', measurement_valid=True,
                       profile_key='synthetic-C1-b32', stage='canary',
                       resources=dict(per_gpu_peak_vram_mib={'0': 7900}, peak_rss_mib=31000))
        requested = dict(job('train', False), formal=True, requires_profile='/synthetic/profile.json')
        with patch.object(dispatch, 'read_json', return_value=profile), patch.object(dispatch, '_profile_binding') as bind:
            self.assertTrue(dispatch.measured_reservation(requested)['profiled_second_train'])
            bind.assert_called_once()
            profile['resources']['per_gpu_peak_vram_mib']['0'] = 8001
            with self.assertRaisesRegex(ValueError, 'below a measured'):
                dispatch.measured_reservation(requested)

    def test_monitor_includes_entire_gpu_and_project_memory(self):
        state, g = usage([0]), rows(1)
        state['actual_cuda_pids'][0] = 3
        state['actual_vram_mib'][0] = 18000
        state['project_rss_mib'] = 300000
        g[0]['memory_free_mib'] = 1024
        errors = dispatch.monitor_violations(state, g, {'gpus': [0]}, reservation())
        self.assertEqual(len(errors), 4)

    def test_formal_training_actual_completed_status_is_supported(self):
        resources = dict(per_gpu_peak_vram_mib={'0': 7900}, peak_rss_mib=31000)
        with patch.object(dispatch, 'read_json', return_value={'status': 'training_completed'}):
            receipt, binding = dispatch.stage_measurement(job('train', False), resources)
        self.assertEqual(receipt['status'], 'training_completed')
        self.assertIsNone(binding)

    def test_missing_receipt_or_training_config_rejected_before_launch(self):
        for key in ('config', 'result_receipt'):
            bad = job()
            bad.pop(key)
            with self.assertRaises(ValueError):
                dispatch.validate_job(bad)

    def test_two_batch_profile_only_is_resource_measurement_not_full_calibration(self):
        resources = dict(per_gpu_peak_vram_mib={'0': 7900}, peak_rss_mib=31000)
        receipt = dict(status='PROFILED', total_batches=2, optimizer_updates=0,
                       execution_binding='/synthetic/binding.json')
        short = dict(job('calibration'), calibration_profile_only=True)
        with patch.object(dispatch, 'read_json', return_value=receipt), patch.object(dispatch, '_profile_binding'):
            self.assertEqual(dispatch.stage_measurement(short, resources)[0]['status'], 'PROFILED')
            with self.assertRaisesRegex(ValueError, 'complete measured exposure'):
                dispatch.stage_measurement(job('calibration'), resources)
            receipt['optimizer_updates'] = 1
            with self.assertRaisesRegex(ValueError, 'complete measured exposure'):
                dispatch.stage_measurement(short, resources)

    def test_execution_binding_reads_yaml_config(self):
        fake = types.ModuleType('evidence_bindings')
        seen = []
        fake.validate_execution_binding = lambda path, cfg, stage, root: seen.append(cfg)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text('arm: C1\nclassification_coefficient: null\n', encoding='utf-8')
            with patch.dict(sys.modules, {'evidence_bindings': fake}):
                dispatch._profile_binding(dict(job(), config=str(path)),
                                           dict(stage='calibration', execution_binding='fixture'))
        self.assertEqual(seen[0]['arm'], 'C1')
        self.assertIsNone(seen[0]['classification_coefficient'])

    def test_xml_includes_graphics_and_compute(self):
        xml = '''<nvidia_smi_log><gpu><minor_number>3</minor_number>
        <fb_memory_usage><total>24564 MiB</total><used>19 MiB</used><free>24545 MiB</free></fb_memory_usage>
        <processes><process_info><pid>77</pid><type>G</type></process_info>
        <process_info><pid>88</pid><type>C</type></process_info></processes></gpu></nvidia_smi_log>'''
        self.assertEqual(dispatch.parse_gpu_xml(xml)[3]['all_process_pids'], [77, 88])
        with self.assertRaises(ValueError):
            dispatch.parse_gpu_xml(xml.replace('<processes>', '<missing>').replace('</processes>', '</missing>'))


class ActualGuardAtomicTests(unittest.TestCase):
    """Run the actual installed-local guard code, replacing only OS/GPU samplers."""
    @classmethod
    def setUpClass(cls):
        cls.lock = threading.RLock()
        fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_UN=2)
        fake_fcntl.flock = lambda fd, action: cls.lock.acquire() if action == 1 else cls.lock.release()
        path = Path(__file__).resolve().parents[2] / 'tools/project_resource_guard.py'
        if not path.is_file():
            # Server release copies live under artifacts, not experiments.
            path = dispatch.DEFAULT_REPO / 'tools/project_resource_guard.py'
        spec = importlib.util.spec_from_file_location('_cpu_actual_guard_fixture', path)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        with patch.dict(sys.modules, {'fcntl': fake_fcntl}):
            spec.loader.exec_module(cls.module)
        cls.module.MAX_ACTIVE_GPUS = 4  # matches read-only verified 94 constant

    def setUp(self):
        # Real flock is released by file close even on a queued exception;
        # the portable test lock has no file-close hook, so isolate each test.
        type(self).lock = threading.RLock()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.processes = {os.getpid(): {'ppid': 0, 'rss_mib': 100}}
        self.guard = self.module.ProjectResourceGuard(Path(self.temp.name) / 'leases.json',
                process_sampler=lambda: copy.deepcopy(self.processes), gpu_process_sampler=lambda: [])
        self.snapshots_in_lock = []
        self.gpus = rows()

    def sample(self):
        # The test lock exposes its owner; this catches a pre-lock GPU sample.
        self.snapshots_in_lock.append(self.lock._is_owned())
        return copy.deepcopy(self.gpus)

    def acquire(self, name):
        return dispatch.atomic_acquire(self.guard, self.module, dict(job(), id=name),
                                       reservation(), self.sample)

    def test_actual_guard_schema_and_reserved_lease_seen_by_next_acquire(self):
        first, audit1 = self.acquire('first')
        second, audit2 = self.acquire('second')
        self.assertNotEqual(first['gpus'], second['gpus'])
        self.assertEqual(audit2['active_gpus_after'], [0, 1])
        state = json.loads(self.guard.lease_file.read_text())
        self.assertEqual(state['schema'], 'jstars-project-resource-leases-v1')
        self.assertEqual(len(state['leases']), 2)
        self.assertTrue(all(self.snapshots_in_lock))

    def test_concurrent_bootstrap_calls_do_not_choose_same_empty_gpu(self):
        result, failures = [], []
        def one(i):
            try:
                result.append(self.acquire('thread_' + str(i))[0]['gpus'][0])
            except Exception as error:
                failures.append(repr(error))
        workers = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
        self.assertFalse(any(w.is_alive() for w in workers))
        self.assertEqual(failures, [])
        self.assertEqual(set(result), {0, 1})

    def test_atomic_fourth_rule_records_basis_and_rejects_fifth(self):
        for i in range(3):
            self.acquire('first_' + str(i))
        lease, audit = self.acquire('fourth')
        self.assertTrue(audit['four_gpu_exception'])
        self.assertGreaterEqual(len(audit['fully_empty_gpus_after']), 2)
        with self.assertRaises(self.module.ResourceUnavailable):
            self.acquire('fifth')

    def test_external_process_arrival_is_sampled_inside_lock(self):
        self.gpus[0]['all_process_pids'] = [654321]
        self.gpus[0].update(memory_free_mib=17000,memory_used_mib=7564)
        lease, _ = self.acquire('arrival')
        self.assertNotEqual(lease['gpus'], [0])
        self.assertTrue(all(self.snapshots_in_lock))


if __name__ == '__main__':
    unittest.main()
