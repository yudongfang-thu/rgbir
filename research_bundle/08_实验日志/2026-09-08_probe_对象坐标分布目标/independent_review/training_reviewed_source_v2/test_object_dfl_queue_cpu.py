"""Queue contracts with a mock executor, never resource_dispatch/GPU/SSH."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import yaml
import run_object_dfl_queue as q

HERE = Path(__file__).resolve().parent


def calibration(dataset, checkpoint, blocked=False):
    coefficients = {'N': 0.}
    coefficients.update({a: q.C1_COEFFICIENT if a == 'C1' else .125 for a in q.ARMS[dataset][1:]})
    reasons = {}
    if blocked:
        for a in q.ARMS[dataset][1:]:
            if a != 'C1':
                coefficients[a] = None
                reasons[a] = 'Known invalid support fixture'
    return dict(status='OBJECT_DFL_CALIBRATION_COMPLETED', dataset=dataset, coefficients=coefficients,
                blocked=reasons, scope='OBJECT_DFL_FIXED8_BNFROZEN', batches=8, seed=42,
                reset_all_parameters_buffers_each_batch=True, optimizer_updates=0, ema_updates=0,
                bn_running_buffers_unchanged=True, formal64_admission=False, new_hash_computed=False,
                official_test_accessed=False, student_initial_checkpoint=q.file_stat(checkpoint))


def stream(path, wrong=False):
    with path.open('x', encoding='utf-8') as f:
        for i in range(30):
            row = dict(batch=i + 1, im_file=['fixture_' + str(j) for j in range(32)], cls=[], bboxes=[],
                       teacher_cls=[], teacher_bboxes=[])
            if wrong and i == 10:
                row['im_file'][0] = 'changed'
            f.write(json.dumps(row) + '\n')


class Tests(unittest.TestCase):
    def fixture(self, dataset, blocked=False, fail=None, wrong_stream=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        out = root / 'output';out.mkdir();(out / 'queue').mkdir();(out / 'effective_configs').mkdir()
        checkpoint = root / 'fixture_checkpoint.txt';checkpoint.write_text('no actual model')
        binding = root / 'binding.json';binding.write_text('{}')
        original_binding = q.NATIVE_BINDING;q.NATIVE_BINDING = binding
        self.addCleanup(setattr, q, 'NATIVE_BINDING', original_binding)
        cfgs = {a: yaml.safe_load((HERE / 'configs' / (dataset + '_' + a + '_s42_FT3.yaml')).read_text()) for a in q.ARMS[dataset]}
        for c in cfgs.values():c['model'] = str(checkpoint.resolve())
        calls = []
        def execute(job, queue):
            stage, arm = job['stage'], job['arm'];calls.append((stage, arm))
            if fail == (stage, arm):raise RuntimeError('Known mocked stage error')
            path = Path(job['expected_receipt']);path.parent.mkdir(parents=True)
            if stage == 'calibration':
                q.write_new(path, calibration(dataset, checkpoint, blocked));return
            if stage == 'eval':
                value = dict(status='OBJECT_DFL_EVALUATION_COMPLETED', scope=q.SCOPE, endpoint=q.ENDPOINT,
                    dataset=dataset, arm=arm, seed=42, single_seed=True, formal_e200_complete=False,
                    new_hash_computed=False, official_test_accessed=False, epochs=3,
                    full_dev_images=q.DEV[dataset][0], full_dev_gt_objects=q.DEV[dataset][1])
                q.write_new(path, value);return
            source = out / 'effective_configs' / (dataset + '_' + arm + '_s42_FT3.yaml')
            cfg = yaml.safe_load(source.read_text())
            config_copy = path.parent / 'direction_config.yaml';shutil.copyfile(source, config_copy)
            value = dict(status='OBJECT_DFL_CANARY_COMPLETED' if stage == 'canary' else 'OBJECT_DFL_TRAINING_COMPLETED',
                scope=q.SCOPE, endpoint=q.ENDPOINT, dataset=dataset, arm=arm, seed=42, single_seed=True,
                formal_e200_complete=False, new_hash_computed=False, official_test_accessed=False,
                model=cfg['model'], successful_updates=24, bn_running_buffers_unchanged=True,
                bn_buffer_count=3, bn_affine_trainable=True, config_copy=str(config_copy),
                epochs_configured=3, last_epoch=3, batches=192,
                resources=dict(per_gpu_peak_vram_mib={'4': 1024}, peak_rss_mib=4096),
                gpu_allocated_peak_mib=800, gpu_reserved_peak_mib=900,
                **{k: cfg[k] for k in ['kd_coefficient', 'classification_coefficient', 'localization_coefficient']})
            q.write_new(path, value)
            q.write_new(path.parent / 'initialization_check.json', dict(status='PASS_FULL_STATE_WARM_START',
                initial_checkpoint=q.file_stat(checkpoint), fresh_optimizer=True, fresh_ema=True,
                teacher_reference_isolated=True, head_included=True))
            stream(path.parent / 'sample_stream.jsonl', wrong=wrong_stream and stage == 'train')
        return out, cfgs, execute, calls

    def test_actual_seven_configs_and_fixed_coefficients(self):
        for dataset in q.DATASETS:
            cfgs = {a: yaml.safe_load((HERE / 'configs' / (dataset + '_' + a + '_s42_FT3.yaml')).read_text()) for a in q.ARMS[dataset]}
            q.validate_configs(dataset, cfgs)
        cfg = q.effective_config({'arm': 'F-rel'}, .25, Path('receipt.json'))
        self.assertEqual((cfg['kd_coefficient'],cfg['classification_coefficient'],cfg['localization_coefficient']),(.25,0.,0.))

    def test_full_both_dataset_sequences(self):
        for dataset in q.DATASETS:
            out, cfgs, execute, calls = self.fixture(dataset)
            result = q.process_dataset(HERE, out, dataset, cfgs, execute)
            self.assertEqual(result['status'], 'COMPLETED', result)
            self.assertTrue(all(x['receipt_validated'] for x in result['completed']))
            expected = [('calibration','N')] + [('canary',a) for a in q.ARMS[dataset]]
            expected += [(s,a) for a in q.ARMS[dataset] for s in ['train','eval']]
            self.assertEqual(calls,expected)

    def test_invalid_calibration_blocks_all_three_arms(self):
        out,cfgs,execute,calls=self.fixture('llvip',blocked=True)
        result=q.process_dataset(HERE,out,'llvip',cfgs,execute)
        self.assertEqual(result['status'],'BLOCKED_CALIBRATION',result)
        self.assertEqual(calls,[('calibration','N')])

    def test_technical_failure_stops_only_dataset(self):
        out,cfgs,execute,calls=self.fixture('llvip',fail=('canary','L3-DFL'))
        result=q.process_dataset(HERE,out,'llvip',cfgs,execute)
        self.assertEqual(result['status'],'DATASET_STOPPED_ON_TECHNICAL_FAILURE')
        self.assertNotIn(('train','N'),calls)

    def test_training_stream_mismatch_blocks_eval(self):
        out,cfgs,execute,calls=self.fixture('llvip',wrong_stream=True)
        result=q.process_dataset(HERE,out,'llvip',cfgs,execute)
        self.assertEqual(result['status'],'DATASET_STOPPED_ON_TECHNICAL_FAILURE')
        self.assertNotIn(('eval','N'),calls)

    def test_malformed_coefficients_fail_closed(self):
        out,cfgs,_,_=self.fixture('llvip')
        receipt=calibration('llvip',Path(cfgs['N']['model']))
        for invalid in [float('nan'),0.,1.1,True,None]:
            bad=copy.deepcopy(receipt);bad['coefficients']['L3-DFL']=invalid
            with self.assertRaises(ValueError):q.coefficient_plan('llvip',bad)

    def test_eval_CLI_and_resource_ceilings(self):
        for dataset in q.DATASETS:
            job=q.make_job(HERE,Path('output'),dataset,'N','eval',cfg={'native_contract_config':None})
            self.assertEqual('--native-profile-binding' in job['command'],dataset=='drone')
            self.assertEqual((job['vram_mib'],job['rss_mib']),(2048,8192))
        for stage in ['calibration','canary']:
            job=q.make_job(HERE,Path('output'),'llvip','N',stage)
            self.assertEqual((job['vram_mib'],job['rss_mib']),(8192,32768))
        with self.assertRaises(ValueError):q.training_reservation(dict(resources=dict(per_gpu_peak_vram_mib={'0':8192},peak_rss_mib=32000),gpu_allocated_peak_mib=8100,gpu_reserved_peak_mib=8192))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),scope='CPU_MOCK_QUEUE_ONLY',
        real_GPU_or_SSH_started=False,new_hash_computed=False)
    with (HERE/'OBJECT_DFL_QUEUE_CPU_CHECKS.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
