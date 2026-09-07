"""CPU fixtures test protocol capture and recovery, never GPU/AP equivalence."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import yaml

import evaluate_independent as evaluation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ValueTensor:
    def __init__(self, values):
        self.values = copy.deepcopy(values)
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return copy.deepcopy(self.values)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'synthetic_run'
        self.run.mkdir()
        self.checkpoint = self.run / 'weights/last.pt'
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b'SYNTHETIC NOT A MODEL')
        self.cfg = dict(dataset='dronevehicle', epochs=200, seed=42, arm='C1', source='paired',
            model='/synthetic/init.pt', teacher='/synthetic/IR.pt', reference='/synthetic/R.pt',
            method_id='SYNTHETIC_FIXTURE', classification_coefficient=.1, localization_coefficient=0.,
            paths={'student_data_yaml': '/synthetic/data.yaml'}, augmentation={'scale': .5})
        self.config = self.run / 'protocol_config.yaml'
        self.config.write_text(yaml.safe_dump(self.cfg))
        self.complete = dict(self.cfg, status='training_completed', last_epoch=200,
            epochs_configured=200, official_test_accessed=False, checkpoint=str(self.checkpoint))
        write_json(self.run / 'completion_receipt.json', self.complete)
        self.train_receipt = dict(terminal_status='COMPLETED', run_kind='train',
            data_role='development_train', dataset='dronevehicle', seed=42,
            source_snapshots={'config': ['config.yaml']}, metric_snapshots=['complete.json'])
        write_json(self.run / 'run_evidence/run_receipt.json', self.train_receipt)
        (self.run / 'run_evidence/config.yaml').write_bytes(self.config.read_bytes())
        write_json(self.run / 'run_evidence/complete.json', self.complete)

    def make_attempt(self):
        attempt = self.run / 'eval_val'
        attempt.mkdir()
        objects = attempt / 'objects.jsonl.gz'
        with gzip.open(objects, 'wt') as stream:
            stream.write('{"image": "/synthetic/a.jpg"}\n')
        result = {key: self.cfg[key] for key in ('dataset', 'arm', 'source', 'seed', 'method_id')}
        result.update(status='completed', checkpoint=str(self.checkpoint), endpoint='fixed_budget_last_ema',
            split='val', official_test_accessed=False, metric_units='fraction_0_to_1', mAP50_95=.123,
            objects=str(objects))
        write_json(attempt / 'evaluation_val.json', result)
        receipt = dict(terminal_status='COMPLETED', run_kind='eval', data_role='development_val',
            dataset=self.cfg['dataset'], seed=self.cfg['seed'],
            inputs={key: result[key] for key in ('method_id', 'arm', 'source', 'checkpoint', 'endpoint')},
            source_snapshots={'config': ['config.yaml'], 'trainer': ['evaluator.py']},
            metric_snapshots=['metric.json', 'objects.gz'])
        write_json(attempt / 'eval_evidence/run_receipt.json', receipt)
        (attempt / 'eval_evidence/config.yaml').write_bytes(self.config.read_bytes())
        (attempt / 'eval_evidence/evaluator.py').write_text('# synthetic evidence fixture\n')
        (attempt / 'eval_evidence/metric.json').write_bytes((attempt / 'evaluation_val.json').read_bytes())
        (attempt / 'eval_evidence/objects.gz').write_bytes(objects.read_bytes())
        return attempt

    def validator(self, images, quantize=None):
        return types.SimpleNamespace(args=types.SimpleNamespace(imgsz=640, batch=32, workers=4,
            quantize=quantize, conf=.001, iou=.7, max_det=300, agnostic_nms=False,
            single_cls=False, rect=True, augment=False),
            dataloader=types.SimpleNamespace(dataset=types.SimpleNamespace(im_files=images)))

    def test_run_identity_checks_actual_training_snapshot(self):
        cfg, complete, checkpoint = evaluation.load_run_configuration(self.run, self.config)
        self.assertEqual(cfg, self.cfg)
        self.assertEqual(checkpoint, self.checkpoint)
        changed = copy.deepcopy(self.cfg)
        changed['augmentation']['scale'] = .9
        self.config.write_text(yaml.safe_dump(changed))
        with self.assertRaisesRegex(ValueError, 'training receipt snapshots'):
            evaluation.load_run_configuration(self.run, self.config)

    def test_incomplete_training_wrong_endpoint_or_unknown_dataset_rejected(self):
        for changes in ({'last_epoch': 199}, {'official_test_accessed': True}, {'checkpoint': '/synthetic/best.pt'}):
            write_json(self.run / 'completion_receipt.json', dict(self.complete, **changes))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluation.load_run_configuration(self.run, self.config)
        self.config.write_text(yaml.safe_dump(dict(self.cfg, dataset='unknown')))
        with self.assertRaisesRegex(ValueError, 'authorized'):
            evaluation.load_run_configuration(self.run, self.config)

    def test_actual_rect_order_preserved_but_canonical_roster_stable(self):
        expected = [str((self.run / name).resolve()) for name in ('a.jpg', 'b.jpg')]
        contract = evaluation.capture_contract(self.validator(expected[::-1]), expected, {'torch': 'fixture'})
        self.assertEqual(contract['roster'], expected)
        self.assertEqual(contract['actual_loader_roster'], expected[::-1])
        self.assertIs(contract['effective_kwargs']['half'], False)
        half = evaluation.capture_contract(self.validator(expected, 16), expected, {'torch': 'fixture'})
        self.assertIs(half['effective_kwargs']['half'], True)

    def test_duplicate_missing_or_wrong_actual_loader_rejected(self):
        expected = [str((self.run / name).resolve()) for name in ('a.jpg', 'b.jpg')]
        for actual in ([expected[0]] * 2, expected[:1], [expected[0], str(self.run / 'c.jpg')]):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                evaluation.capture_contract(self.validator(actual), expected, {})

    def test_seen_and_object_capture_population_checked(self):
        expected = [str((self.run / name).resolve()) for name in ('a.jpg', 'b.jpg')]
        validator = self.validator(expected)
        validator.save_dir, validator.seen = self.run, 2
        contract = evaluation.capture_contract(validator, expected, {})
        for images in (expected[::-1], [expected[0]] * 2, expected[:1]):
            with gzip.open(self.run / 'objects.jsonl.gz', 'wt') as stream:
                stream.write(''.join(json.dumps({'image': image}) + '\n' for image in images))
            if images == expected[::-1]:
                evaluation.verify_population(validator, contract)
                self.assertEqual(contract['observed_images'], 2)
            else:
                with self.assertRaises(ValueError):
                    evaluation.verify_population(validator, contract)

    def test_wrapper_invokes_native_first_without_tensor_or_metric_mutation(self):
        events = []
        class NativeFixture:
            def update_metrics(self, preds, batch):
                events.append('native')
                self.native_metric = {'synthetic_value': .123}
            def _prepare_batch(self, si, batch):
                events.append('capture')
                self.native_metric_before_capture = copy.deepcopy(self.native_metric)
                return batch[si]
        validator = evaluation.make_evidence_validator(NativeFixture)()
        validator.save_dir = self.run
        prediction = dict(bboxes=ValueTensor([[1., 2., 3., 4.]]), cls=ValueTensor([0]), conf=ValueTensor([.9]))
        target = dict(im_file='/synthetic/a.jpg', imgsz=(640, 640), ori_shape=(720, 960),
                      bboxes=ValueTensor([[1., 2., 3., 4.]]), cls=ValueTensor([0]))
        before = {key: value.tolist() for key, value in prediction.items()}
        validator.update_metrics([prediction], [target])
        self.assertEqual(events, ['native', 'capture'])
        self.assertEqual(validator.native_metric, validator.native_metric_before_capture)
        self.assertEqual(before, {key: value.tolist() for key, value in prediction.items()})

    def test_complete_attempt_publish_is_idempotent_and_preserves_originals(self):
        attempt = self.make_attempt()
        originals = {path: path.read_bytes() for path in attempt.rglob('*') if path.is_file()}
        evaluation.publish_evaluation_attempt(self.run, attempt)
        evaluation.publish_evaluation_attempt(self.run, attempt)
        self.assertEqual((self.run / 'evaluation_val.json').read_bytes(), (attempt / 'evaluation_val.json').read_bytes())
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_interrupted_publication_recovers_without_metric_recompute(self):
        attempt = self.make_attempt()
        original_copy = evaluation.copy_missing_or_equal
        count = [0]
        def fail_second(source, destination):
            count[0] += 1
            if count[0] == 2:
                raise OSError('synthetic publication interruption')
            return original_copy(source, destination)
        with patch.object(evaluation, 'copy_missing_or_equal', side_effect=fail_second):
            with self.assertRaises(OSError):
                evaluation.publish_evaluation_attempt(self.run, attempt)
        self.assertFalse((self.run / 'evaluation_val.json').exists())
        args = types.SimpleNamespace(run=self.run, config=self.config, attempt=1)
        with patch('builtins.print'):
            result = evaluation.run(args)
        self.assertEqual(result['mAP50_95'], .123)
        self.assertTrue((self.run / 'evaluation_val.json').is_file())

    def test_attempt_without_receipt_does_not_publish_success(self):
        attempt = self.run / 'eval_val'
        attempt.mkdir()
        write_json(attempt / 'evaluation_val.json', {'synthetic': 'partial'})
        args = types.SimpleNamespace(run=self.run, config=self.config, attempt=1)
        with self.assertRaisesRegex(FileExistsError, 'incomplete attempt'):
            evaluation.run(args)
        self.assertFalse((self.run / 'evaluation_val.json').exists())

    def test_changed_objects_or_conflicting_canonical_copy_rejected(self):
        attempt = self.make_attempt()
        objects = attempt / 'objects.jsonl.gz'
        original = objects.read_bytes()
        objects.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            evaluation.publish_evaluation_attempt(self.run, attempt)
        objects.write_bytes(original)
        canonical = self.run / 'eval_evidence/config.yaml'
        canonical.parent.mkdir()
        canonical.write_text('synthetic conflicting old evidence')
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            evaluation.publish_evaluation_attempt(self.run, attempt)
        self.assertEqual(canonical.read_text(), 'synthetic conflicting old evidence')


if __name__ == '__main__':
    unittest.main()
