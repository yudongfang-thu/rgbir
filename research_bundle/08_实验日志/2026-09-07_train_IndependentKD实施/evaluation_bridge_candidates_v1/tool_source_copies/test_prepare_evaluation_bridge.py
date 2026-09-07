"""CPU/source/file tests only. Temporary eval receipts are synthetic, no AP run."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import prepare_evaluation_bridge as bridge

HERE = Path(__file__).resolve().parent
PROBE = HERE/'evaluation_bridge_candidates_v1/probe'
CANDIDATE = HERE/'evaluation_bridge_candidates_v1/prepared_a1/evaluation_compatibility_candidate.json'


class EvaluationBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_actual_completed_probe_and_seven_formal_sources(self):
        receipt, _, _, _, records = bridge.validate_probe(PROBE)
        formal = next(r for r in records if r['original'].endswith('/evaluate_independent.py'))
        result = bridge.formal_source_order(formal['local_copy'], records)
        self.assertEqual(len(result), 7)
        self.assertEqual([Path(r['expected_formal_receipt_relative']).name for r in result],
            ['01_evaluate_independent.py', '02_val.py', '03_validator.py', '04_model.py',
             '05_utils.py', '06_metrics.py', '07_nms.py'])
        self.assertFalse(any(r['original'].endswith('/evaluator_profile.py') for r in result))
        self.assertFalse(receipt['baseline_training_receipt_created'])

    def test_changed_formal_implementation_order_is_rejected(self):
        records = bridge.validate_probe(PROBE)[-1]
        formal = next(r for r in records if r['original'].endswith('/evaluate_independent.py'))
        source = Path(formal['local_copy']).read_text(encoding='utf-8')
        altered = self.root/'formal.py'
        altered.write_text(source.replace('DetectionValidator, BaseValidator, YOLO.val,',
                                          'BaseValidator, DetectionValidator, YOLO.val,'),encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'object order changed'):
            bridge.formal_source_order(altered, records)

    def test_mismatched_actual_probe_metrics_are_rejected(self):
        for name in ('evaluator_profile_receipt.json','native_metrics.json','evidence_metrics.json',
                     'native_contract.json','evidence_contract.json'):
            (self.root/name).write_bytes((PROBE/name).read_bytes())
        value=bridge.read(self.root/'evidence_metrics.json')
        value['AP50'] += .01
        (self.root/'evidence_metrics.json').write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'five-metric parity failed'):
            bridge.validate_probe(self.root)

    def test_legacy_kwarg_change_is_not_hidden_by_protocol_label(self):
        candidate=bridge.read(CANDIDATE)
        source=CANDIDATE.parent/next(iter(candidate['entries'][0]['evaluation_source_copies'].values()))
        self.assertIn('batch',bridge.legacy_val_arguments(source))
        changed=self.root/'changed_old_eval.py'
        changed.write_text(source.read_text(encoding='utf-8').replace("split='val',imgsz=", "split='val',conf=0.1,imgsz="),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'kwargs differ'):
            bridge.legacy_val_arguments(changed)

    def test_future_formal_source_verification_checks_set_order_and_bytes(self):
        candidate=bridge.read(CANDIDATE)
        paths=[row['expected_formal_receipt_relative'] for row in candidate['canonical_source_order']]
        # Explicit synthetic receipt fixture tests source compatibility only.
        for relative,copy in zip(paths,candidate['canonical_evaluator_source_copies']):
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((CANDIDATE.parent/copy).read_bytes())
        receipt={'status':'SYNTHETIC_SOURCE_ORDER_TEST_ONLY','source_snapshots':{'trainer':paths}}
        (self.root/'run_receipt.json').write_text(json.dumps(receipt),encoding='utf-8')
        self.assertFalse(bridge.verify_future_sources(CANDIDATE,self.root)['bridge_acceptance_created'])
        target=self.root/paths[-1];target.write_bytes(target.read_bytes()+b'\n# synthetic mutation\n')
        with self.assertRaisesRegex(ValueError,'source bytes differ'):
            bridge.verify_future_sources(CANDIDATE,self.root)
        receipt['source_snapshots']['trainer']=paths[:-1]
        (self.root/'run_receipt.json').write_text(json.dumps(receipt),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'set/order differs'):
            bridge.verify_future_sources(CANDIDATE,self.root)

    def test_real_prepared_six_remain_draft_and_preserve_missing_old_diagnostics(self):
        candidate=bridge.read(CANDIDATE)
        summary=bridge.read(CANDIDATE.parent/'candidate_summary.json')
        self.assertEqual(candidate['status'],'DRAFT_AWAITING_INDEPENDENT_REVIEW')
        self.assertIsNone(candidate['reviewer'])
        self.assertEqual(len(candidate['entries']),6)
        self.assertTrue(summary['draft_rejected_by_analyzer_for_all_six'])
        self.assertEqual(summary['frozen_protocol_ids'],['oev1_frozen_drone_e200'])
        for entry in candidate['entries']:
            self.assertFalse(entry['old_objects_recorded'])
            self.assertFalse(entry['old_per_class_recorded'])
            self.assertNotIn('actual_loader_roster',entry['evaluation_contract'])
            self.assertNotIn('observed_images',entry['evaluation_contract'])


if __name__=='__main__':
    unittest.main(verbosity=2)
