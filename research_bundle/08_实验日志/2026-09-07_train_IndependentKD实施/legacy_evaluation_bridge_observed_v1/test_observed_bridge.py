"""Read-only actual evidence plus explicitly synthetic CPU rejection fixtures."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import build_observed_bridge as bridge

HERE = Path(__file__).resolve().parent
LOG = HERE.parent
WORKSPACE = LOG.parents[1]
ANALYZER = WORKSPACE/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
DIAGNOSTICS = LOG/'legacy_diagnostics_snapshot_20260907_161415'
CANDIDATE = LOG/'evaluation_bridge_candidates_v1/prepared_a1/evaluation_compatibility_candidate.json'
REVIEW = LOG/'legacy_endpoint_eval_independent_review_v1/review_receipt.json'
OLD_MANIFEST = LOG/'old_endpoint_manifest_1407.json'


class ObservedBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper = bridge.load_module(LOG/'prepare_evaluation_bridge.py', '_truth_bridge_helper')
        cls.analyzer = helper.load_analyzer(ANALYZER)
        cls.wrapper = bridge.load_module(LOG/'legacy_endpoint_eval_prepare_v1/legacy_checkpoint_evaluate.py', '_truth_wrapper')
        cls.candidate = bridge.read(CANDIDATE)
        cls.review = dict(cls.wrapper.require_review(REVIEW), _receipt_path=str(REVIEW))
        cls.old = Path(next(r['path'] for r in bridge.read(OLD_MANIFEST)['runs'] if r['arm']=='N' and r['seed']==42))
        cls.folder = DIAGNOSTICS/'N_s42_attempt1'
        cls.remote = bridge.read(DIAGNOSTICS/'source_manifest.json')['remote_root']+'/N_s42_attempt1'
        cls.order = [r['expected_formal_receipt_relative'] for r in cls.candidate['canonical_source_order']]

    def validate(self, candidate_root=None):
        return bridge.validate_observed(self.old, {'arm':'N','seed':42}, self.folder, self.remote,
            self.analyzer, self.wrapper, self.order, candidate_root or CANDIDATE.parent,
            self.candidate, self.review, ANALYZER)

    def test_real_same_checkpoint_observation_has_full_nine_sources_and_population(self):
        result = self.validate()
        self.assertEqual(result['comparison']['status'], 'EXACT')
        self.assertEqual(len(result['geometry']), 1469)
        self.assertEqual(len(result['sources']), 9)
        self.assertEqual(result['metric']['seed'], 42)
        self.assertEqual(len(result['metric']['per_class']), 5)
        self.assertEqual({tuple(v['canvas_shape']) for v in result['geometry'].values()}, {(544,672)})

    def test_one_changed_aggregate_metric_is_rejected_even_when_flag_says_exact(self):
        original = bridge.read
        def changed(path):
            value = original(path)
            if Path(path) == self.folder/'evaluation_val.json':
                value['AP75'] += .001
            return value
        with patch.object(bridge, 'read', side_effect=changed), self.assertRaisesRegex(ValueError, 'not exactly equal'):
            self.validate()

    def test_incomplete_runtime_seen_is_rejected(self):
        original = bridge.read
        def changed(path):
            value = original(path)
            if Path(path) == self.folder/'evaluation_contract.json':
                value['observed_images'] = 1468
            return value
        with patch.object(bridge, 'read', side_effect=changed), self.assertRaisesRegex(ValueError, 'Incomplete actual dev1469'):
            self.validate()

    def test_additional_executed_wrapper_source_cannot_be_hidden(self):
        original = bridge.read
        def changed(path):
            value = original(path)
            if Path(path) == self.folder/'eval_evidence/run_receipt.json':
                value['source_snapshots']['trainer'] = value['source_snapshots']['trainer'][:7]
            return value
        with patch.object(bridge, 'read', side_effect=changed), self.assertRaisesRegex(ValueError, 'source set/order differs'):
            self.validate()

    def test_receipt_seed_mismatch_is_rejected(self):
        original = bridge.read
        def changed(path):
            value = original(path)
            if Path(path) == self.folder/'eval_evidence/run_receipt.json':
                value['seed'] = 0
            return value
        with patch.object(bridge, 'read', side_effect=changed), self.assertRaisesRegex(ValueError, 'execution identity is incomplete'):
            self.validate()

    def test_actual_common_source_cannot_differ_from_formal_canonical_copy(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_bridge_source_') as temporary:
            root = Path(temporary)
            for relative in self.candidate['canonical_evaluator_source_copies']:
                destination = root/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((CANDIDATE.parent/relative).read_bytes())
            destination = root/self.candidate['canonical_evaluator_source_copies'][0]
            destination.write_bytes(destination.read_bytes()+b'\n# SYNTHETIC changed source\n')
            with self.assertRaisesRegex(ValueError, 'canonical source differs'):
                self.validate(root)

    def test_changed_derived_evaluation_recipe_is_rejected(self):
        original = bridge.yaml.safe_load
        def changed(value):
            result = original(value)
            if isinstance(result,dict) and result.get('evaluation_kind')=='posthoc_legacy_checkpoint_diagnostics':
                result['batch'] = 16
            return result
        with patch.object(bridge.yaml, 'safe_load', side_effect=changed), self.assertRaisesRegex(ValueError, 'original bound recipe'):
            self.validate()

    def test_missing_seed_is_rejected_before_any_output_created(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_missing_seed_') as temporary:
            root = Path(temporary)
            manifest = bridge.read(OLD_MANIFEST)
            manifest['runs'] = [r for r in manifest['runs'] if not (r['arm']=='N' and r['seed']==123)]
            bridge.write_new(root/'manifest.json', manifest)
            args = argparse.Namespace(old_manifest=root/'manifest.json', candidate=CANDIDATE, diagnostics=DIAGNOSTICS,
                analyzer_root=ANALYZER, wrapper_review=REVIEW, output=root/'never_written')
            with self.assertRaisesRegex(ValueError, 'exactly six unique'):
                bridge.prepare(args)
            self.assertFalse(args.output.exists())

    def test_synthetic_objects_must_have_full_unique_population(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_object_population_') as temporary:
            root = Path(temporary)
            roster = ['/SYNTHETIC/%04d.jpg'%i for i in range(1469)]
            contract = dict(roster=roster,actual_loader_roster=list(reversed(roster)),
                            expected_val_images=1469,observed_images=1469)
            path = root/'objects.jsonl.gz'
            with gzip.open(path,'xt',encoding='utf-8') as stream:
                for image in roster[:-1]:
                    stream.write(json.dumps(dict(image=image,canvas_shape=[640,640],original_shape=[640,640],
                        gt_boxes=[],gt_classes=[],pred_boxes=[],pred_classes=[],pred_confidence=[]))+'\n')
            with self.assertRaisesRegex(ValueError, 'objects population differs'):
                bridge.validate_population(contract,path,roster)


if __name__ == '__main__':
    unittest.main(verbosity=2)
