import argparse
import copy
import json
from pathlib import Path
import unittest
import yaml
import analyze_feature_gm as a
from dose_readout import distribution


def fixtures():
    parent=Path(__file__).resolve().parent.parent
    configs={arm:yaml.safe_load((parent/'release'/'configs'/('drone_'+arm+'_s42_FT3.yaml')).read_text(encoding='utf-8')) for arm in ('N','C1')}
    f=copy.deepcopy(configs['C1']);f.update(arm='F-rel-GM',scope=a.SCOPE,kd_coefficient=a.DOSES['F-rel-GM'],classification_coefficient=a.DOSES['F-rel-GM'])
    configs['F-rel-GM']=f
    receipts={}
    for i,arm in enumerate(a.ARMS):
        c=configs[arm];c['kd_coefficient']=a.DOSES[arm];c['classification_coefficient']=a.DOSES[arm]
        scope=a.SCOPE if arm=='F-rel-GM' else a.CONTROL_SCOPE
        metrics={k:.4+i*.01 for k in a.METRICS}
        receipts[arm]=dict(status='DIRECTION_EVALUATION_COMPLETED',scope=scope,endpoint=scope+'_LAST_EMA',dataset='drone',arm=arm,
            seed=42,single_seed=True,epochs=3,independent_lr_horizon=3,full_dev_images=1469,full_dev_gt_objects=22462,
            observed_images=1469,gt_objects_captured=22462,metric_units='fraction_0_to_1',formal_e200_complete=False,
            formal_paper_gain_claim=False,accepted_endpoint_claim=False,official_test_accessed=False,new_hash_computed=False,
            kd_coefficient=a.DOSES[arm],**metrics,per_class=[dict(class_id=j,name='class'+str(j),**{k:metrics[k] for k in a.AP}) for j in range(5)],
            training_model=c['model'],training_configuration='/'+arm+'/cfg.yaml',training_completion='/'+arm+'/completion.json',
            evaluation_identity_projection=dict(dataset='drone',subset_dev_roster_exact_to_full=True,expected_dev_images=1469,
                expected_dev_gt_objects=22462,training_model=c['model'],training_data_yaml=c['paths']['student_data_yaml'],
                actual_evaluation_data_yaml=c['auxiliary_data_identity']['student_data_yaml']),
            bn_training_evidence=dict(bn_running_buffers_unchanged=True,bn_affine_trainable=True,bn_buffer_count=2),
            matched_control_projection_required=True,matched_control_scope=a.CONTROL_SCOPE,expected_train_images=2048,
            training_subset_identity=c['paths'])
    controls={}
    for arm in ('N','C1'):
        controls[arm]=dict(common_config_fields_exact=True,first30_canary_exact=True,initial_checkpoint_stat_exact=True,
            control_full_train_first30_exact=True,control_training_receipt=dict(path=receipts[arm]['training_completion']),
            control_config=dict(path=receipts[arm]['training_configuration']),control_evaluation_receipt=dict(path='/'+arm+'/eval.json',bytes=42))
    projection=dict(status='FEATURE_GM_MATCHED_CONTROL_VERIFIED',dataset='drone',seed=42,candidate_arm='F-rel-GM',
        candidate_scope=a.SCOPE,control_scope=a.CONTROL_SCOPE,fixed_candidate_coefficient=a.DOSES['F-rel-GM'],controls=controls,
        batch_records=30,batch_size=32,new_hash_computed=False,
        candidate_configuration=dict(path=receipts['F-rel-GM']['training_configuration'],bytes=42),
        candidate_canary_receipt=dict(path='/F/canary.json'),
        initial_checkpoint=dict(path=configs['N']['model'],bytes=42,mtime_ns=42))
    return receipts,configs,projection


class Truths(unittest.TestCase):
    def test_valid_raw_and_differences(self):
        r,c,p=fixtures();v=a.summarize(r,c,p)
        self.assertAlmostEqual(v['F_minus_control_pp']['N']['mAP50_95'],2.)
        self.assertAlmostEqual(v['F_minus_control_pp']['C1']['mAP50_95'],1.)
        self.assertFalse(v['formal_paper_gain_claim'])
    def test_wrong_scope_or_missing_projection_rejected(self):
        for change in ('scope','proof','missing','dose','candidate'):
            r,c,p=fixtures()
            if change=='scope':r['F-rel-GM']['scope']=a.CONTROL_SCOPE
            if change=='proof':p['controls']['C1']['first30_canary_exact']=False
            if change=='missing':del r['C1']
            if change=='dose':r['F-rel-GM']['kd_coefficient']=1.
            if change=='candidate':p['candidate_configuration']['path']='/other.yaml'
            with self.assertRaises(ValueError):a.summarize(r,c,p)
    def test_recipe_or_metric_rejected(self):
        for change in ('lr','subset','macro','bool'):
            r,c,p=fixtures()
            if change=='lr':c['F-rel-GM']['lr0']=.001
            if change=='subset':c['F-rel-GM']['paths']=dict(c['F-rel-GM']['paths'],paired_train_mapping='/different')
            if change=='macro':r['F-rel-GM']['AP50']+=.1
            if change=='bool':r['F-rel-GM']['AP50']=True
            with self.assertRaises(ValueError):a.summarize(r,c,p)
    def test_reciprocal_median_and_small_signed_difference(self):
        self.assertEqual(distribution([2.,.5])['median'],1.25)
        r,c,p=fixtures()
        for k in a.METRICS:r['F-rel-GM'][k]=r['N'][k]-1e-10
        for row in r['F-rel-GM']['per_class']:
            for k in a.AP:row[k]=r['F-rel-GM'][k]
        v=a.summarize(r,c,p)
        self.assertLess(v['F_minus_control_pp']['N']['AP50'],0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Truths);r=unittest.TextTestRunner(verbosity=2).run(suite)
    with args.receipt.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),real_F_AP_read=False,GPU_used=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(0 if r.wasSuccessful() else 1)
