import copy
import math
import unittest
from posthoc_class_adapter import class_values, validate_identity


def percent(value, units):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('not finite')
    scale = {'fraction_0_to_1': 100, 'percent_0_to_100': 1}[units]
    if not 0 <= value*scale <= 100:
        raise ValueError('range')
    return value*scale


class ClassTests(unittest.TestCase):
    def setUp(self):
        self.metric = dict(metric_units='fraction_0_to_1', mAP50_95=.4, AP50=.6, AP75=.5,
            per_class=[dict(class_id=0,mAP50_95=.2,AP50=.4,AP75=.3),
                       dict(class_id=1,mAP50_95=.6,AP50=.8,AP75=.7)])

    def test_known_truth(self):
        self.assertEqual(class_values(self.metric,['0','1'],percent)['mAP50_95'], {'0':20.,'1':60.})

    def test_percent_units(self):
        self.metric['metric_units']='percent_0_to_100'
        for key in ('mAP50_95','AP50','AP75'):
            self.metric[key]*=100
            for row in self.metric['per_class']: row[key]*=100
        self.test_known_truth()

    def test_duplicate(self):
        self.metric['per_class'][1]['class_id']=0
        with self.assertRaises(ValueError): class_values(self.metric,['0','1'],percent)

    def test_missing_class(self):
        with self.assertRaises(ValueError): class_values(self.metric,['0','1','2'],percent)

    def test_bool_id(self):
        self.metric['per_class'][0]['class_id']=False
        with self.assertRaises(ValueError): class_values(self.metric,['0','1'],percent)

    def test_macro_disagreement(self):
        self.metric['AP75']=.4
        with self.assertRaises(ValueError): class_values(self.metric,['0','1'],percent)

    def test_nonfinite_and_range(self):
        for value in (float('nan'),float('inf'),-1,1.1,True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                row=copy.deepcopy(self.metric);row['per_class'][0]['mAP50_95']=value
                class_values(row,['0','1'],percent)

    def test_single_class(self):
        row=dict(metric_units='fraction_0_to_1',mAP50_95=.3,AP50=.6,AP75=.4,
                 per_class=[dict(class_id=0,mAP50_95=.3,AP50=.6,AP75=.4)])
        self.assertEqual(class_values(row,['0'],percent)['mAP50_95'],{'0':30.})


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.record=dict(seed=42,arm='C0',source='paired',dataset='fixture',roster=['a','b'])
        self.old=dict(checkpoint='/fixed/last.pt',endpoint='fixed_budget_last_ema',method_id='C0')
        self.contract=dict(roster=['a','b'],expected_val_images=2,observed_images=2,actual_loader_roster=['b','a'])
        self.entry=dict(actual_source_arm='paired',evaluation_contract=copy.deepcopy(self.contract))
        self.metric=dict(self.old,seed=42,arm='paired',source='paired',dataset='fixture',split='val',
            normalized_method_arm='C0',status='completed',official_test_accessed=False,
            evaluation_kind='posthoc_legacy_checkpoint_diagnostics')
        self.receipt=dict(inputs=copy.deepcopy(self.metric),terminal_status='COMPLETED',run_kind='eval',
                          data_role='development_val',seed=42,dataset='fixture')
        self.complete=dict(self.metric,new_training_receipt_created=False,old_results_modified=False)

    def check(self):
        validate_identity(self.record,self.old,self.entry,self.metric,self.receipt,self.complete,self.contract,['a','b'])

    def test_exact_identity(self): self.check()

    def test_seed_mismatch(self):
        self.metric['seed']=0
        with self.assertRaises(ValueError): self.check()

    def test_checkpoint_mismatch(self):
        self.metric['checkpoint']='/other/best.pt'
        with self.assertRaises(ValueError): self.check()

    def test_receipt_mismatch(self):
        self.receipt['inputs']['checkpoint']='/other/last.pt'
        with self.assertRaises(ValueError): self.check()

    def test_method_relabel(self):
        self.metric['normalized_method_arm']='C1'
        with self.assertRaises(ValueError): self.check()

    def test_test_exposure(self):
        self.metric['official_test_accessed']=True
        with self.assertRaises(ValueError): self.check()

    def test_population_missing(self):
        self.contract['actual_loader_roster']=['a','a']
        self.entry['evaluation_contract']=copy.deepcopy(self.contract)
        with self.assertRaises(ValueError): self.check()

    def test_failure_receipt(self):
        self.receipt['terminal_status']='FAILED'
        with self.assertRaises(ValueError): self.check()

    def test_original_unchanged(self):
        names=('record','old','entry','metric','receipt','complete','contract')
        before=copy.deepcopy([getattr(self,k) for k in names])
        self.check()
        self.assertEqual(before,[getattr(self,k) for k in names])


if __name__=='__main__': unittest.main()
