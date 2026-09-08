"""Small synthetic-only regression checks; no actual new AP is read."""
import argparse,copy,json,tempfile,unittest
from pathlib import Path
import analyze_subset_e8 as a

def file(path):return dict(path=path,bytes=100,mtime_ns=123)
def receipt(arm,value=.3):
    rows=[dict(class_id=i,name='class'+str(i),mAP50_95=value+i*.01,AP50=value+.2+i*.01,AP75=value+.1+i*.01) for i in range(5)]
    return dict(status='SUBSET_SCREEN_EVALUATION_COMPLETED',scope=a.SCOPE,endpoint=a.ENDPOINT,dataset='dronevehicle',arm=arm,seed=42,
        single_seed=True,epochs=8,independent_lr_horizon=8,full_dev_images=1469,full_dev_gt_objects=22462,expected_train_images=2048,
        metric_units='fraction_0_to_1',bn_running_statistics='normal_training',formal_e200_complete=False,formal_paper_gain_claim=False,
        official_test_accessed=False,new_hash_computed=False,classification_coefficient=a.COEFFICIENTS[arm],
        mAP50_95=value+.02,AP50=value+.22,AP75=value+.12,precision=.8,recall=.6,per_class=rows,seconds=1.,
        initialization={k:file('/fixture/'+k+'.pt') for k in ('model','teacher','reference')},
        training_subset_identity={k:'/fixture/'+k for k in a.SUBSET_KEYS},
        training_configuration=file('/fixture/'+arm+'.yaml'),training_completion=file('/fixture/'+arm+'_completion.json'))

class Truths(unittest.TestCase):
    def test_units_sign_and_permuted_five_classes(self):
        n,c=receipt('N'),receipt('C0',.31);c['per_class'].reverse()
        r=a.summarize({'N':n,'C0':c})
        self.assertEqual(r['status'],'COMPLETE_SUBSET_E8_READOUT')
        self.assertAlmostEqual(r['arms'][0]['display_percent']['mAP50_95'],32.)
        self.assertAlmostEqual(r['comparison']['delta_pp']['mAP50_95'],1.)
        self.assertTrue(all(abs(x['delta_pp']['AP50']-1)<1e-10 for x in r['comparison']['per_class']))
        self.assertEqual(r['primary_direction'],'positive');self.assertIsNone(r['standard_deviation'])
        self.assertFalse(r['automatically_admit_e200']);self.assertFalse(r['new_method_causal_claim'])
        self.assertEqual(a.summarize({'N':n,'C0':receipt('C0')})['primary_direction'],'zero')
        self.assertEqual(a.summarize({'N':n,'C0':receipt('C0',.29)})['primary_direction'],'negative')
    def test_fixed_E8_scope_coefficient_BN_and_population(self):
        for key,value in [('epochs',3),('independent_lr_horizon',200),('scope','OBJECT_DFL_FT3_BNFROZEN'),('classification_coefficient',.2),
                          ('bn_running_statistics','frozen'),('expected_train_images',17990),('full_dev_gt_objects',80)]:
            r=receipt('C0');r[key]=value
            with self.assertRaises(ValueError):a.validate(r,'C0')
    def test_nonfinite_wrong_units_and_macro_rejected(self):
        for key,value in [('AP50',float('nan')),('metric_units','percent'),('AP75',True),('mAP50_95',30.)]:
            r=receipt('N');r[key]=value
            with self.assertRaises(ValueError):a.validate(r,'N')
        r=receipt('N');r['per_class'][2]['AP50']+=.01
        with self.assertRaises(ValueError):a.validate(r,'N')
        r=receipt('N');r['per_class'][2]['class_id']=1
        with self.assertRaises(ValueError):a.validate(r,'N')
    def test_matched_initialization_subset_and_class_identity(self):
        for change in ('init','subset','class'):
            n,c=receipt('N'),receipt('C0')
            if change=='init':c['initialization']['model']['mtime_ns']+=1
            elif change=='subset':c['training_subset_identity']['paired_train_mapping']='/other'
            else:c['per_class'][0]['name']='other'
            with self.assertRaises(ValueError):a.summarize({'N':n,'C0':c})
    def test_missing_and_failed_not_zero(self):
        for unavailable in ({},{'C0':dict(status='FAILED',reason='synthetic')}):
            r=a.summarize({'N':receipt('N')},unavailable)
            self.assertIsNone(r['comparison']['delta_pp']);self.assertIsNone(r['arms'][1]['raw_fraction'])
            self.assertEqual(r['status'],'PARTIAL_SUBSET_E8_READOUT')
    def test_file_conflict_never_selects_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'evaluations/N';folder.mkdir(parents=True)
            (folder/'short_evaluation_receipt.json').write_text(json.dumps(receipt('N')))
            (folder/'short_evaluation_failure.json').write_text(json.dumps(dict(status='SUBSET_SCREEN_EVALUATION_FAILED',scope=a.SCOPE,error='fixture')))
            r=a.collect(Path(tmp));self.assertEqual(r['arms'][0]['status'],'CONFLICT');self.assertIsNone(r['comparison']['delta_pp'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    with args.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),scope='SYNTHETIC_RECEIPTS_ONLY',actual_new_AP_read=False,GPU_used=False,SSH_used=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(not r.wasSuccessful())
