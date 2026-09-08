"""Synthetic truths only. Standard library; never reads actual experiment AP."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import analyze_direction as a


def receipt(ds,arm,shift=0.):
    images,objects,nc=a.POPULATIONS[ds]
    r=dict(status='DIRECTION_EVALUATION_COMPLETED',scope=a.SCOPE,endpoint=a.ENDPOINT,
        dataset=ds,arm=arm,seed=42,single_seed=True,epochs=3,independent_lr_horizon=3,
        full_dev_images=images,full_dev_gt_objects=objects,observed_images=images,gt_objects_captured=objects,
        metric_units='fraction_0_to_1',formal_e200_complete=False,formal_paper_gain_claim=False,
        accepted_endpoint_claim=False,official_test_accessed=False,new_hash_computed=False,
        kd_coefficient=0. if arm=='N' else .1,seconds=12.3,training_model='/fixture/'+ds+'/initial.pt',
        bn_training_evidence=dict(bn_running_buffers_unchanged=True),
        evaluation_identity_projection=dict(dataset=ds,subset_dev_roster_exact_to_full=True,
            expected_dev_images=images,expected_dev_gt_objects=objects,actual_evaluation_data_yaml='/fixture/'+ds+'/full.yaml'),
        precision=.7+shift,recall=.8+shift)
    r['per_class']=[dict(class_id=i,name='class_'+str(i),mAP50_95=.3+shift+i*.001,
        AP50=.6+shift+i*.001,AP75=.4+shift+i*.001) for i in range(nc)]
    for key in a.AP:r[key]=sum(row[key] for row in r['per_class'])/nc
    return r


def matrix():
    shifts={'N':0.,'C1':.01,'C2':-.005,'F-rel':.002,'L2-box':.02,'L2-GT':.005}
    return {(ds,arm):receipt(ds,arm,shifts[arm]) for ds,arms in a.ARMS.items() for arm in arms}


class Truths(unittest.TestCase):
    def test_known_percent_pp_direction_and_fixed_controls(self):
        result=a.summarize(matrix());self.assertEqual(result['status'],'COMPLETE_DIRECTION_READOUT')
        rows={r['contrast']:r for r in result['datasets']['drone']['comparisons']}
        self.assertAlmostEqual(rows['C1 - N']['delta_pp']['mAP50_95'],1.)
        self.assertAlmostEqual(rows['C2 - C1']['delta_pp']['mAP50_95'],-1.5)
        self.assertEqual(rows['C2 - C1']['direction']['mAP50_95'],'negative')
        ll={r['contrast']:r for r in result['datasets']['llvip']['comparisons']}
        self.assertAlmostEqual(ll['L2-box - L2-GT']['delta_pp']['AP50'],1.5)
        raw=result['datasets']['drone']['arms'][0]
        self.assertAlmostEqual(raw['raw_fraction']['mAP50_95'],.302)
        self.assertAlmostEqual(raw['display_percent']['mAP50_95'],30.2)
        self.assertIsNone(result['standard_deviation']);self.assertFalse(result['automatically_admit_e200'])
        text=a.markdown(result);self.assertIn('+1.000000',text);self.assertIn('-1.500000',text)

    def test_class_id_pairing_is_independent_of_list_order(self):
        data=matrix();data[('drone','C1')]['per_class'].reverse()
        result=a.summarize(data)
        row=result['datasets']['drone']['comparisons'][0]
        self.assertEqual([r['class_id'] for r in row['per_class']],[0,1,2,3,4])
        for r in row['per_class']:self.assertAlmostEqual(r['delta_pp']['AP75'],1.)

    def test_missing_failed_and_absent_control_are_null_not_zero(self):
        data={('drone','C1'):receipt('drone','C1')}
        result=a.summarize(data,{('drone','C2'):dict(status='FAILED',reason='fixture')})
        self.assertEqual(result['status'],'PARTIAL_DIRECTION_READOUT')
        rows={r['arm']:r for r in result['datasets']['drone']['arms']}
        self.assertEqual(rows['N']['status'],'MISSING');self.assertEqual(rows['C2']['status'],'FAILED')
        self.assertIsNone(rows['N']['raw_fraction']);self.assertIsNone(rows['C2']['display_percent'])
        for r in result['datasets']['drone']['comparisons']:self.assertIsNone(r['delta_pp'])

    def test_bad_units_values_classes_or_macro_rejected(self):
        for field,value in (('metric_units','percent'),('AP50',60.),('AP75',float('nan')),('recall',True)):
            r=receipt('drone','N');r[field]=value
            with self.assertRaises(ValueError):a.validate(r,'drone','N')
        for change in ('duplicate','missing','macro','name'):
            r=receipt('drone','N')
            if change=='duplicate':r['per_class'][1]['class_id']=0
            elif change=='missing':r['per_class'].pop()
            elif change=='macro':r['per_class'][0]['AP50']+=.01
            else:r['per_class'][0]['name']=''
            with self.assertRaises(ValueError):a.validate(r,'drone','N')

    def test_scope_endpoint_population_control_and_dose_rejected(self):
        for field,value in (('scope','HOURLY_SCREEN_FT'),('endpoint','best'),('epochs',8),
                            ('dataset','llvip'),('full_dev_gt_objects',22461),('new_hash_computed',0),('kd_coefficient',.1)):
            r=receipt('drone','N');r[field]=value
            with self.assertRaises(ValueError):a.validate(r,'drone','N')
        for change in ('initial','full_data','class_map'):
            data=matrix();r=data[('drone','C1')]
            if change=='initial':r['training_model']='/wrong.pt'
            elif change=='full_data':r['evaluation_identity_projection']['actual_evaluation_data_yaml']='/different.yaml'
            else:r['per_class'][0]['name']='different'
            with self.assertRaises(ValueError):a.summarize(data)

    def test_small_file_collection_preserves_conflict_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/'evaluations/drone/N';p.mkdir(parents=True)
            (p/'direction_evaluation_receipt.json').write_text(json.dumps(receipt('drone','N')))
            failure=dict(status='DIRECTION_EVALUATION_FAILED',scope=a.SCOPE,error='fixture failure')
            (p/'direction_evaluation_failure.json').write_text(json.dumps(failure))
            q=root/'evaluations/llvip/L2-box';q.mkdir(parents=True)
            (q/'direction_evaluation_failure.json').write_text(json.dumps(failure))
            result=a.collect(root)
            self.assertEqual(result['datasets']['drone']['arms'][0]['status'],'CONFLICT')
            self.assertEqual(result['datasets']['llvip']['arms'][1]['status'],'FAILED')
            self.assertEqual(len(result['input_sources']),3)
            json.dumps(result,allow_nan=False)
            self.assertIn('—',a.markdown(result))

    def test_llvip_l2_control_requires_exact_shared_coefficient_only(self):
        data=matrix();data[('llvip','L2-GT')]['kd_coefficient']=.10000000000000002
        with self.assertRaisesRegex(ValueError,'must share'):a.summarize(data)
        data=matrix();data[('drone','C2')]['kd_coefficient']=.2
        data[('drone','F-rel')]['kd_coefficient']=.3
        result=a.summarize(data)
        self.assertEqual(result['status'],'COMPLETE_DIRECTION_READOUT')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),synthetic_only=True,actual_direction_AP_read=False,
        new_hash_computed=False,gpu_used=False,failure_details=[dict(test=str(t),traceback=e) for t,e in result.failures+result.errors])
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
