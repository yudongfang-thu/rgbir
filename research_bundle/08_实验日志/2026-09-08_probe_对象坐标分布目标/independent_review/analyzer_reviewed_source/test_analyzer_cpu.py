"""Known synthetic differences and provenance failures; no actual L3 AP read."""
import argparse,copy,json,tempfile,unittest
from pathlib import Path
import analyze_object_dfl as a

def receipt(arm,v=.3):
 return dict(status='OBJECT_DFL_EVALUATION_COMPLETED',scope=a.SCOPE,endpoint=a.ENDPOINT,method_identity=a.SCOPE,dataset='llvip',arm=arm,seed=42,
  single_seed=True,epochs=3,independent_lr_horizon=3,full_dev_images=2406,full_dev_gt_objects=7879,observed_images=2406,gt_objects_captured=7879,
  metric_units='fraction_0_to_1',formal_e200_complete=False,formal_paper_gain_claim=False,accepted_endpoint_claim=False,official_test_accessed=False,new_hash_computed=False,
  kd_coefficient=0. if arm=='N' else .1,bn_training_evidence=dict(bn_running_buffers_unchanged=True),seconds=1.,training_model='/fixture/init.pt',
  evaluation_identity_projection=dict(dataset='llvip',subset_dev_roster_exact_to_full=True,expected_dev_images=2406,expected_dev_gt_objects=7879,actual_evaluation_data_yaml='/fixture/data.yaml'),
  mAP50_95=v,AP50=v+.2,AP75=v+.1,precision=.7,recall=.8,per_class=[dict(class_id=0,name='person',mAP50_95=v,AP50=v+.2,AP75=v+.1)])

class Truths(unittest.TestCase):
 def test_known_units_and_control_directions(self):
  r=a.summarize({('llvip','N'):receipt('N'),('llvip','L3-DFL'):receipt('L3-DFL',.31),('llvip','L3-GT'):receipt('L3-GT',.32)})
  self.assertEqual(r['status'],'COMPLETE_OBJECT_DFL_READOUT');c=r['datasets']['llvip']['comparisons']
  self.assertAlmostEqual(c[0]['delta_pp']['mAP50_95'],1.);self.assertAlmostEqual(c[2]['delta_pp']['mAP50_95'],-1.)
  self.assertIsNone(r['standard_deviation']);self.assertFalse(r['automatically_admit_e200'])
 def test_missing_control_is_null(self):
  r=a.summarize({('llvip','L3-DFL'):receipt('L3-DFL')},{('llvip','L3-GT'):dict(status='FAILED',reason='fixture')})
  self.assertEqual(r['status'],'PARTIAL_OBJECT_DFL_READOUT')
  self.assertTrue(all(x['delta_pp'] is None for x in r['datasets']['llvip']['comparisons']))
 def test_wrong_units_nonfinite_or_foreign_identity_rejected(self):
  for key,value in [('metric_units','percent'),('mAP50_95',30.),('AP75',float('nan')),('seed',0),('endpoint','best'),('scope','DIRECTION_FT3_BNFROZEN'),('method_identity','other_method')]:
   r=receipt('N');r[key]=value
   with self.assertRaises(ValueError):a.validate(r,'llvip','N')
 def test_class_and_population_rejected(self):
  for key,value in [('full_dev_gt_objects',80),('observed_images',32),('per_class',[])]:
   r=receipt('N');r[key]=value
   with self.assertRaises(ValueError):a.validate(r,'llvip','N')
 def test_shared_dose_and_initialization_required(self):
  for field,value in [('kd_coefficient',.2),('training_model','/wrong.pt')]:
   t=receipt('L3-DFL');g=receipt('L3-GT');g[field]=value
   with self.assertRaises(ValueError):a.summarize({('llvip','L3-DFL'):t,('llvip','L3-GT'):g})
 def test_conflicting_receipts_are_not_selected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'evaluations/llvip/N';p.mkdir(parents=True)
   (p/'direction_evaluation_receipt.json').write_text(json.dumps(receipt('N')))
   (p/'direction_evaluation_failure.json').write_text(json.dumps(dict(status='OBJECT_DFL_EVALUATION_FAILED',scope=a.SCOPE,error='fixture')))
   r=a.collect(Path(tmp));self.assertEqual(r['datasets']['llvip']['arms'][0]['status'],'CONFLICT')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
 with args.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),scope='SYNTHETIC_RECEIPTS_ONLY',actual_AP_read=False,new_hash_computed=False),f,indent=2)
 raise SystemExit(not r.wasSuccessful())
