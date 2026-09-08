"""Portable pure CPU input/metric contract truths; never calls run or imports runtime."""
from pathlib import Path
import argparse, copy, importlib.util, json, math, unittest
SOURCE=Path(__file__).with_name('export_drone_teacher.py')
spec=importlib.util.spec_from_file_location('capture_test_target',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def metrics(ids):
    return dict(AP50=.6,AP75=.4,mAP50_95=.35,precision=.7,recall=.8,
                per_class=[dict(class_id=i,name=m.CLASS_NAMES[i],AP50=.6,AP75=.4,mAP50_95=.35) for i in ids])

class Contracts(unittest.TestCase):
    def test_empty_label_preserved(self):
        self.assertEqual(m.label_array([]).shape,(0,5))
    def test_all_five_classes(self):
        rows=[[str(i),'.5','.5','.2','.2'] for i in range(5)]
        self.assertEqual(m.label_array(rows)[:,0].tolist(),list(range(5)))
    def test_invalid_labels_rejected(self):
        for row in [['.5','.5','.5','.2','.2'],['5','.5','.5','.2','.2'],['0','nan','.5','.2','.2'],['0','.5','.5','0','.2']]:
            with self.assertRaises(ValueError):m.label_array([row])
    def test_first32_absent_van_valid(self):
        m.validate_metrics(metrics([0,1,2,3]),[0,1,2,3])
        with self.assertRaises(ValueError):m.validate_metrics(metrics([0,1,2,3]),[0,1,2,3,4])
    def test_invalid_metrics_rejected(self):
        for target,value in [('AP75',float('nan')),('recall',1.01)]:
            x=metrics([0]);x[target]=value
            with self.assertRaises(ValueError):m.validate_metrics(x,[0])
        x=metrics([0]);x['per_class'][0]['AP50']=float('inf')
        with self.assertRaises(ValueError):m.validate_metrics(x,[0])
        x=metrics([0]);x['per_class'][0]['name']='person'
        with self.assertRaises(ValueError):m.validate_metrics(x,[0])
    def test_fixed_actual_input_spec(self):
        x=json.loads(SOURCE.with_name('drone_teacher_spec.json').read_text(encoding='utf-8'))
        self.assertEqual(x['scope'],m.SCOPE);self.assertEqual(x['canary_images'],32)
        t=x['models']['T42'];self.assertEqual(len(t['aliases']),1469)
        self.assertEqual(sum(t['label_counts']),24490);self.assertEqual(sum(t['label_counts'][:32]),514)
        self.assertEqual(sum(n==0 for n in t['label_counts']),2)
        self.assertEqual(t['class_counts'],[20588,918,1470,789,725])
        self.assertEqual(Path(t['aliases'][0]).name,'00001.jpg')
        self.assertEqual(Path(t['aliases'][31]).name,'00032.jpg')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                 failures=len(result.failures),errors=len(result.errors),scope='PURE_CPU_INPUT_METRIC_CONTRACTS',
                 GPU_used=False,checkpoint_loaded=False,new_hash_computed=False,native_evaluator_executed=False)
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
