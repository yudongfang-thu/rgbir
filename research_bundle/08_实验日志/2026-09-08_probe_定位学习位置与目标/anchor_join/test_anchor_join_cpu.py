"""Identity-vs-value and missing-history truth checks; no real cache reads."""
import argparse,json,unittest
from pathlib import Path
from analyze_anchor_join import compare,point,learning_anchor

class Truths(unittest.TestCase):
    def test_same_box_different_anchor_is_different(self):
        a=dict(anchor=(0,10),box=[1,2,3,4]);b=dict(anchor=(0,11),box=a['box'])
        self.assertEqual(compare(a['anchor'],b['anchor'],'missing')['status'],'different')
    def test_different_box_same_anchor_is_same_identity(self):
        a=dict(anchor=(2,10),box=[1,2,3,4]);b=dict(anchor=(2,10),box=[1.1,2,3,4])
        self.assertEqual(compare(a['anchor'],b['anchor'],'missing')['status'],'same');self.assertNotEqual(a['box'],b['box'])
    def test_cross_frame_rejected(self):
        with self.assertRaises(ValueError):compare((0,10),(1,10),'missing')
    def test_nonselected_has_no_actual_learning_anchor(self):
        self.assertIsNone(learning_anchor(dict(reference_anchor=55,selected=False),dict(selected=False)))
        self.assertEqual(learning_anchor(dict(reference_anchor=55,selected=True),dict(selected=True)),55)
        with self.assertRaises(ValueError):learning_anchor(None,dict(selected=True))
    def test_null_and_layout_boundaries(self):
        self.assertEqual(compare(None,(0,2),'teacher_not_executed'),dict(status='unavailable',reason='teacher_not_executed'))
        shape=dict(feats=[[32,64,80,80],[32,128,40,40],[32,256,20,20]])
        self.assertEqual((point(6399,shape)['level'],point(6400,shape)['level'],point(8000,shape)['level']),(0,1,2))
        self.assertEqual(point(6400,shape)['center_xy'],[8.,8.])
        with self.assertRaises(ValueError):point(8400,shape)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),real_input_read=False,new_GPU=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(not r.wasSuccessful())
