import argparse
import copy
import json
from pathlib import Path
import unittest
from bridge_l2_records import bind, gates, coverage

FIXED=dict(reference_iou_max=.7,mapped_rgb_iou_min=.6,localization_margin=.05,reliable_conf=.25,teacher_ir_iou_min=.5)

class Truths(unittest.TestCase):
    def test_same_image_wrong_row_and_geometry_rejected(self):
        r=dict(batch_index=0,rgb_gt_index=1,ir_gt_index=2,class_id=0,rgb_gt=[0,0,2,2],ir_gt=[0,0,3,3],pair_iou=.8)
        w=dict(image_index=0,rgb_global_row=1,ir_global_row=2,gt_class=0,rgb_gt_xyxy=[0,0,2,2],ir_gt_xyxy=[0,0,3,3],pair_iou=.8)
        bind(r,w)
        for key,value in [('ir_global_row',1),('rgb_gt_xyxy',[0,0,3,3])]:
            bad=copy.deepcopy(w);bad[key]=value
            with self.assertRaises(AssertionError): bind(r,bad)
    def test_downstream_not_evaluated_is_not_false(self):
        r=dict(reference_reliable=True,reference_gap=False,reference_iou=.72,teacher_anchor=None,selected=False)
        g=gates(r,FIXED)
        self.assertIsNone(g['teacher_own_quality']);self.assertEqual(g['exit'],'reference_iou_not_below_070')
        self.assertIsNone(gates(None,FIXED)['reference_gap'])
    def test_record_mask_not_native_proxy(self):
        r=dict(reference_reliable=True,reference_gap=True,reference_iou=.69,teacher_anchor=123,
               teacher_conf=.4,teacher_own_iou=.85,mapped_teacher_rgb_iou=.85,selected=True)
        g=gates(r,FIXED);self.assertTrue(g['selected'])
        r['selected']=False
        with self.assertRaises(AssertionError):gates(r,FIXED)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    a.output.write_text(json.dumps(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,errors=len(r.errors),failures=len(r.failures),GPU_used=False,new_hash_computed=False),indent=2),encoding='utf-8')
    raise SystemExit(0 if r.wasSuccessful() else 1)
