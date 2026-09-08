"""Known-truth portable CPU tests; no experiment cache or checkpoint is read."""
import argparse,copy,json,unittest
from pathlib import Path
import torch
from pairing_core import load_original,gt_pair,bind_images,original_image_fallback,validate_row
from native_cached_match import load_native,match_row
HERE=Path(__file__).resolve().parent

def row(boxes=(),classes=(),image='/rgb/val/a.jpg',pred=(),pclass=(),scores=()):
    return dict(image=image,canvas_shape=[100,100],original_shape=[100,100],gt_boxes=list(boxes),gt_classes=list(classes),pred_boxes=list(pred),pred_classes=list(pclass),pred_confidence=list(scores))

class Truths(unittest.TestCase):
    def setUp(self):self.original=load_original(HERE/'frozen_sources/object_evidence_loss.py')
    def test_five_classes_and_reordered_teacher(self):
        b=[[i*15,0,i*15+10,10] for i in range(5)]
        p=gt_pair(row(b,list(range(5))),row(b[::-1],list(range(5))[::-1]),self.original)
        self.assertEqual([(x['N_gt_row'],x['T_gt_row']) for x in p],[(i,4-i) for i in range(5)])
    def test_wrong_class_and_unpaired(self):
        n=row([[0,0,10,10],[20,0,30,10]],[0,1]);t=row([[0,0,10,10],[20,0,30,10],[50,0,60,10]],[4,1,2])
        p=gt_pair(n,t,self.original);self.assertEqual([(x['N_gt_row'],x['T_gt_row']) for x in p],[(1,1)])
    def test_maximum_cardinality_not_raw_iou_sum_or_greedy(self):
        self.original['_iou']=lambda a,b:torch.tensor([[.95,.5],[.5,.49]])
        x=row([[0,0,10,10],[0,0,10,10]],[0,0]);p=gt_pair(x,x,self.original)
        self.assertEqual([(r['N_gt_row'],r['T_gt_row']) for r in p],[(0,1),(1,0)])
    def test_all_empty_and_one_side_empty(self):
        e=row();validate_row(e);self.assertEqual(gt_pair(e,e,self.original),[])
        self.assertEqual(gt_pair(row([[0,0,10,10]],[0]),e,self.original),[])
    def test_explicit_image_mapping_boundaries(self):
        models={'N':{'nr':row()},'T':{'tr':row()}};aliases={'N':{'nr':'/rgb/val/a.jpg'},'T':{'tr':'/ir/val/a.jpg'}}
        self.assertEqual(bind_images({'/rgb/val/a.jpg':'/ir/val/a.jpg'},models,aliases)[0]['ir_image'],'tr')
        with self.assertRaises(ValueError):bind_images({'/rgb/train/a.jpg':'/ir/val/a.jpg'},models,aliases)
    def test_original_none_constructor_and_ambiguity(self):
        n='/rgb/val/a.jpg';t='/ir/val/a.jpg';models={'N':{n:row()},'T':{t:row()}};aliases={'N':{n:n},'T':{t:t}}
        self.assertEqual(original_image_fallback(models,aliases,HERE/'frozen_sources/paired_rgbir_data.py'),{n:t})
        models['T']['/ir/val/a.png']=row();aliases['T']['/ir/val/a.png']='/ir/val/a.png'
        with self.assertRaises(ValueError):original_image_fallback(models,aliases,HERE/'frozen_sources/paired_rgbir_data.py')
    def test_actual_native_order_and_strict_confidence(self):
        # Native unique-pred then unique-GT chooses prediction 0 despite greater IoU of prediction 1.
        r=row([[0,0,10,10]],[0],pred=[[0,0,8,10],[0,0,9,10]],pclass=[0,0],scores=[.25,.8])
        native=load_native(HERE/'frozen_sources');low=match_row(r,None,.5,native);high=match_row(r,.25,.5,native)
        self.assertEqual(low['gt_matches'][0]['prediction_id'],0);self.assertEqual(high['gt_matches'][0]['prediction_id'],1)
        self.assertEqual(high['kept_prediction_ids'],[1])
    def test_shapes_and_invalid_classes_rejected(self):
        n=row();t=row();t['canvas_shape']=[101,100]
        with self.assertRaises(ValueError):gt_pair(n,t,self.original)
        with self.assertRaises(ValueError):validate_row(row([[0,0,1,1]],[5]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),scope='CPU_SYNTHETIC_DRONE_PAIRING_AND_NATIVE_MATCH',new_GPU=False,real_cache_read=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(not r.wasSuccessful())
