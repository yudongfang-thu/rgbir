"""Known-truth L3 tests with pinned original matching/base helpers, CPU only."""
import argparse,copy,json,math,sys,unittest
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args,_=p.parse_known_args()
sys.path[:0]=[str(Path(__file__).parent),str(args.reference_dir),str(args.reference_dir/'task_conditional_reference')]
import l3_distribution_loss as l3
import localization_box_v2 as l2
torch.set_num_threads(1)

def labels(boxes=((16.,16.,40.,40.),),classes=None,indices=None):
    b=torch.tensor(boxes).reshape(-1,4)
    return dict(bboxes=torch.cat(((b[:,:2]+b[:,2:])/2,b[:,2:]-b[:,:2]),-1)/64,
        cls=torch.tensor(classes if classes is not None else [0]*len(b)).reshape(-1,1),
        batch_idx=torch.tensor(indices if indices is not None else [0]*len(b)))

def set_box(raw,anchor,box,bi=0):
    center=torch.tensor(((anchor%8+.5)*8,(anchor//8+.5)*8));b=torch.tensor(box)
    distances=torch.cat((center-b[:2],b[2:]-center))/8
    assert bool(((distances>=0)&(distances<15)).all())
    view=raw['boxes'].reshape(raw['boxes'].shape[0],4,16,84)
    for side,d in enumerate(distances.tolist()):
        lo,part=int(d),d-int(d);view[bi,side,:,anchor]=-40.
        view[bi,side,lo,anchor]=math.log(1-part)
        if part:view[bi,side,lo+1,anchor]=math.log(part)

def raw(anchor=27,box=(20.,20.,36.,36.),grad=True):
    r=dict(boxes=torch.full((1,64,84),-40.),scores=torch.full((1,2,84),-20.),
        feats=[torch.zeros(1,2,k,k) for k in (8,4,2)])
    r['boxes'].reshape(1,4,16,84)[:,:,1,:]=0.
    set_box(r,anchor,box);r['scores'][0,0,anchor]=4.
    for k in ('boxes','scores'):r[k].requires_grad_(grad)
    return r

def fixture():
    b=labels();b['teacher_batch']=labels()
    return raw(),raw(box=(16.,16.,40.,40.)),raw(),b

class Truths(unittest.TestCase):
    def test_learning_dtype_positive_tail_rejected_zeros_legal(self):
        q=[[0.]*16 for _ in range(4)]
        for side in q:side[1]=1.
        self.assertIsNotNone(l3.float32_target_or_none(q,'cpu'))
        q[0][15]=1e-100
        self.assertIsNone(l3.float32_target_or_none(q,'cpu'))
        s,t,r,b=fixture();t['boxes']=t['boxes'].detach()
        t['boxes'][t['boxes']==-40.]=-240.
        loss,stats=l3.compute(s,t,r,b)
        self.assertEqual((stats['base_count'],stats['target_float32_rejected'],stats['selected_count']),(1,1,0))
        self.assertTrue(stats['base_records'][0]['full_probability_support'])
        self.assertFalse(stats['base_records'][0]['target_float32_supported'])
        self.assertEqual(float(loss),0.)
    def test_old_R_base_and_gap_exact(self):
        s,t,r,b=fixture();_,old=l2.compute(s,t,r,b);_,new=l3.compute(s,t,r,b)
        for k in ('common_count','pair_iou_count','reference_support_count','reference_unique_owner_count','base_count','reference_reliable_count','reference_gap_count','normalizer'):
            self.assertEqual(old[k],new[k])
        keys=('batch_index','rgb_gt_index','ir_gt_index','class_id','reference_anchor','pair_iou','rgb_gt','ir_gt','reference_box','reference_iou','reference_conf','reference_reliable','reference_gap')
        self.assertEqual([{k:r[k] for k in keys} for r in old['base_records']],[{k:r[k] for k in keys} for r in new['base_records']])
    def test_finite_gradient_only_actual_student_anchor(self):
        s,t,r,b=fixture();loss,stats=l3.compute(s,t,r,b);loss.backward()
        self.assertEqual(stats['selected_anchors'],[(0,27,27,0,0)])
        self.assertTrue(torch.isfinite(s['boxes'].grad).all());self.assertGreater(float(s['boxes'].grad.abs().sum()),0)
        self.assertEqual(torch.nonzero(s['boxes'].grad.abs().sum((0,1))).flatten().tolist(),[27])
        for v in (s['scores'],t['scores'],t['boxes'],r['scores'],r['boxes']):self.assertIsNone(v.grad)
        json.dumps(stats,allow_nan=False)
    def test_GT_and_DFL_identical_selection(self):
        s,t,r,b=fixture();_,a=l3.compute(s,t,r,b,variant='L3-DFL');_,g=l3.compute(s,t,r,b,variant='L3-GT')
        for k in ('base_records','selected_anchors','selected_object_ids','base_count','selected_count','normalizer'):
            self.assertEqual(a[k],g[k])
    def test_exact_KL_temperature_fourmean_base_not_selected(self):
        raw_s={'boxes':torch.zeros(1,64,2,requires_grad=True)}
        ids=torch.tensor([[0,1,1,0,0]]);q=torch.zeros(1,4,16);q[:,:,3]=1
        loss=l3.distribution_loss(raw_s,ids,q,2)
        self.assertAlmostEqual(float(loss),2*math.log(16),places=6)
        loss.backward();self.assertTrue(raw_s['boxes'].grad[:,:,0].eq(0).all())
    def test_GT_sqrt_after_adjacent_bins_no_clamp(self):
        q=l3.gt_temperature2_target(torch.tensor([[0.,1.25,2.5,14.99]]))
        self.assertEqual(float(q[0,0,0]),1.)
        self.assertAlmostEqual(float(q[0,1,1]),math.sqrt(.75)/(math.sqrt(.75)+.5),places=6)
        self.assertEqual(float(q[0,2,2]),.5)
        for d in (-.001,15.):
            with self.assertRaises(ValueError):l3.gt_temperature2_target(torch.full((1,4),d))
    def test_no_independent_teacher_anchor_substitution(self):
        s,t,r,b=fixture();t=raw(anchor=18,box=(16.,16.,40.,40.))
        _,a=l3.compute(s,t,r,b);_,old=l2.compute(s,t,r,b)
        self.assertEqual(a['selected_count'],0);self.assertEqual(old['selected_count'],1)
        self.assertEqual(a['base_records'][0]['teacher_anchor'],27)
    def test_full_tail_shift_rejects_without_shrinking_base(self):
        s,t,r,b=fixture();b['teacher_batch']=labels(((16.5,16.,40.5,40.),));t=raw(box=(16.5,16.,40.5,40.))
        _,a=l3.compute(s,t,r,b)
        self.assertEqual((a['base_count'],a['fullsupport_rejected'],a['selected_count']),(1,1,0))
    def test_temperature_before_nonidentity_transport(self):
        s,t,r,b=fixture();b['teacher_batch']=labels(((15.,15.,41.,41.),));t=raw(box=(15.,15.,41.,41.))
        orig,native=l3._helpers();ids,gt,target,stats,c,st=l3._select(t,r,b,(8,16,32),orig,native)
        self.assertEqual(stats['selected_count'],1)
        logits=t['boxes'][0,:,27].detach().reshape(4,16).tolist()
        arguments=dict(teacher_gt=[15.,15.,41.,41.],student_gt=[16.,16.,40.,40.],teacher_center=[28.,28.],student_center=[28.,28.],teacher_stride=8.,student_stride=8.)
        t1=l3.transport_logits(logits,**arguments)['target_probabilities']
        wrong=torch.tensor(t1).sqrt();wrong/=wrong.sum(-1,keepdim=True)
        self.assertFalse(torch.allclose(target[0],wrong,atol=1e-5,rtol=1e-5))
    def test_student_outputs_not_used_for_selection(self):
        s,t,r,b=fixture();_,a=l3.compute(s,t,r,b);_,z=l3.compute(raw(box=(18.,18.,38.,38.)),t,r,b)
        self.assertEqual(a['base_records'],z['base_records'])
    def test_base_denominator_before_R_reliability(self):
        s,t,r,b=fixture();one,_=l3.compute(s,t,r,b)
        for v in (s,t,r):
            for k in ('boxes','scores'):v[k]=v[k].detach().repeat(2,1,1).requires_grad_(v is s and k=='boxes')
            v['feats']=[x.repeat(2,1,1,1) for x in v['feats']]
        r['scores'][1,0,27]=math.log(.1/.9)
        b=labels(((16.,16.,40.,40.),)*2,indices=[0,1]);b['teacher_batch']=copy.deepcopy(b)
        two,a=l3.compute(s,t,r,b)
        self.assertEqual((a['base_count'],a['selected_count'],a['normalizer']),(2,1,2))
        self.assertAlmostEqual(float(two),float(one)/2,places=6)
    def test_all_class_IR_unique_owner_gate(self):
        s,t,r,b=fixture();b['teacher_batch']=labels(((16.,16.,40.,40.),)*2,classes=[0,1])
        _,a=l3.compute(s,t,r,b);self.assertEqual((a['base_count'],a['selected_count']),(1,0))
    def test_R_gap_point70_unchanged(self):
        s,t,r,b=fixture();r=raw(box=(16.,16.,40.,40.));_,a=l3.compute(s,t,r,b)
        self.assertEqual((a['base_count'],a['reference_gap_count'],a['selected_count']),(1,0,0))
        self.assertIsNone(a['base_records'][0]['teacher_same_anchor_unique'])
    def test_empty_differentiable_zero(self):
        s,t,r,b=fixture();b=labels(());b['teacher_batch']=labels(())
        loss,a=l3.compute(s,t,r,b);loss.backward()
        self.assertEqual(float(loss),0);self.assertEqual(a['normalizer'],1);self.assertTrue(s['boxes'].grad.eq(0).all())
        with self.assertRaises(ValueError):l3.compute(s,t,r,b,variant='L1')

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    source=Path(__file__).with_name('l3_distribution_loss.py');s=source.stat()
    with args.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),source=dict(path=str(source),bytes=s.st_size,mtime_ns=s.st_mtime_ns),reference_dir=str(args.reference_dir),scope='CPU_SYNTHETIC_LOSS_SELECTION_ONLY',GPU_used=False,new_AP_read=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(not result.wasSuccessful())
