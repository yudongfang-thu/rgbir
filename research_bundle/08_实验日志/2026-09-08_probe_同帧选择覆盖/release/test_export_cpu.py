"""Portable CPU truths against supplied pinned selector and selected-only source."""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import torch
from raw_object_state import state,spatial_assign
from export_selection import export_selection


def prepare_modules(reference,candidate):
    sys.path.insert(0,str(reference))
    import selection_adapter,classification_logit
    for module,name in ((selection_adapter,'selection_adapter.py'),(classification_logit,'classification_logit.py')):
        if Path(module.__file__).resolve()!=reference.resolve()/name:raise ValueError('Wrong pinned test module')
    name='_same_frame_cpu_candidate';spec=importlib.util.spec_from_file_location(name,candidate)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return selection_adapter,module.make_api(selection_adapter,classification_logit)


def fixture(dtype=torch.float32):
    # Three images retain one empty image; GT global order is intentionally interleaved.
    rgb=dict(batch_idx=torch.tensor([1.,0.,0.]),cls=torch.zeros(3,1),
             bboxes=torch.tensor([[.5,.5,.375,.375],[.3125,.3125,.25,.25],[.6875,.6875,.25,.25]]))
    ir=dict(batch_idx=torch.tensor([0.,1.,1.]),cls=torch.zeros(3,1),
            bboxes=torch.tensor([[.3125,.3125,.25,.25],[.5,.5,.375,.375],[.8125,.1875,.25,.25]]))
    batch=dict(rgb,img=torch.zeros(3,3,64,64),strong_img=torch.zeros(3,3,64,64),teacher_batch=ir,
        im_file=['/rgb/0.png','/rgb/1.png','/rgb/empty.png'],
        pair_info=[dict(weak_source='/rgb/'+str(i)+'.png',strong_source='/ir/'+str(i)+'.png') for i in range(3)])
    cfg=ADAPTER.evidence_config(dict(input_size=64,levels=(0,1)))
    def raw():
        d=torch.full((3,4,16,84),-30.);d[:,:,0,:]=0
        return dict(scores=torch.full((3,1,84),-4.,dtype=dtype),boxes=d.reshape(3,64,84).to(dtype),
                    feats=[torch.zeros(3,4,h,h,dtype=dtype) for h in (8,4,2)])
    student,teacher,reference=raw(),raw(),raw()
    centers,sv,ranges,size=ADAPTER.original._layout(student,cfg,(8,16,32))
    def fill(out,labels,fgscore):
        idx,cl,boxes=ADAPTER.original._labels(labels,3,1,torch.device('cpu'),size)
        for row,box in enumerate(boxes):
            bi=int(idx[row]);inside=ADAPTER.original._inside(box[None],centers)[0]
            out['scores'][bi,0,inside]=fgscore
            # Place an exact decoded box at the nearest P3 center inside this GT.
            valid=torch.nonzero(inside[:64],as_tuple=False).flatten()
            center=(box[:2]+box[2:])/2
            anchor=int(valid[((centers[valid]-center)**2).sum(1).argmin()])
            distances=torch.cat((centers[anchor]-box[:2],box[2:]-centers[anchor]))/sv[anchor]
            for edge,distance in enumerate(distances):
                d=float(distance);low=int(math.floor(d));fraction=d-low
                logits=out['boxes'][bi,edge*16:(edge+1)*16,anchor];logits.fill_(-30.)
                logits[low]=math.log(1-fraction) if fraction else 0.
                if fraction:logits[low+1]=math.log(fraction)
    fill(student,batch,-2.);fill(reference,batch,-1.);fill(teacher,ir,3.)
    def ids(labels,prefix):
        return [dict(global_gt_row=i,image_index=int(bi),stable_gt_id=prefix+str(int(bi))+'::native_gt:'+str(i),
            source_gt_row=i,image_path=prefix+str(int(bi))) for i,bi in enumerate(labels['batch_idx'])]
    identity=dict(status='STABLE_GT_IDENTITY_VERIFIED',frame_id='fixture_forward',rgb_rows=ids(rgb,'/rgb/'),ir_rows=ids(ir,'/ir/'))
    return batch,student,teacher,reference,cfg,identity


def export(fix):
    b,s,t,r,c,identity=fix
    return export_selection(b,s,t,r,api=API,evidence_config=c,strides=(8,16,32),selection_seed=43,identity_contract=identity)


class Truths(unittest.TestCase):
    def test_fixed_state_priority_and_separate_own_cross_coordinates(self):
        base=dict(confidence=.5,**{'class':0},own_gt_iou=.8,anchor_index=1,box=[0,0,1,1])
        self.assertEqual(state(None,0)['state'],'no_candidate')
        cases=[(.1,1,.1,'low_confidence'),(.5,1,.1,'class_and_localization'),(.5,1,.8,'class_only'),(.5,0,.1,'localization_only'),(.5,0,.8,'correct')]
        for conf,cl,iou,name in cases:
            p=dict(base,confidence=conf,own_gt_iou=iou);p['class']=cl
            self.assertEqual(state(p,0)['state'],name)
        self.assertTrue(state(base,0)['correct'])
        self.assertFalse(state(base,0,evaluation_iou=.2,coordinate_scope='rgb')['correct'])

    def test_true_selector_all_gt_unmatched_empty_frame_and_gate_closure(self):
        result=export(fixture())
        self.assertEqual(len(result['records']),3);self.assertEqual(len(result['ir_records']),3)
        self.assertEqual(len(result['frames']),3);self.assertEqual(result['frames'][2]['rgb_gt_count'],0)
        unmatched=result['records'][2];self.assertFalse(unmatched['gates']['matched'])
        self.assertIsNone(unmatched['states']['T']);self.assertIsNone(unmatched['gates']['selected'])
        matched=[r for r in result['records'] if r['gates']['matched']]
        self.assertEqual(len(matched),2);self.assertEqual(result['selector_counts']['base_count'],2)
        self.assertEqual(result['selector_counts']['eligible_count'],2);self.assertEqual(result['selector_counts']['selected_count'],1)
        self.assertEqual(sum(r['gates']['selected'] for r in matched),1)
        for row in matched:
            self.assertEqual(row['states']['S']['state'],'low_confidence');self.assertTrue(row['states']['T']['correct'])
        json.dumps(result,allow_nan=False)

    def test_exact_thin_fp16_and_full_fp32_selector_crosscheck(self):
        a=export(fixture(torch.float32));b=export(fixture(torch.float16))
        self.assertFalse(a['thin_path_used']);self.assertTrue(b['thin_path_used'])
        self.assertEqual(a['selector_counts'],b['selector_counts'])
        for result in (a,b):self.assertTrue(result['exact_selector_crosscheck'])

    def test_nonbase_and_zero_quality_remain_explicit(self):
        fix=fixture();fix[3]['scores'].fill_(-10.)
        r=export(fix);self.assertEqual(r['selector_counts']['base_count'],0)
        for row in r['records']:
            if row['gates']['matched']:
                self.assertFalse(row['gates']['reference_candidate']);self.assertFalse(row['gates']['base'])
                self.assertIsNotNone(row['gates']['quality_q'])
        fix=fixture();fix[2]['scores'].copy_(fix[3]['scores'])
        r=export(fix);self.assertEqual(r['selector_counts']['eligible_count'],0)
        self.assertEqual(r['selector_counts']['selected_count'],0)

    def test_any_candidate_gate_is_not_one_to_one_assignment(self):
        boxes=torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]])
        predictions=boxes[:1];confidence=torch.tensor([.9]);classes=torch.tensor([0])
        assigned,_=spatial_assign(boxes,predictions,confidence,classes,original=ADAPTER.original)
        gate=ADAPTER.original._has_candidate(predictions,confidence[None],boxes,.05,.1)
        self.assertEqual(len(assigned),1);self.assertEqual(int(gate.sum()),2)

    def test_wrong_identity_rejected_inputs_and_gradients_unchanged(self):
        fix=fixture();fix[-1]['rgb_rows'][0]['image_index']=0
        with self.assertRaisesRegex(ValueError,'order differs'):export(fix)
        fix=fixture();tensors=[raw[k] for raw in fix[1:4] for k in ('scores','boxes')]
        before=[v.clone() for v in tensors]
        result=export(fix)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,tensors)))
        self.assertTrue(all(t.grad is None for t in tensors));self.assertFalse(result['new_model_forward'])
        self.assertEqual(result['optimizer_updates'],0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--candidate-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();ADAPTER,API=prepare_modules(args.reference_dir,args.candidate_source)
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        torch=str(torch.__version__),reference=str(args.reference_dir),candidate=str(args.candidate_source),
        GPU_used=False,new_model_forward=False,new_hash_computed=False,
        details=[dict(test=str(t),traceback=e) for t,e in result.failures+result.errors])
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
