"""Portable tests: installed pinned native APIs, or local saved-source AST subset."""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
import detector_witness as w


def source_functions(path,names,env):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    found=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:found.append(node)
        if isinstance(node,ast.ClassDef):
            found.extend(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in names)
    code=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+found,type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(path),'exec'),env)


def xywh2xyxy(x):
    y=x.clone();y[...,0]=x[...,0]-x[...,2]/2;y[...,1]=x[...,1]-x[...,3]/2
    y[...,2]=x[...,0]+x[...,2]/2;y[...,3]=x[...,1]+x[...,3]/2
    return y


def saved_native(path):
    env=dict(torch=torch,np=np,sys=sys,time=time,LOGGER=NS(warning=lambda text:(_ for _ in ()).throw(AssertionError(text))),xywh2xyxy=xywh2xyxy)
    source_functions(path/'7_metrics.py',{'box_iou'},env)
    source_functions(path/'4_validator.py',{'match_predictions'},env)
    Base=type('SavedBase',(),{'match_predictions':env['match_predictions']})
    source_functions(path/'3_val.py',{'_process_batch'},env)
    Validator=type('SavedDetection',(Base,),{'_process_batch':env['_process_batch']})
    source_functions(path/'6_nms.py',{'non_max_suppression','TorchNMS'},env)
    return Validator,Base,env['box_iou'],env['non_max_suppression'],xywh2xyxy


def fixture():
    batch=dict(img=torch.zeros(3,3,64,64),batch_idx=torch.tensor([0.,1.]),cls=torch.zeros(2,1),
        bboxes=torch.tensor([[.5,.5,.5,.5],[.5,.5,.5,.5]]))
    batch['teacher_batch']={k:v.clone() for k,v in batch.items() if k!='img'}
    cfg=ADAPTER.evidence_config(dict(input_size=64,levels=(0,1)))
    def raw(logit):
        d=torch.full((3,4,16,84),-30.);d[:,:,0]=0.
        r=dict(boxes=d.reshape(3,64,84),scores=torch.full((3,1,84),-10.),feats=[torch.zeros(3,4,h,h) for h in (8,4,2)])
        for bi in (0,1):
            for edge,value in enumerate((.5,.5,3.5,3.5)):
                r['boxes'][bi,edge*16:(edge+1)*16,18].fill_(-30.)
                r['boxes'][bi,edge*16+int(value),18]=math.log(.5);r['boxes'][bi,edge*16+int(value)+1,18]=math.log(.5)
            r['scores'][bi,0,18]=logit
        return r
    raws={'S':raw(-2.),'T':raw(3.),'R':raw(-2.)}
    records=[dict(rgb_global_row=i,ir_global_row=i,stable_rgb_gt_id='rgb:'+str(i),stable_ir_gt_id='ir:'+str(i),
        frame_id='frame:'+str(i),gates=dict(teacher_correct_own=True)) for i in (0,1)]
    existing=dict(scope='SAME_FORWARD_SELECTION_COVERAGE',records=records,
        ir_records=[dict(ir_global_row=i,stable_ir_gt_id='ir:'+str(i)) for i in (0,1)],frames=[dict(frame_id='frame:'+str(i)) for i in range(3)])
    def head():
        if INSTALLED:
            from ultralytics.nn.modules.head import Detect
            h=Detect(nc=1,ch=(4,4,4));h.stride=torch.tensor([8.,16.,32.]);return h.eval()
        class Head:
            end2end=False;xyxy=False;export=False;nc=1
            def _inference(self,r):
                centers,sv,_,_=ADAPTER.original._layout(r,cfg,(8,16,32));b=ADAPTER.original._decode_boxes(r,centers,sv)
                y=torch.cat(((b[...,:2]+b[...,2:])/2,b[...,2:]-b[...,:2]),-1).transpose(1,2)
                return torch.cat((y,r['scores'].sigmoid()),1)
        return Head()
    return batch,raws,{n:NS(model=[head()]) for n in raws},cfg,existing


class Truths(unittest.TestCase):
    def test_dense_threshold_class_and_highest_confidence(self):
        boxes=torch.tensor([[0.,0,10,10],[0.,0,10,10],[20.,20,30,30]])
        p=torch.tensor([[.25,.8,.1],[.1,.1,.9]])
        result=w.dense_witness(boxes,p,boxes[:1],torch.tensor([0]),[0],['a'],COMP[2])
        self.assertEqual(result[0]['dense_correct_count'],2);self.assertEqual(result[0]['dense_witness']['anchor_index'],1)
        strict=w.dense_witness(boxes,p,boxes[:1],torch.tensor([0]),[0],['a'],COMP[2],strict=True)
        self.assertEqual(strict[0]['dense_correct_count'],1)
    def test_other_gt_overlap_does_not_claim_one_to_one(self):
        gt=torch.tensor([[0.,0,10,10],[0.,0,10,10]])
        x=w.dense_witness(gt[:1],torch.tensor([[.9]]),gt,torch.tensor([0,0]),[2,3],['a','b'],COMP[2])
        self.assertEqual(x[0]['dense_witness']['other_gt_overlaps'][0]['global_gt_row'],3)
        self.assertTrue(all(y['dense_any_correct'] for y in x))
    def test_actual_native_non_greedy_order_and_profile_restore(self):
        # Native unique-det reorders by detection index before unique-GT: det0 wins.
        gt=dict(bboxes=torch.tensor([[0.,0,10,10]]),cls=torch.tensor([0.]))
        pred=dict(bboxes=torch.tensor([[0.,0,8,10],[0.,0,9,10]]),cls=torch.tensor([0.,0.]))
        before=sys.getprofile();pairs,tp=w.native_matches(pred,gt,COMP)
        self.assertEqual(pairs.tolist(),[[0,0]]);self.assertEqual(tp.tolist(),[[True],[False]])
        self.assertIs(sys.getprofile(),before)
        # Nonempty all-false TP cannot hide a failed native return-frame capture.
        wrong_base=type('OtherBase',(),{'match_predictions':lambda *args:None})
        empty_iou_pred=dict(bboxes=pred['bboxes']+100,cls=pred['cls'])
        with self.assertRaises(AssertionError):w.native_matches(empty_iou_pred,gt,(COMP[0],wrong_base,*COMP[2:]))
    def test_native_nms_indices_mutation_and_empty(self):
        y=torch.zeros(2,5,7);y[0,:4,:2]=torch.tensor([[5,5],[5,5],[10,10],[10,10.]])
        y[0,4,:2]=torch.tensor([.9,.8]);saved=y.clone()
        det,ids=COMP[3](y.clone(),.25,.7,return_idxs=True,multi_label=True,max_det=300)
        self.assertEqual(ids[0].tolist(),[0]);self.assertEqual(len(det[1]),0);self.assertTrue(torch.equal(y,saved))
    def test_whole_api_own_gt_frames_and_no_grad(self):
        batch,raws,models,cfg,old=fixture();before=copy.deepcopy(raws)
        result=w.analyze_witness(batch,raws,models,cfg,(8,16,32),old)
        self.assertEqual(len(result['records']),2);self.assertEqual(len(result['frames']),3)
        for r in result['records']:
            self.assertTrue(r['detector']['T']['dense_any_correct']);self.assertTrue(r['detector']['T']['native_postnms_matched'])
            self.assertFalse(r['detector']['S']['dense_any_correct']);self.assertFalse(r['detector']['S']['native_postnms_matched'])
        self.assertEqual(result['frames'][2]['native_detections']['T'],[])
        for n in raws:
            for k in ('scores','boxes'):self.assertTrue(torch.equal(raws[n][k],before[n][k]));self.assertIsNone(raws[n][k].grad)
        json.dumps(result,allow_nan=False)
    def test_original_teacher_gate_mismatch_reject(self):
        batch,raws,models,cfg,old=fixture();old['records'][0]['gates']['teacher_correct_own']=False
        with self.assertRaises(AssertionError):w.analyze_witness(batch,raws,models,cfg,(8,16,32),old)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--native-source-dir',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.reference_dir));import selection_adapter as ADAPTER
    assert Path(ADAPTER.__file__).resolve()==args.reference_dir.resolve()/'selection_adapter.py'
    INSTALLED=args.native_source_dir is None;COMP=w.native_components() if INSTALLED else saved_native(args.native_source_dir)
    w.native_components=lambda:COMP;torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Truths))
    with args.output.open('x',encoding='utf-8') as f:json.dump(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),installed_native_APIs=INSTALLED,
        native_head_actual=INSTALLED,saved_native_source_AST=not INSTALLED,torch=str(torch.__version__),
        GPU_used=False,new_model_forward=False,new_hash_computed=False),f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
