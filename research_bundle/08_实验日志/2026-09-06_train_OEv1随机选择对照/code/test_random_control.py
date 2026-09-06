"""CPU integration/scientific tests for the isolated paired_random route."""
import ast
import copy
import math
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import torch
import object_evidence_loss as loss_module
from object_evidence_loss import EvidenceConfig, object_evidence_loss
from test_object_evidence_loss import raw, labels


def read_source(name):
    embedded=globals().get('EMBEDDED_SOURCES')
    if embedded is not None:return embedded[name]
    here=Path(__file__).resolve().parent
    return (here/name if name!='baseline_trainer' else here.parent/'source_baseline/train_object_evidence.py').read_text(encoding='utf-8')


def criterion_class(source, capture):
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('raw_prediction','combine_loss','EvidenceCriterion')]
    def stub_loss(s,t,r,batch,**kwargs):
        capture.append(kwargs['arm'])
        return s['scores'].square().mean(), {'selected_count':2}
    namespace={'torch':torch,'EvidenceConfig':EvidenceConfig,'object_evidence_loss':stub_loss,
               'time':time,'append_json':lambda *_:None}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<criterion-under-test>','exec'),namespace)
    return namespace['EvidenceCriterion']


class RandomControlTests(unittest.TestCase):
    def test_existing_pn_criterion_loss_gradient_unchanged_and_random_forwarded(self):
        cfg={'evidence':{'input_size':64},'seed':42,'kd_weight':.1,'log_every_batches':100}
        class Frozen(torch.nn.Module):
            def forward(self,image):return raw(1)
        def run(source,arm):
            seen=[]
            criterion=criterion_class(source,seen)
            trainer=SimpleNamespace(model=SimpleNamespace(stride=(8,16,32)),epoch=0,real_updates=0,wall_started=time.time(),save_dir=Path('.'))
            native=lambda prediction,batch:(prediction['scores'].sum(),torch.zeros(3))
            c=criterion(native,Frozen(),Frozen(),cfg,arm,trainer,False)
            prediction=raw(2,requires_grad=True)
            batch={'teacher_batch':labels(),'img':torch.zeros(1,3,64,64),'strong_img':torch.zeros(1,3,64,64)}
            total,_=c(prediction,batch)
            gradient=torch.autograd.grad(total,prediction['scores'])[0]
            return total.detach(),gradient,seen
        original=read_source('baseline_trainer'); revised=read_source('train_object_evidence.py')
        for arm in ('paired','weight0'):
            a=run(original,arm); b=run(revised,arm)
            self.assertTrue(torch.equal(a[0],b[0]))
            self.assertTrue(torch.equal(a[1],b[1]))
            self.assertEqual(a[2],b[2])
            self.assertEqual(b[2],['paired'])
        self.assertEqual(run(revised,'paired_random')[2],['paired_random'])

    def test_same_k_and_dose_from_sparse_base_and_ineligible_can_be_selected(self):
        q=torch.tensor([4.,0.,100.,2.,0.,0.,3.,0.])
        base=torch.tensor([1,1,0,1,1,0,1,0],dtype=torch.bool)
        teacher_correct=torch.tensor([1,0,1,1,1,1,0,1],dtype=torch.bool)
        eligible=base & teacher_correct & (q>0)
        p=loss_module._choose(q,eligible,base,'paired',.5,42)
        seen_ineligible=False
        for seed in range(32):
            before=torch.random.get_rng_state().clone()
            r=loss_module._choose(q,eligible,base,'paired_random',.5,seed)
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            self.assertEqual(len(r),len(p))
            self.assertEqual(len(set(r.tolist())),len(r))
            self.assertTrue(bool(base[r].all()))
            self.assertEqual(len(r)/int(base.sum()),len(p)/int(base.sum()))
            self.assertTrue(torch.equal(r,loss_module._choose(q,eligible,base,'paired_random',.5,seed)))
            seen_ineligible|=bool((~eligible[r]).any())
        self.assertTrue(seen_ineligible)

    def test_full_loss_can_select_unreliable_object_but_keeps_k_and_normalizer(self):
        targets=labels(((20/64,28/64,24/64,24/64),(44/64,28/64,24/64,24/64)))
        targets['teacher_batch']=copy.deepcopy(targets)
        s,t,r=raw(0,targets,True),raw(4,targets),raw(1,targets)
        cfg=EvidenceConfig(input_size=64)
        centers,_,_,_=loss_module._layout(s,cfg,(8,16,32))
        second=loss_module._inside(torch.tensor([[32.,16.,56.,40.]]),centers)[0]
        t['scores'][0,0,second]=-1.
        _,pstats=object_evidence_loss(s,t,r,targets,config=cfg,arm='paired')
        alternative=False
        for seed in range(8):
            value,stats=object_evidence_loss(s,t,r,targets,config=cfg,arm='paired_random',seed=seed)
            for key in ('base_count','teacher_correct_base_count','eligible_count','selected_count','normalizer','nominal_dose'):
                self.assertEqual(stats[key],pstats[key])
            alternative|=stats['selected_object_ids']!=pstats['selected_object_ids']
            grad=torch.autograd.grad(value,s['scores'],retain_graph=True)[0]
            self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertTrue(alternative)

    def test_random_zero_eligible_is_zero_even_with_nonempty_base(self):
        selected=loss_module._choose(torch.zeros(4),torch.zeros(4,dtype=torch.bool),torch.ones(4,dtype=torch.bool),'paired_random',.5,0)
        self.assertEqual(len(selected),0)


if __name__=='__main__':unittest.main(verbosity=2)
