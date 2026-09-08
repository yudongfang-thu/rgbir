"""Exploratory one-branch losses. No teacher or reference in deployed model."""
import time
import torch
from runtime import legacy,ORIGINAL_CRITERION
from gradient_observation import shared_parameter_set

def make_type(api):
    class DirectionCriterion(ORIGINAL_CRITERION):
        def losses(self,student,teacher,reference,batch,arms):
            result={}; strides=tuple(int(x) for x in self.trainer.model.stride)
            if any(a in ('N','C1','C2','F-rel') for a in arms):
                payload=api.build(student,teacher,reference,batch,strides=strides,config=self.evidence_cfg,
                                  selection_seed=self.cfg['seed']+self.calls,full_diagnostics=False)
                if 'N' in arms or 'C1' in arms:
                    loss,stats=api.loss(payload)
                    for a in ('N','C1'):
                        if a in arms:result[a]=(loss,stats)
                if 'C2' in arms:
                    from direction_losses import class_balanced_loss
                    result['C2']=class_balanced_loss(payload.learning)
                if 'F-rel' in arms:
                    from direction_losses import feature_relation_loss
                    result['F-rel']=feature_relation_loss(payload.learning,student,teacher,batch)
            if any(a.startswith('L2') for a in arms):
                from localization_box_v2 import compute
                for arm in arms:
                    if arm.startswith('L2'):result[arm]=compute(student,teacher,reference,batch,strides,variant=arm)
            return result

        def __call__(self,prediction,batch):
            native,items=self.native(prediction,batch); self.calls+=1
            student=legacy.raw_prediction(prediction)
            self.teacher.eval();self.reference.eval()
            with torch.no_grad():
                teacher=legacy.raw_prediction(self.teacher(batch['strong_img']))
                reference=legacy.raw_prediction(self.reference(batch['img']))
            arm=self.cfg['arm']
            # N retains same auxiliary control computation and literal zero dose.
            request='L2-box' if arm=='N' and self.cfg['dataset']=='llvip' else arm
            if arm=='F-rel-GM':request='F-rel'  # Explicit alias; identical frozen operator.
            kd,stats=self.losses(student,teacher,reference,batch,[request])[request]
            weight=float(self.cfg['kd_coefficient']);b=int(batch['img'].shape[0])
            total=native.sum()+b*weight*kd
            if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite direction loss')
            selected=int(stats.get('selected_count',0));self.selected_total+=selected
            if self.calls==1 or (self.cfg.get('canary_execution',False) and selected and not any(x['kd_gradient_l2']>0 for x in self.gradient_checks)):
                indices,named=shared_parameter_set(self.trainer.model)
                names=[n for n,p in named];params=[p for n,p in named]
                g=torch.autograd.grad(kd,params,retain_graph=True,allow_unused=True)
                norm=float(sum((v.detach().double().square().sum() for v in g if v is not None),torch.zeros((),device=kd.device)).sqrt())
                if not torch.isfinite(torch.tensor(norm)):raise FloatingPointError('Nonfinite KD gradient')
                row=dict(batch=self.calls,kd_gradient_l2=norm,parameter_names=names,
                    actual_B=b,coefficient=weight,teacher_has_grad=any(p.grad is not None for p in self.teacher.parameters()),
                    reference_has_grad=any(p.grad is not None for p in self.reference.parameters()))
                if row['teacher_has_grad'] or row['reference_has_grad']:raise AssertionError('Auxiliary gradient leak')
                if arm=='N' and not torch.equal(total,native.sum()):raise AssertionError('N changed native loss')
                if norm>0 or self.calls==1:self.gradient_checks.append(row)
                legacy.append_json(self.trainer.save_dir/'gradient_checks.jsonl',row)
            self.last_stats=dict(stats,arm=arm,batch=self.calls,epoch=int(self.trainer.epoch),
                actual_B=b,coefficient=weight,loss_unweighted=float(kd.detach()),weighted_kd_total=float(kd.detach())*b*weight,
                native_total=float(native.detach().sum()),total_loss=float(total.detach()))
            if self.calls<=3 or self.calls%16==0:
                legacy.append_json(self.trainer.save_dir/'kd_batches.jsonl',self.last_stats)
            return total,items
    return DirectionCriterion
