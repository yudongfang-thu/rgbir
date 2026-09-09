"""New native-only execution path, preserving paired loader and student setup.

No production module changes on import. Auxiliary models remain initialized by
the common trainer but are not forwarded. This is a separately identified N
implementation; production trajectory equivalence is not presumed.
"""
import time
import torch


def make_api(selection, classification):
    return None


def make_criterion_type(criterion_module, api=None):
    base=criterion_module.IndependentCriterion
    legacy=criterion_module.legacy
    def call(self,prediction,batch):
        if self.method_arm!='N':raise ValueError('Native-fast is only for N/weight0')
        native,items=self.native(prediction,batch)
        self.calls+=1
        total=native.sum()
        if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite native loss')
        # No teacher-derived quantities have been measured. Zero is the known
        # applied dose; omitted selection and evidence values remain null.
        self.last_stats=dict(arm='N',source='native_rgb_gt',batch=self.calls,
            epoch=int(self.trainer.epoch),actual_B=int(batch['img'].shape[0]),
            coefficient=0.,native_total=float(total.detach()),total_loss=float(total.detach()),
            loss_unweighted=None,weighted_kd_total=0.,base_count=None,
            selected_count=0,selection_computed=False,teacher_forward=False,
            reference_forward=False,optimizer_updates=self.trainer.real_updates,
            seconds=time.time()-self.trainer.wall_started,
            path='native_fast_v1_paired_loader_student_setup_unchanged')
        if self.calls==1:
            self.gradient_checks.append(dict(batch=1,scope='native_sum_identity_only',
                teacher_forward=False,reference_forward=False,
                note='No independent cross-implementation gradient measurement in this callback'))
        if self.calls<=3 or self.calls%self.cfg['log_every_batches']==0:
            self.last_stats['student_files']=list(batch['im_file'])
            legacy.append_json(self.trainer.save_dir/'kd_batches.jsonl',self.last_stats)
        return total,items
    return type('NativeFastCriterion',(base,),{'__call__':call})
