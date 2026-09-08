"""Bounded independent proof of the retained final-loss graph; CPU only."""
from pathlib import Path
import weakref,json,math,sys
import torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
attempt=sys.argv[1] if len(sys.argv)>1 else 'attempt1'
suffix='' if attempt=='attempt1' else '_'+attempt
def two_batches(clear_final_loss):
    p=torch.nn.Parameter(torch.tensor([.1,.2,.3]))
    references=[];live_at_next_forward=[];rows=[]
    class TrackedSin(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):
            ctx.save_for_backward(x)
            references.append(weakref.ref(ctx))
            return x.sin()
        @staticmethod
        def backward(ctx,grad):
            (x,)=ctx.saved_tensors
            return grad*x.cos()
    for batch_index in range(2):
        if references:live_at_next_forward.append(references[-1]() is not None)
        activation=p.repeat(13,1)*1.01
        prediction=TrackedSin.apply(activation)
        losses={'DFL':prediction.square().mean(),'GT':(prediction-.5).square().mean()}
        grads={};norms={}
        for arm,loss in losses.items():
            grads[arm]=torch.autograd.grad(loss,[p],retain_graph=True)
            norms[arm]=float(grads[arm][0].double().norm())
        rows.append({'batch':batch_index+1,'norms':norms})
        del activation,prediction,losses,grads
        if clear_final_loss:del loss
    still_live_at_loop_end=references[-1]() is not None
    return rows,live_at_next_forward,still_live_at_loop_end
old=two_batches(False);fixed=two_batches(True)
result={'status':'PASS' if old[0]==fixed[0] and old[1]==[True] and fixed[1]==[False] and old[2] and not fixed[2] else 'FAIL',
 'actual_old_cleanup_retains_last_loss_graph':old[1]==[True],
 'explicit_loss_deletion_releases_graph_before_next_forward':fixed[1]==[False],
 'same_gradient_norms_exact':old[0]==fixed[0],'gradient_norms':old[0],
 'last_autograd_context_live_after_legacy_cleanup':old[2],'last_autograd_context_live_after_fixed_cleanup':fixed[2],
 'measurement':'Weak references track custom autograd context owning saved tensors. Python Tensor-wrapper weakrefs cannot establish C++ saved-storage lifetime.',
 'scope':'Synthetic autograd graph-lifetime mechanism, not a measurement of production VRAM or complete proof of its magnitude.',
 'torch_version':torch.__version__,'device':'cpu','new_GPU':False,'new_hash_computed':False}
with (HERE/('CALIBRATION_LIFETIME_INDEPENDENT_CPU'+suffix+'.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
raise SystemExit(result['status']!='PASS')
