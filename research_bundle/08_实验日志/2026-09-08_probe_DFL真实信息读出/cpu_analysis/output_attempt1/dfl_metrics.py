"""Distribution descriptors and unclamped, own-anchor adjacent-bin DFL diagnostic."""
import numpy as np


def distribution(logits):
    x=np.asarray(logits,dtype=np.float64)
    if x.shape!=(4,16) or not np.isfinite(x).all():raise ValueError('Finite 4x16 logits required')
    shifted=x-x.max(axis=1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
    p=np.exp(logp);bins=np.arange(16,dtype=np.float64)
    mean=(p*bins).sum(1)
    variance=(p*(bins[None,:]-mean[:,None])**2).sum(1)
    entropy=-(p*logp).sum(1)
    return dict(probabilities=p.tolist(),log_probabilities=logp.tolist(),expectation_bin=mean.tolist(),
                entropy_nat=entropy.tolist(),variance_bin2=variance.tolist())


def own_anchor_readout(desc,distances):
    target=np.asarray(distances,dtype=np.float64)
    if target.shape!=(4,):raise ValueError('Four own-GT LTRB distances required')
    logp=np.asarray(desc['log_probabilities']);mean=desc['expectation_bin']
    edges=[]
    for i,d in enumerate(target):
        reason='nonfinite_GT_distance' if not np.isfinite(d) else 'negative_GT_distance' if d<0 else 'GT_distance_ge_15' if d>=15 else None
        if reason:
            edges.append(dict(valid=False,reason=reason,DFL_CE_nat=None,expectation_minus_GT_bin=None,absolute_expectation_error_bin=None))
        else:
            lo=int(np.floor(d));hi=lo+1;wl=float(hi-d);wr=float(d-lo)
            ce=float(-wl*logp[i,lo]-wr*logp[i,hi]);error=float(mean[i]-d)
            edges.append(dict(valid=True,reason=None,DFL_CE_nat=ce,expectation_minus_GT_bin=error,
                              absolute_expectation_error_bin=abs(error),left_bin=lo,right_bin=hi,left_weight=wl,right_weight=wr))
    valid=all(e['valid'] for e in edges)
    return dict(edges=edges,all_four_edges_valid=valid,
                four_edge_mean_DFL_CE_nat=sum(e['DFL_CE_nat'] for e in edges)/4 if valid else None,
                four_edge_mean_absolute_expectation_error_bin=sum(e['absolute_expectation_error_bin'] for e in edges)/4 if valid else None,
                target_clamped=False,training_loss_reconstructed=False)
