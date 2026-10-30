"""No-Rasch evidence-adaptive smoothing plus support-conditioned residuals."""
import numpy as np
from scipy.special import betaln,logsumexp

BUDGETS=np.array([0,1,3,7,15,31])
KERNEL_WEIGHTS=np.array([0,0,.2,.4,1,1])


def probabilities(parts, config, ablation=None):
    n,s=parts['n'],parts['s']
    alpha=1. if ablation=='no_prior_shrink' else config['prior_shrink']
    center=config.get('prior_center',.5)
    q=np.clip(center+alpha*(parts['prior']-center),.03,.97)
    u=np.clip(center+alpha*(parts['support_prior']-center),.03,.97)
    if ablation=='neutral_metadata_prior':
        q=np.full_like(q,.5);u=np.full_like(u,.5)
    concentrations=np.asarray(config['concentrations'])
    masses=np.asarray(config['masses'])
    a=u[...,None]*concentrations;b=(1-u[...,None])*concentrations
    logits=np.broadcast_to(np.log(masses),a.shape).copy()
    if ablation!='no_evidence_weighting':
        logits+=betaln(a+s[...,None],b+n[...,None]-s[...,None])-betaln(a,b)
    weights=np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
    smoothed=np.sum(weights*(s[...,None]+concentrations*q[...,None])/(n[...,None]+concentrations),axis=-1)
    scale=0. if ablation=='no_kernel' else config['residual_scale']
    w=np.interp(n,BUDGETS,KERNEL_WEIGHTS)*scale
    if config['combine']=='rebase' and ablation!='original_kernel_blend':
        prediction=smoothed+w*parts['residual']
    else:
        kernel=np.clip(parts['kernel_base']+parts['residual'],0,1)
        prediction=(1-w)*smoothed+w*kernel
    decay=None if ablation=='no_topic_decay' else config['topic_decay']
    topic_weight=0. if ablation=='no_topic' else (1. if decay is None else decay/(decay+n))
    return np.clip(prediction+topic_weight*parts['topic'],0,1)


def ablations(config):
    names=['no_kernel','no_topic','neutral_metadata_prior']
    if config['combine']=='rebase':names.append('original_kernel_blend')
    if config['topic_decay'] is not None:names.append('no_topic_decay')
    if config['prior_shrink']!=1.:names.append('no_prior_shrink')
    if len(config['concentrations'])>1:names.append('no_evidence_weighting')
    return names
