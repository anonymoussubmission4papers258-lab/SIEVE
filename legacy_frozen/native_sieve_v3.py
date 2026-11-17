"""SIEVE v3: RGB/VFM spectral proposals with HSI residual correction and checks.

The frozen predictor proposes a full spectrum, including directions not retained
by an HSI low-rank model. HSI fits a low-rank *correction*. A held-out comparison
checks observable spectral agreement, not unobservable HR null-space accuracy.
No function accepts a reference target. All proposal weights use training only.
"""
import torch
import torch.nn.functional as F
from native_sieve_v2 import prepare,solve,up


@torch.inference_mode()
def prepare_residual(h,r,feature,prior,srf,sigma,rank,valid=None,fit_mask=None):
    assert prior.shape==(h.shape[0],h.shape[1],r.shape[-2],r.shape[-1])
    p=prepare(h,r,feature,srf,sigma,rank,valid,fit_mask)
    residual=h-F.avg_pool2d(prior,p['sf'])
    fit=residual[0].flatten(1)[:,p['mask'].flatten().bool()]
    # Uncentered second moment includes a systematic spectral bias of the RGB
    # proposal. Its eigenspace and every correction statistic exclude check HSI.
    moment=fit@fit.T/fit.shape[1]
    eigen,basis=torch.linalg.eigh(moment);rank=min(rank,h.shape[1],max(1,fit.shape[1]))
    basis=basis[:,-rank:];signal=(eigen[-rank:]-sigma*sigma).clamp_min(0)
    p.update(basis=basis,signal=signal,mu=torch.zeros(h.shape[1],1,device=h.device,dtype=h.dtype),
             z=torch.einsum('ck,bchw->bkhw',basis,residual),
             initial=torch.zeros(1,rank,*h.shape[-2:],device=h.device,dtype=h.dtype),prior=prior)
    if srf is not None:
        s=torch.as_tensor(srf,device=h.device,dtype=h.dtype)
        p['g']=s@basis;p['rc']=r-torch.einsum('kc,bchw->bkhw',s,prior)
    return p


@torch.inference_mode()
def solve_residual(p,config,steps=100):
    correction,diag=solve(p,config,steps)
    return p['prior']+correction,{**diag,'parameterization':'full-spectrum RGB proposal plus low-rank HSI residual'}


def check_masks(h,valid,scale,fold):
    yy,xx=torch.meshgrid(torch.arange(h.shape[-2],device=h.device),torch.arange(h.shape[-1],device=h.device),indexing='ij')
    low=torch.ones_like(h[:,:1]) if valid is None else (F.avg_pool2d(valid.float(),scale)>.999).float()
    check=(((yy%4)*4+xx%4)==fold).float()[None,None]*low
    return check,low-check


@torch.inference_mode()
def infer(h,r,feature,prior,srf,sigma,proposal_config,base_config,valid=None,steps=100):
    """Two fixed 1/16 checkerboard folds; paired Gaussian-noise evidence test.

    Predictions entering each decision exclude its check observations. Accepted
    outputs are not refitted. The 2.326 threshold is a per-block conditional
    decision rule; it is not a dataset-level multiple-testing guarantee.
    """
    sf=r.shape[-1]//h.shape[-1];outputs=[];diagnostics=[]
    for fold in (0,10):
        check,fit=check_masks(h,valid,sf,fold)
        if not fit.any():
            b=prepare(h,r,None,None,sigma,base_config['rank'],valid)
            output,di=solve(b,{**base_config,'visual':'hsi'},steps)
            outputs.append(output);diagnostics.append(dict(fold=fold,acceptance=0.,no_fit_evidence=True,**di));continue
        p=prepare_residual(h,r,feature,prior,srf,sigma,proposal_config['rank'],valid,fit)
        b=prepare(h,r,None,None,sigma,base_config['rank'],valid,fit)
        proposed,pdiag=solve_residual(p,proposal_config,steps)
        baseline,bdiag=solve(b,{**base_config,'visual':'hsi'},steps)
        ph=F.avg_pool2d(proposed,sf);bh=F.avg_pool2d(baseline,sf)
        gain=((bh-h).square()-(ph-h).square()).sum(1,keepdim=True)*check
        power=(bh-ph).square().sum(1,keepdim=True)*check
        kernel=min(16,h.shape[-2],h.shape[-1])
        count=F.avg_pool2d(check,kernel,ceil_mode=True,divisor_override=1)
        total=F.avg_pool2d(gain,kernel,ceil_mode=True,divisor_override=1)
        sd=2*sigma*F.avg_pool2d(power,kernel,ceil_mode=True,divisor_override=1).clamp_min(0).sqrt()
        accepted=(total>float(proposal_config.get('check_z',2.326))*sd)&(count>=8)
        if not p['accepted']:accepted.zero_()
        a=up(accepted.float(),r.shape[-2:]);outputs.append(a*proposed+(1-a)*baseline)
        weight=torch.ones_like(a) if valid is None else valid
        diagnostics.append(dict(fold=fold,acceptance=float((a*weight).sum()/weight.sum().clamp_min(1)),
             check_observations=int(check.sum()),eligible_blocks=int((count>=8).sum()),accepted_blocks=int(accepted.sum()),
             max_relative_solver_residual=max(pdiag['relative_solver_residual'],bdiag['relative_solver_residual']),
             rgb_sensor_compatible=p['accepted']))
    return torch.stack(outputs).mean(0),dict(crossfit_folds=[0,10],heldout_fraction_per_fold=1/16,
             mean_acceptance=sum(q['acceptance'] for q in diagnostics)/len(diagnostics),folds=diagnostics,
             refit_after_check=False,evidence_scope='agreement in observed LR spectral space')
