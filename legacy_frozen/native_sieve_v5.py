"""HSI-calibrated visual proposals with independent HSI acceptance checks.

The new component is a local affine RGB residual fit, a classical linear-model
construction. It is not claimed as a novel regressor. Each fit excludes the HSI
observations later used to decide whether its proposal should be adopted.
"""
import torch
import torch.nn.functional as F
from native_sieve_v2 import prepare,solve,up
from native_sieve_v3 import prepare_residual,solve_residual,check_masks


@torch.inference_mode()
def calibrate_visual_prior(h,r,prior,fit_mask,ridge):
    sf=r.shape[-1]//h.shape[-1]
    assert h.shape[0]==1 and prior.shape==(1,h.shape[1],*r.shape[-2:])
    chosen=fit_mask.flatten().bool();count=int(chosen.sum())
    if not count:raise ValueError('No fitting HSI for local visual calibration')
    low_rgb=F.avg_pool2d(r,sf)[0].flatten(1).T
    residual=(h-F.avg_pool2d(prior,sf))[0].flatten(1).T
    # Double-precision3x3 normal equations make the diagnostic closed form
    # stable without changing the FP32 spectral inference and stored outputs.
    x=low_rgb[chosen].double();y=residual[chosen].double()
    center=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(.02)
    x=(x-center)/scale;offset=y.mean(0);centered=y-offset
    gram=x.T@x/count;rhs=x.T@centered/count
    system=gram+float(ridge)*torch.eye(3,device=x.device,dtype=x.dtype)
    weights=torch.linalg.solve(system,rhs)
    high=(r[0].permute(1,2,0).double()-center)/scale
    correction=(high@weights+offset).permute(2,0,1)[None].to(h.dtype)
    result=prior+correction
    assert torch.isfinite(result).all()
    return result,dict(fitting_observations=count,ridge=float(ridge),features='RGB affine,3slopes plus an unpenalized intercept',
                       condition_number=float(torch.linalg.cond(system)),
                       correction_rms=float(correction.square().mean().sqrt()),check_observations_used=False)


@torch.inference_mode()
def infer(h,r,feature,prior,srf,sigma,proposal_config,base_config,valid=None,steps=100):
    sf=r.shape[-1]//h.shape[-1];outputs=[];diagnostics=[]
    for fold in (0,10):
        check,fit=check_masks(h,valid,sf,fold)
        if not fit.any():
            b=prepare(h,r,None,None,sigma,base_config['rank'],valid)
            output,di=solve(b,{**base_config,'visual':'hsi'},steps)
            outputs.append(output);diagnostics.append(dict(fold=fold,acceptance=0.,no_fit_evidence=True,**di));continue
        calibrated,local=calibrate_visual_prior(h,r,prior,fit,proposal_config['local_ridge'])
        p=prepare_residual(h,r,feature,calibrated,srf,sigma,proposal_config['rank'],valid,fit)
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
            rgb_sensor_compatible=p['accepted'],local_calibration=local))
    return torch.stack(outputs).mean(0),dict(crossfit_folds=[0,10],heldout_fraction_per_fold=1/16,
        mean_acceptance=sum(q['acceptance'] for q in diagnostics)/len(diagnostics),folds=diagnostics,refit_after_check=False,
        evidence_scope='Local spectral/color relationships are priors estimated from fitting HSI; independent held-out HSI checks LR agreement only')
