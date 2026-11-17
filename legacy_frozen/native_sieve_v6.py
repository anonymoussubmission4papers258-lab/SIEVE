"""HSI calibrates fixed RGB/VFM features before an independent acceptance check.

The local design extends the RGB affine calibration with training-PCA VFM
coordinates and their RGB interactions. Regression is established prior art;
only fitting HSI determines its coefficients. No new backbone is introduced.
"""
import torch
import torch.nn.functional as F
from native_sieve_v2 import prepare,solve,up
from native_sieve_v3 import prepare_residual,solve_residual,check_masks


@torch.inference_mode()
def calibrate_visual_prior(h,r,prior,fit_mask,ridge,feature,components):
    sf=r.shape[-1]//h.shape[-1]
    assert h.shape[0]==1 and prior.shape==(1,h.shape[1],*r.shape[-2:])
    chosen=fit_mask.flatten().bool();count=int(chosen.sum())
    if not count:raise ValueError('No fitting HSI for local visual calibration')
    if components:
        assert feature is not None and feature.shape[1]>=components
        f=feature[:,-components:]
        assert f.shape[-2:]==r.shape[-2:]
        interaction=(f[:,:,None]*r[:,None]).flatten(1,2)
        design=torch.cat([r,f,interaction],1)
    else:design=r
    low_rgb=F.avg_pool2d(design,sf)[0].flatten(1).T
    residual=(h-F.avg_pool2d(prior,sf))[0].flatten(1).T
    # Double-precision normal equations make the diagnostic closed form
    # stable without changing the FP32 spectral inference and stored outputs.
    x=low_rgb[chosen].double();y=residual[chosen].double()
    center=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(.02)
    x=(x-center)/scale;offset=y.mean(0);centered=y-offset
    gram=x.T@x/count;rhs=x.T@centered/count
    system=gram+float(ridge)*torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
    weights=torch.linalg.solve(system,rhs)
    high=(design[0].permute(1,2,0).double()-center)/scale
    correction=(high@weights+offset).permute(2,0,1)[None].to(h.dtype)
    result=prior+correction
    assert torch.isfinite(result).all()
    return result,dict(fitting_observations=count,ridge=float(ridge),features='RGB, training-PCA VFM, RGBxVFM with an unpenalized intercept',
                       feature_dimensions=x.shape[1],vfm_components=components,
                       condition_number=float(torch.linalg.cond(system)),
                       correction_rms=float(correction.square().mean().sqrt()),check_observations_used=False)


@torch.inference_mode()
def infer(h,r,feature,prior,srf,sigma,proposal_config,base_config,valid=None,steps=100,calibration_feature=None):
    sf=r.shape[-1]//h.shape[-1];outputs=[];diagnostics=[]
    for fold in (0,10):
        check,fit=check_masks(h,valid,sf,fold)
        if not fit.any():
            b=prepare(h,r,None,None,sigma,base_config['rank'],valid)
            output,di=solve(b,{**base_config,'visual':'hsi'},steps)
            outputs.append(output);diagnostics.append(dict(fold=fold,acceptance=0.,no_fit_evidence=True,**di));continue
        calibrated,local=calibrate_visual_prior(h,r,prior,fit,proposal_config['local_ridge'],calibration_feature,proposal_config['local_vfm_components'])
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
