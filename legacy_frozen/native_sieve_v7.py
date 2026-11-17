"""Spatially varying local spectral calibration, before the unchanged HSI check.

Overlapping local affine models use only fitting HSI. Their coefficients are
blended with fixed spatial weights. This is an established local-guidance design
choice, evaluated as a proposal change, not a new regression or new backbone.
"""
import torch
from native_sieve_v6 import calibrate_visual_prior as global_calibration


def starts(length,window,stride):
    positions=list(range(0,length-window+1,stride))
    if positions[-1]!=length-window:positions.append(length-window)
    return positions


@torch.inference_mode()
def calibrate_local_windows(h,r,prior,fit_mask,ridge,feature,components,window):
    height,width=r.shape[-2:];scale=width//h.shape[-1]
    global_prior,global_diagnostic=global_calibration(h,r,prior,fit_mask,ridge,feature,components)
    if window>=max(height,width):
        return global_prior,{**global_diagnostic,'local_window':window,'windows':1,'global_control':True}
    assert window%scale==0 and window>=scale*4 and window<=min(height,width)
    stride=window//2;assert stride%scale==0
    coordinate=torch.arange(window,device=r.device,dtype=r.dtype)
    taper=torch.minimum(coordinate+1,window-coordinate)/(window/2)
    taper=(taper[:,None]*taper[None,:])[None,None]
    accumulated=torch.zeros_like(prior);weight=torch.zeros_like(r[:,:1]);diagnostics=[];fallback=0
    for y in starts(height,window,stride):
        for x in starts(width,window,stride):
            sl=(slice(y//scale,(y+window)//scale),slice(x//scale,(x+window)//scale))
            mask=fit_mask[...,sl[0],sl[1]];count=int(mask.sum())
            if count<16:
                fitted=global_prior[...,y:y+window,x:x+window];fallback+=1
                diagnostic=dict(fitting_observations=count,global_fallback=True)
            else:
                fitted,diagnostic=global_calibration(h[...,sl[0],sl[1]],r[...,y:y+window,x:x+window],
                    prior[...,y:y+window,x:x+window],mask,ridge,
                    feature[...,y:y+window,x:x+window] if feature is not None else None,components)
            accumulated[...,y:y+window,x:x+window]+=fitted*taper
            weight[...,y:y+window,x:x+window]+=taper
            diagnostics.append(dict(y=y,x=x,**diagnostic))
    assert torch.all(weight>0)
    calibrated=accumulated/weight
    assert torch.isfinite(calibrated).all()
    return calibrated,dict(local_window=window,stride=stride,windows=len(diagnostics),global_fallback_windows=fallback,
        ridge=float(ridge),vfm_components=components,minimum_fitting_observations=16,
        fixed_taper='Separable positive triangular weights; no target-based spatial weights',
        check_observations_used=False,global_calibration=global_diagnostic,local_fits=diagnostics)


import torch.nn.functional as F
from native_sieve_v2 import prepare,solve,up
from native_sieve_v3 import prepare_residual,solve_residual,check_masks

@torch.inference_mode()
def infer(h,r,feature,prior,srf,sigma,proposal_config,base_config,valid=None,steps=100,calibration_feature=None):
    sf=r.shape[-1]//h.shape[-1];outputs=[];diagnostics=[]
    for fold in (0,10):
        check,fit=check_masks(h,valid,sf,fold)
        if not fit.any():
            b=prepare(h,r,None,None,sigma,base_config['rank'],valid)
            output,di=solve(b,{**base_config,'visual':'hsi'},steps)
            outputs.append(output);diagnostics.append(dict(fold=fold,acceptance=0.,no_fit_evidence=True,**di));continue
        calibrated,local=calibrate_local_windows(h,r,prior,fit,proposal_config['local_ridge'],calibration_feature,proposal_config['local_vfm_components'],proposal_config['local_window'])
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
