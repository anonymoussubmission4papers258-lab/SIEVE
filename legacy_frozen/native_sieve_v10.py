"""Sensor-consistent RBF extension of local visual calibration.

The RBF expansion is standard regression, not a claimed new foundation model.
Its coefficients see fitting HSI only. The original held-out evidence check is
retained; kernel features are area-averaged after nonlinear expansion.
"""
import math
import torch
import torch.nn.functional as F
from native_sieve_v2 import prepare,solve,up
from native_sieve_v3 import prepare_residual,solve_residual,check_masks
from native_sieve_v6 import calibrate_visual_prior


def farthest_centers(x,limit=32):
    assert x.ndim==2 and len(x)>0
    selected=[int((x-x.mean(0)).square().sum(1).argmin())]
    distance=(x-x[selected[0]]).square().sum(1)
    while len(selected)<min(limit,len(x)):
        index=int(distance.argmax())
        if float(distance[index])<=1e-12:break
        selected.append(index);distance=torch.minimum(distance,(x-x[index]).square().sum(1))
    return x[selected],selected


@torch.inference_mode()
def calibrate_kernel_prior(h,r,prior,fit_mask,ridge,feature,components,bandwidth):
    if bandwidth is None:return calibrate_visual_prior(h,r,prior,fit_mask,ridge,feature,components)
    assert bandwidth in [.5,1.,2.] and ridge>0 and 0<=components<=feature.shape[1]
    sf=r.shape[-1]//h.shape[-1];height,width=r.shape[-2:]
    chosen=fit_mask.flatten().bool();count=int(chosen.sum());assert count>0
    f=feature[:,-components:] if components else feature[:,:0]
    assert f.shape[-2:]==r.shape[-2:] and h.shape[0]==1
    visual=torch.cat([r,f],1).double()
    low=F.avg_pool2d(visual,sf)[0].flatten(1).T
    center=low[chosen].mean(0);scale=low[chosen].std(0,unbiased=False).clamp_min(.02)
    # Give RGB and the PCA coordinates equal total feature weight.
    group=torch.tensor([1/math.sqrt(3)]*3+([1/math.sqrt(components)]*components if components else []),
        device=h.device,dtype=torch.float64)
    norm=(low[chosen]-center)/scale*group
    anchors,selected=farthest_centers(norm,32)
    distances=torch.pdist(anchors).square();positive=distances[distances>1e-12]
    median=positive.median() if len(positive) else torch.tensor(1.,device=h.device,dtype=torch.float64)
    high=(visual[0].permute(1,2,0).reshape(-1,visual.shape[1])-center)/scale*group
    d2=(high.square().sum(1,keepdim=True)+anchors.square().sum(1)[None]-2*high@anchors.T).clamp_min(0)
    kernel=torch.exp(-d2/(2*float(bandwidth)**2*median)).T.reshape(1,len(anchors),height,width)
    interaction=(f[:,:,None]*r[:,None]).flatten(1,2)
    design=torch.cat([r.double(),f.double(),interaction.double(),kernel],1)
    # The exact declared sensor acts on each HR basis function before fitting.
    low_design=F.avg_pool2d(design,sf)[0].flatten(1).T
    x=low_design[chosen];y=(h-F.avg_pool2d(prior,sf))[0].flatten(1).T[chosen].double()
    dc=x.mean(0);ds=x.std(0,unbiased=False).clamp_min(.02);x=(x-dc)/ds
    offset=y.mean(0);system=x.T@x/count+float(ridge)*torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
    weights=torch.linalg.solve(system,x.T@(y-offset)/count)
    normalized=(design[0].permute(1,2,0)-dc)/ds
    correction=(normalized@weights+offset).permute(2,0,1)[None]
    low_from_hr=F.avg_pool2d(correction,sf)[0].flatten(1).T
    low_from_design=(low_design-dc)/ds@weights+offset
    identity=float((low_from_hr-low_from_design).abs().max());assert identity<1e-10
    result=prior+correction.to(h.dtype);assert torch.isfinite(result).all()
    return result,dict(fitting_observations=count,ridge=float(ridge),vfm_components=components,
        feature_dimensions=int(design.shape[1]),rbf_centers=len(anchors),center_indices_in_fitting_set=selected,
        bandwidth_multiplier=float(bandwidth),center_squared_distance_median=float(median),
        condition_number=float(torch.linalg.cond(system)),correction_rms=float(correction.square().mean().sqrt()),
        sensor_basis_identity_max_abs=identity,check_observations_used=False,
        features='Unchanged RGB/PCA/interactions plus32deterministic RBF basis functions; sensor acts after nonlinear expansion')


@torch.inference_mode()
def infer(h,r,feature,prior,srf,sigma,proposal_config,base_config,valid=None,steps=100,calibration_feature=None,bandwidth=None):
    sf=r.shape[-1]//h.shape[-1];outputs=[];diagnostics=[]
    for fold in (0,10):
        check,fit=check_masks(h,valid,sf,fold)
        if not fit.any():
            b=prepare(h,r,None,None,sigma,base_config['rank'],valid)
            output,di=solve(b,{**base_config,'visual':'hsi'},steps)
            outputs.append(output);diagnostics.append(dict(fold=fold,acceptance=0.,no_fit_evidence=True,**di));continue
        calibrated,local=calibrate_kernel_prior(h,r,prior,fit,proposal_config['local_ridge'],calibration_feature,
            proposal_config['local_vfm_components'],bandwidth)
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
