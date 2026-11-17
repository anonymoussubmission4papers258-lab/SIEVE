"""The existing independent HSI check, with a fit-only conditional proposal."""
import torch
import torch.nn.functional as F
from native_sieve_v2 import prepare,solve,up
from native_sieve_v3 import check_masks,prepare_residual,solve_residual
from native_sieve_v6 import calibrate_visual_prior


@torch.inference_mode()
def fold_proposal(h,r,feature,projected,model,srf,sigma,config,valid,fit,steps=100):
    prior=model.predict(h,r,projected,fit)
    local=None
    if config.get('local_ridge') is not None:
        prior,local=calibrate_visual_prior(h,r,prior,fit,config['local_ridge'],projected,config['local_vfm_components'])
    state=prepare_residual(h,r,feature,prior,srf,sigma,config['rank'],valid,fit)
    proposed,diagnostics=solve_residual(state,config,steps)
    return proposed,diagnostics,state,local


@torch.inference_mode()
def infer(h,r,feature,projected,model,srf,sigma,proposal_config,base_config,valid=None,steps=100):
    sf=r.shape[-1]//h.shape[-1];outputs=[];diagnostics=[]
    for fold in (0,10):
        check,fit=check_masks(h,valid,sf,fold)
        if not fit.any():
            b=prepare(h,r,None,None,sigma,base_config['rank'],valid)
            output,di=solve(b,{**base_config,'visual':'hsi'},steps)
            outputs.append(output);diagnostics.append(dict(fold=fold,acceptance=0.,no_fit_evidence=True,**di));continue
        proposed,pdiag,p,local=fold_proposal(h,r,feature,projected,model,srf,sigma,proposal_config,valid,fit,steps)
        b=prepare(h,r,None,None,sigma,base_config['rank'],valid,fit)
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
            rgb_sensor_compatible=p['accepted'],local_calibration=local,proposal_condition='Fitting HSI only; check observations excluded before interpolation'))
    return torch.stack(outputs).mean(0),dict(crossfit_folds=[0,10],heldout_fraction_per_fold=1/16,
        mean_acceptance=sum(q['acceptance'] for q in diagnostics)/len(diagnostics),folds=diagnostics,refit_after_check=False,
        evidence_scope='Conditional proposal and calibration use only fitting HSI; the unchanged acceptance rule checks observable LR agreement')
