"""Full-spectrum, image-regularized SIEVE residual inference.

v3 regularized only a truncated residual, leaving proposal directions outside
that correction space unchanged. Here every spectral direction can be corrected,
and RGB/VFM edge weights regularize the final reconstructed image P + E Z.
The prior, HSI checks and two cross-fit masks are otherwise unchanged.
"""
import torch
import torch.nn.functional as F
from native_sieve_v2 import prepare,solve,up,lap
from native_sieve_v3 import check_masks


@torch.inference_mode()
def prepare_full(h,r,feature,prior,srf,sigma,valid=None,fit_mask=None):
    p=prepare(h,r,feature,srf,sigma,h.shape[1],valid,fit_mask)
    residual=h-F.avg_pool2d(prior,p['sf'])
    fit=residual[0].flatten(1)[:,p['mask'].flatten().bool()]
    moment=fit.double()@fit.double().T/fit.shape[1]
    magnitude=moment.abs().max().clamp_min(torch.finfo(torch.float64).tiny)
    eigen,basis=torch.linalg.eigh(moment/magnitude)
    eigen=(eigen*magnitude).to(h.dtype);basis=basis.to(h.dtype)
    if not torch.isfinite(basis).all():raise FloatingPointError('Invalid full spectral covariance basis')
    # Keep the complete orthonormal basis, including directions with no inferred
    # residual signal. Continuous precision regularizes them instead of making
    # their RGB/VFM-proposed high-frequency values immutable.
    signal=(eigen-sigma*sigma).clamp_min(0)
    p.update(basis=basis,signal=signal,mu=torch.zeros(h.shape[1],1,device=h.device,dtype=h.dtype),
             z=torch.einsum('ck,bchw->bkhw',basis,residual),
             initial=torch.zeros_like(h),prior=prior)
    if srf is not None:
        s=torch.as_tensor(srf,device=h.device,dtype=h.dtype)
        p['g']=s@basis;p['rc']=r-torch.einsum('kc,bchw->bkhw',s,prior)
    return p


def system(p,config):
    """Return an SPD normal operator and RHS of the declared quadratic."""
    shape=p['r'].shape[-2:];sf=p['sf'];basis=p['basis'];channels=basis.shape[0]
    visual=config.get('visual','vfm');vw=config.get('vfm_edge',.3) if visual=='vfm' else 0.
    rw=config.get('rgb_edge',4.) if visual!='hsi' else 0.
    wx=torch.exp(-rw*p['rx']-vw*p['vx']).clamp_min(.005)
    wy=torch.exp(-rw*p['ry']-vw*p['vy']).clamp_min(.005)
    gamma=float(config['smooth']);anchor=float(config['anchor'])
    rgb=float(config['rgb']) if p['g'] is not None and visual!='hsi' else 0.
    spectral=float(config.get('spectral',.01))*p['sigma']**2/(p['signal']+p['sigma']**2+1e-12)
    precision=spectral.reshape(1,channels,1,1)+anchor+1e-6
    rhs=up(p['z']*p['mask'],shape)+gamma*torch.einsum('ck,bchw->bkhw',basis,lap(p['prior'],wx,wy))
    if rgb:rhs=rhs+rgb*torch.einsum('ik,bihw->bkhw',p['g'],p['rc'])
    def op(z):
        out=up(F.avg_pool2d(z,sf)*p['mask'],shape)-gamma*lap(z,wx,wy)+precision*z
        if rgb:out=out+rgb*torch.einsum('ik,bihw->bkhw',p['g'],torch.einsum('ik,bkhw->bihw',p['g'],z))
        return out
    degree=F.pad(wx,(0,1))+F.pad(wx,(1,0))+F.pad(wy,(0,0,0,1))+F.pad(wy,(0,0,1,0))
    diagonal=up(p['mask'],shape)/(sf*sf)+gamma*degree+precision
    if rgb:diagonal=diagonal+rgb*p['g'].square().sum(0).reshape(1,channels,1,1)
    return op,rhs,diagonal,dict(wx=wx,wy=wy,gamma=gamma,precision=precision,rgb=rgb)


@torch.inference_mode()
def solve_full(p,config,steps=150):
    op,rhs,diagonal,_=system(p,config)
    z=torch.zeros_like(rhs);res=rhs.clone();pre=res/diagonal;direction=pre.clone();rz=(res*pre).sum()
    for _ in range(steps):
        ad=op(direction);alpha=rz/(direction*ad).sum().clamp_min(1e-20)
        z+=alpha*direction;res-=alpha*ad;pre=res/diagonal;next_rz=(res*pre).sum()
        direction=pre+(next_rz/rz.clamp_min(1e-20))*direction;rz=next_rz
    pred=p['prior']+torch.einsum('ck,bkhw->bchw',p['basis'],z)
    relative=float((rhs-op(z)).norm()/rhs.norm().clamp_min(1e-10))
    return pred,dict(relative_solver_residual=relative,rank=p['basis'].shape[1],
                     parameterization='full-spectrum residual, spatial regularity on final P+EZ')


@torch.inference_mode()
def infer(h,r,feature,prior,srf,sigma,proposal_config,base_config,valid=None,steps=150):
    sf=r.shape[-1]//h.shape[-1];outputs=[];diagnostics=[]
    for fold in (0,10):
        check,fit=check_masks(h,valid,sf,fold)
        if not fit.any():
            b=prepare(h,r,None,None,sigma,base_config['rank'],valid)
            output,di=solve(b,{**base_config,'visual':'hsi'},100)
            outputs.append(output);diagnostics.append(dict(fold=fold,acceptance=0.,no_fit_evidence=True,**di));continue
        p=prepare_full(h,r,feature,prior,srf,sigma,valid,fit)
        b=prepare(h,r,None,None,sigma,base_config['rank'],valid,fit)
        proposed,pdiag=solve_full(p,proposal_config,steps)
        baseline,bdiag=solve(b,{**base_config,'visual':'hsi'},100)
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
             mean_acceptance=sum(d['acceptance'] for d in diagnostics)/len(diagnostics),folds=diagnostics,
             refit_after_check=False,evidence_scope='agreement in observed LR spectral space; image regularity is a prior, not an evidence certificate')
