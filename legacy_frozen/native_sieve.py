"""SIEVE native-band core: HSI subspace + proposed spatial metric + soft observations.

No target parameter. Optional known SRF is disabled for real RGB-SWIR data.
All bands and rank are dynamic; spectral covariance is estimated from HSI only.
"""
import torch
import torch.nn.functional as F

def up(x,shape):return F.interpolate(x,size=shape,mode='nearest')

def lap(x,wx,wy):
    dx=(x[...,1:]-x[...,:-1])*wx;dy=(x[...,1:,:]-x[...,:-1,:])*wy
    return F.pad(dx,(0,1))-F.pad(dx,(1,0))+F.pad(dy,(0,0,0,1))-F.pad(dy,(0,0,1,0))

@torch.inference_mode()
def prepare(h,r,feature,srf,sigma,rank,valid=None,fit_mask=None):
    channels=h.shape[1];sf=r.shape[-1]//h.shape[-1];shape=r.shape[-2:]
    assert h.shape[0]==1 and shape[0]==h.shape[-2]*sf
    valid=torch.ones_like(r[:,:1]) if valid is None else valid.float()
    mask=(F.avg_pool2d(valid,sf)>.999).float()
    if fit_mask is not None:mask=mask*fit_mask
    obs=h[0].flatten(1);chosen=mask.flatten().bool()
    if int(chosen.sum())<max(8,rank):raise ValueError('Insufficient observed HSI evidence')
    fit=obs[:,chosen];mu=fit.mean(1,keepdim=True);center=fit-mu
    covariance=center@center.T/max(center.shape[1]-1,1)
    eig,vectors=torch.linalg.eigh(covariance)
    rank=min(rank,channels);basis=vectors[:,-rank:];signal=(eig[-rank:]-sigma**2).clamp_min(0)
    z=(basis.T@(obs-mu)).reshape(1,rank,*h.shape[-2:])
    # Mask BEFORE filling; held-out HSI values cannot enter initialization.
    known=z*mask
    denominator=F.avg_pool2d(F.pad(mask,(2,2,2,2),mode='replicate'),5,stride=1)
    numerator=F.avg_pool2d(F.pad(known,(2,2,2,2),mode='replicate'),5,stride=1)
    initial=torch.where(mask.bool(),known,numerator/denominator.clamp_min(1e-8))
    shrink=signal/(signal+sigma**2+1e-12)
    initial=initial*shrink.reshape(1,rank,1,1)
    rgbscale=r.std((-2,-1),keepdim=True).clamp_min(.02);rz=r/rgbscale
    rx=(rz[...,1:]-rz[...,:-1]).square().mean(1,keepdim=True);ry=(rz[...,1:,:]-rz[...,:-1,:]).square().mean(1,keepdim=True)
    if feature is None:vx=torch.zeros_like(rx);vy=torch.zeros_like(ry)
    else:
        f=F.normalize(F.interpolate(feature.float(),size=shape,mode='bilinear',align_corners=False),dim=1)
        vx=(f[...,1:]-f[...,:-1]).square().sum(1,keepdim=True);vy=(f[...,1:,:]-f[...,:-1,:]).square().sum(1,keepdim=True)
        energy=(vx.sum()+vy.sum())/(vx.numel()+vy.numel());vx/=energy.clamp_min(1e-5);vy/=energy.clamp_min(1e-5)
    g=rc=None;qmean=None;accepted=True
    if srf is not None:
        s=torch.as_tensor(srf,device=h.device,dtype=h.dtype);g=s@basis;rc=r-(s@mu).reshape(1,3,1,1)
        delta=F.avg_pool2d(r,sf)-torch.einsum('kc,bchw->bkhw',s,h)
        covariance_rgb=sigma**2*(s@s.T)+1e-10*torch.eye(3,device=h.device)
        q=(delta*torch.einsum('ij,bjhw->bihw',torch.linalg.inv(covariance_rgb),delta)).sum(1,keepdim=True)
        qmean=float((q*mask).sum()/mask.sum());accepted=qmean<=9
    return dict(h=h,r=r,basis=basis,mu=mu,signal=signal,z=z,initial=initial,mask=mask,sf=sf,sigma=sigma,
                rx=rx,ry=ry,vx=vx,vy=vy,g=g,rc=rc,qmean=qmean,accepted=accepted)

@torch.inference_mode()
def solve(p,config,steps=100):
    shape=p['r'].shape[-2:];sf=p['sf'];basis=p['basis'];rank=basis.shape[1];c=basis.shape[0]
    use=config.get('visual','vfm');vweight=config.get('vfm_edge',.3) if use=='vfm' else 0
    rweight=config.get('rgb_edge',4.) if use!='hsi' else 0
    wx=torch.exp(-rweight*p['rx']-vweight*p['vx']).clamp_min(.005)
    wy=torch.exp(-rweight*p['ry']-vweight*p['vy']).clamp_min(.005)
    strength=float(config['smooth']);anchor=float(config.get('anchor',.01))
    precision=float(config.get('rgb',1.)) if p['g'] is not None and use!='hsi' else 0
    lowup=up(p['z']*p['mask'],shape);initial=up(p['initial'],shape)
    # Noise-aware shrinkage in the prior rather than a blind rank cutoff alone.
    spectral=float(config.get('spectral',0.))*p['sigma']**2/(p['signal']+p['sigma']**2+1e-12)
    spectral=spectral.reshape(1,rank,1,1)+1e-6
    rhs=lowup+anchor*initial
    if precision:rhs=rhs+precision*torch.einsum('ik,bihw->bkhw',p['g'],p['rc'])
    def op(x):
        out=up(F.avg_pool2d(x,sf)*p['mask'],shape)-strength*lap(x,wx,wy)+(anchor+spectral)*x
        if precision:out=out+precision*torch.einsum('ik,bihw->bkhw',p['g'],torch.einsum('ik,bkhw->bihw',p['g'],x))
        return out
    degree=F.pad(wx,(0,1))+F.pad(wx,(1,0))+F.pad(wy,(0,0,0,1))+F.pad(wy,(0,0,1,0))
    diagonal=up(p['mask'],shape)/(sf*sf)+strength*degree+anchor+spectral
    if precision:diagonal=diagonal+precision*p['g'].square().sum(0).reshape(1,rank,1,1)
    x=initial.clone();res=rhs-op(x);z=res/diagonal;direction=z.clone();rz=(res*z).sum()
    for it in range(steps):
        ad=op(direction);alpha=rz/(direction*ad).sum().clamp_min(1e-20);x+=alpha*direction;res-=alpha*ad
        z=res/diagonal;new=(res*z).sum();direction=z+(new/rz.clamp_min(1e-20))*direction;rz=new
    pred=torch.einsum('ck,bkhw->bchw',basis,x)+p['mu'].reshape(1,c,1,1)
    residual=float((rhs-op(x)).norm()/rhs.norm().clamp_min(1e-10))
    return pred,dict(relative_solver_residual=residual,compatibility=p['qmean'],rgb_accepted=p['accepted'],rank=rank)

@torch.inference_mode()
def infer(h,r,feature,srf,sigma,proposal_config,base_config,valid=None,steps=100):
    """Known-response compatibility or independent held-out checks for SWIR.

    Base parameters must be independently selected on validation. Unknown-response
    decisions compare cross-fitted predictions without refitting after acceptance.
    """
    if srf is not None:
        p=prepare(h,r,feature,srf,sigma,proposal_config['rank'],valid)
        if p['accepted']:return solve(p,proposal_config,steps)
        b=prepare(h,r,None,None,sigma,base_config['rank'],valid)
        out,diag=solve(b,{**base_config,'visual':'hsi'},steps)
        return out,{**diag,'rgb_accepted':False,'compatibility':p['qmean'],'fallback':True}
    yy,xx=torch.meshgrid(torch.arange(h.shape[-2],device=h.device),torch.arange(h.shape[-1],device=h.device),indexing='ij')
    outputs=[];fractions=[];sf=r.shape[-1]//h.shape[-1]
    validlow=torch.ones_like(h[:,:1]) if valid is None else (F.avg_pool2d(valid.float(),sf)>.999).float()
    for fold in (0,3):
        check=(((yy%2)*2+xx%2)==fold).float()[None,None]*validlow;fit=1-check
        p=prepare(h,r,feature,None,sigma,proposal_config['rank'],valid,fit)
        b=prepare(h,r,None,None,sigma,base_config['rank'],valid,fit)
        zp,_=solve(p,proposal_config,steps);zb,_=solve(b,{**base_config,'visual':'hsi'},steps)
        ph=F.avg_pool2d(zp,sf);bh=F.avg_pool2d(zb,sf)
        gain=((bh-h).square()-(ph-h).square()).sum(1,keepdim=True)*check
        # Conditional paired-loss SD for independent Gaussian check observations.
        power=(bh-ph).square().sum(1,keepdim=True)*check
        kernel=min(8,h.shape[-2],h.shape[-1]);area=kernel*kernel
        count=F.avg_pool2d(check,kernel,ceil_mode=True,divisor_override=1)
        total=F.avg_pool2d(gain,kernel,ceil_mode=True,divisor_override=1)
        sd=2*sigma*F.avg_pool2d(power,kernel,ceil_mode=True,divisor_override=1).clamp_min(0).sqrt()
        accepted=(total>float(proposal_config.get('check_z',2.326))*sd)&(count>=8)
        a=up(accepted.float(),r.shape[-2:]);outputs.append(a*zp+(1-a)*zb);fractions.append(float(a.mean()))
    return torch.stack(outputs).mean(0),dict(sensor_model='unknown/nonoverlapping',crossfit_folds=[0,3],mean_acceptance=sum(fractions)/len(fractions),refit_after_check=False)
