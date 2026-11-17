import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.metrics import confusion_matrix,cohen_kappa_score,roc_auc_score
from skimage.metrics import structural_similarity

def spectral_metrics(pred,ref,valid=None,scale=1,h=None):
    p=np.asarray(pred,dtype=np.float64);x=np.asarray(ref,dtype=np.float64)
    assert p.shape==x.shape and p.ndim==3 and np.isfinite(p).all()
    v=np.ones(p.shape[:2],bool) if valid is None else valid
    e=p-x;a=p[v];b=x[v];rmse=np.sqrt(np.mean(e[v]**2));mae=np.abs(e[v]).mean()
    norms=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);ok=norms>1e-10
    sam=np.full(v.sum(),np.nan);sam[ok]=np.degrees(np.arccos(np.clip((a[ok]*b[ok]).sum(1)/norms[ok],-1,1)))
    sammap=np.full(v.shape,np.nan);sammap[v]=sam
    # Spectral Pearson correlation over wavelengths, undefined flat spectra excluded.
    ac=a-a.mean(1,keepdims=True);bc=b-b.mean(1,keepdims=True)
    denom=np.linalg.norm(ac,axis=1)*np.linalg.norm(bc,axis=1);sel=denom>1e-12
    corr=((ac[sel]*bc[sel]).sum(1)/denom[sel]).mean() if sel.any() else np.nan
    ssims=[]
    for k in range(x.shape[-1]):
        _,sm=structural_similarity(x[:,:,k],p[:,:,k],data_range=1.,full=True)
        interior=v.copy();interior[:3]=False;interior[-3:]=False;interior[:,:3]=False;interior[:,-3:]=False
        ssims.append(sm[interior].mean())
    gradx=np.sqrt(sum(np.gradient(x,axis=d)**2 for d in (0,1)))
    gradp=np.sqrt(sum(np.gradient(p,axis=d)**2 for d in (0,1)))
    mag=gradx.mean(-1);edge=mag>=np.quantile(mag[v],.85);err=(e**2).mean(-1)
    region_rmse=lambda mask:float(np.sqrt(err[mask].mean())) if mask.any() else np.nan
    out={'rmse':float(rmse),'mae':float(mae),'psnr_db':float(-20*np.log10(max(rmse,1e-15))),'sam_deg':float(np.nanmean(sam)),'sam_valid_fraction':float(ok.mean()),'ssim_mean_band':float(np.mean(ssims)),'spectral_correlation':float(corr),'gradient_mae':float(np.abs(gradp-gradx)[v].mean()),'boundary_rmse':region_rmse(v&edge),'interior_rmse':region_rmse(v&~edge)}
    means=b.mean(0);bandrmse=np.sqrt((e[v]**2).mean(0));nonzero=np.abs(means)>1e-8
    out['ergas']=float(100/scale*np.sqrt(np.mean((bandrmse[nonzero]/means[nonzero])**2))) if scale>1 else np.nan
    if h is not None:
        hp=p.reshape(p.shape[0]//scale,scale,p.shape[1]//scale,scale,p.shape[2]).mean((1,3)) if scale>1 else p
        out['observation_rmse']=float(np.sqrt(np.mean((hp-h)**2)))
    return out,{'sam':sammap,'rmse':np.sqrt(err),'per_band_rmse':bandrmse}

def downstream_metrics(prob,labels,valid=None):
    v=np.ones(labels.shape,bool) if valid is None else valid
    y=labels[v].astype(int);p=prob[v];pred=p.argmax(-1);cm=confusion_matrix(y,pred,labels=np.arange(p.shape[-1])).astype(float)
    tp=np.diag(cm);support=cm.sum(1);count=cm.sum(0)
    recall=np.divide(tp,support,out=np.full(4,np.nan),where=support>0)
    precision=np.divide(tp,count,out=np.zeros(4),where=count>0)
    f1=np.divide(2*tp,support+count,out=np.zeros(4),where=(support+count)>0)
    iou=np.divide(tp,support+count-tp,out=np.full(4,np.nan),where=(support+count-tp)>0)
    conf=p.max(-1);acc=(pred==y).astype(float);ece=0
    for j in range(15):
        mask=(conf>=j/15)&(conf<(j+1)/15 if j<14 else conf<=1)
        if mask.any():ece+=mask.mean()*abs(conf[mask].mean()-acc[mask].mean())
    order=np.argsort(-conf);risks=1-np.cumsum(acc[order])/np.arange(1,len(y)+1)
    out={'oa':float(acc.mean()),'aa_balanced_accuracy':float(np.nanmean(recall)),'macro_f1':float(f1[support>0].mean()),'weighted_f1':float((f1*support).sum()/support.sum()),'macro_precision':float(precision[support>0].mean()),'macro_recall':float(np.nanmean(recall)),'kappa':float(cohen_kappa_score(y,pred)),'miou':float(np.nanmean(iou)),'dice_macro':float(f1[support>0].mean()),'ece_15':float(ece),'brier':float(((p-np.eye(4)[y])**2).sum(1).mean()),'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-9)).mean()),'aurc':float(risks.mean())}
    # Tolerance-one-pixel classwise boundary F1. Undefined classes remain NaN.
    bfs=[]
    def edge(m):return m & ~(__import__('scipy').ndimage.binary_erosion(m))
    fullpred=prob.argmax(-1)
    for c in range(4):
        et=edge((labels==c)&v);ep=edge((fullpred==c)&v)
        if not et.any():bfs.append(np.nan);continue
        pr=(ep&binary_dilation(et)).sum()/max(ep.sum(),1);re=(et&binary_dilation(ep)).sum()/et.sum()
        bfs.append(2*pr*re/max(pr+re,1e-15))
    out['boundary_f1_tol1']=float(np.nanmean(bfs))
    return out,cm.astype(int),[dict(class_id=c,support=int(support[c]),precision=float(precision[c]),recall=float(recall[c]),f1=float(f1[c]),iou=float(iou[c]),boundary_f1=float(bfs[c])) for c in range(4)]
