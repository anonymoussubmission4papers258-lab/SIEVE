"""Versioned native-grid metrics, streaming bands/rows; no target used for fitting."""
import numpy as np
from scipy.ndimage import binary_erosion,gaussian_filter

VERSION='R6-native-v1'

def reconstruction_metrics(pred,reference,valid,scale=4,data_range=1.):
    if pred.shape!=reference.shape or pred.ndim!=3 or valid.shape!=pred.shape[:2]:raise ValueError('Mismatched arrays')
    height,width,channels=pred.shape;v=np.asarray(valid,bool);n=int(v.sum())
    if n==0:raise ValueError('No valid evaluation pixels')
    sq=np.zeros(channels,np.float64);absolute=np.zeros(channels,np.float64);mean=np.zeros(channels,np.float64)
    sam_sum=0.;sam_count=0;zero_pred_count=0;rmsemap=np.full((height,width),np.nan,np.float32);sammap=np.full_like(rmsemap,np.nan)
    for y in range(0,height,64):
        p=np.asarray(pred[y:y+64],np.float64);x=np.asarray(reference[y:y+64],np.float64);mask=v[y:y+64]
        if not np.isfinite(p[mask]).all() or not np.isfinite(x[mask]).all():raise ValueError('Nonfinite valid prediction/reference')
        a=p[mask];b=x[mask];err=a-b;sq+=np.square(err).sum(0);absolute+=np.abs(err).sum(0);mean+=b.sum(0)
        refnorm=np.linalg.norm(b,axis=1);pnorm=np.linalg.norm(a,axis=1);has_ref=refnorm>1e-12
        # A zero prediction against a nonzero target counts as90degrees, not an
        # omitted difficult pixel. Only undefined zero-reference spectra omit.
        cos=np.divide((a*b).sum(1),pnorm*refnorm,out=np.zeros(len(a)),where=(pnorm*refnorm)>1e-24)
        angles=np.degrees(np.arccos(np.clip(cos,-1,1)));sam_sum+=angles[has_ref].sum();sam_count+=int(has_ref.sum());zero_pred_count+=int(((pnorm<=1e-12)&has_ref).sum())
        mapview=rmsemap[y:y+64];mapview[mask]=np.sqrt(np.square(err).mean(1));mapview=sammap[y:y+64]
        values=np.full(len(a),np.nan);values[has_ref]=angles[has_ref];mapview[mask]=values
    mse=sq/n;means=mean/n;global_mse=float(mse.mean())
    band_psnr=np.full(channels,np.inf);positive=mse>0;band_psnr[positive]=10*np.log10(data_range**2/mse[positive])
    nonzero=np.abs(means)>1e-8
    ergas=float(100/scale*np.sqrt(np.mean(mse[nonzero]/means[nonzero]**2))) if nonzero.any() else None
    # Gaussian SSIM exactly matches skimage's population-covariance convention.
    # Every11x11window must have valid sensor support; boundary pixels excluded.
    interior=binary_erosion(v,structure=np.ones((11,11),bool),border_value=0)
    ssim=[];c1=(.01*data_range)**2;c2=(.03*data_range)**2
    if interior.any():
        for k in range(channels):
            a=np.asarray(pred[:,:,k],np.float64);b=np.asarray(reference[:,:,k],np.float64)
            ux=gaussian_filter(a,1.5,truncate=3.5);uy=gaussian_filter(b,1.5,truncate=3.5)
            vx=gaussian_filter(a*a,1.5,truncate=3.5)-ux*ux;vy=gaussian_filter(b*b,1.5,truncate=3.5)-uy*uy
            cov=gaussian_filter(a*b,1.5,truncate=3.5)-ux*uy
            sm=((2*ux*uy+c1)*(2*cov+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2));ssim.append(float(sm[interior].mean()))
    metrics=dict(metric_version=VERSION,valid_pixels=n,bands=channels,rmse=float(np.sqrt(global_mse)),mse=global_mse,mae=float((absolute/n).mean()),
        psnr_global_db=float(10*np.log10(data_range**2/global_mse)) if global_mse>0 else None,
        psnr_mean_band_db=float(band_psnr.mean()) if np.isfinite(band_psnr).all() else None,perfect_reconstruction=global_mse==0,
        sam_deg=float(sam_sum/sam_count) if sam_count else None,sam_defined_pixels=sam_count,sam_zero_prediction_pixels=zero_pred_count,
        ssim_mean_band=float(np.mean(ssim)) if ssim else None,ssim_valid_windows=int(interior.sum()),ergas=ergas,ergas_defined_bands=int(nonzero.sum()),
        data_range=data_range,scale=scale,band_mse=mse.tolist(),band_reference_mean=means.tolist())
    return metrics,dict(rmse=rmsemap,sam=sammap)

def confusion_metrics(cm,include=None):
    cm=np.asarray(cm,np.int64)
    if cm.ndim!=2 or cm.shape[0]!=cm.shape[1] or cm.sum()==0:raise ValueError('Invalid confusion matrix')
    tp=np.diag(cm).astype(float);support=cm.sum(1);pred=cm.sum(0);union=support+pred-tp
    iou=np.divide(tp,union,out=np.zeros(len(tp)),where=union>0);f1=np.divide(2*tp,support+pred,out=np.zeros(len(tp)),where=(support+pred)>0)
    recall=np.divide(tp,support,out=np.zeros(len(tp)),where=support>0);include=np.arange(len(tp)) if include is None else np.asarray(include)
    active=include[support[include]>0];union_active=include[union[include]>0]
    return dict(oa=float(tp.sum()/cm.sum()),miou_fixed=float(iou[include].mean()),macro_f1_fixed=float(f1[include].mean()),
                macro_recall_supported=float(recall[active].mean()) if len(active) else None,
                miou_union_present=float(iou[union_active].mean()) if len(union_active) else None,
                class_iou=iou.tolist(),class_f1=f1.tolist(),class_recall=recall.tolist(),class_support=support.tolist(),prediction_count=pred.tolist(),
                evaluated_classes=include.tolist(),absent_reference_classes=[int(i) for i in include if support[i]==0],confusion=cm.tolist())
