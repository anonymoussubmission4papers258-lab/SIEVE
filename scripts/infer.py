"""Portable observation-only inference and optional, separate scoring."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sieve3.config import from_dict
from sieve3.model import CoupledNet
from sieve3.pipeline import Fusion
from sieve3 import audit


def read_checkpoint(path):
    # Only NumPy's RNG-state containers supplement PyTorch's restricted loader.
    core = getattr(np, '_core', np.core)
    safe = [core.multiarray._reconstruct,
            (core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
            np.ndarray, np.dtype, type(np.dtype('uint32'))]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location='cpu', weights_only=True)


def load_model(path, device):
    state = read_checkpoint(path)
    cfg = from_dict(state['config'])
    net = CoupledNet(cfg.model, vit_config=state.get('vit_config'))
    net.load_state_dict(state['model'], strict=True)
    net.eval().requires_grad_(False).to(device)
    return Fusion(net, cfg)


def tensor(a, device):
    return torch.as_tensor(np.ascontiguousarray(a.transpose(2, 0, 1)),
                           dtype=torch.float32, device=device)[None]


def predict(fusion, input_path, device, scale, sigma, tile=256, mode='default'):
    with np.load(input_path, allow_pickle=False) as z:
        h, r = z['h'].copy(), z['r'].copy()
        v = z['valid'].copy().astype(bool)
        srf = z['srf'].copy() if 'srf' in z else None
    assert h.ndim == r.ndim == 3 and r.shape[2] == 3
    assert v.shape == r.shape[:2] and h.shape[:2] == (r.shape[0]//scale, r.shape[1]//scale)
    assert np.isfinite(h).all() and np.isfinite(r).all() and v.any()
    if srf is not None: assert srf.shape == (3, h.shape[2]) and np.isfinite(srf).all()
    # audit.run builds only temporary observation-dependent fitting state.
    result = audit.run(fusion, tensor(r,device), tensor(h,device),
                       torch.as_tensor(v,dtype=torch.float32,device=device)[None,None],
                       scale, sigma,
                       srf=None if srf is None else torch.as_tensor(srf,dtype=torch.float32,device=device),
                       mode=mode, tile=tile, amp=False, out_device=torch.device('cpu'))
    prediction = result['x'][0].detach().float().permute(1,2,0).contiguous().numpy()
    assert np.isfinite(prediction).all()
    return prediction, h, v


def score(prediction, h, valid, target, scale, metric):
    # The reference is opened only after prediction has returned.
    with np.load(target, allow_pickle=False) as z:
        x, target_valid = z['x'].copy(), z['valid'].copy().astype(bool)
    assert np.array_equal(valid,target_valid) and x.shape == prediction.shape
    if metric == 'legacy32':
        from metrics.legacy32 import spectral_metrics
        result,_ = spectral_metrics(prediction,x,valid,scale=scale,h=h)
    else:
        from metrics.native import reconstruction_metrics
        result,_ = reconstruction_metrics(prediction,x,valid,scale=scale)
    return result


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if hasattr(v,'item'):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',help='NPZ with HWC h/r, HW valid, and optional 3xB srf')
    p.add_argument('--reference',help='optional NPZ with HWC x and HW valid; scoring only')
    p.add_argument('--manifest',help='portable case manifest, e.g. protocols/matched79.json')
    p.add_argument('--data-root',default='data')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--scale',type=int,default=4)
    p.add_argument('--sigma',type=float,default=.01)
    p.add_argument('--tile',type=int,default=256)
    p.add_argument('--mode',choices=['default','fast','strict'],default='default')
    p.add_argument('--metric',choices=['legacy32','native'],default='native')
    args=p.parse_args()
    if bool(args.input)==bool(args.manifest):p.error('choose exactly one of --input and --manifest')
    out=Path(args.output)
    if out.exists():p.error('output already exists; choose a new directory')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    fusion=load_model(args.checkpoint,args.device)
    if args.manifest:
        manifest=json.loads(Path(args.manifest).read_text())
        cases=manifest['cases'];metric=manifest.get('metric',args.metric)
    else:
        cases=[dict(case_id='single',input={'path':args.input},reference={'path':args.reference} if args.reference else None,scale=args.scale,sigma=args.sigma)]
        metric=args.metric
    rows=[]
    for case in cases:
        def path(key):
            row=case.get(key)
            if not row:return None
            q=Path(row['path']);q=q if q.is_absolute() or not args.manifest else Path(args.data_root)/q
            if row.get('sha256'):
                digest=hashlib.sha256(q.read_bytes()).hexdigest()
                if digest!=row['sha256']:raise ValueError('manifest input SHA256 mismatch: '+case['case_id'])
            return q
        inp=path('input')
        prediction,h,valid=predict(fusion,inp,args.device,case['scale'],case['sigma'],args.tile,args.mode)
        folder=out/case['case_id'];folder.mkdir(parents=True)
        np.save(folder/'prediction.npy',prediction,allow_pickle=False)
        target=path('reference')
        metrics=score(prediction,h,valid,target,case['scale'],metric) if target else None
        row=dict(case_id=case['case_id'],shape=list(prediction.shape),metric=metric,metrics=metrics)
        (folder/'result.json').write_text(json.dumps(clean(row),indent=2)+'\n')
        rows.append(row);print(case['case_id'],flush=True)
    (out/'results.json').write_text(json.dumps(clean(rows),indent=2)+'\n')


if __name__ == '__main__':main()
