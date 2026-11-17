"""Reconstruction metrics and the observability decomposition of the error."""
from __future__ import annotations


import torch
import torch.nn.functional as F

from .ops import block_valid, lift, pool


@torch.no_grad()
def reconstruction_metrics(pred: torch.Tensor, ref: torch.Tensor, valid: torch.Tensor, s: int) -> dict:
    """pred/ref [C,H,W] float tensors on the same device; valid [H,W] bool.

    RMSE, unit-peak PSNR, SAM (degrees, zero-norm pairs excluded), mean-band SSIM (7x7 uniform window,
    centres whose window is fully valid, 3-pixel border excluded), ERGAS (100/s).
    """
    pred, ref = pred.float(), ref.float()
    C = ref.shape[0]
    m = valid.float()
    n = m.sum().clamp_min(1.0)
    err = (pred - ref) * m
    mse = (err ** 2).sum() / (n * C)
    out = {"rmse": float(mse.sqrt()), "psnr": float(10 * torch.log10(1.0 / mse.clamp_min(1e-20)))}
    dot = (pred * ref).sum(0)
    npred, nref = pred.norm(dim=0), ref.norm(dim=0)
    ok = valid & (npred > 0) & (nref > 0)
    cos = (dot[ok] / (npred[ok] * nref[ok])).clamp(-1.0, 1.0)
    out["sam"] = float(torch.rad2deg(torch.acos(cos)).mean()) if cos.numel() else float("nan")
    rmse_b = ((err ** 2).sum((1, 2)) / n).sqrt()
    mean_b = (ref * m).sum((1, 2)) / n
    out["ergas"] = float(100.0 / s * ((rmse_b / mean_b.abs().clamp_min(1e-12)) ** 2).mean().sqrt())
    x, y = pred.unsqueeze(1), ref.unsqueeze(1)
    k = 7
    mu_x, mu_y = F.avg_pool2d(x, k, 1), F.avg_pool2d(y, k, 1)
    sxx = F.avg_pool2d(x * x, k, 1) - mu_x ** 2
    syy = F.avg_pool2d(y * y, k, 1) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, k, 1) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    window_ok = (F.avg_pool2d(m[None, None], k, 1) > 0.999).float()
    out["ssim"] = float(((ssim * window_ok).sum((1, 2, 3)) / window_ok.sum().clamp_min(1.0)).mean())
    return out


@torch.no_grad()
def observability(pred: torch.Tensor, ref: torch.Tensor, valid: torch.Tensor, s: int, srf) -> dict:
    """Energy fractions of P_A P_S, P_A (I-P_S), (I-P_A) P_S, (I-P_A)(I-P_S) of the error (fully valid blocks)."""
    if srf is None or s == 1:
        return {}
    S = torch.as_tensor(srf, dtype=torch.float32, device=pred.device)
    PS = S.T @ torch.linalg.solve(S @ S.T, S)
    blocks = lift(block_valid(valid.float()[None, None], s), s)[0]
    D = (pred.float() - ref.float()) * blocks
    PA = lift(pool(D[None], s), s)[0]
    ps = lambda X: torch.einsum("cd,dhw->chw", PS, X)
    comps = {"A_S": ps(PA), "A_null": PA - ps(PA), "null_S": ps(D) - ps(PA), "joint_null": D - PA - ps(D) + ps(PA)}
    total = (D ** 2).sum().clamp_min(1e-30)
    return {k: float((v ** 2).sum() / total) for k, v in comps.items()}
