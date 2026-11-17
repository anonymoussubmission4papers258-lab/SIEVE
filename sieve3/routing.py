"""Material-level routing and the observable correction (SIEVE-2 parts reused by SIEVE-3: sieve3.coupled adds
the part columns). Spectra are solved from measurements, never predicted.

Identity routing (SER):  x_i = sum_k Pi_ik (theta_k + Theta_k r_i)  ==  X = M Theta,
with Theta solved in closed form from the LR-HSI on fitting cells only:
    Theta = argmin || W^{1/2} (A M Theta - H) ||^2 + ridge (column-normalised).
Given the identities Pi, the output is a linear function of the measured spectra H.

Local regression: a guided-filter style affine RGB law fitted in a small LR window from fitting
cells only, applied at HR. The pipeline applies it to the identity residual H - A x_id (observable
correction): the measurements correct what they can observe, the identities keep the within-block
structure they cannot. Also linear in H.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .ops import box_sum, fill_from_fit, pool, upsample_window


# ----------------------------------------------------------------------------- identity routing
def pooled_design(Pi: torch.Tensor, rstd: torch.Tensor, s: int) -> torch.Tensor:
    """D = A M with M = [Pi, Pi*r1, Pi*r2, Pi*r3], built without materialising M. -> [B, 4K, h, w]."""
    parts = [pool(Pi, s)] + [pool(Pi * rstd[:, c:c + 1], s) for c in range(rstd.shape[1])]
    return torch.cat(parts, 1)


def normal_equations(D: torch.Tensor, lr: torch.Tensor, fit: torch.Tensor):
    """G = D W D^T, b = D W H^T over fitting cells. D [B,P,h,w], lr [B,C,h,w], fit [B,1,h,w]."""
    Df = D.flatten(2)
    Dw = Df * fit.to(D.dtype).flatten(2)
    return Dw @ Df.transpose(1, 2), Dw @ lr.flatten(2).transpose(1, 2)


def solve_ridge(G: torch.Tensor, b: torch.Tensor, ridge: float):
    """Column-normalised ridge solve in float64. Returns theta [B,P,C] (float32) and aux for noise maps."""
    Gd, bd = G.double(), b.double()
    d = torch.diagonal(Gd, dim1=1, dim2=2).clamp_min(0.0).sqrt() + 1e-6
    A = Gd / (d.unsqueeze(2) * d.unsqueeze(1))
    A = A + ridge * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    theta = torch.linalg.solve(A, bd / d.unsqueeze(2)) / d.unsqueeze(2)
    return theta.float(), (A, d, float(ridge))


def identity_apply(Pi: torch.Tensor, rstd: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """X = M Theta evaluated without materialising M. Pi [B,K,H,W], theta [B,4K,C] -> [B,C,H,W]."""
    K = Pi.shape[1]
    x = torch.einsum("bkhw,bkc->bchw", Pi, theta[:, :K])
    for c in range(rstd.shape[1]):
        x = x + rstd[:, c:c + 1] * torch.einsum("bkhw,bkc->bchw", Pi, theta[:, K * (c + 1):K * (c + 2)])
    return x


def identity_routing(Pi, rstd, lr, fit, s: int, ridge: float):
    D = pooled_design(Pi, rstd, s)
    G, b = normal_equations(D, lr, fit)
    theta, aux = solve_ridge(G, b, ridge)
    return identity_apply(Pi, rstd, theta), theta, aux


def identity_noise_std(Pi, rstd, aux, sigma: float) -> torch.Tensor:
    """Per-pixel noise std of identity routing given the identities: sigma * sqrt(m_i^T Q m_i).

    Theta_c = S D^T W h_c with S = diag(1/d) A^{-1} diag(1/d); Cov = sigma^2 S G S and
    S G S = diag(1/d) A^{-1} (A - ridge I) A^{-1} diag(1/d).
    """
    A, d, ridge = aux
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    Ainv = torch.linalg.inv(A)
    Q = Ainv @ (A - ridge * eye) @ Ainv
    Q = (Q / (d.unsqueeze(2) * d.unsqueeze(1))).float()
    K = Pi.shape[1]
    feats = [None] + [rstd[:, c:c + 1] for c in range(rstd.shape[1])]
    var = torch.zeros_like(Pi[:, :1])
    for a in range(len(feats)):
        for b in range(a, len(feats)):
            q = torch.einsum("bkhw,bkl,blhw->bhw", Pi, Q[:, a * K:(a + 1) * K, b * K:(b + 1) * K], Pi).unsqueeze(1)
            weight = 1.0 if a == b else 2.0
            if feats[a] is not None:
                q = q * feats[a]
            if feats[b] is not None:
                q = q * feats[b]
            var = var + weight * q
    return sigma * var.clamp_min(0.0).sqrt()


# ----------------------------------------------------------------------------- local routing
def solve3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Closed-form solve of many 3x3 systems A X = B (A [...,3,3] positive definite, B [...,3,C]).

    Adjugate / determinant: no batched LAPACK/cuBLAS call, so there is no limit on the number of systems
    (s = 1 panels have one system per HR pixel) and it is faster than torch.linalg.solve.
    """
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    c00, c01, c02 = e * i - f * h, -(d * i - f * g), d * h - e * g
    c10, c11, c12 = -(b * i - c * h), a * i - c * g, -(a * h - b * g)
    c20, c21, c22 = b * f - c * e, -(a * f - c * d), a * e - b * d
    det = a * c00 + b * c01 + c * c02
    adj = ((c00, c10, c20), (c01, c11, c21), (c02, c12, c22))            # adj = cofactor matrix transposed
    b0, b1, b2 = B[..., 0, :], B[..., 1, :], B[..., 2, :]
    rows = [r0[..., None] * b0 + r1[..., None] * b1 + r2[..., None] * b2 for r0, r1, r2 in adj]
    return torch.stack(rows, -2) / det[..., None, None]


def local_coefficients(rbar, lr, fit, radius: int = 2, eps: float = 1e-2, smooth: int = 1, min_count: int = 4):
    """Affine law h ~ a + B^T rbar per LR cell from fitting cells in a (2r+1)^2 window.

    rbar [B,3,h,w] = A r_std, lr [B,C,h,w], fit [B,1,h,w]. Returns a [B,C,h,w], Bc [B,3,C,h,w].
    Coefficients never depend on non-fitting cells' measurements.
    """
    B, C, h, w = lr.shape
    fit = fit.to(lr.dtype)
    N = box_sum(fit, radius)
    Nc = N.clamp_min(1.0)
    mr = box_sum(fit * rbar, radius) / Nc
    mh = box_sum(fit * lr, radius) / Nc
    rr = (rbar.unsqueeze(2) * rbar.unsqueeze(1)).reshape(B, 9, h, w)
    Crr = box_sum(fit * rr, radius).reshape(B, 3, 3, h, w) / Nc.unsqueeze(1) - mr.unsqueeze(2) * mr.unsqueeze(1)
    rh = (rbar.unsqueeze(2) * lr.unsqueeze(1)).reshape(B, 3 * C, h, w)
    Crh = box_sum(fit * rh, radius).reshape(B, 3, C, h, w) / Nc.unsqueeze(1) - mr.unsqueeze(2) * mh.unsqueeze(1)
    eye = torch.eye(3, device=lr.device, dtype=lr.dtype)
    Bc = solve3(Crr.permute(0, 3, 4, 1, 2) + eps * eye, Crh.permute(0, 3, 4, 1, 2))
    ok = (N >= min_count).permute(0, 2, 3, 1).unsqueeze(-1)
    Bc = torch.where(ok, Bc, torch.zeros_like(Bc)).permute(0, 3, 4, 1, 2).contiguous()
    global_mean = (fit * lr).sum((2, 3), keepdim=True) / fit.sum((2, 3), keepdim=True).clamp_min(1.0)
    a = mh - (Bc * mr.unsqueeze(2)).sum(1)
    a = torch.where(N > 0, a, global_mean.expand_as(a))
    if smooth > 0:
        count = box_sum(torch.ones_like(N), smooth)
        a = box_sum(a, smooth) / count
        Bc = (box_sum(Bc.reshape(B, 3 * C, h, w), smooth) / count).reshape(B, 3, C, h, w)
    return a, Bc


def local_apply(a, Bc, rstd):
    """Training path (differentiable, F.interpolate)."""
    B, _, C, h, w = Bc.shape
    size = rstd.shape[-2:]
    x = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
    for c in range(3):
        x = x + rstd[:, c:c + 1] * F.interpolate(Bc[:, c], size=size, mode="bilinear", align_corners=False)
    return x


def local_apply_window(a, Bc, rstd_window, out_hw, y0, y1, x0, x1):
    """Inference path: identical values to local_apply on the full image, for any HR window."""
    x = upsample_window(a, out_hw, y0, y1, x0, x1)
    for c in range(3):
        x = x + rstd_window[:, c:c + 1] * upsample_window(Bc[:, c], out_hw, y0, y1, x0, x1)
    return x


# ----------------------------------------------------------------------------- HSI-only (scene fallback)
def hsi_only(lr, fit, size):
    return F.interpolate(fill_from_fit(lr, fit), size=size, mode="bilinear", align_corners=False)
