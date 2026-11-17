"""Coupled routing: the joint null space is removed by correspondence, not filled by a prior.

    x_c = sum_k Pi_k (theta_k + Theta_k r)  +  coup * sum_m Gamma_m delta_m,        coup = beta * shading

Pi (material slots), Gamma (part codes: "the same point of the same kind of thing", over a global codebook)
and beta (does this pixel couple to its copies?) come from the VFM. Every spectral weight [theta, Theta, delta]
is solved in closed form from the LR-HSI fitting cells. A copy seen at another sub-block phase of the LR grid
measures a different mixture of the same part spectra, so phase diversity across copies turns the
RGB-invisible sub-block detail (null(A) x null(S)) into measured unknowns -- the VFM never predicts a spectrum.

Solve (float64, sufficient statistics G = D W D^T, b = D W H^T, Y^T W Y, so tiled == whole image):
  material columns  column-normalised ridge;
  part columns      Bayesian ridge: prior std tau = part_prior * RMS(H)  ->  lambda = sigma^2 / tau^2,
                    restricted to the `rank` leading spectral directions of the material-level residual
                    (the subspace is treated as data: no gradient through the eigenvectors);
  measured          per code, 1 - posterior/prior variance = 1 - reg_c [A^-1]_cc  (model-relative: how much of
                    the coupling's own detail the measurements pin down -- correctness is judged by the audit).
Mirrors tests/reference_numpy.py (coupled_design / solve_coupled / coupled_apply / measured_map).
"""
from __future__ import annotations

import torch

from .ops import pool
from .routing import identity_apply


def coupled_design(Pi, rstd, Gamma, coup, s: int) -> torch.Tensor:
    """D = A [Pi, Pi*r1, Pi*r2, Pi*r3, coup*Gamma] -> [B, 4K+M, h, w] (no HR design is materialised)."""
    parts = [pool(Pi, s)] + [pool(Pi * rstd[:, c:c + 1], s) for c in range(rstd.shape[1])]
    if Gamma is not None and Gamma.shape[1] > 0:
        parts.append(pool(Gamma * coup, s))
    return torch.cat(parts, 1)


def normal_equations64(D: torch.Tensor, lr: torch.Tensor, fit: torch.Tensor):
    """G = D W D^T [B,P,P], b = D W H^T [B,P,C] over fitting cells, accumulated in float64."""
    Df = D.double().flatten(2)
    Dw = Df * fit.double().flatten(2)
    return Dw @ Df.transpose(1, 2), Dw @ lr.double().flatten(2).transpose(1, 2)


def lr_gram64(lr: torch.Tensor, fit: torch.Tensor) -> torch.Tensor:
    """Y^T W Y [B,C,C] over fitting cells (needed for the residual subspace of the part detail)."""
    Y = lr.double().flatten(2)
    return (Y * fit.double().flatten(2)) @ Y.transpose(1, 2)


def batch_scalar(value, batch: int, device) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=torch.float64).to(device).flatten()
    return t.expand(batch) if t.numel() == 1 else t


def part_lambda(lr: torch.Tensor, fit: torch.Tensor, sigma, prior: float) -> torch.Tensor:
    """Bayesian ridge of the part deviations per image: sigma^2 / (prior * RMS(H on fitting cells))^2 -> [B]."""
    f = fit.double()
    n = (f.sum((1, 2, 3)) * lr.shape[1]).clamp_min(1.0)
    rms = ((lr.double() ** 2) * f).sum((1, 2, 3)).div(n).sqrt()
    tau = (prior * rms).clamp_min(1e-12)
    return batch_scalar(sigma, lr.shape[0], lr.device) ** 2 / tau ** 2


def _normalise(G, n_mat: int, ridge: float, lam_part, ridge_part: float):
    d = (torch.diagonal(G, dim1=1, dim2=2).clamp_min(0.0) + 1e-12).sqrt()
    A = G / (d.unsqueeze(2) * d.unsqueeze(1))
    reg = torch.full_like(d, float(ridge))
    if G.shape[-1] > n_mat:
        part = (lam_part.reshape(-1, 1) / d[:, n_mat:] ** 2).clamp_min(float(ridge_part))
        reg = torch.cat([reg[:, :n_mat], part], 1)
    return A + torch.diag_embed(reg), d, reg


def material_weights(G, b, n_mat: int, ridge: float) -> torch.Tensor:
    """Material-level solve (coupling off, beta = 0): column-normalised ridge on the first n_mat columns."""
    zero = torch.zeros(G.shape[0], dtype=G.dtype, device=G.device)
    Am, dm, _ = _normalise(G[:, :n_mat, :n_mat], n_mat, ridge, zero, 0.0)
    return torch.linalg.solve(Am, b[:, :n_mat] / dm.unsqueeze(2)) / dm.unsqueeze(2)


def solve_coupled(G, b, n_mat: int, ridge: float, lam_part, ridge_part: float = 1e-3, YtY=None, rank: int = 0,
                  need_measured: bool = True):
    """Joint solve of material laws and part deviations (float64).

    Returns w [B,P,C], the material-level weights w0 [B,n_mat,C] and measured [B,M] (None without parts).
    """
    G, b = G.double(), b.double()
    P, C = b.shape[1], b.shape[2]
    w0 = material_weights(G, b, n_mat, ridge)
    if P == n_mat:
        return w0, w0, None
    lam_part = torch.as_tensor(lam_part, dtype=torch.float64, device=G.device).flatten().expand(G.shape[0])
    A, d, reg = _normalise(G, n_mat, ridge, lam_part, ridge_part)
    measured = None
    if need_measured:                                             # a diagnostic and gate input: no gradient
        with torch.no_grad():
            diag = torch.linalg.inv(A).diagonal(dim1=1, dim2=2)[:, n_mat:]
            measured = (1.0 - reg[:, n_mat:] * diag).clamp(0.0, 1.0)
    if rank <= 0 or rank >= C or YtY is None:
        return torch.linalg.solve(A, b / d.unsqueeze(2)) / d.unsqueeze(2), w0, measured
    with torch.no_grad():                                         # residual directions of the material level
        Gm, bm, v0 = G[:, :n_mat, :n_mat], b[:, :n_mat], w0
        EtE = YtY.double() - bm.transpose(1, 2) @ v0 - v0.transpose(1, 2) @ bm + v0.transpose(1, 2) @ Gm @ v0
        U = torch.linalg.eigh(0.5 * (EtE + EtE.transpose(1, 2)))[1][..., -rank:]       # [B,C,r]
    wU = torch.linalg.solve(A, (b @ U) / d.unsqueeze(2)) / d.unsqueeze(2)              # joint solve in span(U)
    Ut = U.transpose(1, 2)
    w_mat = w0 - (w0 @ U) @ Ut + wU[:, :n_mat] @ Ut
    w_part = wU[:, n_mat:] @ Ut
    return torch.cat([w_mat, w_part], 1), w0, measured


def coupled_apply(Pi, rstd, Gamma, coup, w) -> torch.Tensor:
    """x_c from the solved weights, without materialising the design. w [B,P,C] (float32)."""
    K = Pi.shape[1]
    x = identity_apply(Pi, rstd, w[:, :4 * K])
    if Gamma is not None and Gamma.shape[1] > 0 and w.shape[1] > 4 * K:
        x = x + coup * torch.einsum("bmhw,bmc->bchw", Gamma, w[:, 4 * K:])
    return x


def measured_map(Gamma, beta, measured) -> torch.Tensor:
    """mu = beta * sum_m Gamma_m measured_m -> [B,1,H,W] in [0,1]; 0 where nothing is coupled."""
    if Gamma is None or measured is None:
        return torch.zeros_like(beta[:, :1])
    return beta * torch.einsum("bmhw,bm->bhw", Gamma, measured.to(Gamma.dtype)).unsqueeze(1)


def coupled_routing(Pi, rstd, Gamma, coup, lr, fit, s: int, rcfg, sigma, need_measured: bool = True) -> dict:
    """Whole-image coupled routing (training path; inference accumulates the same statistics over tiles)."""
    D = coupled_design(Pi, rstd, Gamma, coup, s)
    G, b = normal_equations64(D, lr, fit)
    lam = part_lambda(lr, fit, sigma, rcfg.part_prior)
    YtY = lr_gram64(lr, fit) if (Gamma is not None and rcfg.part_rank > 0) else None
    w, w0, measured = solve_coupled(G, b, 4 * Pi.shape[1], rcfg.ridge, lam, rcfg.ridge_part, YtY, rcfg.part_rank,
                                    need_measured)
    w, w0 = w.float(), w0.float()
    x_c = coupled_apply(Pi, rstd, Gamma, coup, w)
    x_m = x_c if Gamma is None or Gamma.shape[1] == 0 else identity_apply(Pi, rstd, w0)
    return {"x_c": x_c, "x_m": x_m, "w": w, "w0": w0, "measured": measured}
