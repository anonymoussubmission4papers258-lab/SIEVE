"""Band-agnostic spectral tokens (fitting cells only) for the material slots and the VFM's spectral-twin adapter."""
from __future__ import annotations

import torch

from .ops import fill_from_fit


@torch.no_grad()
def spectral_tokens(lr: torch.Tensor, fit: torch.Tensor, dims: int) -> torch.Tensor:
    """Per-image PCA of the LR-HSI estimated on fitting cells, scaled by the mean leading eigenvalue.

    Any number of bands maps to `dims` channels. Only distances between tokens are used downstream,
    so the arbitrary sign/rotation of the per-image basis is irrelevant. Non-fitting cells are filled
    from fitting neighbours, so a held-out cell's own spectrum never enters its token.
    Returns [B, dims, h, w] (no gradient).
    """
    B, C, h, w = lr.shape
    X = lr.float().flatten(2)
    weight = fit.float().flatten(2)
    n = weight.sum(-1, keepdim=True).clamp_min(1.0)
    mean = (X * weight).sum(-1, keepdim=True) / n
    centred = (X - mean) * weight
    cov = centred @ centred.transpose(1, 2) / n
    evals, evecs = torch.linalg.eigh(cov.double())
    k = min(dims, C)
    basis = evecs[..., -k:].flip(-1).float()
    scale = evals[..., -k:].clamp_min(1e-12).mean(-1).sqrt().float().view(B, 1, 1)
    z = (torch.einsum("bck,bcn->bkn", basis, X - mean) / scale).view(B, k, h, w)
    if k < dims:
        z = torch.cat([z, z.new_zeros(B, dims - k, h, w)], 1)
    return fill_from_fit(z, fit.float())
