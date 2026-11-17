"""Coupling baselines that are *not* a VFM (ablation: is the VFM replaceable?).

part_source = "color": part codes from k-means on RGB, i.e. "pixels of the same colour share sub-block detail".
Colour cannot say *which point of which object* a pixel is, so it cannot couple copies of an object point by
point; tests/demo_copies.py shows it stays at chance on RGB-invisible patterns, while object-relative codes
recover them. Everything else (closed-form solve, gates, audit) is identical.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def colour_centroids(rgb: torch.Tensor, valid: torch.Tensor, M: int, iters: int = 20,
                     max_points: int = 200_000) -> torch.Tensor:
    """Deterministic k-means on the RGB of valid pixels (init: luminance quantiles). -> [B, M, 3]."""
    out = []
    for b in range(rgb.shape[0]):
        x = rgb[b].float().flatten(1).T[valid[b, 0].flatten() > 0.5]
        if x.shape[0] == 0:
            x = rgb[b].float().flatten(1).T
        if x.shape[0] > max_points:
            x = x[::math.ceil(x.shape[0] / max_points)]
        order = torch.argsort(x.mean(1), stable=True)
        pick = ((torch.arange(M, device=x.device, dtype=torch.float32) + 0.5) / M * (x.shape[0] - 1)).long()
        cent = x[order[pick]].clone()
        for _ in range(iters):
            onehot = F.one_hot(torch.cdist(x, cent).argmin(1), M).to(x.dtype)     # deterministic sums (no atomics)
            sums, count = onehot.T @ x, onehot.sum(0).unsqueeze(1)
            cent = torch.where(count > 0, sums / count.clamp_min(1.0), cent)
        out.append(cent)
    return torch.stack(out)


def colour_codes(rgb: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """One-hot code of the nearest colour centroid. rgb [B,3,H,W], centroids [B,M,3] -> [B,M,H,W]."""
    d = ((rgb.float().unsqueeze(1) - centroids[:, :, :, None, None]) ** 2).sum(2)
    return F.one_hot(d.argmin(1), centroids.shape[1]).permute(0, 3, 1, 2).float()
