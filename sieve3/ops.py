"""Observation operators and small tensor utilities.

Conventions: HR tensors are [B, C, H, W]; LR tensors are [B, C, h, w] with H = s*h.
A is non-overlapping s x s area averaging (identity when s == 1). Masks are float {0,1}.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pool(x: torch.Tensor, s: int) -> torch.Tensor:
    """Spatial observation operator A (block mean)."""
    return x if s == 1 else F.avg_pool2d(x, s)


def lift(x: torch.Tensor, s: int) -> torch.Tensor:
    """Nearest-neighbour replication (U = s^2 A^T up to the factor s^2)."""
    return x if s == 1 else F.interpolate(x, scale_factor=s, mode="nearest")


def block_valid(valid_hr: torch.Tensor, s: int) -> torch.Tensor:
    """An LR cell is valid only if every HR pixel of its block is valid."""
    v = valid_hr.float()
    if s == 1:
        return (v > 0.5).float()
    return (F.avg_pool2d(v, s) > 0.999).float()


def observable_hr(valid_hr: torch.Tensor, s: int) -> torch.Tensor:
    return lift(block_valid(valid_hr, s), s)


def fold_index(h: int, w: int, period: int, device=None) -> torch.Tensor:
    yy = torch.arange(h, device=device).view(h, 1) % period
    xx = torch.arange(w, device=device).view(1, w) % period
    return yy * period + xx


def fold_mask(h: int, w: int, period: int, fold: int, device=None) -> torch.Tensor:
    """Interleaved fold: cells with ((y mod p)*p + x mod p) == fold. Shape [1,1,h,w]."""
    return (fold_index(h, w, period, device) == fold).float().view(1, 1, h, w)


def random_check(batch: int, h: int, w: int, period: int, device, generator=None) -> torch.Tensor:
    """One random interleaved fold per sample (density 1/period^2)."""
    folds = torch.randint(0, period * period, (batch,), generator=generator)
    idx = fold_index(h, w, period, device).view(1, 1, h, w)
    return (idx == folds.to(device).view(batch, 1, 1, 1)).float()


def standardize_rgb(rgb: torch.Tensor, valid: torch.Tensor, floor: float = 0.02) -> torch.Tensor:
    """Per-image, per-channel standardisation over valid pixels; invalid pixels set to 0."""
    w = valid.float()
    n = w.sum((2, 3), keepdim=True).clamp_min(1.0)
    mean = (rgb * w).sum((2, 3), keepdim=True) / n
    var = (((rgb - mean) ** 2) * w).sum((2, 3), keepdim=True) / n
    return (rgb - mean) / var.sqrt().clamp_min(floor) * w


def box_sum(x: torch.Tensor, r: int) -> torch.Tensor:
    """Sum over a (2r+1)^2 window with zero padding (same spatial size)."""
    if r <= 0:
        return x
    return F.avg_pool2d(F.pad(x, (r, r, r, r)), 2 * r + 1, stride=1, divisor_override=1)


def fill_from_fit(x: torch.Tensor, fit: torch.Tensor) -> torch.Tensor:
    """Keep fitting cells; fill all other cells from 3x3 fitting neighbours, else the global fitting mean.

    The output at a non-fitting cell never depends on that cell's own value.
    """
    fit = fit.to(x.dtype)
    kept = x * fit
    count = box_sum(fit, 1)
    total = box_sum(kept, 1)
    global_mean = kept.sum((2, 3), keepdim=True) / fit.sum((2, 3), keepdim=True).clamp_min(1.0)
    neighbour = torch.where(count > 0, total / count.clamp_min(1.0), global_mean.expand_as(x))
    return torch.where(fit > 0.5, x, neighbour)


def _bilinear_index(out_len: int, in_len: int, start: int, stop: int, device):
    """Source indices/weights of PyTorch bilinear interpolation (align_corners=False)."""
    scale = in_len / out_len
    dst = torch.arange(start, stop, device=device, dtype=torch.float64)
    src = ((dst + 0.5) * scale - 0.5).clamp_min(0.0)
    i0 = src.floor().long().clamp_max(in_len - 1)
    i1 = (i0 + 1).clamp_max(in_len - 1)
    lam = (src - i0.double()).float()
    return i0, i1, lam


def upsample_window(x: torch.Tensor, out_hw, y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
    """Rows [y0,y1) and columns [x0,x1) of F.interpolate(x, out_hw, 'bilinear', align_corners=False).

    Exact for any window, so tiled and whole-image inference agree.
    """
    H, W = out_hw
    h, w = x.shape[-2:]
    r0, r1, ly = _bilinear_index(H, h, y0, y1, x.device)
    c0, c1, lx = _bilinear_index(W, w, x0, x1, x.device)
    ly = ly.to(x.dtype).view(-1, 1)
    lx = lx.to(x.dtype)
    rows = x[..., r0, :] * (1 - ly) + x[..., r1, :] * ly
    return rows[..., c0] * (1 - lx) + rows[..., c1] * lx


def vfm_normalize(rgb: torch.Tensor, quantile: float = 0.995) -> torch.Tensor:
    """Scale an RGB image to [0,1] by a robust per-image maximum (for foundation-model inputs)."""
    flat = rgb.detach().flatten(1)
    if flat.shape[1] > 1_000_000:
        step = math.ceil(flat.shape[1] / 1_000_000)
        flat = flat[:, ::step]
    top = torch.quantile(flat.float(), quantile, dim=1).clamp_min(1e-6)
    return (rgb / top.view(-1, 1, 1, 1).to(rgb.dtype)).clamp(0.0, 1.0)


@torch.no_grad()
def estimate_sigma(lr: torch.Tensor, fit: torch.Tensor) -> torch.Tensor:
    """Robust white-noise estimate per image from deviations to the 3x3 fitting-neighbour mean.

    Each deviation is divided by sqrt(1 + 1/n) for its own number n of fitting neighbours (4-6 inside
    interleaved audit folds, fewer at borders), so the estimate is unbiased for any fit mask.
    """
    fit = fit.to(lr.dtype)
    neighbours = box_sum(lr * fit, 1) - lr * fit
    count = box_sum(fit, 1) - fit
    ok = (fit > 0.5) & (count > 0)
    resid = (lr - neighbours / count.clamp_min(1.0)) / (1.0 + 1.0 / count.clamp_min(1.0)).sqrt()
    out = []
    for b in range(lr.shape[0]):
        values = resid[b][ok[b].expand_as(resid[b])].abs()
        if values.numel() == 0:
            out.append(torch.tensor(float("nan"), device=lr.device))
            continue
        if values.numel() > 2_000_000:
            values = values[:: math.ceil(values.numel() / 2_000_000)]
        out.append(1.4826 * values.median())
    return torch.stack(out)


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def luminance_factor(rgb: torch.Tensor, valid: torch.Tensor, lo: float = 0.05, hi: float = 20.0) -> torch.Tensor:
    """Shading factor l = (mean of RGB channels) / (its image mean over valid pixels), clamped. -> [B,1,H,W].

    Part detail is coupled as coup * delta with coup = beta * l, so copies under different illumination
    share one reflectance-like deviation.
    """
    lum = rgb.float().mean(1, keepdim=True)
    w = valid.float()
    mean = (lum * w).sum((2, 3), keepdim=True) / w.sum((2, 3), keepdim=True).clamp_min(1.0)
    return (lum / mean.clamp_min(1e-12)).clamp(lo, hi)


def sincos_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    """[h*w, dim] sinusoidal encoding of normalised cell centres ((i+0.5)/h, (j+0.5)/w), row-major.

    Grids of different resolution (ViT patches, LR cells) share one coordinate frame.
    """
    n = max(1, dim // 4)
    freq = (2.0 ** torch.arange(n, device=device, dtype=torch.float32)) * math.pi
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    ay, ax = gy.reshape(-1, 1) * freq, gx.reshape(-1, 1) * freq
    pe = torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], 1)
    if pe.shape[1] < dim:
        pe = torch.cat([pe, pe.new_zeros(pe.shape[0], dim - pe.shape[1])], 1)
    return pe[:, :dim]


def key_grid(h: int, w: int, max_keys: int):
    """Largest grid of at most max_keys cells with the aspect ratio of h x w."""
    if h * w <= max_keys:
        return h, w
    r = math.sqrt(max_keys / float(h * w))
    return max(1, int(h * r)), max(1, int(w * r))
