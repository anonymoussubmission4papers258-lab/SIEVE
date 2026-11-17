"""Independent NumPy reference of the SIEVE-2 measurement-side mathematics.

Mirrors sieve3.ops / routing / evidence on single images without a batch dimension:
HR arrays are [C, H, W], LR arrays [C, h, w], masks [h, w] (float 0/1).
Used (1) to verify the mathematics without PyTorch and (2) by tests/test_torch.py to
cross-check the PyTorch implementation.
"""
from __future__ import annotations

import math

import numpy as np


def pool(x, s):
    if s == 1:
        return x.copy()
    C, H, W = x.shape
    return x.reshape(C, H // s, s, W // s, s).mean((2, 4))


def lift(x, s):
    return x if s == 1 else x.repeat(s, 1).repeat(s, 2)


def box_sum(x, r):
    """x [C,h,w]; zero-padded (2r+1)^2 window sums."""
    if r <= 0:
        return x.copy()
    C, h, w = x.shape
    p = np.zeros((C, h + 2 * r, w + 2 * r), x.dtype)
    p[:, r:r + h, r:r + w] = x
    out = np.zeros_like(x)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out += p[:, dy:dy + h, dx:dx + w]
    return out


def fill_from_fit(x, fit):
    fit = fit[None].astype(x.dtype)
    kept = x * fit
    count = box_sum(fit, 1)
    total = box_sum(kept, 1)
    gmean = kept.sum((1, 2), keepdims=True) / max(fit.sum(), 1.0)
    neigh = np.where(count > 0, total / np.maximum(count, 1.0), np.broadcast_to(gmean, x.shape))
    return np.where(fit > 0.5, x, neigh)


def standardize_rgb(rgb, valid, floor=0.02):
    w = valid[None].astype(rgb.dtype)
    n = max(w.sum(), 1.0)
    mean = (rgb * w).sum((1, 2), keepdims=True) / n
    var = (((rgb - mean) ** 2) * w).sum((1, 2), keepdims=True) / n
    return (rgb - mean) / np.maximum(np.sqrt(var), floor) * w


def _bilinear_index(out_len, in_len, start, stop):
    scale = in_len / out_len
    dst = np.arange(start, stop, dtype=np.float64)
    src = np.maximum((dst + 0.5) * scale - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), in_len - 1)
    i1 = np.minimum(i0 + 1, in_len - 1)
    return i0, i1, (src - i0)


def upsample(x, out_hw, window=None):
    """PyTorch-equivalent bilinear (align_corners=False). x [..., h, w]."""
    H, W = out_hw
    y0, y1, x0, x1 = window if window is not None else (0, H, 0, W)
    h, w = x.shape[-2:]
    r0, r1, ly = _bilinear_index(H, h, y0, y1)
    c0, c1, lx = _bilinear_index(W, w, x0, x1)
    rows = x[..., r0, :] * (1 - ly)[:, None] + x[..., r1, :] * ly[:, None]
    return rows[..., c0] * (1 - lx) + rows[..., c1] * lx


# ----------------------------------------------------------------------------- identity routing
def pooled_design(Pi, rstd, s):
    parts = [pool(Pi, s)] + [pool(Pi * rstd[c:c + 1], s) for c in range(rstd.shape[0])]
    return np.concatenate(parts, 0)


def normal_equations(D, lr, fit):
    Df = D.reshape(D.shape[0], -1)
    Dw = Df * fit.reshape(1, -1)
    return Dw @ Df.T, Dw @ lr.reshape(lr.shape[0], -1).T


def solve_ridge(G, b, ridge):
    d = np.sqrt(np.maximum(np.diag(G), 0.0)) + 1e-6
    A = G / np.outer(d, d) + ridge * np.eye(G.shape[0])
    theta = np.linalg.solve(A, b / d[:, None]) / d[:, None]
    return theta


def identity_apply(Pi, rstd, theta):
    K = Pi.shape[0]
    x = np.einsum("khw,kc->chw", Pi, theta[:K])
    for c in range(rstd.shape[0]):
        x = x + rstd[c:c + 1] * np.einsum("khw,kc->chw", Pi, theta[K * (c + 1):K * (c + 2)])
    return x


def identity_routing(Pi, rstd, lr, fit, s, ridge):
    G, b = normal_equations(pooled_design(Pi, rstd, s), lr, fit)
    theta = solve_ridge(G, b, ridge)
    return identity_apply(Pi, rstd, theta), theta


# ----------------------------------------------------------------------------- local routing
def local_coefficients(rbar, lr, fit, radius=2, eps=1e-2, smooth=1, min_count=4):
    C, h, w = lr.shape
    f = fit[None]
    N = box_sum(f, radius)[0]
    Nc = np.maximum(N, 1.0)
    mr = box_sum(f * rbar, radius) / Nc
    mh = box_sum(f * lr, radius) / Nc
    Crr = np.zeros((3, 3, h, w))
    for i in range(3):
        for j in range(3):
            Crr[i, j] = box_sum(f * rbar[i:i + 1] * rbar[j:j + 1], radius)[0] / Nc - mr[i] * mr[j]
    Crh = np.zeros((3, C, h, w))
    for i in range(3):
        Crh[i] = box_sum(f * rbar[i:i + 1] * lr, radius) / Nc - mr[i][None] * mh
    Bc = np.zeros((3, C, h, w))
    for y in range(h):
        for x in range(w):
            if N[y, x] >= min_count:
                Bc[:, :, y, x] = np.linalg.solve(Crr[:, :, y, x] + eps * np.eye(3), Crh[:, :, y, x])
    gmean = (f * lr).sum((1, 2)) / max(fit.sum(), 1.0)
    a = mh - (Bc * mr[:, None]).sum(0)
    a = np.where(N[None] > 0, a, gmean[:, None, None])
    if smooth > 0:
        count = box_sum(np.ones((1, h, w)), smooth)
        a = box_sum(a, smooth) / count
        Bc = (box_sum(Bc.reshape(3 * C, h, w), smooth) / count).reshape(3, C, h, w)
    return a, Bc


def local_apply(a, Bc, rstd, window=None):
    H, W = rstd.shape[-2:]
    if window is None:
        out_hw, window = (H, W), (0, H, 0, W)
        rs = rstd
    else:
        out_hw = window[4]
        window = window[:4]
        rs = rstd
    x = upsample(a, out_hw, window)
    for c in range(3):
        x = x + rs[c:c + 1] * upsample(Bc[c], out_hw, window)
    return x


def corrected_routing(Pi, rstd, lr, fit, s, ridge, **local):
    """x_id and x_cor = x_id + local regression of the identity residual (observable correction)."""
    x_id, _ = identity_routing(Pi, rstd, lr, fit, s, ridge)
    a, Bc = local_coefficients(pool(rstd, s), lr - pool(x_id, s), fit, **local)
    return x_id, x_id + local_apply(a, Bc, rstd)


# ----------------------------------------------------------------------------- evidence
def region_sum(x, R):
    """x [h,w] -> [ceil(h/R), ceil(w/R)] sums."""
    h, w = x.shape
    Rh, Rw = math.ceil(h / R), math.ceil(w / R)
    out = np.zeros((Rh, Rw))
    for i in range(Rh):
        for j in range(Rw):
            out[i, j] = x[i * R:(i + 1) * R, j * R:(j + 1) * R].sum()
    return out


def kappa_cells(x_a, x_b, s, valid_lr):
    d = x_a - x_b
    obs = (pool(d, s) ** 2).sum(0) * s * s
    tot = pool(d ** 2, s).sum(0) * s * s
    return obs * valid_lr, tot * valid_lr


def region_tests(pred_a, pred_b, lr, valid_lr, masks, sigma, R, statistic="z", min_cells=4):
    delta = (((pred_b - lr) ** 2).sum(0) - ((pred_a - lr) ** 2).sum(0)) * valid_lr
    power = ((pred_a - pred_b) ** 2).sum(0) * valid_lr
    out = []
    for mask in masks:
        m = mask * valid_lr
        n = region_sum(m, R)
        T = region_sum(delta * m, R)
        if statistic == "z":
            V = 4 * sigma ** 2 * region_sum(power * m, R)
            z = T / np.sqrt(np.maximum(V, 1e-30))
            ok = (n >= min_cells) & (V > 0)
        else:
            S2 = region_sum(delta ** 2 * m, R)
            mean = T / np.maximum(n, 1)
            var = (S2 - n * mean ** 2) / np.maximum(n - 1, 1)
            z = mean / np.sqrt(np.maximum(var, 1e-30) / np.maximum(n, 1))
            ok = (n >= max(min_cells, 3)) & (var > 0)
        out.append(np.where(ok, z, np.nan))
    return np.stack(out)


def _ndtr(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def combine_folds(Z):
    F = Z.shape[0]
    nan = np.isnan(Z)
    zz = np.where(nan, 0.0, Z)
    p_plus = np.where(nan, 1.0, _ndtr(-zz)).min(0)
    p_minus = np.where(nan, 1.0, _ndtr(zz)).min(0)
    return np.minimum(F * p_plus, 1.0), np.minimum(F * p_minus, 1.0), (~nan).any(0)


def fold_mask(h, w, period, fold):
    yy = np.arange(h)[:, None] % period
    xx = np.arange(w)[None, :] % period
    return ((yy * period + xx) == fold).astype(np.float64)


# ----------------------------------------------------------------------------- coupled routing (SIEVE-3)
def luminance_factor(rgb, valid, lo=0.05, hi=20.0):
    """Shading factor l = mean RGB / image mean over valid pixels -> [1,H,W]."""
    lum = rgb.mean(0)
    m = (lum * valid).sum() / max(valid.sum(), 1.0)
    return np.clip(lum / max(m, 1e-12), lo, hi)[None]


def coupled_design(Pi, rstd, Gamma, coup, s):
    """D = A [Pi, Pi*r1, Pi*r2, Pi*r3, coup*Gamma]  -> [4K+M, h, w]."""
    parts = [pool(Pi, s)] + [pool(Pi * rstd[c:c + 1], s) for c in range(rstd.shape[0])]
    if Gamma is not None and Gamma.shape[0] > 0:
        parts.append(pool(Gamma * coup, s))
    return np.concatenate(parts, 0)


def part_lambda(lr, fit, sigma, prior):
    """Bayesian ridge of the part deviations: sigma^2 / tau^2 with tau = prior * RMS(H on fitting cells)."""
    rms = np.sqrt((lr ** 2 * fit[None]).sum() / max(fit.sum() * lr.shape[0], 1.0))
    tau = max(prior * rms, 1e-12)
    return sigma ** 2 / tau ** 2


def _normalised(G, n_mat, ridge, lam_part, ridge_part):
    P = G.shape[0]
    d = np.sqrt(np.maximum(np.diag(G), 0.0) + 1e-12)     # (sqrt(x + eps^2): finite gradient for empty columns)
    A = G / np.outer(d, d)
    reg = np.full(P, float(ridge))
    reg[n_mat:] = np.maximum(lam_part / d[n_mat:] ** 2, ridge_part)
    return A + np.diag(reg), d, reg


def material_weights(G, b, n_mat, ridge):
    """Material-level solve (coupling off): column-normalised ridge on the first n_mat columns."""
    Am, dm, _ = _normalised(G[:n_mat, :n_mat], n_mat, ridge, 0.0, 0.0)
    return np.linalg.solve(Am, b[:n_mat] / dm[:, None]) / dm[:, None]


def solve_coupled(G, b, n_mat, ridge, lam_part, ridge_part=1e-3, YtY=None, rank=0):
    """Joint solve of material laws (column-normalised ridge) and part deviations (Bayesian ridge).

    Part deviations live in the `rank` leading spectral directions of the material-level residual
    (computed from the sufficient statistics G, b, YtY, so tiling stays exact); rank <= 0 or >= C keeps
    all bands. Returns weights [P,C] and, per part code, 1 - posterior/prior variance (measured fraction).
    """
    P, C = b.shape
    A, d, reg = _normalised(G, n_mat, ridge, lam_part, ridge_part)
    measured = np.clip(1.0 - reg[n_mat:] * np.diag(np.linalg.inv(A))[n_mat:], 0.0, 1.0)
    if n_mat == P or rank <= 0 or rank >= C or YtY is None:
        return np.linalg.solve(A, b / d[:, None]) / d[:, None], measured
    w0 = material_weights(G, b, n_mat, ridge)                               # material-only weights [4K,C]
    EtE = YtY - b[:n_mat].T @ w0 - w0.T @ b[:n_mat] + w0.T @ G[:n_mat, :n_mat] @ w0
    U = np.linalg.eigh(0.5 * (EtE + EtE.T))[1][:, -rank:]                  # residual directions [C,r]
    wU = np.linalg.solve(A, (b @ U) / d[:, None]) / d[:, None]             # joint solve in span(U) [P,r]
    w = np.zeros((P, C))
    w[:n_mat] = w0 - (w0 @ U) @ U.T + wU[:n_mat] @ U.T
    w[n_mat:] = wU[n_mat:] @ U.T
    return w, measured


def coupled_apply(Pi, rstd, Gamma, coup, w):
    K = Pi.shape[0]
    x = identity_apply(Pi, rstd, w[:4 * K])
    if Gamma is not None and Gamma.shape[0] > 0:
        x = x + coup * np.einsum("mhw,mc->chw", Gamma, w[4 * K:])
    return x


def gram_lr(lr, fit):
    Y = lr.reshape(lr.shape[0], -1) * fit.reshape(1, -1)
    return Y @ lr.reshape(lr.shape[0], -1).T


def coupled_routing(Pi, rstd, Gamma, coup, lr, fit, s, ridge, lam_part, ridge_part=1e-3, rank=4):
    """x_c = sum_k Pi_k (theta_k + Theta_k r) + coup * sum_m Gamma_m delta_m, all weights solved from H.

    Returns x_c, weights and the measured fraction per part code. coup = beta * shading.
    """
    D = coupled_design(Pi, rstd, Gamma, coup, s)
    G, b = normal_equations(D, lr, fit)
    w, measured = solve_coupled(G, b, 4 * Pi.shape[0], ridge, lam_part, ridge_part, gram_lr(lr, fit), rank)
    return coupled_apply(Pi, rstd, Gamma, coup, w), w, measured


def measured_map(Gamma, beta, measured):
    if Gamma is None or Gamma.shape[0] == 0:
        return np.zeros(beta.shape[1:])
    return beta[0] * np.einsum("mhw,m->hw", Gamma, measured)


def coupling_residual(x_c, lr, fit, s, sigma):
    """In-sample normalised residual per LR cell, ~1 where the coupled model explains the measurement."""
    r = ((pool(x_c, s) - lr) ** 2).sum(0) / (sigma ** 2 * lr.shape[0])
    return r * fit


def coupled_candidates(Pi, rstd, Gamma, coup, lr, fit, s, ridge, lam_part, ridge_part=1e-3, rank=4, **local):
    """All SIEVE-3 candidates of one image (mirrors sieve3.pipeline with gate g = 0):
    x_c   coupled routing;             x_m    material level (coupling off, beta = 0);
    x_cor x_c + observable correction; x_dcor x_m + observable correction (fallback where coupling is refuted).
    """
    D = coupled_design(Pi, rstd, Gamma, coup, s)
    G, b = normal_equations(D, lr, fit)
    n_mat = 4 * Pi.shape[0]
    w, measured = solve_coupled(G, b, n_mat, ridge, lam_part, ridge_part, gram_lr(lr, fit), rank)
    w0 = material_weights(G, b, n_mat, ridge)
    x_c = coupled_apply(Pi, rstd, Gamma, coup, w)
    x_m = identity_apply(Pi, rstd, w0)
    rbar = pool(rstd, s)
    x_cor = x_c + local_apply(*local_coefficients(rbar, lr - pool(x_c, s), fit, **local), rstd)
    x_dcor = x_m + local_apply(*local_coefficients(rbar, lr - pool(x_m, s), fit, **local), rstd)
    return {"x_c": x_c, "x_m": x_m, "x_cor": x_cor, "x_dcor": x_dcor, "w": w, "w0": w0, "measured": measured,
            "G": G, "b": b}
