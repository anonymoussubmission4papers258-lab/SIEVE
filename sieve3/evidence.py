"""Held-out spectral evidence: the single quantity behind learning, deciding and certifying.

Evidence of a candidate on a held-out LR cell l:  -||h_l - (A x)_l||^2, where x was produced
without using h_l (cross-fitting). Under independent Gaussian noise its expectation is the
observable prediction risk plus a constant (J-invariance), so

* training maximises it                       -> heldout_loss
* the gate is fed its in-sample analogue      -> fit_features
* the audit tests it per region and fold      -> region_tests / combine_folds
  (SIEVE-3: coupled vs material level = are the coupled copies the same? and gated vs corrected output)
* and first asks whether it can exist at all  -> kappa_cells (observable fraction of a disagreement)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .ops import fill_from_fit, pool

UNTESTABLE, CERTIFIED, INCONCLUSIVE, OVERRIDDEN, SCENE_FALLBACK = 0, 1, 2, 3, 4
CODE_NAMES = {UNTESTABLE: "untestable(VFM-only)", CERTIFIED: "certified", INCONCLUSIVE: "inconclusive",
              OVERRIDDEN: "overridden(corrected)", SCENE_FALLBACK: "scene_fallback(HSI-only)"}


def _sigma2(sigma, like: torch.Tensor):
    if torch.is_tensor(sigma):
        return (sigma.to(like.dtype).to(like.device) ** 2).view(-1, 1, 1, 1)
    return float(sigma) ** 2


# ----------------------------------------------------------------------------- learning
def heldout_loss(x_hr: torch.Tensor, lr: torch.Tensor, check: torch.Tensor, s: int, sigma) -> torch.Tensor:
    """Mean held-out squared error per entry divided by sigma^2 (pure noise gives 1)."""
    err = ((pool(x_hr, s) - lr) ** 2) * check
    num = err.sum((1, 2, 3))
    den = (check.sum((1, 2, 3)) * lr.shape[1]).clamp_min(1.0)
    sig2 = _sigma2(sigma, lr)
    if torch.is_tensor(sig2):
        sig2 = sig2.view(-1)
    return (num / den / sig2).mean()


# ----------------------------------------------------------------------------- deciding
@torch.no_grad()
def fit_features(lr, pred_id, pred_cor, fit, sigma) -> torch.Tensor:
    """In-sample evidence of identity routing and of the corrected candidate on fitting cells.

    pred_* are LR predictions (A x); non-fitting cells are filled from fitting neighbours.
    Returns log1p of per-cell normalised residual energy, [B,2,h,w].
    """
    sig2 = _sigma2(sigma, lr) * lr.shape[1]
    rid = ((pred_id - lr) ** 2).sum(1, keepdim=True) / sig2
    rco = ((pred_cor - lr) ** 2).sum(1, keepdim=True) / sig2
    return torch.log1p(fill_from_fit(torch.cat([rid, rco], 1), fit))


# ----------------------------------------------------------------------------- certifying
def region_sum(x: torch.Tensor, R: int) -> torch.Tensor:
    return F.avg_pool2d(x, R, stride=R, ceil_mode=True, divisor_override=1)


def kappa_cells(x_a: torch.Tensor, x_b: torch.Tensor, s: int, valid_lr: torch.Tensor):
    """Per LR cell: observable energy ||P_A d||^2 and total energy ||d||^2 of d = x_a - x_b."""
    diff = x_a - x_b
    obs = (pool(diff, s) ** 2).sum(1, keepdim=True) * float(s * s)
    tot = pool(diff ** 2, s).sum(1, keepdim=True) * float(s * s)
    return obs * valid_lr, tot * valid_lr


def region_tests(pred_a, pred_b, lr, valid_lr, fold_masks, sigma, R: int, statistic: str = "z", min_cells: int = 4):
    """Paired held-out comparison of candidate a versus b, per region and fold.

    delta_l = ||b_l - h_l||^2 - ||a_l - h_l||^2 (positive favours a). Each fold's statistic uses only
    cells of that fold, whose predictions were made without them:
      z: T / sqrt(4 sigma^2 sum ||a_l - b_l||^2) -- exact under Gaussian noise with known sigma;
      t: self-normalised mean/sd of delta over cells -- for unknown or heteroscedastic noise.
    Returns [F, B, 1, Rh, Rw] with NaN where a fold has too few cells or no power.
    """
    delta = (((pred_b - lr) ** 2).sum(1, keepdim=True) - ((pred_a - lr) ** 2).sum(1, keepdim=True)) * valid_lr
    power = ((pred_a - pred_b) ** 2).sum(1, keepdim=True) * valid_lr
    sig2 = _sigma2(sigma, lr)
    out = []
    for mask in fold_masks:
        m = mask * valid_lr
        n = region_sum(m, R)
        T = region_sum(delta * m, R)
        if statistic == "z":
            V = 4.0 * sig2 * region_sum(power * m, R)
            z = T / V.clamp_min(1e-30).sqrt()
            ok = (n >= min_cells) & (V > 0)
        elif statistic == "t":
            S2 = region_sum(delta ** 2 * m, R)
            mean = T / n.clamp_min(1.0)
            var = (S2 - n * mean ** 2) / (n - 1.0).clamp_min(1.0)
            z = mean / (var.clamp_min(1e-30) / n.clamp_min(1.0)).sqrt()
            ok = (n >= max(min_cells, 3)) & (var > 0)
        else:
            raise ValueError(f"unknown statistic {statistic}")
        out.append(torch.where(ok, z, torch.full_like(z, float("nan"))))
    return torch.stack(out)


def combine_folds(Z: torch.Tensor):
    """Bonferroni over folds (valid under any dependence between folds).

    Returns p_plus (a better), p_minus (b better), any_ok, mean z over available folds.
    """
    folds = Z.shape[0]
    nan = torch.isnan(Z)
    zz = torch.where(nan, torch.zeros_like(Z), Z)
    one = torch.ones_like(Z)
    p_plus = torch.where(nan, one, torch.special.ndtr(-zz)).min(0).values
    p_minus = torch.where(nan, one, torch.special.ndtr(zz)).min(0).values
    any_ok = (~nan).any(0)
    z_mean = zz.sum(0) / (~nan).sum(0).clamp_min(1)
    return (folds * p_plus).clamp_max(1.0), (folds * p_minus).clamp_max(1.0), any_ok, z_mean


def decide(p_plus, p_minus, any_ok, kappa, alpha: float, kappa_min: float) -> torch.Tensor:
    """Four outcomes per region. Overriding has priority over certifying."""
    code = torch.full(kappa.shape, INCONCLUSIVE, dtype=torch.long, device=kappa.device)
    code = torch.where(p_plus <= alpha, torch.full_like(code, CERTIFIED), code)
    code = torch.where(p_minus <= alpha, torch.full_like(code, OVERRIDDEN), code)
    testable = any_ok & (kappa >= kappa_min)
    return torch.where(testable, code, torch.full_like(code, UNTESTABLE))
