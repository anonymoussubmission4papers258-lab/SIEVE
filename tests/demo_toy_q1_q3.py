"""Toy illustration of the three research sub-questions with the NumPy reference (no PyTorch).

Scene: 4 materials; materials 2 and 3 are metamers (identical RGB, different spectra).
Q1  correct identities make the unobserved (joint-null) content recoverable from the measurements;
Q3  a visually plausible identity hypothesis that merges the metamers is exposed by held-out evidence;
    the region audit overrides it -- only where the merged identity lives -- to the corrected candidate
    (identities keep the within-block structure, measurements correct what they observe);
    with correct identities nothing is overridden;
    differences confined inside LR blocks are flagged as untestable (kappa ~ 0).
Run: python tests/demo_toy_q1_q3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_numpy as R  # noqa: E402

rng = np.random.default_rng(7)
C, s, H, W, sigma = 16, 4, 96, 96, 0.01
h, w = H // s, W // s
S = np.abs(rng.standard_normal((3, C)))
S /= S.sum(1, keepdims=True)
PS = S.T @ np.linalg.solve(S @ S.T, S)

base = 0.2 + 0.6 * rng.random((3, C))
null_dir = (np.eye(C) - PS) @ rng.standard_normal(C)
null_dir *= 0.15 / np.abs(null_dir).max()
spectra = np.vstack([base, base[2] + null_dir])          # material 3 = metamer of material 2
assert np.allclose(S @ spectra[2], S @ spectra[3])

labels = np.zeros((H, W), int)
yy, xx = np.mgrid[0:H, 0:W]
labels[(xx // 24 + yy // 24) % 2 == 1] = 1
labels[(yy > 40) & (yy < 70) & (xx > 10) & (xx < 50)] = 2
labels[(yy > 40) & (yy < 70) & (xx >= 50) & (xx < 86)] = 3   # metamer region next to material 2
labels[(np.abs(xx - yy) < 2)] = 1                            # thin diagonal structure inside blocks
shade = 0.75 + 0.25 * np.cos(xx / 11.0) * np.sin(yy / 17.0)
X = spectra[labels].transpose(2, 0, 1) * shade[None]
rgb = np.einsum("kc,chw->khw", S, X)
lr = R.pool(X, s) + sigma * rng.standard_normal((C, h, w))
valid_hr, valid_lr = np.ones((H, W)), np.ones((h, w))
rstd = R.standardize_rgb(rgb, valid_hr)
masks = [R.fold_mask(h, w, 2, f) for f in range(4)]


def onehot(lab, K):
    return np.eye(K)[lab].transpose(2, 0, 1)


def joint_null_fraction(err):
    PA = R.lift(R.pool(err, s), s)
    ps = lambda Z: np.einsum("cd,dhw->chw", PS, Z)
    jn = err - PA - ps(err) + ps(PA)
    return (jn ** 2).sum() / max((err ** 2).sum(), 1e-30)


def rmse(x):
    return float(np.sqrt(((x - X) ** 2).mean()))


def crossfit(predict):
    """Held-out LR predictions: each fold predicted from the other three folds only."""
    out = np.zeros_like(lr)
    for m in masks:
        out = np.where(m[None] > 0.5, R.pool(predict(1 - m), s), out)
    return out


def identity(Pi):
    return lambda fit: R.identity_routing(Pi, rstd, lr, fit, s, 1e-3)[0]


def corrected(Pi):
    return lambda fit: R.corrected_routing(Pi, rstd, lr, fit, s, 1e-3)[1]


true_pi, merged_pi = onehot(labels, 4), onehot(np.minimum(labels, 2), 3)
candidates = {
    "identity: true materials": identity(true_pi),
    "identity: metamers merged (visual)": identity(merged_pi),
    "corrected: metamers merged": corrected(merged_pi),
    "local regression only": lambda fit: R.local_apply(*R.local_coefficients(R.pool(rstd, s), lr, fit), rstd),
    "HSI-only bilinear": lambda fit: R.upsample(R.fill_from_fit(lr, fit), (H, W)),
}
print(f"{'candidate':36s} {'HR RMSE':>9s} {'joint-null share':>17s} {'held-out MSE/sigma^2':>21s}")
for name, f in candidates.items():
    x = f(valid_lr)
    print(f"{name:36s} {rmse(x):9.5f} {joint_null_fraction(x - X):17.3f} "
          f"{((crossfit(f) - lr) ** 2).mean() / sigma ** 2:21.3f}")

merged_support = R.region_sum(R.pool((labels >= 2)[None].astype(float), s)[0], 6) > 0
print("\naudit: identity routing vs corrected candidate, per 6x6-cell region (Bonferroni over 4 folds)")
for name, Pi in (("true identities", true_pi), ("metamers merged", merged_pi)):
    fid, fco = identity(Pi), corrected(Pi)
    xi, xc = fid(valid_lr), fco(valid_lr)
    p_plus, p_minus, _ = R.combine_folds(R.region_tests(crossfit(fid), crossfit(fco), lr, valid_lr, masks, sigma, 6))
    over = p_minus <= 0.05
    cert = (p_plus <= 0.05) & ~over
    audited = np.where(np.kron(over, np.ones((6 * s, 6 * s)))[None] > 0.5, xc, xi)
    print(f"  {name:16s} overridden {int(over.sum()):2d} (outside merged support: {int((over & ~merged_support).sum())}), "
          f"certified {int(cert.sum()):2d}, inconclusive {int((~over & ~cert).sum()):2d} | "
          f"HR RMSE {rmse(xi):.5f} -> audited {rmse(audited):.5f}")

pert = rng.standard_normal(X.shape) * 0.02
pert -= R.lift(R.pool(pert, s), s)
obs, tot = R.kappa_cells(pert, 0 * pert, s, valid_lr)
print(f"\nkappa of a within-block (null(A)) difference: {obs.sum() / tot.sum():.1e} -> untestable, labelled VFM-only")
