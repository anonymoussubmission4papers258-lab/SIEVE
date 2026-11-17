"""Mathematical checks of SIEVE-2 that need only NumPy (run: python tests/test_reference_numpy.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_numpy as R  # noqa: E402


def scene(seed=0, C=16, h=24, w=24, s=4, K=6):
    rng = np.random.default_rng(seed)
    H, W = h * s, w * s
    labels = rng.integers(0, K, (6, 6)).repeat(H // 6 + 1, 0).repeat(W // 6 + 1, 1)[:H, :W]
    yy, xx = np.mgrid[0:H, 0:W]
    labels = (labels + ((xx // 7 + yy // 11) % 2) * 0).astype(int)
    Pi = np.eye(K)[labels].transpose(2, 0, 1)                           # hard identities [K,H,W]
    shading = 0.6 + 0.4 * np.sin(xx / 9.0) * np.cos(yy / 13.0)
    rgb = np.stack([shading * (0.2 + 0.1 * k) for k in range(3)])       # crude RGB
    rgb = rgb + 0.05 * rng.standard_normal((3, H, W))
    return rng, Pi, rgb, labels


def test_identifiability_and_linearity():
    rng, Pi, rgb, _ = scene()
    s, C = 4, 16
    valid = np.ones(rgb.shape[1:])
    rstd = R.standardize_rgb(rgb, valid)
    theta_true = rng.standard_normal((4 * Pi.shape[0], C)) * 0.1
    X = R.identity_apply(Pi, rstd, theta_true)
    H = R.pool(X, s)
    fit = np.ones(H.shape[1:])
    x_hat, _ = R.identity_routing(Pi, rstd, H, fit, s, 1e-9)
    rel = np.linalg.norm(x_hat - X) / np.linalg.norm(X)
    assert rel < 1e-4, f"identity routing should recover an exactly structured scene, rel={rel:.2e}"
    H2 = rng.standard_normal(H.shape)
    a, _ = R.identity_routing(Pi, rstd, H, fit, s, 1e-3)
    b, _ = R.identity_routing(Pi, rstd, H2, fit, s, 1e-3)
    ab, _ = R.identity_routing(Pi, rstd, H + 2 * H2, fit, s, 1e-3)
    assert np.allclose(ab, a + 2 * b, atol=1e-8), "identity routing must be linear in H"
    print(f"identifiability: rel error {rel:.2e}; linearity ok")


def test_heldout_invariance():
    rng, Pi, rgb, _ = scene(1)
    s = 4
    valid = np.ones(rgb.shape[1:])
    rstd = R.standardize_rgb(rgb, valid)
    H = rng.standard_normal((16, 24, 24))
    check = R.fold_mask(24, 24, 2, 3)
    fit = 1 - check
    H_perturbed = H + 5.0 * check[None] * rng.standard_normal(H.shape)
    x1, _ = R.identity_routing(Pi, rstd, H, fit, s, 1e-3)
    x2, _ = R.identity_routing(Pi, rstd, H_perturbed, fit, s, 1e-3)
    assert np.allclose(x1, x2), "identity routing must not depend on held-out cells"
    rbar = R.pool(rstd, s)
    a1, B1 = R.local_coefficients(rbar, H, fit)
    a2, B2 = R.local_coefficients(rbar, H_perturbed, fit)
    assert np.allclose(a1, a2) and np.allclose(B1, B2), "local routing must not depend on held-out cells"
    f1, f2 = R.fill_from_fit(H, fit), R.fill_from_fit(H_perturbed, fit)
    assert np.allclose(f1, f2), "fill_from_fit must not depend on held-out cells"
    print("held-out invariance (J-invariance) ok")


def test_local_exact_and_linear():
    rng, _, rgb, _ = scene(2)
    s = 4
    rstd = R.standardize_rgb(rgb, np.ones(rgb.shape[1:]))
    a0 = rng.standard_normal((8, 1, 1))
    B0 = rng.standard_normal((3, 8, 1, 1)) * 0.3
    X = a0 + (B0 * rstd[:, None]).sum(0)
    H = R.pool(X, s)
    fit = np.ones(H.shape[1:])
    a, Bc = R.local_coefficients(R.pool(rstd, s), H, fit, eps=1e-10, smooth=0)
    x_hat = R.local_apply(a, Bc, rstd)
    rel = np.linalg.norm(x_hat - X) / np.linalg.norm(X)
    assert rel < 1e-6, f"local routing should reproduce a global affine law, rel={rel:.2e}"
    H2 = rng.standard_normal(H.shape)
    l1 = R.local_apply(*R.local_coefficients(R.pool(rstd, s), H, fit), rstd)
    l2 = R.local_apply(*R.local_coefficients(R.pool(rstd, s), H2, fit), rstd)
    l12 = R.local_apply(*R.local_coefficients(R.pool(rstd, s), H + 2 * H2, fit), rstd)
    assert np.allclose(l12, l1 + 2 * l2, atol=1e-8), "local routing must be linear in H"
    print(f"local routing: affine recovery rel {rel:.2e}; linearity ok")


def test_corrected_candidate():
    rng, Pi, rgb, _ = scene(6)
    s = 4
    rstd = R.standardize_rgb(rgb, np.ones(rgb.shape[1:]))
    fit = 1 - R.fold_mask(24, 24, 2, 1)
    H1, H2 = rng.standard_normal((2, 12, 24, 24))
    c1 = R.corrected_routing(Pi, rstd, H1, fit, s, 1e-3)[1]
    c2 = R.corrected_routing(Pi, rstd, H2, fit, s, 1e-3)[1]
    c12 = R.corrected_routing(Pi, rstd, H1 + 2 * H2, fit, s, 1e-3)[1]
    assert np.allclose(c12, c1 + 2 * c2, atol=1e-8), "corrected candidate must be linear in H"
    Hp = H1 + 5.0 * (1 - fit)[None] * rng.standard_normal(H1.shape)
    assert np.allclose(c1, R.corrected_routing(Pi, rstd, Hp, fit, s, 1e-3)[1]), "corrected candidate leaks held-out cells"
    print("corrected candidate: linear in H, independent of held-out cells")


def test_window_upsampling():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 12, 9))
    full = R.upsample(x, (48, 36))
    win = R.upsample(x, (48, 36), (8, 40, 12, 36))
    assert np.allclose(full[:, 8:40, 12:36], win), "windowed bilinear must equal the full upsampling"
    print("windowed bilinear = full bilinear ok")


def test_kappa():
    rng = np.random.default_rng(4)
    s = 4
    d = rng.standard_normal((6, 32, 32))
    d_null = d - R.lift(R.pool(d, s), s)                                # zero block means: invisible
    d_obs = R.lift(R.pool(d, s), s)                                     # block constant: fully visible
    valid = np.ones((8, 8))
    o, t = R.kappa_cells(d_null, 0 * d_null, s, valid)
    assert o.sum() / t.sum() < 1e-20
    o, t = R.kappa_cells(d_obs, 0 * d_obs, s, valid)
    assert abs(o.sum() / t.sum() - 1) < 1e-12
    o, t = R.kappa_cells(d, 0 * d, s, valid)
    print(f"kappa: null-space difference 0, block-constant 1, white noise {o.sum() / t.sum():.3f} (~1/s^2={1 / s ** 2:.3f})")


def test_audit_calibration(trials=400, alpha=0.05):
    """Equal-risk candidates: false certify/override rates must stay <= alpha (Bonferroni over 4 folds)."""
    rng = np.random.default_rng(5)
    C, h, w, sigma, Rg = 16, 32, 32, 0.01, 8
    masks = [R.fold_mask(h, w, 2, f) for f in range(4)]
    valid = np.ones((h, w))
    x = rng.random((C, h, w))
    dirs = rng.standard_normal((2, C, h, w))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    amp = 0.004
    pa, pb = x + amp * dirs[0], x + amp * dirs[1]                       # same per-cell error norm
    false_cert = false_over = 0
    regions = 0
    for _ in range(trials):
        lr = x + sigma * rng.standard_normal(x.shape)
        Z = R.region_tests(pa, pb, lr, valid, masks, sigma, Rg, "z")
        p_plus, p_minus, ok = R.combine_folds(Z)
        false_cert += int((p_plus <= alpha).sum())
        false_over += int((p_minus <= alpha).sum())
        regions += p_plus.size
    rate_c, rate_o = false_cert / regions, false_over / regions
    assert rate_c <= alpha * 1.5 and rate_o <= alpha * 1.5, (rate_c, rate_o)
    better = x + 0.5 * amp * dirs[0]
    power = 0
    for _ in range(50):
        lr = x + sigma * rng.standard_normal(x.shape)
        p_plus, _, _ = R.combine_folds(R.region_tests(better, pb, lr, valid, masks, sigma, Rg, "z"))
        power += int((p_plus <= alpha).sum())
    print(f"audit null: false certify {rate_c:.3f}, false override {rate_o:.3f} (alpha {alpha}); "
          f"power when 4x lower excess risk: {power / (50 * 16):.2f}")
    Zt = R.region_tests(better, pb, x + sigma * rng.standard_normal(x.shape), valid, masks, sigma, Rg, "t")
    assert np.isfinite(Zt).any()


# ----------------------------------------------------------------------------- SIEVE-3 coupled routing
def copies_scene(n_copies=40, seed=0, C=12, H=128, W=128, s=4, T=7, rank_delta=2, noise=0.0):
    """Exactly coupled scene: background + n copies of a 7x7 object whose interior carries part detail
    delta_m (m = object-relative position) living in a rank_delta spectral subspace."""
    rng = np.random.default_rng(seed)
    occ = np.zeros((H, W), bool)
    relpos = -np.ones((H, W), int)
    for _ in range(20000):
        if (relpos >= 0).sum() >= n_copies * T * T:
            break
        y, x = rng.integers(0, H - T), rng.integers(0, W - T)
        if occ[max(0, y - 1):y + T + 1, max(0, x - 1):x + T + 1].any():
            continue
        occ[y:y + T, x:x + T] = True
        relpos[y:y + T, x:x + T] = np.arange(T * T).reshape(T, T)
    label = (relpos >= 0).astype(int)
    Pi = np.eye(2)[label].transpose(2, 0, 1)
    Gamma = np.zeros((T * T, H, W))
    m = relpos >= 0
    Gamma[relpos[m], np.nonzero(m)[0], np.nonzero(m)[1]] = 1.0
    yy, xx = np.mgrid[0:H, 0:W]
    rgb = np.stack([0.5 + 0.2 * np.sin(xx / (7.0 + 3 * c)) * np.cos(yy / (11.0 + c)) for c in range(3)])
    rgb = rgb + 0.1 * label[None]
    valid = np.ones((H, W))
    rstd = R.standardize_rgb(rgb, valid)
    coup = R.luminance_factor(rgb, valid)
    basis = np.linalg.qr(rng.standard_normal((C, rank_delta)))[0]
    w = np.zeros((8 + T * T, C))
    w[:8] = rng.standard_normal((8, C)) * 0.2 + np.r_[np.ones(2), np.zeros(6)][:, None] * 0.5
    w[8:] = rng.standard_normal((T * T, rank_delta)) @ basis.T * 0.05
    X = R.coupled_apply(Pi, rstd, Gamma, coup, w)
    lr = R.pool(X, s) + noise * rng.standard_normal((C, H // s, W // s))
    return dict(X=X, Pi=Pi, Gamma=Gamma, rstd=rstd, coup=coup, lr=lr, w=w, s=s, basis=basis, label=label)


def test_coupled_reduces_to_identity_routing():
    rng, Pi, rgb, _ = scene(7)
    s = 4
    rstd = R.standardize_rgb(rgb, np.ones(rgb.shape[1:]))
    lr = rng.standard_normal((10, 24, 24))
    fit = 1 - R.fold_mask(24, 24, 2, 0)
    x_id, _ = R.identity_routing(Pi, rstd, lr, fit, s, 1e-3)
    x_c, _, measured = R.coupled_routing(Pi, rstd, None, None, lr, fit, s, 1e-3, 1.0)
    assert np.allclose(x_id, x_c), "without part codes coupled routing must equal identity routing"
    assert measured.size == 0
    print("coupled routing without parts = identity routing (SIEVE-2) ok")


def test_coupled_exact_recovery_and_lowrank():
    sc = copies_scene()
    fit = np.ones(sc["lr"].shape[1:])
    args = (sc["Pi"], sc["rstd"], sc["Gamma"], sc["coup"], sc["lr"], fit, sc["s"])
    rel = {}
    for rank in (0, 2, 12):
        x, w, _ = R.coupled_routing(*args, 1e-10, 1e-14, ridge_part=1e-12, rank=rank)
        rel[rank] = np.linalg.norm(x - sc["X"]) / np.linalg.norm(sc["X"])
        assert rel[rank] < 1e-5, f"coupled routing must recover an exactly coupled scene (rank {rank}): {rel[rank]:.2e}"
    x_mat, _ = R.identity_routing(sc["Pi"], sc["rstd"], sc["lr"], fit, sc["s"], 1e-10)
    rel_mat = np.linalg.norm(x_mat - sc["X"]) / np.linalg.norm(sc["X"])
    assert rel_mat > 100 * max(rel.values()), "the part detail must be invisible to material-level routing"
    obj = sc["label"] == 1
    det = np.linalg.norm((x_mat - sc["X"])[:, obj]) / np.linalg.norm(sc["X"][:, obj])
    print(f"exact coupled recovery: rel {max(rel.values()):.1e} (full, rank 2, rank C); "
          f"material level misses the part detail (object rel {det:.1e})")


def test_coupled_heldout_invariance_and_tiling():
    sc = copies_scene(seed=1, noise=0.01)
    h, w = sc["lr"].shape[1:]
    rng = np.random.default_rng(2)
    check = R.fold_mask(h, w, 2, 1)
    fit = 1 - check
    lam = R.part_lambda(sc["lr"], fit, 0.01, 0.15)
    perturbed = sc["lr"] + 5.0 * check[None] * rng.standard_normal(sc["lr"].shape)
    lam_p = R.part_lambda(perturbed, fit, 0.01, 0.15)
    assert np.isclose(lam, lam_p), "part prior must use fitting cells only"
    args = (sc["Pi"], sc["rstd"], sc["Gamma"], sc["coup"])
    for rank in (0, 4):
        a = R.coupled_candidates(*args, sc["lr"], fit, sc["s"], 1e-3, lam, rank=rank)
        b = R.coupled_candidates(*args, perturbed, fit, sc["s"], 1e-3, lam, rank=rank)
        for key in ("x_c", "x_m", "x_cor", "x_dcor", "measured"):
            assert np.allclose(a[key], b[key]), f"{key} (rank {rank}) depends on held-out cells"
    # sufficient statistics add over tiles -> the tiled solve equals the whole-image solve
    D = R.coupled_design(*args, sc["s"])
    G, b = R.normal_equations(D, sc["lr"], fit)
    Gt, bt, Yt = 0, 0, 0
    for ys in (slice(0, 12), slice(12, h)):
        for xs in (slice(0, 20), slice(20, w)):
            Gi, bi = R.normal_equations(D[:, ys, xs], sc["lr"][:, ys, xs], fit[ys, xs])
            Gt, bt, Yt = Gt + Gi, bt + bi, Yt + R.gram_lr(sc["lr"][:, ys, xs], fit[ys, xs])
    n_mat = 8
    w_whole, m_whole = R.solve_coupled(G, b, n_mat, 1e-3, lam, 1e-3, R.gram_lr(sc["lr"], fit), 4)
    w_tiled, m_tiled = R.solve_coupled(Gt, bt, n_mat, 1e-3, lam, 1e-3, Yt, 4)
    assert np.allclose(w_whole, w_tiled) and np.allclose(m_whole, m_tiled), "tiled sufficient statistics differ"
    x1, _, _ = R.coupled_routing(*args, sc["lr"], fit, sc["s"], 1e-3, lam, rank=0)
    x2, _, _ = R.coupled_routing(*args, perturbed * 0 + 2 * sc["lr"], fit, sc["s"], 1e-3, lam, rank=0)
    assert np.allclose(x2, 2 * x1), "full-band coupled routing must be linear in H"
    print("coupled routing: held-out invariance (all candidates, measured), exact tiling, linearity ok")


def test_measured_fraction():
    out = []
    for n in (2, 8, 32):
        sc = copies_scene(n_copies=n, seed=3, noise=0.01)
        fit = np.ones(sc["lr"].shape[1:])
        lam = R.part_lambda(sc["lr"], fit, 0.01, 0.15)
        _, _, measured = R.coupled_routing(sc["Pi"], sc["rstd"], sc["Gamma"], sc["coup"], sc["lr"], fit, sc["s"],
                                           1e-3, lam, rank=4)
        assert ((measured >= 0) & (measured <= 1)).all()
        out.append(float(measured.mean()))
    assert out[0] < out[1] < out[2], f"measured fraction must grow with the number of copies: {out}"
    sc = copies_scene(n_copies=8, seed=3, noise=0.01)
    extra = np.concatenate([sc["Gamma"], np.zeros((3, *sc["Gamma"].shape[1:]))], 0)   # 3 unused codes
    fit = np.ones(sc["lr"].shape[1:])
    lam = R.part_lambda(sc["lr"], fit, 0.01, 0.15)
    _, _, measured = R.coupled_routing(sc["Pi"], sc["rstd"], extra, sc["coup"], sc["lr"], fit, sc["s"], 1e-3, lam)
    assert np.all(measured[-3:] < 1e-6), "unused codes must have measured fraction 0"
    print(f"measured fraction: {out[0]:.2f} / {out[1]:.2f} / {out[2]:.2f} for 2 / 8 / 32 copies; unused codes 0")


if __name__ == "__main__":
    test_identifiability_and_linearity()
    test_heldout_invariance()
    test_local_exact_and_linear()
    test_corrected_candidate()
    test_window_upsampling()
    test_kappa()
    test_audit_calibration()
    test_coupled_reduces_to_identity_routing()
    test_coupled_exact_recovery_and_lowrank()
    test_coupled_heldout_invariance_and_tiling()
    test_measured_fraction()
    print("ALL NUMPY REFERENCE TESTS PASSED")
