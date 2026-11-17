"""The SIEVE-3 claim on a controlled scene (NumPy only, ~1 min):  python tests/demo_copies.py [--split]

A 7x7 object whose interior hides a pattern made of two METAMERS (identical RGB, different spectra) is
repeated N times at random positions, i.e. at random sub-block phases of the LR grid. Every method uses the
same closed-form coupled solver; only the coupling (which pixels share spectral detail) differs:

  no coupling      materials only (one law per material)                      -> SIEVE-2 level
  colour coupling  part codes from k-means on RGB colour                      -> what colour can link
  VFM coupling     part codes = "same point on the same kind of object"       -> what a VFM can link

Claims checked:
  (1) only VFM coupling recovers the RGB-invisible pattern, and it improves with the number of copies;
  (2) the measured map (posterior/prior variance) says how much of that detail is measured, per pixel;
  (3) Q3: copies that look identical but differ inside are exposed by held-out measurements: strongly when a
      split is offered (paired test merged vs split), and locally by the audit's region test (coupled vs material
      level), which refutes the merged coupling where it predicts unseen cells worse than no coupling, never
      refutes a correct coupling, and refutes harmful couplings.
  --split  prototype of a data-driven split (hard EM over copies, proposal never sees the test cells, ~2 min):
           recovers most kinds but the test is not yet powerful -- an open item (see HANDOFF_ZH.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_numpy as R  # noqa: E402

C, s, H, W, T, SIGMA, PRIOR, RANK = 31, 4, 128, 128, 7, 0.01, 0.15, 4
lam = np.linspace(400, 700, C)
g = lambda m, sd: np.exp(-0.5 * ((lam - m) / sd) ** 2)  # noqa: E731
S = np.stack([g(600, 35), g(540, 38), g(455, 30)])
S /= S.sum(1, keepdims=True)
PS = S.T @ np.linalg.solve(S @ S.T, S)
MAT_A = 0.30 + 0.20 * g(520, 60)
DELTA = (np.eye(C) - PS) @ (0.3 * (g(470, 15) - g(620, 18)) + 0.2 * (g(560, 12) - g(505, 12)))
DELTA *= 0.12 / np.abs(DELTA).max()
MAT_B = MAT_A + DELTA                                       # metamer of A: same RGB
BG = 0.15 + 0.5 / (1 + np.exp(-(lam - 580) / 15))
assert np.allclose(S @ MAT_A, S @ MAT_B)
P1 = np.zeros((T, T), bool); P1[:, 3] = True; P1[3, :] = True; P1[1, 1] = P1[5, 5] = True      # noqa: E702
P2 = np.zeros((T, T), bool); np.fill_diagonal(P2, True); P2[np.arange(T), T - 1 - np.arange(T)] = True  # noqa: E702


def scene(N, seed, mixed=False):
    rng = np.random.default_rng(seed)
    occ = np.zeros((H, W), bool)
    pos = []
    for _ in range(50000):
        if len(pos) == N:
            break
        y, x = rng.integers(0, H - T), rng.integers(0, W - T)
        if occ[max(0, y - 1):y + T + 1, max(0, x - 1):x + T + 1].any():
            continue
        occ[y:y + T, x:x + T] = True
        pos.append((y, x))
    yy, xx = np.mgrid[0:H, 0:W]
    shade = 0.9 + 0.1 * np.cos(xx / 23.0) * np.sin(yy / 31.0)
    X = BG[:, None, None] * shade[None]
    label = np.zeros((H, W), int)
    relpos = -np.ones((H, W), int)
    ctype = -np.ones((H, W), int)
    isB = np.zeros((H, W), bool)
    inst = -np.ones((H, W), int)
    for i, (y, x) in enumerate(pos):
        kind = (i % 2) if mixed else 0
        pat = P1 if kind == 0 else P2
        spec = np.where(pat[..., None], MAT_B, MAT_A).transpose(2, 0, 1)
        X[:, y:y + T, x:x + T] = spec * shade[None, y:y + T, x:x + T]
        label[y:y + T, x:x + T] = 1
        relpos[y:y + T, x:x + T] = np.arange(T * T).reshape(T, T)
        inst[y:y + T, x:x + T] = i
        ctype[y:y + T, x:x + T] = kind
        isB[y:y + T, x:x + T] = pat
    rgb = np.einsum("kc,chw->khw", S, X)
    lr = R.pool(X, s) + SIGMA * rng.standard_normal((C, H // s, W // s))
    return dict(X=X, rgb=rgb, lr=lr, label=label, relpos=relpos, ctype=ctype, isB=isB, inst=inst, n=len(pos))


def colour_codes(rgb, M=6, iters=25, seed=0):
    """Part codes a colour-based method could form: k-means on RGB (pixels of equal colour share a code)."""
    rng = np.random.default_rng(seed)
    f = rgb.reshape(3, -1).T
    cent = f[rng.choice(len(f), M, replace=False)]
    for _ in range(iters):
        lab = ((f[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        for m in range(M):
            if (lab == m).any():
                cent[m] = f[lab == m].mean(0)
    return np.eye(M)[lab].T.reshape(M, *rgb.shape[1:])


def vfm_codes(sc, split_types=False):
    """Codes a VFM supplies: same point of the same kind of object (object-relative position)."""
    n_types = 2 if split_types else 1
    G = np.zeros((n_types * T * T, H, W))
    m = sc["relpos"] >= 0
    idx = sc["relpos"][m] + (sc["ctype"][m] * T * T if split_types else 0)
    G[idx, np.nonzero(m)[0], np.nonzero(m)[1]] = 1.0
    return G


def solve(sc, Gamma, fit=None):
    valid = np.ones((H, W))
    rstd = R.standardize_rgb(sc["rgb"], valid)
    Pi = np.eye(2)[sc["label"]].transpose(2, 0, 1)                    # materials: background / object
    beta = np.ones((1, H, W))
    coup = beta * R.luminance_factor(sc["rgb"], valid)
    fit = np.ones((H // s, W // s)) if fit is None else fit
    lam_p = R.part_lambda(sc["lr"], fit, SIGMA, PRIOR)
    x, _, measured = R.coupled_routing(Pi, rstd, Gamma, coup, sc["lr"], fit, s, 1e-3, lam_p, rank=RANK)
    mu = R.measured_map(Gamma, beta, measured)
    return x, mu


def crossfit_residual(sc, Gamma):
    """Held-out residual per LR cell (4 interleaved folds): each cell predicted without its own value."""
    h, w = H // s, W // s
    out = np.zeros((h, w))
    for f in range(4):
        m = R.fold_mask(h, w, 2, f)
        x, _ = solve(sc, Gamma, fit=1 - m)
        r = ((R.pool(x, s) - sc["lr"]) ** 2).sum(0) / (SIGMA ** 2 * C)
        out = np.where(m > 0.5, r, out)
    return out


def scores(sc, x, mu, res):
    obj = sc["label"] == 1
    err = np.sqrt(((x - sc["X"])[:, obj] ** 2).mean())
    v = x[:, obj] / np.linalg.norm(x[:, obj], axis=0, keepdims=True)
    proj = (v - v.mean(1, keepdims=True)).T @ DELTA                   # RGB-invisible direction
    truth = sc["isB"][obj]
    pred = proj > 0
    bal = 0.5 * ((pred & truth).sum() / max(truth.sum(), 1) + (~pred & ~truth).sum() / max((~truth).sum(), 1))
    touched = R.pool(obj[None].astype(float), s)[0] > 0
    return err, bal, float(mu[obj].mean()), float(res[touched].mean())


def table(seeds=8):
    print("Claim (1)-(2): RGB-invisible pattern inside repeated objects (balanced accuracy: chance = 50%)")
    print(f"{'copies':>6} | {'method':16s} | {'object RMSE':>11} | {'pattern bal.acc':>15} | {'measured':>8}")
    for N in (1, 4, 16, 64):
        for name in ("no coupling", "colour coupling", "VFM coupling"):
            rows = []
            for sd in range(seeds):
                sc = scene(N, sd)
                Gm = None if name == "no coupling" else colour_codes(sc["rgb"]) if name == "colour coupling"\
                    else vfm_codes(sc)
                x, mu = solve(sc, Gm)
                rows.append(scores(sc, x, mu, np.zeros((H // s, W // s))))
            r = np.mean(rows, 0)
            print(f"{N:6d} | {name:16s} | {r[0]:11.5f} | {r[1]:15.1%} | {r[2]:8.2f}")
        print("-" * 66)


def q3(seeds=8, N=48):
    print(f"Claim (3): {N} copies, half hide a different pattern (still identical in RGB)")
    print(f"{'scene':28s} | {'coupling':24s} | {'held-out residual/noise':>23} | {'pattern bal.acc':>15}")
    paired = []
    for mixed in (False, True):
        for split in ((False, True) if mixed else (False,)):
            rows = []
            for sd in range(seeds):
                sc = scene(N, 100 + sd, mixed=mixed)
                Gm = vfm_codes(sc, split_types=split)
                x, mu = solve(sc, Gm)
                res = crossfit_residual(sc, Gm)
                rows.append(scores(sc, x, mu, res))
                if mixed:
                    touched = R.pool((sc["label"] == 1)[None].astype(float), s)[0] > 0
                    paired.append((sd, split, res[touched]))
            r = np.mean(rows, 0)
            label = "all copies identical" if not mixed else "two hidden variants"
            how = "same point (one kind)" if not split else "same point, kinds split"
            print(f"{label:28s} | {how:24s} | {r[3]:23.2f} | {r[1]:15.1%}")
    z, zm = [], []
    for sd in range(seeds):
        one = [p[2] for p in paired if p[0] == sd and not p[1]][0]
        two = [p[2] for p in paired if p[0] == sd and p[1]][0]
        d = one - two
        z.append(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
        sc = scene(N, 100 + sd, mixed=True)                              # material level (coupling off)
        touched = R.pool((sc["label"] == 1)[None].astype(float), s)[0] > 0
        dm = crossfit_residual(sc, None)[touched] - one
        zm.append(dm.mean() / (dm.std(ddof=1) / np.sqrt(dm.size)))
    print(f"paired held-out test, merged vs split coupling on the same scenes: z = {np.mean(z):.1f} "
          f"(min {np.min(z):.1f}) -> the measurements expose the look-alike copies")
    print(f"paired held-out test, material level vs merged coupling (whole image): z = {np.mean(zm):.1f} "
          f"(min {np.min(zm):.1f})\n   -> globally the merged coupling still beats no coupling; the audit therefore "
          f"tests per region (next table)")
    print("note: 'measured' is model-relative (how much of the coupling's own detail H pins down); colour coupling\n"
          "      pins coarse colour clusters, which cannot represent the hidden pattern at all.")


def shuffled_codes(sc, rng):
    """A harmful coupling: every copy links each of its points to a random *other* point of the other copies."""
    G = np.zeros((T * T, H, W))
    for i in range(sc["n"]):
        m = sc["inst"] == i
        perm = rng.permutation(T * T)
        G[perm[sc["relpos"][m]], np.nonzero(m)[0], np.nonzero(m)[1]] = 1.0
    return G


def audit_regions(sc, Gamma, region=8, alpha=0.05):
    """The audit's coupling test (mirrors sieve3.audit): per region of 8x8 LR cells, paired held-out comparison of
    the coupled x_c and the material-level x_m on 4 cross-fitted folds, Bonferroni over folds."""
    h, w = H // s, W // s
    folds = [R.fold_mask(h, w, 2, f) for f in range(4)]
    pc, pm = np.zeros_like(sc["lr"]), np.zeros_like(sc["lr"])
    for f in range(4):
        held = folds[f] > 0.5
        pc[:, held] = R.pool(solve(sc, Gamma, fit=1 - folds[f])[0], s)[:, held]
        pm[:, held] = R.pool(solve(sc, None, fit=1 - folds[f])[0], s)[:, held]
    p_plus, p_minus, ok = R.combine_folds(R.region_tests(pc, pm, sc["lr"], np.ones((h, w)), folds, SIGMA, region))
    obj = R.region_sum((R.pool((sc["label"] == 1)[None].astype(float), s)[0] > 0).astype(float), region) > 0
    return ((p_plus <= alpha) & ok & obj).sum(), ((p_minus <= alpha) & ok & obj).sum(), obj.sum()


def q3_audit(seeds=6, N=48):
    print(f"Claim (3) as the audit sees it: coupled vs material level per region ({N} copies, alpha 0.05)")
    print(f"{'coupling':42s} | {'certified':>9} | {'refuted':>7}")
    rng = np.random.default_rng(0)
    cases = (("correct, all copies identical", False, lambda sc: vfm_codes(sc)),
             ("merged, two hidden variants (look-alikes)", True, lambda sc: vfm_codes(sc)),
             ("harmful: points shuffled across copies", False, lambda sc: shuffled_codes(sc, rng)))
    for name, mixed, make in cases:
        tot = np.zeros(3)
        for sd in range(seeds):
            sc = scene(N, 300 + sd, mixed=mixed)
            tot += audit_regions(sc, make(sc))
        print(f"{name:42s} | {tot[0] / tot[2]:9.0%} | {tot[1] / tot[2]:7.0%}   ({int(tot[2])} object regions)")


def _paired_z(a, b):
    d = a - b
    return d.mean() / (d.std(ddof=1) / np.sqrt(d.size))


def _solve_w(sc, Gamma, fit):
    valid = np.ones((H, W))
    rstd = R.standardize_rgb(sc["rgb"], valid)
    Pi = np.eye(2)[sc["label"]].transpose(2, 0, 1)
    coup = R.luminance_factor(sc["rgb"], valid)
    lam_p = R.part_lambda(sc["lr"], fit, SIGMA, PRIOR)
    _, w, _ = R.coupled_routing(Pi, rstd, Gamma, coup, sc["lr"], fit, s, 1e-3, lam_p, rank=RANK)
    return w, (Pi, rstd, coup)


def em_split(sc, Gamma, fit, cover, rng, iters=10, restarts=6):
    """Hard EM over copies: two coupled models, each copy assigned to the one that explains its fitting cells.

    Uses fitting cells only (the test fold is never touched)."""
    n, M = cover.shape[0], Gamma.shape[0]
    best, best_e = None, np.inf
    for _ in range(restarts):
        group = rng.random(n) < 0.5
        for _ in range(iters):
            hmap = np.isin(sc["inst"], np.nonzero(group)[0])
            split = np.concatenate([Gamma * ~hmap[None], Gamma * hmap[None]], 0)
            w, (Pi, rstd, coup) = _solve_w(sc, split, fit)
            e = []
            for g in (0, 1):                                            # every copy under each group's detail
                wg = np.concatenate([w[:8], w[8 + g * M:8 + (g + 1) * M]], 0)
                r = ((R.pool(R.coupled_apply(Pi, rstd, Gamma, coup, wg), s) - sc["lr"]) ** 2).sum(0) * fit
                e.append(np.einsum("nhw,hw->n", cover, r))
            new = e[1] < e[0]
            if (new == group).all():
                break
            group = new
        total = np.minimum(e[0], e[1]).sum()
        if total < best_e:
            best, best_e = group.copy(), total
    return best


def split_test(sc, Gamma, alpha=0.05, seed=0):
    """Data-driven Q3 test without knowing the kinds.

    For each fold t (4 rotations): propose a two-way split of the copies by hard EM on the cells outside t, then
    fit the merged and the split coupling without t and record both residuals on t. The proposal never sees the
    cells it is tested on, so with identical copies the split cannot win (it only loses a little to its extra
    parameters). One-sided paired t over all cross-fitted cells.
    """
    from math import erf, sqrt
    h, w = H // s, W // s
    rng = np.random.default_rng(seed)
    folds = [R.fold_mask(h, w, 2, f) for f in range(4)]
    n = sc["n"]
    cover = np.stack([R.pool((sc["inst"] == i)[None].astype(float), s)[0] for i in range(n)])
    touched = cover.sum(0) > 0
    kinds = np.array([sc["ctype"][sc["inst"] == i][0] for i in range(n)])
    diffs, accs = [], []
    for tf in range(4):
        test = folds[tf]
        group = em_split(sc, Gamma, 1 - test, cover, rng)
        accs.append(max((group == (kinds == 1)).mean(), (group != (kinds == 1)).mean()))
        hmap = np.isin(sc["inst"], np.nonzero(group)[0])
        split = np.concatenate([Gamma * ~hmap[None], Gamma * hmap[None]], 0)
        res = {}
        for name, G in (("merged", Gamma), ("split", split)):
            xt, _ = solve(sc, G, fit=1 - test)
            res[name] = (((R.pool(xt, s) - sc["lr"]) ** 2).sum(0) / (SIGMA ** 2 * C))[(test > 0.5) & touched]
        diffs.append(res["merged"] - res["split"])
    d = np.concatenate(diffs)
    z = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
    return z, float(np.mean(accs)), 0.5 * (1 - erf(z / sqrt(2))) <= alpha


def q3_split(seeds=8, N=48):
    print(f"Claim (3) without knowing the kinds: data-driven split test ({N} copies; one-sided alpha 0.05)")
    print(f"{'scene':28s} | {'z':>6} | {'exposed':>8} | {'split = true kinds':>18}")
    for mixed in (False, True):
        rows = []
        for sd in range(seeds):
            sc = scene(N, 200 + sd, mixed=mixed)
            rows.append(split_test(sc, vfm_codes(sc)))
        z, acc, hit = (np.array([r[i] for r in rows], float) for i in range(3))
        label = "all copies identical" if not mixed else "two hidden variants"
        print(f"{label:28s} | {z.mean():6.1f} | {hit.mean():8.0%} | {acc.mean() if mixed else float('nan'):18.0%}")


if __name__ == "__main__":
    table()
    print()
    q3()
    print()
    q3_audit()
    if "--split" in sys.argv:
        print()
        q3_split()
