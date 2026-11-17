"""Audit: the VFM proposes couplings, held-out spectral evidence tests them.

Scene level (as SIEVE-2): known SRF -> RGB/HSI compatibility; unknown SRF -> corrected fusion vs HSI-only.
Region level (R x R LR cells), two paired held-out tests on cross-fitted folds (Bonferroni over folds):
  coupling   (Q3)  coupled x_c  vs  material-level x_m.  Refuted (x_m significantly better): in this region,
                   coupling the pixels to their copies predicts measurements it never saw worse than not
                   coupling them -- e.g. copies that look alike but are measured differently. The region falls
                   back to the material level with the observable correction (x_dcor = x_m + correction: the
                   measurements correct what they observe, as for SIEVE-2 overrides). Certified: coupling to the
                   copies predicts unseen measurements better. Untestable (kappa < kappa_min): the coupling
                   changes only sub-block detail no measurement can see -> kept, labelled VFM-only.
                   (tests/demo_copies.py: correct couplings are refuted in 0% of object regions, merged
                   look-alike couplings in 31%, harmful couplings in 68%; a coupling that is worse than a split
                   but still better than none is not refuted -- see HANDOFF_ZH.md.)
  correction       gated x vs fully corrected x_cor (SIEVE-2): overridden -> x_cor.
Maps returned with every result:
  mu     measured map: beta * sum_m Gamma_m measured_m (model-relative share of the coupled detail pinned by H)
  resid  cross-fitted coupling residual per LR cell: ||A x_c - h||^2 / (sigma^2 C) at cells held out (~1 when
         the coupled model explains what it never saw; > 1 where coupled copies disagree)
Every statistic that judges a cell comes from a fold run in which that cell was held out.
"""
from __future__ import annotations

import math

import torch

from .evidence import (CERTIFIED, INCONCLUSIVE, OVERRIDDEN, SCENE_FALLBACK, UNTESTABLE, combine_folds, decide,
                       region_sum, region_tests)
from .ops import block_valid, estimate_sigma, fold_mask, pool

CODE_COUNTS = (("untestable", UNTESTABLE), ("certified", CERTIFIED), ("inconclusive", INCONCLUSIVE),
               ("overridden", OVERRIDDEN))
COUPLING_COUNTS = (("untestable", UNTESTABLE), ("certified", CERTIFIED), ("inconclusive", INCONCLUSIVE),
                   ("refuted", OVERRIDDEN))


@torch.no_grad()
def compatibility(rgb, lr, valid_lr, srf, s: int, sigma: float) -> float:
    """Mean whitened discrepancy of A R and S H (approximately chi^2_3 with mean 3 when consistent)."""
    S = torch.as_tensor(srf, dtype=torch.float32, device=lr.device)
    d = pool(rgb.float(), s) - torch.einsum("kc,bchw->bkhw", S, lr.float())
    cov = (sigma ** 2) * (S @ S.T) + 1e-10 * torch.eye(3, device=lr.device)
    q = (d * torch.einsum("ij,bjhw->bihw", torch.linalg.inv(cov), d)).sum(1, keepdim=True)
    return float((q * valid_lr).sum() / valid_lr.sum().clamp_min(1.0))


def _to_hr(code: torch.Tensor, cell: int, H: int, W: int) -> torch.Tensor:
    return code.repeat_interleave(cell, -2).repeat_interleave(cell, -1)[..., :H, :W]


def _fallback(fusion, lr, valid_lr, H, W, h, w, region, out_device, report, reason):
    x = fusion.hsi_only(lr, valid_lr, (H, W), out_device)
    code = torch.full((1, 1, math.ceil(h / region), math.ceil(w / region)), SCENE_FALLBACK, dtype=torch.long,
                      device=lr.device)
    report["scene"] = f"fallback_hsi_only({reason})"
    return {"x": x, "code": code, "code_coupling": code.clone(), "g": None, "mu": None, "beta": None,
            "resid": None, "report": report}


def _mean(x, mask):
    return float((x * mask).sum() / mask.sum().clamp_min(1.0))


@torch.no_grad()
def run(fusion, rgb, lr, valid_hr, s: int, sigma, srf=None, mode: str = "default", tile: int = 0,
        amp: bool = False, out_device=None):
    """Full audited inference for one image.

    sigma: noise std (float) or "auto" (estimated from fitting cells, separately for every fold).
    mode : 'fast' (VFM decisions only, no folds), 'default' (audited), 'strict' (audited, cross-fitted average).
    """
    acfg = fusion.cfg.audit
    H, W = rgb.shape[-2:]
    C, h, w = lr.shape[1:]
    dev = lr.device
    out_device = out_device or dev
    valid_lr = block_valid(valid_hr, s)
    auto = isinstance(sigma, str)
    if auto and sigma != "auto":
        raise ValueError("sigma must be a number or 'auto'")

    def sigma_for(fit):
        return float(estimate_sigma(lr.float(), fit)[0]) if auto else float(sigma)

    sigma_full = sigma_for(valid_lr)
    report = {"mode": mode, "sigma": sigma_full}

    # ---------------------------------------------------------------- scene level (known SRF)
    if srf is not None:
        qbar = compatibility(rgb, lr, valid_lr, srf, s, sigma_full)
        report["compat_qbar"] = qbar
        if qbar > acfg.compat_threshold:
            return _fallback(fusion, lr, valid_lr, H, W, h, w, acfg.region, out_device, report, "compatibility")

    full = fusion.infer(rgb, lr, valid_hr, valid_lr, s, sigma_full, tile=tile, amp=amp, out_device=out_device,
                        keep_decoupled=mode != "fast")
    valid_dev = valid_hr.to(out_device)
    report["mu_mean"] = _mean(full["mu"], valid_dev)
    report["beta_mean"] = _mean(full["beta"], valid_dev)
    report["gate_mean"] = _mean(full["g"], valid_dev)
    maps = {"g": full["g"], "mu": full["mu"], "beta": full["beta"]}
    if mode == "fast":
        report["scene"] = "fused(no audit)"
        return {"x": full["x"], "code": None, "code_coupling": None, "resid": None, "report": report,
                "full": full, **maps}

    period = int(round(math.sqrt(acfg.folds)))
    if period * period != acfg.folds:
        raise ValueError("audit.folds must be a perfect square")
    masks, sigmas = [], []
    pred = {key: torch.zeros_like(lr) for key in ("x", "cor", "c", "m", "hsi")}
    k_obs, k_tot = torch.zeros_like(lr[:, :1]), torch.zeros_like(lr[:, :1])
    c_obs, c_tot = torch.zeros_like(lr[:, :1]), torch.zeros_like(lr[:, :1])
    resid = torch.zeros_like(lr[:, :1])
    strict = {}
    for fold in range(acfg.folds):
        mask = fold_mask(h, w, period, fold, dev) * valid_lr
        fit = valid_lr * (1.0 - mask)
        sig = sigma_for(fit)
        out = fusion.infer(rgb, lr, valid_hr, fit, s, sig, tile=tile, amp=amp, out_device=out_device,
                           keep_decoupled=mode == "strict", hr_outputs=mode == "strict")
        held = mask > 0.5
        for key, src in (("x", "p_x"), ("cor", "p_cor"), ("c", "p_c"), ("m", "p_m"), ("hsi", "p_hsi")):
            pred[key] = torch.where(held, out[src], pred[key])
        k_obs, k_tot = torch.where(held, out["cell_obs"], k_obs), torch.where(held, out["cell_tot"], k_tot)
        c_obs, c_tot = torch.where(held, out["coup_obs"], c_obs), torch.where(held, out["coup_tot"], c_tot)
        r = ((out["p_c"] - lr.float()) ** 2).sum(1, keepdim=True) / (max(sig, 1e-12) ** 2 * C)
        resid = torch.where(held, r, resid)
        masks.append(mask)
        sigmas.append(sig)
        if mode == "strict":
            for key in ("x", "x_cor", "x_dcor"):
                strict[key] = out[key] / acfg.folds if key not in strict else strict[key] + out[key] / acfg.folds
        del out                                             # bounded memory: one fold run alive at a time

    lr32 = lr.float()

    def per_fold_tests(a, b, R):
        return torch.cat([region_tests(a, b, lr32, valid_lr, [m], sg, R=R, statistic=acfg.statistic,
                                       min_cells=acfg.min_cells) for m, sg in zip(masks, sigmas)], 0)

    # ---------------------------------------------------------------- scene level (unknown SRF)
    if srf is None:
        _, p_hsi_better, _, z_scene = combine_folds(per_fold_tests(pred["cor"], pred["hsi"], max(h, w)))
        report["scene_z_fusion_vs_hsi"] = float(z_scene.flatten()[0])
        if float(p_hsi_better.flatten()[0]) <= acfg.alpha:
            res = _fallback(fusion, lr, valid_lr, H, W, h, w, acfg.region, out_device, report, "evidence")
            res.update(maps)
            return res

    # ---------------------------------------------------------------- region level
    def kappa(obs, tot):
        o, t = region_sum(obs, acfg.region), region_sum(tot, acfg.region)
        return torch.where(t > 1e-20, o / t.clamp_min(1e-20), torch.zeros_like(t))

    p_plus, p_minus, any_ok, z_cor = combine_folds(per_fold_tests(pred["x"], pred["cor"], acfg.region))
    kappa_cor = kappa(k_obs, k_tot)
    code = decide(p_plus, p_minus, any_ok, kappa_cor, acfg.alpha, acfg.kappa_min)
    if full["has_parts"]:
        cp_plus, cp_minus, c_ok, z_coup = combine_folds(per_fold_tests(pred["c"], pred["m"], acfg.region))
        kappa_coup = kappa(c_obs, c_tot)
        code_coupling = decide(cp_plus, cp_minus, c_ok, kappa_coup, acfg.alpha, acfg.kappa_min)
    else:
        z_coup = kappa_coup = None
        code_coupling = torch.full_like(code, UNTESTABLE)

    cell = acfg.region * s
    override = _to_hr((code == OVERRIDDEN).to(out_device), cell, H, W)
    refuted = _to_hr((code_coupling == OVERRIDDEN).to(out_device), cell, H, W)
    src = strict if mode == "strict" else full
    x = torch.where(override, src["x_cor"], src["x"])
    if full["has_parts"]:
        x = torch.where(refuted, src["x_dcor"], x)
    report.update({"scene": "fused",
                   "regions_correction": {k: int((code == v).sum()) for k, v in CODE_COUNTS},
                   "regions_coupling": {k: int((code_coupling == v).sum()) for k, v in COUPLING_COUNTS},
                   "kappa_mean": float(kappa_cor.mean()),
                   "resid_mean": _mean(resid, valid_lr)})
    if kappa_coup is not None:
        report["kappa_coupling_mean"] = float(kappa_coup.mean())
    return {"x": x, "code": code, "code_coupling": code_coupling, "z": z_cor, "z_coupling": z_coup,
            "kappa": kappa_cor, "kappa_coupling": kappa_coup, "resid": resid, "report": report, "full": full,
            **maps}
