"""End-to-end SIEVE-3 forward passes (coupled super-resolution fusion).

The VFM proposes couplings, the measurements solve every spectral value, the audit tests the couplings:
  x_c   = coupled routing: material laws + part detail shared by copies, all weights solved from H
  x_m   = material level (coupling off, beta = 0)                                  -> fallback of the audit
  x_cor = x_c + local RGB-guided regression of the LR residual (H - A x_c)        -> observable correction
  x     = x_c + (1 - g) * (x_cor - x_c)                                           (g: correction gate)
  mu    = beta * sum_m Gamma_m measured_m: how much of the coupled detail the measurements pin down

train_forward : differentiable, whole crop.
infer         : no-grad, multi-pass tiled engine (global sufficient statistics -> one solve -> per-tile
                outputs); the whole image is the single-tile special case, so tiled and untiled results agree.
Every quantity that predicts a non-fitting LR cell depends only on RGB and fitting-cell measurements.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from .baselines import colour_centroids, colour_codes
from .coupled import (coupled_apply, coupled_design, coupled_routing, lr_gram64, measured_map, normal_equations64,
                      part_lambda, solve_coupled)
from .evidence import fit_features, kappa_cells
from .ops import block_valid, fill_from_fit, lcm, luminance_factor, pool, standardize_rgb, upsample_window
from .routing import identity_apply, local_apply, local_apply_window, local_coefficients
from .spectral import spectral_tokens


def _autocast(device_type: str, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


class Fusion:
    def __init__(self, net, cfg):
        self.net = net
        self.cfg = cfg

    def _local(self, rstd, residual_lr, fit, s: int):
        r = self.cfg.routing
        return local_coefficients(pool(rstd, s), residual_lr, fit, r.local_radius, r.local_eps, r.local_smooth,
                                  r.local_min_count)

    @property
    def colour(self) -> bool:
        m = self.cfg.model
        return m.parts > 0 and m.part_source == "color"

    def _parts(self, u, rgb, centroids):
        if self.cfg.model.parts <= 0:
            return None
        if self.colour:
            return colour_codes(rgb, centroids)
        return self.net.part_codes(u)

    @staticmethod
    def _beta(beta_logit, like):
        return torch.ones_like(like[:, :1]) if beta_logit is None else torch.sigmoid(beta_logit)

    # ------------------------------------------------------------------ training
    def train_forward(self, rgb, lr, valid_hr, fit, s: int, sigma, amp: bool = False):
        m, r = self.cfg.model, self.cfg.routing
        H, W = rgb.shape[-2:]
        h, w = lr.shape[-2:]
        rgb, lr32 = rgb.float(), lr.float()
        valid_lr = block_valid(valid_hr, s)
        fit = fit * valid_lr
        rstd = standardize_rgb(rgb, valid_hr)
        z = spectral_tokens(lr32, fit, m.spectral_dims)
        with _autocast(rgb.device.type, amp):
            feats = self.net.pixels(rgb, rstd, z, (h, w))
        keys, proto = self.net.slot_attention(feats["f"].float(), z, fit, valid_lr)
        z_hr = F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)
        Pi = self.net.assign(feats["e"], keys, proto, z_hr)
        centroids = colour_centroids(rgb, valid_hr, m.color_codes) if self.colour else None
        Gamma = self._parts(feats["u"], rgb, centroids)
        beta = self._beta(feats["beta_logit"], rgb)
        coup = beta * luminance_factor(rgb, valid_hr, r.shade_lo, r.shade_hi)
        cr = coupled_routing(Pi, rstd, Gamma, coup, lr32, fit, s, r, sigma)
        x_c = cr["x_c"]
        p_c = pool(x_c, s)
        a, Bc = self._local(rstd, lr32 - p_c, fit, s)
        correction = local_apply(a, Bc, rstd)
        x_cor = x_c + correction
        evidence = fit_features(lr32, p_c.detach(), pool(x_cor, s).detach(), fit, sigma)
        evidence = F.interpolate(evidence, size=(H, W), mode="bilinear", align_corners=False)  # = upsample_window
        mu = measured_map(Gamma, beta, cr["measured"])
        g = self.net.gate(feats["e"], Pi, evidence, mu.detach())
        x = x_c + (1.0 - g) * correction
        return {"x": x, "x_c": x_c, "x_m": cr["x_m"], "x_cor": x_cor, "g": g, "beta": beta, "mu": mu, "Pi": Pi,
                "Gamma": Gamma, "e": feats["e"], "u": feats["u"], "w": cr["w"], "measured": cr["measured"]}

    # ------------------------------------------------------------------ inference engine
    @staticmethod
    def tiles(H: int, W: int, s: int, tile: int):
        if tile <= 0 or (tile >= H and tile >= W):
            return [(0, H, 0, W)]
        step = max(lcm(s, 8), tile - tile % lcm(s, 8))
        return [(y, min(y + step, H), x, min(x + step, W)) for y in range(0, H, step) for x in range(0, W, step)]

    @torch.no_grad()
    def infer(self, rgb, lr, valid_hr, fit, s: int, sigma, tile: int = 0, amp: bool = False, out_device=None,
              keep_decoupled: bool = False, hr_outputs: bool = True, halo: int = 8):
        """Single image (batch 1). Returns HR outputs, maps and LR predictions for the audit.

        keep_decoupled: also build x_dcor = x_m + observable correction (fallback where the audit refutes coupling).
        hr_outputs=False: LR predictions and kappa cells only (audit folds) -- no full-resolution cubes are stored.
        """
        assert rgb.shape[0] == 1, "infer processes one image at a time"
        m, r = self.cfg.model, self.cfg.routing
        net = self.net
        dev = rgb.device
        out_device = out_device or dev
        H, W = rgb.shape[-2:]
        C, h, w = lr.shape[1:]
        rgb, lr = rgb.float(), lr.float()
        valid_lr = block_valid(valid_hr, s)
        fit = fit * valid_lr
        rstd = standardize_rgb(rgb, valid_hr)
        z = spectral_tokens(lr, fit, m.spectral_dims)
        with _autocast(dev.type, amp):
            feats = net.pixels(rgb, rstd, z, (h, w))
        e = feats["e"].float()
        u = None if feats["u"] is None else feats["u"].float()
        beta_logit = None if feats["beta_logit"] is None else feats["beta_logit"].float()
        keys, proto = net.slot_attention(feats["f"].float(), z, fit, valid_lr)
        centroids = colour_centroids(rgb, valid_hr, m.color_codes) if self.colour else None
        shade = luminance_factor(rgb, valid_hr, r.shade_lo, r.shade_hi)
        n_mat = 4 * keys.shape[1]
        tiles = self.tiles(H, W, s, tile)
        cache = {}

        def maps(y0, y1, x0, x1):
            """Material slots, part codes and coupling of an HR window (all per-pixel given the features)."""
            key = (y0, y1, x0, x1)
            if key in cache:
                return cache[key]
            win = (..., slice(y0, y1), slice(x0, x1))
            Pi = net.assign(e[win], keys, proto, upsample_window(z, (H, W), y0, y1, x0, x1))
            Gamma = self._parts(None if u is None else u[win], rgb[win], centroids)
            beta = self._beta(None if beta_logit is None else beta_logit[win], e[win])
            out = {"Pi": Pi, "Gamma": Gamma, "beta": beta, "coup": beta * shade[win], "rstd": rstd[win]}
            if len(tiles) == 1:
                cache[key] = out
            return out

        def lr_slice(y0, y1, x0, x1):
            return (..., slice(y0 // s, y1 // s), slice(x0 // s, x1 // s))

        # pass 1: global sufficient statistics of the coupled solve
        G = b = None
        for t in tiles:
            mp = maps(*t)
            D = coupled_design(mp["Pi"], mp["rstd"], mp["Gamma"], mp["coup"], s)
            Gt, bt = normal_equations64(D, lr[lr_slice(*t)], fit[lr_slice(*t)])
            G = Gt if G is None else G + Gt
            b = bt if b is None else b + bt
        has_parts = G.shape[-1] > n_mat
        YtY = lr_gram64(lr, fit) if (has_parts and r.part_rank > 0) else None
        lam = part_lambda(lr, fit, sigma, r.part_prior)
        w64, w0_64, measured = solve_coupled(G, b, n_mat, r.ridge, lam, r.ridge_part, YtY, r.part_rank)
        wc, w0 = w64.float(), w0_64.float()

        # pass 2: LR predictions of the coupled, material-level and HSI-only candidates
        lr_fill = fill_from_fit(lr, fit)
        p_c, p_m, p_hsi = torch.zeros_like(lr), torch.zeros_like(lr), torch.zeros_like(lr)
        for t in tiles:
            mp, sl = maps(*t), lr_slice(*t)
            p_c[sl] = pool(coupled_apply(mp["Pi"], mp["rstd"], mp["Gamma"], mp["coup"], wc), s)
            p_m[sl] = pool(identity_apply(mp["Pi"], mp["rstd"], w0), s)
            p_hsi[sl] = pool(upsample_window(lr_fill, (H, W), *t), s)

        # observable corrections: local regressions of the LR residuals on fitting cells
        keep_decoupled = keep_decoupled and hr_outputs
        a_c, B_c = self._local(rstd, lr - p_c, fit, s)
        a_m, B_m = self._local(rstd, lr - p_m, fit, s) if keep_decoupled else (None, None)

        # pass 3: LR prediction of the corrected candidate (evidence for the gate)
        p_cor = p_c.clone()
        for t in tiles:
            sl, rs = lr_slice(*t), rstd[..., t[0]:t[1], t[2]:t[3]]
            p_cor[sl] = p_cor[sl] + pool(local_apply_window(a_c, B_c, rs, (H, W), *t), s)
        evidence = fit_features(lr, p_c, p_cor, fit, sigma)

        # pass 4: gates and HR outputs (the halo makes the 3x3 gate convolutions exact at tile borders)
        def hr(channels, keep=True):
            return torch.empty(1, channels, H, W, device=out_device) if (hr_outputs and keep) else None

        x_out, x_cor, g_out, beta_out, mu_out = hr(C), hr(C), hr(1), hr(1), hr(1)
        x_dcor = hr(C, keep_decoupled)
        p_x = torch.zeros_like(lr)
        cell_obs, cell_tot = torch.zeros(1, 1, h, w, device=dev), torch.zeros(1, 1, h, w, device=dev)
        coup_obs, coup_tot = torch.zeros(1, 1, h, w, device=dev), torch.zeros(1, 1, h, w, device=dev)
        for (y0, y1, x0, x1) in tiles:
            Y0, Y1, X0, X1 = max(0, y0 - halo), min(H, y1 + halo), max(0, x0 - halo), min(W, x1 + halo)
            mh = maps(Y0, Y1, X0, X1)
            mu_h = measured_map(mh["Gamma"], mh["beta"], measured)
            ev_h = upsample_window(evidence, (H, W), Y0, Y1, X0, X1)
            inner = (..., slice(y0 - Y0, y1 - Y0), slice(x0 - X0, x1 - X0))
            g = net.gate(e[..., Y0:Y1, X0:X1], mh["Pi"], ev_h, mu_h)[inner]
            Pi = mh["Pi"][inner]
            Gamma = None if mh["Gamma"] is None else mh["Gamma"][inner]
            beta, coup, mu = mh["beta"][inner], mh["coup"][inner], mu_h[inner]
            rs = rstd[..., y0:y1, x0:x1]
            xc = coupled_apply(Pi, rs, Gamma, coup, wc)
            correction = local_apply_window(a_c, B_c, rs, (H, W), y0, y1, x0, x1)
            xcor = xc + correction
            xg = xc + (1.0 - g) * correction
            xm = identity_apply(Pi, rs, w0)
            sl = lr_slice(y0, y1, x0, x1)
            p_x[sl] = pool(xg, s)
            cell_obs[sl], cell_tot[sl] = kappa_cells(xg, xcor, s, valid_lr[sl])
            coup_obs[sl], coup_tot[sl] = kappa_cells(xc, xm, s, valid_lr[sl])
            if not hr_outputs:
                continue
            x_out[..., y0:y1, x0:x1] = xg.to(out_device)
            x_cor[..., y0:y1, x0:x1] = xcor.to(out_device)
            g_out[..., y0:y1, x0:x1] = g.to(out_device)
            beta_out[..., y0:y1, x0:x1] = beta.to(out_device)
            mu_out[..., y0:y1, x0:x1] = mu.to(out_device)
            if keep_decoupled:
                x_dcor[..., y0:y1, x0:x1] = (xm + local_apply_window(a_m, B_m, rs, (H, W), y0, y1, x0, x1)).to(out_device)
        return {"x": x_out, "x_cor": x_cor, "x_dcor": x_dcor, "g": g_out, "beta": beta_out, "mu": mu_out,
                "p_x": p_x, "p_c": p_c, "p_m": p_m, "p_cor": p_cor, "p_hsi": p_hsi,
                "cell_obs": cell_obs, "cell_tot": cell_tot, "coup_obs": coup_obs, "coup_tot": coup_tot,
                "w": wc, "w0": w0, "measured": measured, "valid_lr": valid_lr, "has_parts": has_parts}

    @torch.no_grad()
    def hsi_only(self, lr, fit, size, out_device=None):
        x = F.interpolate(fill_from_fit(lr.float(), fit), size=size, mode="bilinear", align_corners=False)
        return x.to(out_device or lr.device)
