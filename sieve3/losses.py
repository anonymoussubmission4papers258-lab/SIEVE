"""Training objectives.

Stage A (joint RGB+HSI VFM):
  physical   w_ho * E_ho(x_c) + w_ho_material * E_ho(x_m) + w_gate * E_ho(g-mix of stop-gradient candidates)
             + w_hr * [ MSE(x) + w_hr_coupled * MSE(x_c) ] / sigma^2
  prior      w_rel  * KL(VFM-L affinities || material-embedding affinities)         (SIEVE-2)
             w_part * cannot-link: distant pixels may share a part code only if the VFM-L sees the same kind
             w_usage * (1 - entropy of the part-code usage of coupled pixels)
E_ho is the held-out LR evidence (sigma^2-normalised; 1 = noise level). The network outputs couplings only:
spectra are always solved from H, so HR supervision shapes *who shares with whom*, never spectral values.

Stage B (edge student): the same physical terms + distillation of the stage-A couplings
  w_kd_part * KL(Gamma_A || Gamma_B) (shared codebook) + w_kd_material * co-assignment BCE of Pi on pixel pairs
  + w_kd_gates * (MSE(beta) + MSE(g)).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .evidence import heldout_loss


def _sigma2_mean(sigma) -> float:
    return float(torch.as_tensor(sigma, dtype=torch.float64).float().mean()) ** 2


def masked_mse(x, ref, valid):
    m = valid.float()
    return (((x - ref) ** 2) * m).sum() / (m.sum() * ref.shape[1]).clamp_min(1.0)


def relational_distill(e, teacher, valid, anchors: int = 256, targets: int = 1024, tau_s: float = 0.1,
                       tau_t: float = 0.07, generator=None):
    """Match the student's pixel-affinity distribution to the teacher's on random (often distant) pairs."""
    t = F.normalize(teacher.float(), dim=1)
    losses = []
    for b in range(e.shape[0]):
        idx = valid[b, 0].flatten().nonzero().squeeze(1)
        if idx.numel() < 16:
            continue
        pick = idx[torch.randint(0, idx.numel(), (anchors + targets,), generator=generator).to(idx.device)]
        ia, it = pick[:anchors], pick[anchors:]
        es, ts = e[b].flatten(1).float(), t[b].flatten(1)
        logit_s = es[:, ia].T @ es[:, it] / tau_s
        logit_t = ts[:, ia].T @ ts[:, it] / tau_t
        p_t = logit_t.softmax(-1)
        losses.append((p_t * (p_t.clamp_min(1e-8).log() - logit_s.log_softmax(-1))).sum(-1).mean())
    if not losses:
        return e.sum() * 0.0
    return torch.stack(losses).mean()


def _pairs(valid_b, pairs: int, width: int, generator):
    idx = valid_b.flatten().nonzero().squeeze(1)
    if idx.numel() < 16:
        return None
    pick = idx[torch.randint(0, idx.numel(), (2 * pairs,), generator=generator).to(idx.device)]
    ia, ib = pick[:pairs], pick[pairs:]
    return ia, ib, ia // width, ia % width, ib // width, ib % width


def part_cannot_link(Gamma, teacher, valid, pairs: int = 4096, min_dist: int = 12, lo: float = 0.2,
                     hi: float = 0.6, generator=None):
    """Distant pixels may share a part code only if the large VFM sees them as the same kind of thing.

    loss = mean over pairs at distance >= min_dist of (1 - allow) * P(coupled),  P(coupled) = Gamma_a . Gamma_b,
    allow = clip((cos_VFM-L(a, b) - lo) / (hi - lo), 0, 1). Which points of same-kind copies couple is left to
    the measurements (held-out evidence / HR terms); near pairs are left to the measurements entirely.
    """
    W = Gamma.shape[-1]
    t = F.normalize(teacher.float(), dim=1)
    losses = []
    for b in range(Gamma.shape[0]):
        sample = _pairs(valid[b, 0], pairs, W, generator)
        if sample is None:
            continue
        ia, ib, ya, xa, yb, xb = sample
        far = ((ya - yb) ** 2 + (xa - xb) ** 2) >= min_dist ** 2
        if int(far.sum()) < 8:
            continue
        ia, ib = ia[far], ib[far]
        Gb, Tb = Gamma[b].flatten(1), t[b].flatten(1)
        coupled = (Gb[:, ia] * Gb[:, ib]).sum(0)
        allow = (((Tb[:, ia] * Tb[:, ib]).sum(0) - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
        losses.append(((1.0 - allow) * coupled).mean())
    if not losses:
        return Gamma.sum() * 0.0
    return torch.stack(losses).mean()


def codebook_usage(Gamma, valid, beta=None):
    """1 - H(usage)/log M, usage = code distribution of coupled pixels over the batch (0: all codes used)."""
    M = Gamma.shape[1]
    w = valid.float() if beta is None else valid.float() * beta.detach()
    p = (Gamma * w).sum((0, 2, 3))
    p = p / p.sum().clamp_min(1e-12)
    return 1.0 - (-(p * p.clamp_min(1e-12).log()).sum()) / math.log(max(M, 2))


def objective(out, lr, ref, valid_hr, check, s, sigma, teacher, tcfg, generator=None):
    """Stage-A objective. Returns (total loss, dict of float terms)."""
    terms = {"ho_coupled": heldout_loss(out["x_c"], lr, check, s, sigma)}
    xg = out["g"] * out["x_c"].detach() + (1.0 - out["g"]) * out["x_cor"].detach()
    terms["ho_gate"] = heldout_loss(xg, lr, check, s, sigma)
    total = tcfg.w_ho * terms["ho_coupled"] + tcfg.w_gate * terms["ho_gate"]
    if tcfg.w_ho_material > 0 and out["Gamma"] is not None:
        terms["ho_material"] = heldout_loss(out["x_m"], lr, check, s, sigma)
        total = total + tcfg.w_ho_material * terms["ho_material"]
    if tcfg.w_hr > 0 and ref is not None:
        sig2 = _sigma2_mean(sigma)
        terms["hr"] = masked_mse(out["x"], ref, valid_hr) / sig2
        terms["hr_coupled"] = masked_mse(out["x_c"], ref, valid_hr) / sig2
        total = total + tcfg.w_hr * (terms["hr"] + tcfg.w_hr_coupled * terms["hr_coupled"])
    if teacher is not None and tcfg.w_rel > 0:
        terms["rel"] = relational_distill(out["e"], teacher, valid_hr, tcfg.distill_anchors, tcfg.distill_targets,
                                          tcfg.tau_student, tcfg.tau_teacher, generator)
        total = total + tcfg.w_rel * terms["rel"]
    if out["Gamma"] is not None and out["u"] is not None:          # learned (VFM) part codes only
        if teacher is not None and tcfg.w_part > 0:
            terms["part"] = part_cannot_link(out["Gamma"], teacher, valid_hr, tcfg.part_pairs, tcfg.part_min_dist,
                                             tcfg.teacher_sim_lo, tcfg.teacher_sim_hi, generator)
            total = total + tcfg.w_part * terms["part"]
        if tcfg.w_usage > 0:
            terms["usage"] = codebook_usage(out["Gamma"], valid_hr, out["beta"])
            total = total + tcfg.w_usage * terms["usage"]
    return total, {k: float(v.detach()) for k, v in terms.items()}


def co_assignment(Pi_s, Pi_t, valid, pairs: int = 4096, generator=None):
    """BCE between the probabilities that two pixels share a material slot (slots need not be aligned)."""
    W = Pi_s.shape[-1]
    losses = []
    for b in range(Pi_s.shape[0]):
        sample = _pairs(valid[b, 0], pairs, W, generator)
        if sample is None:
            continue
        ia, ib = sample[:2]
        Ps, Pt = Pi_s[b].flatten(1), Pi_t[b].flatten(1)
        cs = (Ps[:, ia] * Ps[:, ib]).sum(0).float().clamp(1e-6, 1 - 1e-6)
        ct = (Pt[:, ia] * Pt[:, ib]).sum(0).float().clamp(0.0, 1.0)
        losses.append(-(ct * cs.log() + (1.0 - ct) * (1.0 - cs).log()).mean())
    if not losses:
        return Pi_s.sum() * 0.0
    return torch.stack(losses).mean()


def distill_objective(out_s, out_t, lr, ref, valid_hr, check, s, sigma, tcfg, generator=None):
    """Stage-B objective: physical terms of the student + distillation of the stage-A couplings."""
    total, terms = objective(out_s, lr, ref, valid_hr, check, s, sigma, None, tcfg, generator)
    m = valid_hr.float()
    n = m.sum().clamp_min(1.0)
    extra = {}
    if tcfg.w_kd_part > 0 and out_s["Gamma"] is not None and out_t["Gamma"] is not None:
        Gt = out_t["Gamma"].float()
        kl = (Gt * (Gt.clamp_min(1e-8).log() - out_s["Gamma"].clamp_min(1e-8).log())).sum(1, keepdim=True)
        extra["kd_part"] = (kl * m).sum() / n
    if tcfg.w_kd_material > 0:
        extra["kd_material"] = co_assignment(out_s["Pi"], out_t["Pi"].float(), valid_hr, tcfg.kd_pairs, generator)
    if tcfg.w_kd_gates > 0:
        gates = ((out_s["g"] - out_t["g"].float()) ** 2 + (out_s["beta"] - out_t["beta"].float()) ** 2) * m
        extra["kd_gates"] = gates.sum() / n
    weights = {"kd_part": tcfg.w_kd_part, "kd_material": tcfg.w_kd_material, "kd_gates": tcfg.w_kd_gates}
    for key, value in extra.items():
        total = total + weights[key] * value
        terms[key] = float(value.detach())
    return total, terms
