"""PyTorch checks: cross-check against the NumPy reference, model-level invariances, smoke training.

Run first on the target machine (a few minutes on a 4090):  python tests/test_torch.py
With SIEVE_HF_PATH=<DINOv2/DINOv3-small folder or HF id> the ViT path (LoRA + spectral-twin hook) is tested too.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
import reference_numpy as R  # noqa: E402
from test_reference_numpy import copies_scene  # noqa: E402
from sieve3 import audit  # noqa: E402
from sieve3.baselines import colour_centroids, colour_codes  # noqa: E402
from sieve3.config import Config, RoutingConfig  # noqa: E402
from sieve3.coupled import coupled_design, coupled_routing, part_lambda  # noqa: E402
from sieve3.evidence import combine_folds, region_tests  # noqa: E402
from sieve3.losses import objective  # noqa: E402
from sieve3.model import CoupledNet  # noqa: E402
from sieve3.ops import (block_valid, fill_from_fit, fold_mask, luminance_factor, pool, standardize_rgb,  # noqa: E402
                        upsample_window)
from sieve3.pipeline import Fusion  # noqa: E402
from sieve3.routing import identity_routing, local_apply, local_apply_window, local_coefficients, solve3  # noqa: E402
from sieve3.spectral import spectral_tokens  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEV)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        raise AssertionError(name)


def rel(a, b):
    a, b = torch.as_tensor(a).double().cpu(), torch.as_tensor(b).double().cpu()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


# ----------------------------------------------------------------------------- against the NumPy reference
def test_sieve2_parts_against_reference():
    rng = np.random.default_rng(0)
    s, C, h, w, K = 4, 10, 12, 10, 5
    H, W = h * s, w * s
    Pi = rng.random((K, H, W))
    Pi /= Pi.sum(0, keepdims=True)
    rgb = rng.random((3, H, W))
    valid = np.ones((H, W))
    lr = rng.random((C, h, w))
    fit = 1 - R.fold_mask(h, w, 2, 1)
    rstd_np = R.standardize_rgb(rgb, valid)
    rstd = standardize_rgb(t(rgb)[None], t(valid)[None, None])
    check("standardize_rgb", torch.allclose(rstd[0], t(rstd_np), atol=1e-5))
    x_np, _ = R.identity_routing(Pi, rstd_np, lr, fit, s, 1e-3)
    x_t, _, _ = identity_routing(t(Pi)[None], rstd, t(lr)[None], t(fit)[None, None], s, 1e-3)
    check("identity routing = reference", rel(x_t[0], x_np) < 1e-4, f"rel {rel(x_t[0], x_np):.1e}")
    A3 = torch.randn(500, 3, 3, device=DEV, dtype=torch.float64)
    A3 = A3 @ A3.transpose(1, 2) + 1e-2 * torch.eye(3, device=DEV, dtype=torch.float64)
    B3 = torch.randn(500, 3, 7, device=DEV, dtype=torch.float64)
    check("closed-form 3x3 solve = linalg.solve", rel(solve3(A3, B3), torch.linalg.solve(A3, B3)) < 1e-9)
    a_np, B_np = R.local_coefficients(R.pool(rstd_np, s), lr, fit)
    a_t, B_t = local_coefficients(pool(rstd, s), t(lr)[None], t(fit)[None, None])
    check("local coefficients = reference", max(rel(a_t[0], a_np), rel(B_t[0], B_np)) < 1e-4)
    l_t = local_apply(a_t, B_t, rstd)
    check("local apply = reference", rel(l_t[0], R.local_apply(a_np, B_np, rstd_np)) < 1e-4)
    win = local_apply_window(a_t, B_t, rstd[..., 8:24, 4:36], (H, W), 8, 24, 4, 36)
    check("windowed local apply = full", float((win - l_t[..., 8:24, 4:36]).abs().max()) < 1e-5)
    up_full = F.interpolate(t(lr)[None], size=(H, W), mode="bilinear", align_corners=False)
    up_win = upsample_window(t(lr)[None], (H, W), 4, 36, 8, 40)
    check("upsample_window = F.interpolate", float((up_full[..., 4:36, 8:40] - up_win).abs().max()) < 1e-5)
    check("fill_from_fit = reference",
          torch.allclose(fill_from_fit(t(lr)[None], t(fit)[None, None])[0], t(R.fill_from_fit(lr, fit)), atol=1e-5))
    masks_np = [R.fold_mask(h, w, 2, f) for f in range(4)]
    pa, pb = lr + 0.01 * rng.standard_normal(lr.shape), lr + 0.02 * rng.standard_normal(lr.shape)
    lrn = lr + 0.01 * rng.standard_normal(lr.shape)
    Z_np = R.region_tests(pa, pb, lrn, np.ones((h, w)), masks_np, 0.01, 4)
    Z_t = region_tests(t(pa)[None], t(pb)[None], t(lrn)[None], torch.ones(1, 1, h, w, device=DEV),
                       [fold_mask(h, w, 2, f, DEV) for f in range(4)], 0.01, 4)
    check("region tests = reference", np.allclose(Z_t[:, 0, 0].cpu().numpy(), Z_np, equal_nan=True, atol=1e-3))
    pp, _, _ = R.combine_folds(Z_np)
    pp_t, _, _, _ = combine_folds(Z_t)
    check("Bonferroni = reference", np.allclose(pp_t[0, 0].cpu().numpy(), pp, atol=1e-5))


def test_coupled_against_reference():
    rng = np.random.default_rng(1)
    s, C, h, w, K, M = 4, 12, 16, 14, 3, 9
    H, W = h * s, w * s
    Pi = rng.random((K, H, W))
    Pi /= Pi.sum(0, keepdims=True)
    Gamma = rng.random((M, H, W)) ** 4
    Gamma[-1] = 0.0                                                     # an unused code
    Gamma /= Gamma.sum(0, keepdims=True)
    rgb = 0.2 + rng.random((3, H, W))
    valid = np.ones((H, W))
    beta = rng.random((1, H, W))
    lr = rng.random((C, h, w))
    fit = 1 - R.fold_mask(h, w, 2, 1)
    rstd_np = R.standardize_rgb(rgb, valid)
    coup_np = beta * R.luminance_factor(rgb, valid)
    lam = R.part_lambda(lr, fit, 0.01, 0.15)
    rstd = standardize_rgb(t(rgb)[None], t(valid)[None, None])
    coup = t(beta)[None] * luminance_factor(t(rgb)[None], t(valid)[None, None])
    check("luminance factor = reference", rel(coup[0], coup_np) < 1e-5)
    lam_t = part_lambda(t(lr)[None], t(fit)[None, None], 0.01, 0.15)
    check("part lambda = reference", abs(float(lam_t[0]) - lam) / lam < 1e-5)
    D_np = R.coupled_design(Pi, rstd_np, Gamma, coup_np, s)
    D_t = coupled_design(t(Pi)[None], rstd, t(Gamma)[None], coup, s)
    check("coupled design = reference", rel(D_t[0], D_np) < 1e-5)
    for rank in (0, 4):
        ref = R.coupled_candidates(Pi, rstd_np, Gamma, coup_np, lr, fit, s, 1e-3, lam, 1e-3, rank)
        rc = RoutingConfig(part_rank=rank)
        out = coupled_routing(t(Pi)[None], rstd, t(Gamma)[None], coup, t(lr)[None], t(fit)[None, None], s, rc, 0.01)
        e1, e2 = rel(out["x_c"][0], ref["x_c"]), rel(out["x_m"][0], ref["x_m"])
        e3 = float((out["measured"][0].cpu().double() - torch.as_tensor(ref["measured"])).abs().max())
        check(f"coupled routing = reference (rank {rank})", max(e1, e2) < 1e-3 and e3 < 1e-4,
              f"x_c {e1:.1e}, x_m {e2:.1e}, measured {e3:.1e}")
    check("unused code has measured fraction 0", float(out["measured"][0, -1]) < 1e-6)


def test_coupled_recovers_copies():
    """Oracle part codes on an exactly coupled scene: the torch solve recovers the RGB-invisible detail."""
    sc = copies_scene(n_copies=40, seed=0, C=12)
    fit = torch.ones(1, 1, *sc["lr"].shape[1:], device=DEV)
    rc = RoutingConfig(ridge=1e-10, part_prior=1e6, ridge_part=1e-12, part_rank=2)
    out = coupled_routing(t(sc["Pi"])[None], t(sc["rstd"])[None], t(sc["Gamma"])[None], t(sc["coup"])[None],
                          t(sc["lr"])[None], fit, sc["s"], rc, 1e-6)
    err = float((out["x_c"][0].cpu().double() - torch.as_tensor(sc["X"])).norm() / np.linalg.norm(sc["X"]))
    err_m = float((out["x_m"][0].cpu().double() - torch.as_tensor(sc["X"])).norm() / np.linalg.norm(sc["X"]))
    check("coupled solve recovers sub-block detail from copies", err < 1e-3 and err_m > 10 * err,
          f"coupled {err:.1e} vs material level {err_m:.1e}")


def test_colour_codes():
    rgb = torch.rand(2, 3, 32, 32, device=DEV)
    valid = torch.ones(2, 1, 32, 32, device=DEV)
    cent = colour_centroids(rgb, valid, 6)
    codes = colour_codes(rgb, cent)
    check("colour codes are one-hot", bool(torch.allclose(codes.sum(1), torch.ones_like(codes[:, 0]))))
    check("colour centroids deterministic", bool(torch.allclose(cent, colour_centroids(rgb, valid, 6), atol=1e-6)))


# ----------------------------------------------------------------------------- model level
def tiny_cfg(**model):
    cfg = Config()
    cfg.model.slots = 8
    cfg.model.parts = 16
    for key, value in model.items():
        setattr(cfg.model, key, value)
    cfg.audit.region = 4
    return cfg


def synthetic_image(C=12, H=64, W=64, s=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 5, (8, 8), generator=g).repeat_interleave(H // 8, 0).repeat_interleave(W // 8, 1)
    spectra = torch.rand(5, C, generator=g)
    X = spectra[labels].permute(2, 0, 1)
    X = X * (0.7 + 0.3 * torch.rand(1, H, W, generator=g))
    srf = torch.rand(3, C, generator=g)
    srf = srf / srf.sum(1, keepdim=True)
    rgb = torch.einsum("kc,chw->khw", srf, X)
    lr = pool(X[None], s)[0] + 0.01 * torch.randn(C, H // s, W // s, generator=g)
    return X.to(DEV), rgb[None].to(DEV), lr[None].to(DEV), srf.to(DEV)


VARIANTS = [{"adapter": "xattn"}, {"adapter": "conv"}, {"adapter": "none", "part_source": "color"},
            {"parts": 0}, {"coupling_gate": False}]


def test_model_invariance_and_tiling():
    s = 4
    X, rgb, lr, _ = synthetic_image()
    valid = torch.ones(1, 1, 64, 64, device=DEV)
    vlr = block_valid(valid, s)
    check_mask = fold_mask(16, 16, 2, 2, DEV)
    fit = vlr * (1 - check_mask)
    lr2 = lr + 3.0 * check_mask * torch.randn_like(lr)
    for variant in VARIANTS:
        torch.manual_seed(0)
        cfg = tiny_cfg(**variant)
        fusion = Fusion(CoupledNet(cfg.model).to(DEV).eval(), cfg)
        tag = json.dumps(variant)
        with torch.no_grad():
            a = fusion.train_forward(rgb, lr, valid, fit, s, 0.01)
            b = fusion.train_forward(rgb, lr2, valid, fit, s, 0.01)
        diffs = {k: float((a[k] - b[k]).abs().max()) for k in ("x", "x_c", "x_m", "x_cor", "g", "beta", "mu")}
        check(f"held-out cells never enter any output {tag}", max(diffs.values()) < 1e-5, json.dumps(diffs))
        ia = fusion.infer(rgb, lr, valid, fit, s, 0.01)
        ib = fusion.infer(rgb, lr2, valid, fit, s, 0.01)
        diff = max(float((ia[k] - ib[k]).abs().max()) for k in ("p_x", "p_c", "p_m"))
        check(f"held-out cells never enter inference predictions {tag}", diff < 1e-5, f"{diff:.1e}")
        lo = fusion.infer(rgb, lr, valid, fit, s, 0.01, hr_outputs=False)
        diff = max(float((lo[k] - ia[k]).abs().max()) for k in ("p_x", "cell_obs", "coup_obs"))
        check(f"LR-only audit pass = full pass {tag}", diff < 1e-6 and lo["x"] is None, f"{diff:.1e}")
        whole = fusion.infer(rgb, lr, valid, vlr, s, 0.01, tile=0, keep_decoupled=True)
        tiled = fusion.infer(rgb, lr, valid, vlr, s, 0.01, tile=32, keep_decoupled=True)
        diff = max(float((whole[k] - tiled[k]).abs().max()) for k in ("x", "x_cor", "x_dcor", "mu", "g"))
        check(f"tiled multi-pass inference = whole image {tag}", diff < 1e-4, f"{diff:.1e}")
        with torch.no_grad():
            tr = fusion.train_forward(rgb, lr, valid, vlr, s, 0.01)
        diff = float((tr["x"] - whole["x"]).abs().max())
        check(f"training and inference paths agree {tag}", diff < 1e-3, f"{diff:.1e}")
        mu = whole["mu"]
        check(f"measured map in [0,1] {tag}", bool((mu >= 0).all() and (mu <= 1 + 1e-6).all()))


def test_gradients():
    s = 4
    X, rgb, lr, _ = synthetic_image()
    rgb, lr, X = rgb.repeat(2, 1, 1, 1), lr.repeat(2, 1, 1, 1), X[None].repeat(2, 1, 1, 1)
    valid = torch.ones(2, 1, 64, 64, device=DEV)
    check_mask = fold_mask(16, 16, 2, 0, DEV).repeat(2, 1, 1, 1)
    teacher = torch.randn(2, 16, 64, 64, device=DEV)
    for variant in ({"adapter": "xattn"}, {"adapter": "conv"}):
        torch.manual_seed(0)
        cfg = tiny_cfg(**variant)
        net = CoupledNet(cfg.model).to(DEV).train()
        fusion = Fusion(net, cfg)
        out = fusion.train_forward(rgb, lr, valid, 1 - check_mask, s, 0.01, amp=DEV.type == "cuda")
        loss, terms = objective(out, lr, X, valid, check_mask, s, 0.01, teacher, cfg.train)
        loss.backward()
        bad = [n for n, p in net.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        check(f"finite gradients for all trainable parameters {json.dumps(variant)}", not bad, ", ".join(bad[:6]))
        print("   loss terms:", json.dumps({k: round(v, 4) for k, v in terms.items()}))


def test_audit_runs():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    fusion = Fusion(CoupledNet(cfg.model).to(DEV).eval(), cfg)
    X, rgb, lr, srf = synthetic_image()
    valid = torch.ones(1, 1, 64, 64, device=DEV)
    for mode in ("fast", "default", "strict"):
        res = audit.run(fusion, rgb, lr, valid, 4, 0.01, srf=None, mode=mode)
        ok = bool(torch.isfinite(res["x"]).all()) and res["mu"] is not None
        check(f"audit mode={mode} finite output and maps", ok, json.dumps(res["report"]))
    res = audit.run(fusion, rgb, lr, valid, 4, "auto", srf=None, mode="default")
    check("audit with estimated sigma", bool(torch.isfinite(res["x"]).all()))
    res = audit.run(fusion, rgb, lr, valid, 4, 0.01, srf=srf.cpu().numpy(), mode="default")
    check("scene-level compatibility on consistent data", res["report"].get("compat_qbar", 99) < 9.0,
          f"qbar={res['report'].get('compat_qbar')}")


def test_hf_backbone():
    path = os.environ.get("SIEVE_HF_PATH")
    if not path:
        print("SKIP hf backbone (set SIEVE_HF_PATH to a DINOv2/DINOv3-small folder or HF id)")
        return
    from sieve3.lora import LoRALinear
    torch.manual_seed(0)
    cfg = tiny_cfg(backbone="hf", hf_path=path, adapter="xattn", lora_blocks=2, lora_rank=4)
    net = CoupledNet(cfg.model).to(DEV).train()
    n_lora = sum(isinstance(m, LoRALinear) for m in net.modules())
    vit_trainable = [n for n, p in net.named_parameters() if p.requires_grad and n.startswith("backbone.vit")]
    check("LoRA wraps the adapted blocks; only LoRA factors of the ViT train",
          n_lora > 0 and all("lora_" in n for n in vit_trainable), f"{n_lora} layers")
    X, rgb, lr, _ = synthetic_image()
    s = 4
    valid = torch.ones(1, 1, 64, 64, device=DEV)
    fit = block_valid(valid, s) * (1 - fold_mask(16, 16, 2, 0, DEV))
    rstd = standardize_rgb(rgb, valid)
    torch.nn.init.normal_(net.backbone.adapter.out.weight, std=0.05)
    with torch.no_grad():
        z1 = spectral_tokens(lr, fit, cfg.model.spectral_dims)
        e1 = net.pixels(rgb, rstd, z1, (16, 16))["e"]
        e2 = net.pixels(rgb, rstd, torch.randn_like(z1), (16, 16))["e"]
    check("the ViT reads the HSI (spectral-twin hook active)", float((e1 - e2).abs().max()) > 1e-4)
    net.eval()
    fusion = Fusion(net, cfg)
    lr2 = lr + 3.0 * (1 - fit) * torch.randn_like(lr)
    with torch.no_grad():
        a = fusion.train_forward(rgb, lr, valid, fit, s, 0.01)
        b = fusion.train_forward(rgb, lr2, valid, fit, s, 0.01)
    check("hf: held-out cells never enter the outputs", float((a["x"] - b["x"]).abs().max()) < 1e-4)
    net.train()
    out = fusion.train_forward(rgb, lr, valid, fit, s, 0.01, amp=DEV.type == "cuda")
    loss, _ = objective(out, lr, X[None], valid, block_valid(valid, s) - fit, s, 0.01, None, cfg.train)
    loss.backward()
    grads = [p.grad for n, p in net.named_parameters() if "lora_" in n]
    check("hf: LoRA factors receive finite gradients", all(g is not None and torch.isfinite(g).all() for g in grads))


def test_smoke_pipeline():
    """Synthetic manifest -> stage A (tiny backbone) -> stage B -> audited evaluation."""
    import distill_edge
    import evaluate
    import train
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = []
        srf = None
        for i in range(8):
            X, rgb, lr, srf_t = synthetic_image(seed=i)
            srf = srf_t.cpu().numpy().tolist()
            path = tmp / f"img{i}.npy"
            np.save(path, X.permute(1, 2, 0).cpu().numpy())
            rows.append({"id": f"img{i}", "split": "train" if i < 6 else "test", "reference": str(path)})
            (tmp / "teacher" / "SYN").mkdir(parents=True, exist_ok=True)
            np.save(tmp / "teacher" / "SYN" / f"img{i}.npy", np.random.rand(64, 8, 8).astype(np.float16))
        (tmp / "protocol.json").write_text(json.dumps({"observation": {"scale": 4, "noise_sigma": 0.01, "srf": srf}}))
        (tmp / "manifest.json").write_text(json.dumps({"protocol": str(tmp / "protocol.json"), "samples": rows}))
        (tmp / "panels.json").write_text(json.dumps({"panels": [{"name": "SYN", "manifest": str(tmp / "manifest.json")}]}))
        common = {"train": {"batch": 2, "crop": 32, "log_every": 10, "warmup": 5, "time_budget_hours": 0.5,
                            "budget_probe": 5}, "audit": {"region": 4}}
        (tmp / "a.json").write_text(json.dumps({"model": {"backbone": "tiny", "slots": 8, "parts": 16}, **common}))
        (tmp / "b.json").write_text(json.dumps({"model": {"backbone": "tiny", "adapter": "conv", "slots": 8,
                                                          "parts": 16}, **common}))
        train.main(train.parser().parse_args(
            ["--config", str(tmp / "a.json"), "--panels", str(tmp / "panels.json"), "--train-panels", "SYN",
             "--teacher", str(tmp / "teacher"), "--out", str(tmp / "A"), "--iters", "30", "--seed", "0"]))
        check("stage A produced a checkpoint", (tmp / "A" / "model.pt").exists())
        distill_edge.main(distill_edge.parser().parse_args(
            ["--config", str(tmp / "b.json"), "--stageA", str(tmp / "A" / "model.pt"), "--panels",
             str(tmp / "panels.json"), "--train-panels", "SYN", "--out", str(tmp / "B"), "--iters", "20"]))
        check("stage B produced a checkpoint", (tmp / "B" / "model.pt").exists())
        for stage in ("A", "B"):
            summary = evaluate.main(evaluate.parser().parse_args(
                ["--ckpt", str(tmp / stage / "model.pt"), "--panels", str(tmp / "panels.json"), "--eval-panels", "SYN",
                 "--split", "test", "--mode", "default", "--tile", "32", "--save-pred", "--out", str(tmp / f"eval{stage}")]))
            check(f"stage {stage}: audited evaluation finite", np.isfinite(summary["SYN"]["rmse"]),
                  json.dumps(summary["SYN"]))


if __name__ == "__main__":
    print(f"device: {DEV}, torch {torch.__version__}")
    test_sieve2_parts_against_reference()
    test_coupled_against_reference()
    test_coupled_recovers_copies()
    test_colour_codes()
    test_model_invariance_and_tiling()
    test_gradients()
    test_audit_runs()
    test_hf_backbone()
    test_smoke_pipeline()
    print("ALL TORCH TESTS PASSED")
